"""Step 2: quantize to int8, then compare random bit flips with a targeted bit search.

Targeted search (a simplified version of the Progressive Bit Search in
Rakin et al., "Bit-Flip Attack", ICCV 2019): at each step, compute the loss gradient
on a small batch the attacker holds, estimate for every candidate bit how much
flipping it would raise the loss (gradient x weight change), flip the best one, repeat.
"""
import copy, json, os, sys, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ResNet20, loaders, accuracy, quantize, load_q, quant_layers, flip_bit, flip_bits_global, DEV

EVAL_BATCHES = 5      # evaluate on test images 0..4999 (batches of 1000)
ATTACK_N = 128        # attacker's images, taken from test images 9000..9999 -> disjoint from evaluation
RANDOM_N = [1, 10, 100, 1000, 3000, 10000, 30000, 100000]
TRIALS = 5
TARGET_STEPS = 30
TOPK = 10             # candidate weights per layer, chosen by |gradient|


def clone_q(q): return {k: (v.clone(), s) for k, (v, s) in q.items()}


def main(part="all"):
    """part: 'random', 'targeted' or 'all'. Results are merged into results.json."""
    torch.manual_seed(0)
    res = json.load(open("results.json")) if os.path.exists("results.json") else {}
    t0 = time.time()
    _, test = loaders()
    base = ResNet20().to(DEV); base.load_state_dict(torch.load("resnet20.pt", map_location=DEV))
    fp_acc = accuracy(base, test)
    q0 = quantize(base)
    acc0 = accuracy(base, test, EVAL_BATCHES)
    names = list(q0.keys())
    total_bits = sum(qi.numel() for qi, _ in q0.values()) * 8
    print(f"float32 accuracy (10k): {fp_acc:.2f}%   int8 accuracy (eval 5k): {acc0:.2f}%   "
          f"weight bits: {total_bits:,}", flush=True)

    res.update(fp_acc=fp_acc, acc0=acc0, total_bits=total_bits, eval_images=EVAL_BATCHES * 1000,
               attack_images=ATTACK_N)

    # ---------------- random flips ----------------
    random_res = {}
    for n_flips in (RANDOM_N if part in ("random", "all") else []):
        accs = []
        for t in range(TRIALS):
            q = clone_q(q0)
            flip_bits_global(q, names, torch.randperm(total_bits)[:n_flips])   # n distinct bits
            model = copy.deepcopy(base); load_q(model, q)
            accs.append(accuracy(model, test, EVAL_BATCHES))
            print(f"  random {n_flips:6d} flips  trial {t+1}/{TRIALS}: {accs[-1]:.2f}%", flush=True)
        random_res[n_flips] = dict(mean=sum(accs) / len(accs), min=min(accs), max=max(accs), trials=accs)
        print(f"random  {n_flips:6d} flips -> mean {random_res[n_flips]['mean']:.2f}%  "
              f"(min {min(accs):.2f}, max {max(accs):.2f})  [{time.time()-t0:.0f}s]", flush=True)

    if random_res: res["random"] = {str(k): v for k, v in random_res.items()}
    json.dump(res, open("results.json", "w"), indent=2)
    if part == "random":
        print(f"saved random results to results.json  [{time.time()-t0:.0f}s]"); return

    # ---------------- targeted bit search ----------------
    x_last, y_last = list(test)[-1]                       # test images 9000..9999
    x_att, y_att = x_last[:ATTACK_N].to(DEV), y_last[:ATTACK_N].to(DEV)
    q = clone_q(q0); model = copy.deepcopy(base); load_q(model, q)
    lossf = nn.CrossEntropyLoss(); targeted = [(0, acc0)]; log = []; used = set()
    for step in range(1, TARGET_STEPS + 1):
        model.eval(); model.zero_grad()
        loss_before = lossf(model(x_att), y_att); loss_before.backward()
        best = None
        for name, m in quant_layers(model):
            qi, scale = q[name]; g = m.weight.grad.view(-1); flat = qi.view(-1)
            for idx in torch.topk(g.abs(), min(TOPK, g.numel())).indices.tolist():
                v = int(flat[idx].item())
                for bit in range(8):
                    if (name, idx, bit) in used: continue
                    u = (v & 0xFF) ^ (1 << bit); nv = u - 256 if u >= 128 else u
                    gain = g[idx].item() * (nv - v) * scale.item()   # first-order loss increase
                    if best is None or gain > best[0]:
                        best = (gain, name, idx, bit, v, nv)
        gain, name, idx, bit, v, nv = best
        used.add((name, idx, bit))
        flip_bit(q[name][0].view(-1), idx, bit); load_q(model, q)
        with torch.no_grad(): loss_after = lossf(model(x_att), y_att).item()
        acc = accuracy(model, test, EVAL_BATCHES)
        targeted.append((step, acc))
        log.append(dict(step=step, layer=name, index=idx, bit=bit, before=v, after=nv,
                        est_gain=gain, attack_loss_before=loss_before.item(), attack_loss_after=loss_after, acc=acc))
        print(f"targeted step {step:2d}: {name}[{idx}] bit {bit} ({v:4d} -> {nv:4d})  "
              f"attack loss {loss_before.item():.3f} -> {loss_after:.3f}  test acc {acc:.2f}%", flush=True)

    res.update(targeted=targeted, flips=log)
    json.dump(res, open("results.json", "w"), indent=2)
    if "random" not in res:
        print("no random results yet; run `python attack.py random` to draw the plot"); return
    random_res = {int(k): v for k, v in res["random"].items()}
    rn = sorted(random_res)

    # ---------------- plot ----------------
    c_rnd, c_tgt, ink, muted = "#2a78d6", "#eb6834", "#2b2a27", "#8a8980"
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    xr = [0] + rn
    ax.fill_between(xr, [acc0] + [random_res[n]["min"] for n in rn],
                    [acc0] + [random_res[n]["max"] for n in rn], color=c_rnd, alpha=0.15, lw=0)
    ax.plot(xr, [acc0] + [random_res[n]["mean"] for n in rn], "o-", color=c_rnd, lw=2, ms=5,
            label=f"Random bit flips (mean, min–max of {TRIALS})")
    ax.plot([s for s, _ in targeted], [a for _, a in targeted], "^-", color=c_tgt, lw=2, ms=5,
            label="Targeted bit search")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Number of flipped bits (log scale)"); ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(10, ls="--", lw=1, c=muted); ax.text(1.1, 11.5, "random guess (10%)", color=muted, fontsize=8)
    ax.annotate("Random", (rn[-1], random_res[rn[-1]]["mean"]), xytext=(-4, 10),
                textcoords="offset points", ha="right", fontsize=9, color=ink)
    ax.annotate("Targeted", (targeted[-1][0], targeted[-1][1]), xytext=(6, 6),
                textcoords="offset points", fontsize=9, color=ink)
    ax.set_title(f"ResNet-20 (int8, {total_bits:,} weight bits) on CIFAR-10", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.18))
    fig.tight_layout(); fig.savefig("bitflip.png", dpi=150)
    print(f"saved results.json and bitflip.png  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":   # required on Windows: DataLoader workers re-import this file
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
