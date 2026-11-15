"""Step 2: quantize to int8, then compare random bit flips with a targeted bit search.

Targeted search (a simplified version of the Progressive Bit Search in
Rakin et al., "Bit-Flip Attack", ICCV 2019): at each step, compute the loss gradient
on a small batch the attacker holds, estimate for every candidate bit how much
flipping it would raise the loss (gradient x weight change), flip the best one, repeat.
"""
import copy, json, random, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ResNet20, loaders, accuracy, quantize, load_q, quant_layers, flip_bit, DEV

torch.manual_seed(0); random.seed(0)
_, test = loaders()
base = ResNet20().to(DEV); base.load_state_dict(torch.load("resnet20.pt", map_location=DEV))
q0 = quantize(base)
EVAL_BATCHES = 5   # 5,000 test images per evaluation; set None for all 10,000
acc0 = accuracy(base, test, EVAL_BATCHES)
total_bits = sum(qi.numel() for qi, _ in q0.values()) * 8
print(f"int8 model accuracy: {acc0:.2f}%   weight bits: {total_bits:,}")

def clone_q(q): return {k: (v.clone(), s) for k, (v, s) in q.items()}

# ---------------- random flips ----------------
RANDOM_N = [1, 10, 100, 1000, 10000]
TRIALS = 5
names = list(q0.keys()); sizes = [q0[n][0].numel() for n in names]
random_res = {}
for n_flips in RANDOM_N:
    accs = []
    for _ in range(TRIALS):
        q = clone_q(q0)
        for _ in range(n_flips):
            name = random.choices(names, weights=sizes)[0]
            flat = q[name][0].view(-1)
            flip_bit(flat, random.randrange(flat.numel()), random.randrange(8))
        model = copy.deepcopy(base); load_q(model, q)
        accs.append(accuracy(model, test, EVAL_BATCHES))
    random_res[n_flips] = sum(accs) / len(accs)
    print(f"random  {n_flips:6d} flips -> {random_res[n_flips]:.2f}%")

# ---------------- targeted bit search ----------------
TARGET_STEPS = 30
TOPK = 10          # candidate weights per layer, chosen by |gradient|
x_att, y_att = next(iter(test)); x_att, y_att = x_att[:128].to(DEV), y_att[:128].to(DEV)  # attacker's small batch
q = clone_q(q0); model = copy.deepcopy(base); load_q(model, q)
lossf = nn.CrossEntropyLoss(); targeted = [(0, acc0)]; log = []
for step in range(1, TARGET_STEPS + 1):
    model.eval(); model.zero_grad()
    lossf(model(x_att), y_att).backward()
    best = None
    for name, m in quant_layers(model):
        qi, scale = q[name]; g = m.weight.grad.view(-1); flat = qi.view(-1)
        for idx in torch.topk(g.abs(), min(TOPK, g.numel())).indices.tolist():
            v = int(flat[idx].item())
            for bit in range(8):
                u = (v & 0xFF) ^ (1 << bit); nv = u - 256 if u >= 128 else u
                gain = g[idx].item() * (nv - v) * scale.item()   # first-order loss increase
                if best is None or gain > best[0]:
                    best = (gain, name, idx, bit, v, nv)
    _, name, idx, bit, v, nv = best
    flip_bit(q[name][0].view(-1), idx, bit); load_q(model, q)
    acc = accuracy(model, test, EVAL_BATCHES)
    targeted.append((step, acc)); log.append(dict(step=step, layer=name, index=idx, bit=bit, before=v, after=nv, acc=acc))
    print(f"targeted step {step:2d}: {name}[{idx}] bit {bit} ({v} -> {nv}) -> {acc:.2f}%")

json.dump(dict(acc0=acc0, total_bits=total_bits, random=random_res, targeted=targeted, flips=log),
          open("results.json", "w"), indent=2)

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot([0] + RANDOM_N, [acc0] + [random_res[n] for n in RANDOM_N], "o-", label="Random bit flips (avg of 5)")
ax.plot([s for s, _ in targeted], [a for _, a in targeted], "s-", label="Targeted bit search")
ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("Number of flipped bits"); ax.set_ylabel("Test accuracy (%)")
ax.axhline(10, ls="--", lw=1, c="gray"); ax.text(1.1, 11, "random guess (10%)", color="gray", fontsize=8)
ax.set_title(f"ResNet-20 (int8, {total_bits:,} weight bits) on CIFAR-10"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("bitflip.png", dpi=150)
print("saved results.json and bitflip.png")
