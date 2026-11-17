"""Step 3: how much do simple defenses raise the number of bits an attacker must flip?

Attack: Progressive Bit Search closer to Rakin et al. (ICCV 2019) than attack.py.
Each step, every layer proposes its CANDS best (weight, bit) pairs by first-order
estimate; each proposal is actually flipped, the attack-batch loss measured, and
undone. The single flip that raised the loss most is kept. The attack stops when
eval accuracy <= STOP_ACC or after the step limit.

Defenses (each is a config):
  base          no defense
  sign          sign bit (bit 7) cannot be flipped (e.g. stored with ECC/replicas)
  clamp         activations clamped per channel to the max seen on training data
  pc            model trained with the piece-wise clustering penalty (train.py --cluster)
  parity        group parity on bits 7 and 6, flagged groups zeroed. The attacker
                ignores it (same search as base); we report accuracy after recovery.
  parity-adapt  attacker knows parity covers bits 6-7 and only flips bits 0-5

    python defense.py run base sign clamp pc parity-adapt   (parity is measured inside base)
    python defense.py plot
"""
import json, os, sys, time, zlib
import torch, torch.nn as nn, torchvision
from common import ResNet20, quantize, quant_layers, DEV, MEAN, STD

EVAL_N = 5000          # eval images 0..4999, same as attack.py
ATTACK_N = 128         # attacker images drawn from 5000..9999 (disjoint from eval)
SEEDS = 5
MAX_STEPS = 300
STOP_ACC = 11.0        # "collapsed": near 10% random guess
CANDS = 5              # proposals per layer that are actually tried each step
GROUP = 64             # weights per parity group
OUT = "defense_results.json"


def cifar(train):
    ds = torchvision.datasets.CIFAR10("data", train=train, download=False)
    x = torch.tensor(ds.data).permute(0, 3, 1, 2).float() / 255
    m, s = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - m) / s).to(DEV), torch.tensor(ds.targets).to(DEV)


class Net:
    """An int8 model whose weights live in self.q (the attackable memory)."""
    def __init__(self, ckpt):
        self.model = ResNet20().to(DEV)
        self.model.load_state_dict(torch.load(ckpt, map_location=DEV))
        self.model.eval()
        self.q = quantize(self.model)
        self.mods = dict(quant_layers(self.model))
        self.names = list(self.q)
        self.hooks = []

    def sync(self, name, idx=None):
        qi, s = self.q[name]
        w = self.mods[name].weight.data
        if idx is None: w.copy_(qi.float() * s)
        else: w.view(-1)[idx] = qi.view(-1)[idx].float() * s

    def flip(self, name, idx, bit):
        u = self.q[name][0].view(-1).view(torch.uint8)
        u[idx] ^= (1 << bit)
        self.sync(name, idx)

    @torch.no_grad()
    def acc(self, x, y, bs=1000):
        ok = sum((self.model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
        return 100.0 * ok / len(x)

    def add_clamp(self, xtr):
        """Clamp the stem and every block output to the per-channel maximum seen on
        the (unaugmented) training set. min(relu(a), c) = relu(min(a, c)), so the
        stem is clamped at its BatchNorm output."""
        points = [self.model.bn] + list(self.model.layers)
        maxes = [None] * len(points)
        def rec(i):
            def h(_, __, out):
                m = out.detach().amax(dim=(0, 2, 3))
                maxes[i] = m if maxes[i] is None else torch.maximum(maxes[i], m)
            return h
        hs = [p.register_forward_hook(rec(i)) for i, p in enumerate(points)]
        with torch.no_grad():
            for i in range(0, len(xtr), 1000): self.model(xtr[i:i + 1000])
        for h in hs: h.remove()
        for p, m in zip(points, maxes):
            b = m.view(1, -1, 1, 1)
            self.hooks.append(p.register_forward_hook(lambda _, __, out, b=b: torch.minimum(out, b)))


# ---------------- parity detection ----------------
class Parity:
    """Two parity bits (over bit 7 and over bit 6) per group of GROUP weights.
    contiguous: group = index // GROUP.  interleaved: a secret per-layer permutation
    first, so neighbouring weights (one output channel) land in different groups."""
    def __init__(self, net, interleaved, seed=1234):
        self.groups = {}
        g = torch.Generator(device="cpu").manual_seed(seed)
        for name in net.names:
            n = net.q[name][0].numel()
            pos = torch.randperm(n, generator=g) if interleaved else torch.arange(n)
            self.groups[name] = (pos // GROUP).to(DEV)
        self.sig = self.signature(net.q)
        self.clean = {k: v.clone() for k, (v, _) in net.q.items()}

    def signature(self, q):
        out = {}
        for name, (qi, _) in q.items():
            u = qi.view(-1).view(torch.uint8).to(torch.int64)
            grp = self.groups[name]; ng = int(grp.max()) + 1
            out[name] = [torch.zeros(ng, dtype=torch.int64, device=DEV).index_add_(0, grp, (u >> b) & 1) & 1
                         for b in (7, 6)]
        return out

    def recover(self, net):
        """Groups whose parity changed are either zeroed ("zero", needs nothing else)
        or re-read from a clean copy kept elsewhere ("reload", e.g. from flash).
        Returns (groups flagged, {mode: recovered q})."""
        now = self.signature(net.q); flagged = 0; out = {"zero": {}, "reload": {}}
        for name, (qi, s) in net.q.items():
            bad = (now[name][0] != self.sig[name][0]) | (now[name][1] != self.sig[name][1])
            flagged += int(bad.sum())
            m = bad[self.groups[name]].view_as(qi)
            out["zero"][name] = (torch.where(m, torch.zeros_like(qi), qi), s)
            out["reload"][name] = (torch.where(m, self.clean[name], qi), s)
        return flagged, out


def acc_with(net, q2, x, y):
    saved = net.q; net.q = q2
    for n in net.names: net.sync(n)
    a = net.acc(x, y)
    net.q = saved
    for n in net.names: net.sync(n)
    return a


# ---------------- the attack ----------------
BITS = torch.arange(8, device=DEV)

def candidates(net, allowed):
    """First-order gain of every (weight, bit) flip; CANDS best per layer."""
    out = []
    for name in net.names:
        qi, s = net.q[name]
        g = net.mods[name].weight.grad.view(-1)
        v = qi.view(-1).to(torch.int16)
        nu = (v & 0xFF).unsqueeze(1) ^ (1 << BITS).to(torch.int16).unsqueeze(0)
        nv = torch.where(nu >= 128, nu - 256, nu)
        gain = g.unsqueeze(1) * (nv - v.unsqueeze(1)).float() * s
        gain[:, ~allowed] = float("-inf")
        top = torch.topk(gain.view(-1), min(CANDS, gain.numel())).indices.tolist()
        out += [(name, t // 8, t % 8) for t in top]
    return out


def attack(net, xa, ya, xe, ye, allowed, parities=None, log=print):
    lossf = nn.CrossEntropyLoss()
    acc0 = net.acc(xe, ye)
    curve, flips = [acc0], []
    rec = {f"{k}-{m}": [acc0] for k in (parities or {}) for m in ("zero", "reload")}
    for step in range(1, MAX_STEPS + 1):
        for p in net.model.parameters(): p.grad = None
        lossf(net.model(xa), ya).backward()
        best = None
        with torch.no_grad():
            for name, idx, bit in candidates(net, allowed):
                net.flip(name, idx, bit)
                l = lossf(net.model(xa), ya).item()
                net.flip(name, idx, bit)                  # undo
                if best is None or l > best[0]: best = (l, name, idx, bit)
        l, name, idx, bit = best
        before = int(net.q[name][0].view(-1)[idx])
        net.flip(name, idx, bit)
        a = net.acc(xe, ye)
        curve.append(a)
        flips.append(dict(layer=name, index=idx, bit=bit, before=before,
                          after=int(net.q[name][0].view(-1)[idx]), loss=l))
        extra = ""
        for k, par in (parities or {}).items():
            fl, qs = par.recover(net)
            for mode, q2 in qs.items():
                rec[f"{k}-{mode}"].append(acc_with(net, q2, xe, ye))
            extra += f"  {k}: {fl} groups, zero {rec[k + '-zero'][-1]:.2f}% reload {rec[k + '-reload'][-1]:.2f}%"
        if step % 10 == 0 or a <= STOP_ACC:
            log(f"    step {step:3d}  {name}[{idx}] bit {bit}  attack loss {l:.3f}  acc {a:.2f}%{extra}")
        if a <= STOP_ACC: break
    return dict(acc0=acc0, curve=curve, flips=flips,
                to_stop=len(curve) - 1 if curve[-1] <= STOP_ACC else None, recovered=rec)


CONFIGS = {
    "base":        dict(ckpt="resnet20.pt",    bits=range(8), clamp=False, parity=True),
    "sign":        dict(ckpt="resnet20.pt",    bits=range(7), clamp=False, parity=False),
    "clamp":       dict(ckpt="resnet20.pt",    bits=range(8), clamp=True,  parity=False),
    "pc":          dict(ckpt="resnet20_pc.pt", bits=range(8), clamp=False, parity=False),
    "parity-adapt": dict(ckpt="resnet20.pt",   bits=range(6), clamp=False, parity=False),
}


def run(cfg_name, xe, ye, xpool, ypool, xtr):
    cfg = CONFIGS[cfg_name]
    allowed = torch.zeros(8, dtype=torch.bool, device=DEV); allowed[list(cfg["bits"])] = True
    runs = []
    for seed in range(SEEDS):
        net = Net(cfg["ckpt"])
        if cfg["clamp"]: net.add_clamp(xtr)
        pick = torch.randperm(len(xpool), generator=torch.Generator().manual_seed(seed))[:ATTACK_N].to(DEV)
        parities = {"contiguous": Parity(net, False), "interleaved": Parity(net, True)} if cfg["parity"] else None
        t0 = time.time()
        print(f"  [{cfg_name}] seed {seed}: clean acc {net.acc(xe, ye):.2f}%", flush=True)
        r = attack(net, xpool[pick], ypool[pick], xe, ye, allowed, parities,
                   log=lambda s: print(s, flush=True))
        r["seconds"] = time.time() - t0
        print(f"  [{cfg_name}] seed {seed}: {r['to_stop']} flips to <= {STOP_ACC}%  ({r['seconds']:.0f}s)", flush=True)
        runs.append(r)
    return runs


def overhead(xe):
    """Cost of the cheapest 'detect everything' option: CRC32 over all int8 weights."""
    net = Net("resnet20.pt")
    blob = b"".join(net.q[n][0].cpu().numpy().tobytes() for n in net.names)
    t0 = time.perf_counter()
    for _ in range(100): zlib.crc32(blob)
    crc_us = (time.perf_counter() - t0) / 100 * 1e6
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20): net.model(xe[:1000])
    torch.cuda.synchronize()
    inf_us = (time.perf_counter() - t0) / 20 * 1e6
    n = len(blob)
    return dict(weight_bytes=n, crc32_us=crc_us, infer_1000_us=inf_us,
                parity_bits=2 * sum((net.q[k][0].numel() + GROUP - 1) // GROUP for k in net.names))


def main(cfgs):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    x, y = cifar(False)
    xe, ye, xpool, ypool = x[:EVAL_N], y[:EVAL_N], x[EVAL_N:], y[EVAL_N:]
    xtr = cifar(True)[0] if any(CONFIGS[c]["clamp"] for c in cfgs) else None
    res["overhead"] = overhead(xe)
    for c in cfgs:
        print(f"== {c}", flush=True)
        res[c] = run(c, xe, ye, xpool, ypool, xtr)
        json.dump(res, open(OUT, "w"), indent=1)
    print(f"saved {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run": main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        from defense_plot import plot; plot(json.load(open(OUT)))
    else: print(__doc__)
