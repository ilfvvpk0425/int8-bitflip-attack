"""Draw defense.png from defense_results.json only (no model needed)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from statistics import median

INK, MUTED, GRID, DOT = "#2b2a27", "#8a8980", "#e6e5e0", "#2a78d6"
LABELS = {"base": "No defense", "sign": "Sign bit protected", "clamp": "Activation clamp",
          "pc": "Clustering-trained", "parity-adapt": "Parity, adaptive\n(bits 0-5 only)"}
REC = [("raw", "No recovery"), ("contiguous-zero", "Contiguous groups,\nzero group"),
       ("contiguous-reload", "Contiguous groups,\nreload group"),
       ("interleaved-zero", "Interleaved groups,\nzero group"),
       ("interleaved-reload", "Interleaved groups,\nreload group")]


def style(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def plot(res, out="defense.png"):
    from defense import MAX_STEPS, STOP_ACC
    cfgs = [c for c in LABELS if c in res]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1]))

    # left: flips needed to collapse, one dot per attack seed
    style(a1)
    for i, c in enumerate(cfgs):
        ys = len(cfgs) - 1 - i
        vals = [r["to_stop"] for r in res[c]]
        done = [v for v in vals if v is not None]
        cens = len(vals) - len(done)
        a1.scatter(done, [ys] * len(done), s=36, color=DOT, edgecolor="white", lw=1.5, zorder=3)
        if cens:
            a1.scatter([MAX_STEPS] * cens, [ys] * cens, s=36, facecolor="white", edgecolor=DOT, lw=1.5, zorder=3)
        txt = f"median {median(done):g}" if not cens else f"{cens}/{len(vals)} not collapsed after {MAX_STEPS}"
        if done and not cens: a1.plot([median(done)] * 2, [ys - 0.25, ys + 0.25], color=INK, lw=1.5, zorder=4)
        a1.text(1.02, ys, txt, transform=a1.get_yaxis_transform(), va="center", fontsize=8, color=MUTED)
    a1.set_yticks(range(len(cfgs))); a1.set_yticklabels([LABELS[c] for c in reversed(cfgs)], fontsize=9)
    a1.set_xscale("log"); a1.set_xlim(5, MAX_STEPS * 1.3)
    a1.set_xlabel(f"Bit flips until accuracy <= {STOP_ACC:g}% (log scale, 5 attack seeds)")
    a1.set_title("Flips the attacker needs", loc="left", fontsize=11, color=INK)

    # right: accuracy after parity recovery, at the point the undefended model collapsed
    style(a2)
    base = res["base"]
    for i, (k, lab) in enumerate(REC):
        ys = len(REC) - 1 - i
        vals = [r["curve"][-1] if k == "raw" else r["recovered"][k][-1] for r in base]
        a2.scatter(vals, [ys] * len(vals), s=36, color=DOT, edgecolor="white", lw=1.5, zorder=3)
        a2.plot([median(vals)] * 2, [ys - 0.25, ys + 0.25], color=INK, lw=1.5, zorder=4)
        a2.text(1.02, ys, f"median {median(vals):.1f}%", transform=a2.get_yaxis_transform(),
                va="center", fontsize=8, color=MUTED)
    a2.axvline(base[0]["acc0"], ls="--", lw=1, color=MUTED)
    a2.text(base[0]["acc0"], len(REC) - 0.45, "clean", ha="right", fontsize=8, color=MUTED)
    a2.set_yticks(range(len(REC))); a2.set_yticklabels([l for _, l in reversed(REC)], fontsize=9)
    a2.set_xlim(0, 100); a2.set_xlabel("Test accuracy (%) after the undefended attack finished")
    a2.set_title("Parity check on bits 6-7 (64 weights/group)", loc="left", fontsize=11, color=INK)

    fig.tight_layout(w_pad=6); fig.savefig(out, dpi=150)
    print(f"saved {out}")
