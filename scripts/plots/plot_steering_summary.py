#!/usr/bin/env python3
"""Two summary figures for density steering, from analysis/v1_all/test/steering_runs.csv.

1. the layer-7 dose response. One column per sweep, each on its own strength axis --
   the knobs are in different units (arc step, scale, alpha) and do not share one. Top
   row is the property, bottom row is how much output survived, so the trade is visible
   without a second y axis. Point 0 on every x is the same no-injection control.

2. a results table as a PNG, for pasting into a document.

Usage:
    python scripts/plots/plot_steering_summary.py
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "analysis/v1_all/test/steering_runs.csv"
OUT = Path("analysis/v1_all/test/plots")

# Okabe-Ito. Validated for this 4-series case: worst adjacent CVD dE 8.6 (tritan),
# normal-vision 18.7. Every line is also direct-labelled, which the low
# contrast-vs-surface of the orange requires.
COLOR = {"linear": "#D55E00", "manifold": "#0072B2",
         "pca_centroid": "#009E73", "pca_local": "#E69F00"}
MARKER = {"linear": "o", "manifold": "s", "pca_centroid": "^", "pca_local": "D"}


def label(r):
    if r.method == "linear":
        return f"a{r.strength:g}"
    if r.method == "manifold":
        return f"d{r.target:g} s{r.strength:g}"
    if r.method == "pca_local":
        return f"t{r.strength:g}"
    return f"{r.target:g}/t{r.strength:g}"


def dose_response(d, path):
    """Layer 7: property and validity against each sweep's own strength knob."""
    L7 = d[(d.layer == 7) & (d["agg"] == "mean")]
    ctrl = L7[L7.strength == 0].iloc[0]
    c_val, c_dens = ctrl.valid_pct * 100, 10 ** ctrl.control_median

    def sweep(sel, xcol):
        m = sel.sort_values(xcol)
        return ([0] + m[xcol].tolist(),
                [c_dens] + (10 ** m["median"]).tolist(),
                [c_val] + (m.valid_pct * 100).tolist())

    man = L7[L7.method == "manifold"]
    panels = [
        ("manifold: arc step\n(scale fixed at 6)", "delta",
         sweep(man[man.strength == 6], "target"), COLOR["manifold"]),
        ("manifold: scale\n(arc step fixed at 15)", "scale",
         sweep(man[man.target == 15], "strength"), COLOR["manifold"]),
        ("linear: alpha", "alpha",
         sweep(L7[(L7.method == "linear") & (L7.strength != 0)], "strength"),
         COLOR["linear"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey="row")
    for col, (title, xlabel, (x, dens, val), colour) in enumerate(panels):
        top, bot = axes[0][col], axes[1][col]
        top.plot(x, dens, "-o", color=colour, lw=2.2, ms=7)
        top.axhline(c_dens, color="#999", ls=":", lw=1.2)
        top.set_title(title, fontsize=11)
        bot.plot(x, val, "-o", color=colour, lw=2.2, ms=7)
        bot.axhline(c_val, color="#999", ls=":", lw=1.2)
        bot.set_xlabel(xlabel)
        bot.set_ylim(0, 100)
        for ax in (top, bot):
            ax.grid(alpha=0.25, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        # label the arm that keeps both: highest density at >=85% valid
        ok = [(dd, vv, xx) for xx, dd, vv in zip(x, dens, val) if vv >= 85 and xx > 0]
        if ok:
            dd, vv, xx = max(ok)
            top.annotate(f"{dd:.2f}", (xx, dd), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=9,
                         fontweight="bold", color=colour)
    axes[0][0].set_ylabel("volume per atom (\u00c5\u00b3/atom)")
    axes[1][0].set_ylabel("valid output (%)")
    axes[0][0].annotate("control", (0, c_dens), textcoords="offset points",
                        xytext=(10, 6), fontsize=9, color="#777")
    # alpha=80 has no point to plot: it produced 0 valid CIFs, so there is no median.
    # Its absence is the result, so say so rather than letting the line just stop.
    for ax in (axes[0][2], axes[1][2]):
        ax.axvspan(55, 88, color="#D55E00", alpha=0.07)
        ax.set_xlim(-3, 88)
    axes[1][2].annotate("\u03b1=80:\n0% valid\nnothing to plot", (71, 50),
                        ha="center", va="center", fontsize=9.5, color="#D55E00",
                        fontweight="bold")
    fig.suptitle("Layer 7, no space group in the prompt: the property moves with the "
                 "push, and validity pays for it\n"
                 "dotted line = the no-injection control (19.25 \u00c5\u00b3/atom, "
                 "95.7% valid); every sweep starts from it at 0", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")


def table_png(d, path):
    rows, seen = [], set()
    for lay, fam in [(7, "nosg"), (14, "sg")]:
        sub = d[(d.layer == lay) & (d["agg"] == "mean")]
        ctrl = sub[sub.strength == 0]
        pick = pd.concat([ctrl, sub[sub.strength != 0].nlargest(5, "cohens_d")])
        for _, r in pick.iterrows():
            mx = d[(d.run == r.run) & (d["agg"] == "max")]
            rows.append([
                f"L{lay} {fam}", "control" if r.strength == 0 else r.method,
                "--" if r.strength == 0 else label(r),
                f"{r.valid_pct*100:.1f}%",
                f"{10**r.control_median:.2f}" if r.strength == 0 else f"{10**r['median']:.2f}",
                "--" if r.strength == 0 else f"{r.cohens_d:+.3f}",
                "--" if r.strength == 0 else f"{r.p_holm:.1g}",
                "--" if (r.strength == 0 or mx.empty) else f"{mx.cohens_d.iloc[0]:+.3f}",
                "--" if (r.strength == 0 or mx.empty) else f"{mx.p_holm.iloc[0]:.1g}",
            ])
    cols = ["layer", "method", "setting", "valid", "A³/atom",
            "d (mean)", "p (mean)", "d (max)", "p (max)"]
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(rows) + 0.9))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 1.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#dddddd")
        if i == 0:
            c.set_text_props(weight="bold", color="white")
            c.set_facecolor("#40566b")
        elif rows[i-1][1] == "control":
            c.set_facecolor("#eef1f4"); c.set_text_props(style="italic")
        elif i % 2 == 0:
            c.set_facecolor("#fafafa")
    ax.set_title("Density steering — top 5 arms per layer, ranked by effect size (mean)\n"
                 "1,000 prompts × 3 samples; paired against the no-injection control "
                 "on shared prompts",
                 fontsize=12.5, pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    d = d[(d.property == "density_atomic") & (d.source == "raw")]
    OUT.mkdir(parents=True, exist_ok=True)
    dose_response(d, OUT / "density_layer7_dose_response.png")
    table_png(d, OUT / "density_steering_table.png")


if __name__ == "__main__":
    main()
