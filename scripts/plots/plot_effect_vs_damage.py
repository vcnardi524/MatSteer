#!/usr/bin/env python3
"""Does the effect survive the structures, or is it made of them?

THE POINT. Steering that works should move a property without destroying the crystals it
moves. Steering that merely damages them also shifts the measured property, because the
survivors are a biased subset -- broken structures have higher formation energy, and the
truncated ones read ~7% higher volume per atom. Those two stories are indistinguishable
from an effect size alone, and separable the moment effect is plotted against survival.

One point per arm. A method that steers puts points in the upper/lower band at HIGH
validity. A method that only damages traces a line: the more it destroyed, the larger the
"effect". Measured here, linear formation energy runs r = -0.959 and manifold density
r = +0.738 -- both damage. Linear density at layer 24 does not, which is why it is the one
credible result in the sweep.

NO CONNECTING LINES between arms. Each property has three deltas at each injection level
and only two injection levels, so joining them would draw a dose-response curve out of
points that are not a dose sequence.

Usage:
    python scripts/plots/plot_effect_vs_damage.py --model llamat2_cif
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import analysis_dir, MODELS, DEFAULT_MODEL, display_name

PROPS = [("density_atomic", "Volume per atom", "#0072B2"),
         ("band_gap", "Band gap", "#009E73"),
         ("formation_energy_per_atom", "Formation energy", "#D55E00")]
MARKER = {"linear": "o", "manifold": "s"}


def load(model, prop, method):
    suffix = "" if method == "linear" else "_manifold"
    p = analysis_dir("v1_all", None, "test", model=model) / f"{prop}{suffix}_steering_ttest.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[(d.method == method) & d.cohens_d.notna() & d.valid_pct.notna()].copy()
    return d if len(d) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    notes, ctrl = [], None

    for prop, label, colour in PROPS:
        for method in ("linear", "manifold"):
            d = load(args.model, prop, method)
            if d is None:
                continue
            if ctrl is None:
                c = load(args.model, prop, "linear")
                if c is not None and (c.strength == 0).any():
                    ctrl = float(c.loc[c.strength == 0, "valid_pct"].iloc[0]) * 100
            x, y = d.valid_pct * 100, d.cohens_d
            sig = d.p_holm <= 0.05
            ax.scatter(x[sig], y[sig], s=74, marker=MARKER[method], c=colour,
                       edgecolor="white", lw=0.8, zorder=3,
                       label=f"{label} · {method}")
            ax.scatter(x[~sig], y[~sig], s=74, marker=MARKER[method], facecolor="none",
                       edgecolor=colour, lw=1.6, zorder=3)
            if len(d) >= 4:
                r, p = stats.pearsonr(x, y)
                notes.append((label, method, len(d), r, p, colour))
                # Only draw the trend when the correlation is real. Fitting a line through
                # four points with p = 0.45 and drawing it asserts a relationship the data
                # does not support -- the same error as joining non-sequential arms.
                if p <= 0.05:
                    b, a = np.polyfit(x, y, 1)
                    xs = np.linspace(x.min(), x.max(), 20)
                    ax.plot(xs, a + b * xs, color=colour, lw=1.8, ls="--", alpha=0.7,
                            zorder=2)

    ax.axhline(0, color="#555555", lw=1)
    if ctrl:
        ax.axvline(ctrl, color="#555555", lw=1.4, ls=":")
        ax.annotate(f"control\n{ctrl:.0f}% valid", xy=(ctrl, ax.get_ylim()[1]),
                    xytext=(-4, -6), textcoords="offset points", ha="right", va="top",
                    fontsize=8, color="#555555")
    ax.set_xlabel("structures surviving validation (%)   → less damage")
    ax.set_ylabel("Cohen's d against the no-injection control")
    ax.set_title(f"{display_name(args.model)}: is the effect steering, or is it damage?",
                 fontsize=12)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)

    if notes:
        txt = "\n".join(f"{l} · {m}:  r = {r:+.2f}  (p = {p:.3f}, n = {n})"
                        for l, m, n, r, p, _ in notes)
        txt += "\n" + "-" * 52 + "\ndashed fit drawn only where p <= 0.05"
        ax.text(0.985, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, family="monospace",
                bbox=dict(fc="white", ec="#CCCCCC", alpha=0.9, pad=5))

    fig.text(0.5, 0.005,
             "filled = distinguishable from control (p_holm ≤ 0.05); hollow = not.  "
             "A steep fitted line means the effect is explained by how much the arm broke.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = args.out or (analysis_dir("v1_all", None, "test", subdir="plots",
                                    model=args.model) / "effect_vs_damage.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
