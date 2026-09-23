#!/usr/bin/env python3
"""Cohen's d against how hard the linear vector actually pushes, per property.

WHY THE X AXIS IS A FRACTION, NOT ALPHA. The stored vector is unit-norm, so alpha IS the
norm added to the hidden state -- an ABSOLUTE quantity. Hidden states are not the same
size at every layer: measured per-token on answer tokens, |h| is 6.4 at layer 8, 8.7 at
12 and 22.2 at 24. Plotting d against raw alpha therefore puts layer 8 and layer 24 on
axes that mean different things, and a curve drawn through them is meaningless. alpha/|h|
is the common axis, and it is what --alpha-rel was built to hold fixed.

The bottom row is validity, and it is not decoration. For formation energy the effect is
almost perfectly predicted by how many structures the intervention destroyed (Cohen's d
against validity, r = -0.959), which is the signature of damage rather than steering. A
steering plot without the validity panel underneath invites exactly that misreading.

Reads the per-property t-test CSVs, so run those first:
    python scripts/analysis/steering_ttest.py --property <p> --method linear \
        --model llamat2_cif --family cond

Usage:
    python scripts/plots/plot_linear_magnitude_response.py --model llamat2_cif
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import analysis_dir, MODELS, DEFAULT_MODEL, display_name

# Per-token |h| on ANSWER tokens, measured directly from the model (not pooled embedding
# norms, which are a different quantity). Answer tokens are what generation produces and
# what the hook modifies; prompt tokens run larger and would understate the fraction.
HIDDEN_NORM = {"llamat2_cif": {8: 6.4, 12: 8.7, 24: 22.2},
               "crystallm":   {7: 127.1, 14: 165.8}}

PROPS = [("density_atomic", "Volume per atom"),
         ("band_gap", "Band gap"),
         ("formation_energy_per_atom", "Formation energy")]

# Okabe-Ito, keyed on the LAYER ITSELF. Assigning by plotting order made the same colour
# mean layer 12 in one panel and layer 8 in the next, because density is swept at 12/24
# while band gap and formation energy are swept at 8/24.
LAYER_COLOUR = {8: "#0072B2", 12: "#009E73", 24: "#D55E00"}
FALLBACK = "#CC79A7"


def load(model, prop):
    path = analysis_dir("v1_all", None, "test", model=model) / f"{prop}_steering_ttest.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[(d.method == "linear") & d.cohens_d.notna()].copy()
    return d if len(d) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--out", default=None,
                    help="default: analysis/<model>/v1_all/test/plots/"
                         "linear_magnitude_response.png")
    args = ap.parse_args()

    norms = HIDDEN_NORM.get(args.model, {})
    frames = [(p, lab, load(args.model, p)) for p, lab in PROPS]
    frames = [(p, lab, d) for p, lab, d in frames if d is not None]
    if not frames:
        raise SystemExit(f"No linear t-test CSVs for {args.model} -- run steering_ttest first")

    fig, axes = plt.subplots(2, len(frames), figsize=(4.6 * len(frames), 7.2),
                             sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if len(frames) == 1:
        axes = axes.reshape(2, 1)

    for col, (prop, label, d) in enumerate(frames):
        top, bot = axes[0, col], axes[1, col]
        for i, (layer, g) in enumerate(sorted(d.groupby("layer"))):
            h = norms.get(int(layer))
            if not h:
                print(f"  ! no |h| for layer {layer}; skipping")
                continue
            g = g.sort_values("strength")
            x = g.strength / h                       # fraction of the hidden-state norm
            c = LAYER_COLOUR.get(int(layer), FALLBACK)
            top.plot(x, g.cohens_d, "-o", color=c, lw=2, ms=7, label=f"layer {layer}")
            bot.plot(x, g.valid_pct * 100, "-o", color=c, lw=2, ms=7)
            # mark the arms that are not distinguishable from the control
            ns = g[g.p_holm > 0.05]
            top.plot(ns.strength / h, ns.cohens_d, "o", ms=7,
                     mfc="white", mec=c, mew=2, zorder=5)

        top.axhline(0, color="#666666", lw=1)
        for y, t in ((0.2, "small"), (0.5, "medium")):
            for s in (1, -1):
                top.axhline(s * y, color="#BBBBBB", lw=0.8, ls=":")
            top.annotate(t, xy=(0.99, y), xycoords=("axes fraction", "data"),
                         ha="right", va="bottom", fontsize=7, color="#999999")
        top.set_title(label, fontsize=11)
        top.set_ylabel("Cohen's d vs control" if col == 0 else "")
        bot.set_ylabel("valid (%)" if col == 0 else "")
        bot.set_xlabel("injection  alpha / |h|   (signed)")
        bot.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        for a in (top, bot):
            a.grid(alpha=0.25, lw=0.6)
            a.set_axisbelow(True)
        top.legend(frameon=False, fontsize=9, loc="upper left")

    # one survival scale across properties, so "this arm broke more" is readable across
    # panels rather than only within one
    lo = min(f[2].valid_pct.min() for f in frames) * 100
    for c in range(len(frames)):
        axes[1, c].set_ylim(max(0, lo - 4), 74)

    ctrl = frames[0][2]
    base = ctrl.loc[ctrl.strength == 0, "valid_pct"]
    if len(base):
        for col in range(len(frames)):
            axes[1, col].axhline(float(base.iloc[0]) * 100, color="#666666",
                                 lw=1, ls="--")
        axes[1, 0].text(0.02, float(base.iloc[0]) * 100 + 1, "control", fontsize=7,
                        color="#666666", transform=axes[1, 0].get_yaxis_transform())

    fig.suptitle(f"{display_name(args.model)}: linear steering, effect and survival "
                 f"against injection magnitude", fontsize=12)
    fig.text(0.5, 0.005, "hollow markers: not distinguishable from control (p_holm > 0.05)",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out = args.out or (analysis_dir("v1_all", None, "test", subdir="plots",
                                    model=args.model) / "linear_magnitude_response.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
