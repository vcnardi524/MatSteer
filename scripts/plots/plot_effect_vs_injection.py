#!/usr/bin/env python3
"""Effect size against how hard the model was actually pushed, for one property.

"alpha 32" and "scale -10" are knobs in different units, so they cannot be compared
directly. The comparable axis is the measured displacement |h_new - h| at the injection
layer as a share of |h| -- which injection_magnitude.py measures on real per-token states
using the real hooks. This puts Cohen's d against that.

Reading it: a method that steers should climb away from d = 0 as the injection grows. A
method that only perturbs stays flat, or moves in whichever direction output degradation
happens to push the property.

Point size encodes validity, because that is the confound: degradation moves the measured
property on its own, and the arms that move most are usually the arms that broke most.

Usage:
    python scripts/plots/plot_effect_vs_injection.py --property band_gap --layer 4
    python scripts/plots/plot_effect_vs_injection.py --property energy_above_hull --layer 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAG = Path("analysis/v1_all/test/plots")
COLOR = {"linear": "#D55E00", "manifold": "#0072B2", "pca_centroid": "#009E73"}
MARKER = {"linear": "o", "manifold": "s", "pca_centroid": "^"}
NICE = {"band_gap": "band gap", "energy_above_hull": "energy above hull",
        "density_atomic": "volume per atom"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="Keep only runs at this layer. Without it, arms from different "
                         "layers are drawn together and are not comparable.")
    ap.add_argument("--mag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.mag) if args.mag else MAG / f"{args.property}_injection_magnitude.csv"
    d = pd.read_csv(src)
    if args.layer is not None:
        d = d[d.layer == args.layer]
    d = d.dropna(subset=["pct_of_h", "cohens_d"])
    if d.empty:
        raise SystemExit(f"no rows in {src} for layer {args.layer}")
    print(f"{src.name}: {len(d)} runs, layer(s) {sorted(d.layer.unique())}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for meth, g in d.groupby("method"):
        g = g.sort_values("pct_of_h")
        ax.plot(g.pct_of_h, g.cohens_d, "-", color=COLOR.get(meth, "#777"),
                lw=1.6, alpha=0.55, zorder=2)
        ax.scatter(g.pct_of_h, g.cohens_d, s=30 + 170 * g.valid_pct ** 3,
                   color=COLOR.get(meth, "#777"), marker=MARKER.get(meth, "o"),
                   edgecolor="white", linewidth=1.1, zorder=3, label=meth)
        for _, r in g.iterrows():
            ax.annotate(f"{r.strength:g}", (r.pct_of_h, r.cohens_d), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=COLOR.get(meth, "#777"))
    ax.axhline(0, color="#999", lw=1)
    ax.set_xlabel("injection at the steered layer, as % of $|h|$")
    ax.set_ylabel("Cohen's $d$ vs the no-injection control")
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")
    lay = f" — layer {args.layer}" if args.layer is not None else ""
    ax.set_title(f"{NICE.get(args.property, args.property)}: effect size vs how hard the "
                 f"model was pushed{lay}\n"
                 "point labels are the method's own knob; larger point = higher validity",
                 fontsize=11, loc="left")
    out = Path(args.out or MAG / f"{args.property}_effect_vs_injection.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    print(d[["label", "pct_of_h", "cohens_d", "valid_pct"]]
          .sort_values("pct_of_h").to_string(index=False))


if __name__ == "__main__":
    main()
