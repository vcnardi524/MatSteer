#!/usr/bin/env python3
"""Effect size against the layer the injection was applied at.

Every sweep so far fixed the layer and varied the strength. This does the opposite: hold
one coefficient set fixed and walk the layer. It answers whether the layer chosen for a
property was a good one, which nothing else in the repo reports.

One line per SERIES, where a series is (property, method, coefficients):

    --series formation_energy_per_atom:manifold:d=3,s=4
    --series formation_energy_per_atom:linear:alpha=16

Keys map onto the columns of steering_runs.csv: for manifold, d is `target` (the arc
step) and s is `strength` (the scale); for linear, alpha is `strength`. Any key left out
is not filtered on, so `manifold:d=3` draws every scale at that arc step -- usually not
what you want, and the run count printed per series is the check.

MARKER SIZE IS VALIDITY, deliberately. Across 106 density arms corr(validity, the
measured property) was -0.91: degradation moves the property on its own, so a d computed
over broken generations measures breakage. A large point is a trustworthy one; a line
that climbs while its points shrink is reporting damage, not steering.

Usage:
    python scripts/plots/plot_effect_by_layer.py \
        --series formation_energy_per_atom:manifold:d=3,s=4 \
        --series formation_energy_per_atom:linear:alpha=16 \
        --out analysis/<model>/v1_all/test/plots/formation_energy_effect_by_layer.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import analysis_dir

CSV = str(analysis_dir("v1_all", None, "test") / "steering_runs.csv")
COLOR = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#7A3B2E"]
MARKER = ["s", "o", "^", "D", "v", "P", "X"]
# which CSV column each coefficient key lives in, per method
KEYMAP = {"manifold": {"d": "target", "s": "strength"},
          "linear":   {"alpha": "strength"},
          "pca_centroid": {"target": "target", "t": "strength"}}


def parse_series(spec):
    """'prop:method:k=v,k=v' -> (property, method, {column: value}, label)"""
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--series must be prop:method:coeffs, got {spec!r}")
    prop, method, coeffs = parts
    if method not in KEYMAP:
        raise SystemExit(f"unknown method {method!r}; known: {sorted(KEYMAP)}")
    filt, shown = {}, []
    for kv in coeffs.split(","):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        if k not in KEYMAP[method]:
            raise SystemExit(f"{method} takes {sorted(KEYMAP[method])}, got {k!r}")
        filt[KEYMAP[method][k]] = float(v)
        shown.append(f"{k}={v}")
    short = prop.replace("formation_energy_per_atom", "formation energy") \
                .replace("density_atomic", "volume/atom") \
                .replace("energy_above_hull", "hull")
    return prop, method, filt, f"{short}  {method} {' '.join(shown)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help="prop:method:coeffs, repeatable -- one line each")
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--family", default="nosg", choices=("nosg", "sg", "any"))
    ap.add_argument("--source", default="raw", choices=("raw", "relaxed"))
    ap.add_argument("--agg", default="mean", choices=("mean", "max"))
    ap.add_argument("--out", default=None,
                    help="default: analysis/<model>/v1_all/test/plots/effect_by_layer.png")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    d = d[(d["agg"] == args.agg) & (d.source == args.source) & d.cohens_d.notna()]
    if args.family != "any":
        d = d[d.family == args.family]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    drawn = 0
    for i, spec in enumerate(args.series):
        prop, method, filt, label = parse_series(spec)
        g = d[(d.property == prop) & (d.method == method)]
        for col, val in filt.items():
            g = g[np.isclose(g[col].astype(float), val)]
        g = g.sort_values("layer")
        if g.empty:
            print(f"  ! no runs for {spec}"); continue
        print(f"  {label}: {len(g)} runs, layers {g.layer.min():.0f}-{g.layer.max():.0f}")
        c, m = COLOR[i % len(COLOR)], MARKER[i % len(MARKER)]
        ax.plot(g.layer, g.cohens_d, "-", color=c, lw=2, zorder=2, label=label)
        ax.scatter(g.layer, g.cohens_d, s=25 + 160 * g.valid_pct ** 3, color=c,
                   marker=m, edgecolor="white", linewidth=1.1, zorder=3)
        drawn += 1
    if not drawn:
        raise SystemExit("nothing to plot")

    ax.axhline(0, color="#999", lw=1)
    ax.set_xlabel("layer the injection was applied at")
    ax.set_ylabel("Cohen's $d$ vs the no-injection control")
    ax.set_xticks(sorted(d.layer.dropna().unique().astype(int)))
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("Effect size by injection layer, coefficients held fixed\n"
                 "marker size = validity; a line rising while its points shrink is "
                 "reporting degradation, not steering",
                 fontsize=11, loc="left")
    out = Path(args.out or
               analysis_dir("v1_all", None, "test", subdir="plots") /
               "effect_by_layer.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
