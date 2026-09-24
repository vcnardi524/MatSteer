#!/usr/bin/env python3
"""Cohen's d against injection magnitude, one line per RUN, one METHOD per panel.

A run is a series in which exactly one hyperparameter moves. Linear has one factor
(alpha), so a property gives one run per layer. The manifold is a 3 x 2 grid -- three arc
steps at each of two magnitudes -- so it holds two families of runs on different axes.

Methods are NEVER overlaid. Sharing a panel put eight lines in front of a two-entry
legend, and since colour keys the layer rather than the method, the unlabelled manifold
lines read as linear ones. One method per row instead, every line in its own legend:

  ROW 1  linear,   x = injection/|h|          one run per layer
  ROW 2  manifold, x = arc step % of curve    one run per (layer, magnitude),
                                              magnitude carried by line style

There is no "manifold, magnitude varies" row. It would show the SAME twelve points as
row 2 re-connected as six two-point segments instead of four three-point lines -- less
shape, a cramped 20-40% x range, and six lines separated only by colour shade. The
question it would answer, whether the effect grows with magnitude, is read off row 2 by
comparing the solid and dashed line of one colour.

Injection is the x axis in rows 1-2 because the two methods' knobs are not comparable:

    linear    injection = |alpha|                      (the vector is unit-norm)
    manifold  injection = scale * |decode(u+d) - decode(u)|

both over |h| at that layer, with the manifold value recomputed from the curve rather
than assumed.

Usage:
    python scripts/plots/plot_steering_response.py --model llamat2_cif
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import analysis_dir, steering_vectors_dir, MODELS, DEFAULT_MODEL, display_name
from manifold import Manifold

# Per-token |h| on ANSWER tokens, measured from the model -- the states the hook modifies.
HIDDEN_NORM = {"llamat2_cif": {8: 6.4, 12: 8.7, 24: 22.2},
               "crystallm":   {7: 127.1, 14: 165.8}}
PROPS = [("density_atomic", "Volume per atom"),
         ("band_gap", "Band gap"),
         ("formation_energy_per_atom", "Formation energy")]
LAYER_COLOUR = {8: "#0072B2", 12: "#009E73", 24: "#D55E00"}
FALLBACK = "#CC79A7"
# The bucket width each property's sweep actually used -- the WIDER, better-conditioned
# fit of the two available. Named explicitly rather than filtered by pattern: an earlier
# version excluded "_w0.25_", which never matches "..._k32_w0.25.parquet", so it silently
# loaded the narrow curve and every arc-step percentage and injection value computed from
# it was wrong.
SWEEP_WIDTH = {"band_gap": "w0.5_rc", "density_atomic": "w3",
               "formation_energy_per_atom": "w0.5"}
_CURVE = {}


def curve(model, prop, layer):
    key = (model, prop, layer)
    if key not in _CURVE:
        w = SWEEP_WIDTH.get(prop)
        path = steering_vectors_dir(model, "manifolds") / f"{prop}_layer{layer}_k32_{w}.parquet"
        _CURVE[key] = Manifold.load(path) if w and path.exists() else None
    return _CURVE[key]


def step_norm(model, prop, layer, delta):
    m = curve(model, prop, layer)
    if m is None:
        return None
    u = torch.linspace(0, float(m.length), 60).unsqueeze(-1)
    return float((m.decode(u + delta) - m.decode(u)).norm(dim=-1).median())


def load(model, prop, method):
    suffix = "" if method == "linear" else "_manifold"
    p = analysis_dir("v1_all", None, "test", model=model) / f"{prop}{suffix}_steering_ttest.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[(d.method == method) & d.cohens_d.notna()].copy()
    return d if len(d) else None


def enrich(model, prop, d, method):
    """Add injection (share of |h|) and, for the manifold, the arc step as % of curve."""
    norms = HIDDEN_NORM.get(model, {})
    inj, arcpct = [], []
    for _, r in d.iterrows():
        h = norms.get(int(r.layer))
        if not h:
            inj.append(np.nan); arcpct.append(np.nan); continue
        if method == "linear":
            inj.append(r.strength / h); arcpct.append(np.nan)
        else:
            s = step_norm(model, prop, int(r.layer), float(r.target))
            m = curve(model, prop, int(r.layer))
            inj.append(np.nan if s is None else np.sign(r.target) * r.strength * s / h)
            arcpct.append(np.nan if m is None else 100 * r.target / float(m.length))
    d = d.copy(); d["inj"] = inj; d["arcpct"] = arcpct
    return d.dropna(subset=["inj"])


def mark_ns(ax, g, xcol, colour, marker):
    """Hollow the arms that are not distinguishable from the control.

    Without this a panel whose y range is +/-0.08 -- band gap, where every arm has
    p_holm = 1.00 -- draws lines that look as decisive as density's, which are real.
    """
    ns = g[g.p_holm > 0.05]
    if len(ns):
        ax.plot(ns[xcol], ns.cohens_d, marker, ms=7, mfc="white", mec=colour,
                mew=1.8, ls="none", zorder=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = {}
    for prop, label in PROPS:
        lin, man = load(args.model, prop, "linear"), load(args.model, prop, "manifold")
        data[prop] = (label,
                      enrich(args.model, prop, lin, "linear") if lin is not None else None,
                      enrich(args.model, prop, man, "manifold") if man is not None else None)

    fig, axes = plt.subplots(2, len(PROPS), figsize=(5.0 * len(PROPS), 7.8))

    for col, (prop, _) in enumerate(PROPS):
        label, lin, man = data[prop]
        a_lin, a_arc = axes[0, col], axes[1, col]

        # --- row 1: LINEAR. one run per layer, alpha varying ------------------------
        if lin is not None:
            for layer, g in sorted(lin.groupby("layer")):
                g = g.sort_values("inj")
                c = LAYER_COLOUR.get(int(layer), FALLBACK)
                a_lin.plot(g.inj, g.cohens_d, "-o", lw=2.2, ms=7, color=c,
                           label=f"layer {layer}")
                mark_ns(a_lin, g, "inj", c, "o")
            a_lin.legend(frameon=False, fontsize=8.5, loc="best")
        else:
            a_lin.text(0.5, 0.5, "no linear arms", ha="center", va="center",
                       transform=a_lin.transAxes, fontsize=9, color="#999999")
        a_lin.set_title(f"{label}\nlinear", fontsize=10.5)
        a_lin.set_xlabel("injection / |h|")
        a_lin.set_ylabel("Cohen's d" if col == 0 else "")

        # --- row 2: MANIFOLD. arc step varies; magnitude is the line style ----------------------
        if man is not None and man.arcpct.notna().any():
            m2 = man.copy()
            m2["mag"] = (m2.inj.abs() * 100).round(-1)
            for (layer, mag), g in sorted(m2.groupby(["layer", "mag"])):
                if len(g) < 2:
                    continue
                g = g.sort_values("arcpct")
                c = LAYER_COLOUR.get(int(layer), FALLBACK)
                a_arc.plot(g.arcpct, g.cohens_d, marker="s", ms=6, lw=1.8,
                           ls="-" if mag <= 25 else "--", color=c,
                           label=f"L{layer} @ {mag:.0f}% |h|")
                mark_ns(a_arc, g, "arcpct", c, "s")
            a_arc.legend(frameon=False, fontsize=7.5, loc="best")
        else:
            a_arc.text(0.5, 0.5, "no manifold arms yet", ha="center", va="center",
                       transform=a_arc.transAxes, fontsize=9, color="#999999")
        a_arc.set_title("manifold — arc step varies\n"
                        "solid 20% |h|, dashed 40% |h|", fontsize=10.5)
        a_arc.set_xlabel("arc step (% of curve length)")
        a_arc.set_ylabel("Cohen's d" if col == 0 else "")

        for a in (a_lin, a_arc):
            a.axhline(0, color="#555555", lw=1)
            a.grid(alpha=0.25, lw=0.6); a.set_axisbelow(True)
        a_lin.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    fig.suptitle(f"{display_name(args.model)}: steering response, one line per run",
                 fontsize=13)
    fig.text(0.5, 0.006, "colour = layer.  hollow markers are NOT distinguishable from "
             "the control (p_holm > 0.05).  every line varies ONE hyperparameter; "
             "methods are never drawn in the same panel.",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=[0, 0.018, 1, 0.975])
    out = args.out or (analysis_dir("v1_all", None, "test", subdir="plots",
                                    model=args.model) / "steering_response.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
