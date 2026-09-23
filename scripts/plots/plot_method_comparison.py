#!/usr/bin/env python3
"""Linear against manifold at MATCHED PUSH: does the curve buy anything?

Each method's knob is in different units -- alpha for linear, (arc step, scale) for the
manifold -- so neither knob can be the x axis. The common axis is how hard the hook
actually pushes, as a share of the hidden-state norm it is pushing on:

    linear    injection = |alpha|                    (the vector is unit-norm)
    manifold  injection = scale * |decode(u+d) - decode(u)|

divided by |h| at that layer. That is the quantity the sweep was designed around, and it
is what makes "at the same push, which method moves the property further and breaks
less?" a question with an answer.

The survival row is not decoration. Two of the three properties here have an effect that
is largely explained by how many structures the intervention destroyed -- linear
formation energy at r = -0.959 between Cohen's d and validity, manifold density at
r = +0.738 -- so an effect curve shown without survival underneath invites reading damage
as steering.

Reads the t-test CSVs; run steering_ttest.py for each property and method first.

Usage:
    python scripts/plots/plot_method_comparison.py --model llamat2_cif
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

# Per-token |h| on ANSWER tokens, measured from the model. Answer tokens are what the hook
# modifies; prompt tokens run larger and would understate every fraction here.
HIDDEN_NORM = {"llamat2_cif": {8: 6.4, 12: 8.7, 24: 22.2},
               "crystallm":   {7: 127.1, 14: 165.8}}

PROPS = [("density_atomic", "Volume per atom"),
         ("band_gap", "Band gap"),
         ("formation_energy_per_atom", "Formation energy")]

LAYER_COLOUR = {8: "#0072B2", 12: "#009E73", 24: "#D55E00"}
FALLBACK = "#CC79A7"
_STEP_CACHE = {}


def step_norm(model, prop, layer, delta):
    """Median |decode(u+delta) - decode(u)| along the curve, in PCA space.

    The PCA components are orthonormal, so this is also the norm of the displacement once
    it is mapped back to hidden space -- which is what the hook actually adds.
    """
    key = (model, prop, layer)
    if key not in _STEP_CACHE:
        hits = glob.glob(str(steering_vectors_dir(model, "manifolds")
                             / f"{prop}_layer{layer}_k32_w*.parquet"))
        # the wider-bucket fit is the one the sweep used
        hits = [h for h in hits if "_w0.1_" not in h and "_w0.25_" not in h and "_w1." not in h]
        if not hits:
            return None
        _STEP_CACHE[key] = Manifold.load(sorted(hits)[0])
    m = _STEP_CACHE[key]
    u = torch.linspace(0, float(m.length), 60).unsqueeze(-1)
    return float((m.decode(u + delta) - m.decode(u)).norm(dim=-1).median())


def load(model, prop, method):
    suffix = "" if method == "linear" else "_manifold"
    path = analysis_dir("v1_all", None, "test", model=model) / f"{prop}{suffix}_steering_ttest.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[(d.method == method) & d.cohens_d.notna()].copy()
    return d if len(d) else None


def injection(model, prop, d, method):
    norms = HIDDEN_NORM.get(model, {})
    out = []
    for _, r in d.iterrows():
        h = norms.get(int(r.layer))
        if not h:
            out.append(np.nan); continue
        if method == "linear":
            out.append(r.strength / h)                 # alpha IS the injected norm
        else:
            s = step_norm(model, prop, int(r.layer), float(r.target))
            out.append(np.nan if s is None else np.sign(r.target) * r.strength * s / h)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panels = []
    for prop, label in PROPS:
        lin, man = load(args.model, prop, "linear"), load(args.model, prop, "manifold")
        if lin is not None or man is not None:
            panels.append((prop, label, lin, man))
    if not panels:
        raise SystemExit("no t-test CSVs found -- run steering_ttest.py first")

    fig, axes = plt.subplots(2, len(panels), figsize=(4.8 * len(panels), 7.4),
                             sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if len(panels) == 1:
        axes = axes.reshape(2, 1)

    for col, (prop, label, lin, man) in enumerate(panels):
        top, bot = axes[0, col], axes[1, col]
        for method, d, style in (("linear", lin, dict(ls="-", marker="o")),
                                 ("manifold", man, dict(ls="--", marker="s"))):
            if d is None:
                continue
            d = d.copy()
            d["inj"] = injection(args.model, prop, d, method)
            d = d.dropna(subset=["inj"])
            for layer, g in sorted(d.groupby("layer")):
                g = g.sort_values("inj")
                c = LAYER_COLOUR.get(int(layer), FALLBACK)
                top.plot(g.inj, g.cohens_d, color=c, lw=1.8, ms=6, alpha=0.9,
                         label=f"L{layer} {method}", **style)
                bot.plot(g.inj, g.valid_pct * 100, color=c, lw=1.8, ms=6, alpha=0.9, **style)
                ns = g[g.p_holm > 0.05]
                top.plot(ns.inj, ns.cohens_d, style["marker"], ms=6, mfc="white",
                         mec=c, mew=1.6, ls="none", zorder=5)

        top.axhline(0, color="#666666", lw=1)
        for y in (0.2, -0.2):
            top.axhline(y, color="#CCCCCC", lw=0.8, ls=":")
        top.annotate("small", xy=(0.99, 0.2), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7, color="#999999")
        top.set_title(label, fontsize=11)
        top.set_ylabel("Cohen's d vs control" if col == 0 else "")
        bot.set_ylabel("valid (%)" if col == 0 else "")
        bot.set_xlabel("injection / |h|   (signed)")
        bot.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        top.legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
        for a in (top, bot):
            a.grid(alpha=0.25, lw=0.6); a.set_axisbelow(True)

    lo = min(min(f.valid_pct.min() for f in (l, m) if f is not None)
             for _, _, l, m in panels) * 100
    ctrl = None
    for _, _, l, _ in panels:
        if l is not None and (l.strength == 0).any():
            ctrl = float(l.loc[l.strength == 0, "valid_pct"].iloc[0]) * 100
    for c in range(len(panels)):
        axes[1, c].set_ylim(max(0, lo - 4), 74)
        if ctrl:
            axes[1, c].axhline(ctrl, color="#666666", lw=1, ls=":")

    fig.suptitle(f"{display_name(args.model)}: linear vs manifold at matched injection",
                 fontsize=12)
    fig.text(0.5, 0.005, "solid/circles linear, dashed/squares manifold; "
             "hollow = not distinguishable from control; dotted line = control validity",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])

    out = args.out or (analysis_dir("v1_all", None, "test", subdir="plots",
                                    model=args.model) / "method_comparison.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
