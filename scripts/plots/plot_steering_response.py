#!/usr/bin/env python3
"""Cohen's d against injection magnitude, one line per RUN.

A run is a series in which exactly one hyperparameter moves. The manifold sweep is a
3 x 2 grid -- three arc steps at each of two injection magnitudes -- so it contains two
families of runs, and they need different x axes:

  TOP ROW     x = injection / |h|.   One line per (layer, arc step): magnitude varies,
                                     the arc step is held. Linear arms belong here too,
                                     since alpha IS the injected norm.
  BOTTOM ROW  x = arc step, as % of the curve's own length.
                                     One line per (layer, magnitude): the arc step
                                     varies, magnitude is held.

Plotting the grid as a single line sorted by magnitude -- which an earlier version did --
joins points that are not a sequence and draws a dose-response curve that does not exist.

Injection is the common axis because the two methods' knobs are not comparable:

    linear    injection = |alpha|                      (the vector is unit-norm)
    manifold  injection = scale * |decode(u+d) - decode(u)|

both over |h| at that layer. The manifold value is recomputed from the curve here rather
than assumed, so the plot also checks that the per-arm scale solving landed where it was
meant to.

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
_CURVE = {}


def curve(model, prop, layer):
    key = (model, prop, layer)
    if key not in _CURVE:
        hits = [h for h in glob.glob(str(steering_vectors_dir(model, "manifolds")
                                        / f"{prop}_layer{layer}_k32_w*.parquet"))
                if not any(w in h for w in ("_w0.1_", "_w0.25_", "_w1."))]
        _CURVE[key] = Manifold.load(sorted(hits)[0]) if hits else None
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

    fig, axes = plt.subplots(2, len(PROPS), figsize=(5.0 * len(PROPS), 8.4))

    for col, (prop, _) in enumerate(PROPS):
        label, lin, man = data[prop]
        top, bot = axes[0, col], axes[1, col]

        # --- top: magnitude varies, everything else held -----------------------------
        if lin is not None:
            for layer, g in sorted(lin.groupby("layer")):
                g = g.sort_values("inj")
                c = LAYER_COLOUR.get(int(layer), FALLBACK)
                top.plot(g.inj, g.cohens_d, "-o", color=c, lw=2.2, ms=7,
                         label=f"linear L{layer}")
        if man is not None:
            for (layer, delta), g in sorted(man.groupby(["layer", "target"])):
                if len(g) < 2:
                    continue
                g = g.sort_values("inj")
                c = LAYER_COLOUR.get(int(layer), FALLBACK)
                top.plot(g.inj, g.cohens_d, "--s", color=c, lw=1.5, ms=5.5, alpha=0.75)
                top.annotate(f"{g.arcpct.iloc[-1]:.0f}%", xy=(g.inj.iloc[-1], g.cohens_d.iloc[-1]),
                             xytext=(4, 0), textcoords="offset points", fontsize=7,
                             color=c, va="center")
        top.set_title(label, fontsize=11)
        top.set_xlabel("injection / |h|")
        top.set_ylabel("Cohen's d" if col == 0 else "")
        top.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        if lin is not None or man is not None:
            top.legend(frameon=False, fontsize=8, loc="best")

        # --- bottom: arc step varies, magnitude held ---------------------------------
        if man is not None and man.arcpct.notna().any():
            man = man.copy()
            man["mag"] = (man.inj.abs() * 100).round(-1)      # 20% / 40% bands
            for (layer, mag), g in sorted(man.groupby(["layer", "mag"])):
                if len(g) < 2:
                    continue
                g = g.sort_values("arcpct")
                c = LAYER_COLOUR.get(int(layer), FALLBACK)
                ls = "-" if mag <= 25 else "--"
                bot.plot(g.arcpct, g.cohens_d, ls, marker="s", color=c, lw=1.8, ms=6,
                         label=f"L{layer} @ {mag:.0f}% |h|")
            bot.legend(frameon=False, fontsize=8, loc="best")
        else:
            bot.text(0.5, 0.5, "no manifold arms yet", ha="center", va="center",
                     transform=bot.transAxes, fontsize=9, color="#999999")
        bot.set_xlabel("arc step (% of curve length)")
        bot.set_ylabel("Cohen's d" if col == 0 else "")

        for a in (top, bot):
            a.axhline(0, color="#555555", lw=1)
            a.grid(alpha=0.25, lw=0.6); a.set_axisbelow(True)

    fig.suptitle(f"{display_name(args.model)}: steering response, one line per run",
                 fontsize=13)
    fig.text(0.5, 0.008,
             "top: magnitude varies, arc step held (manifold lines labelled with their arc step).  "
             "bottom: arc step varies, magnitude held.",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    out = args.out or (analysis_dir("v1_all", None, "test", subdir="plots",
                                    model=args.model) / "steering_response.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
