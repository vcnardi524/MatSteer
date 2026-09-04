#!/usr/bin/env python3
"""Effect and validity against how hard the intervention actually pushes.

Every method's own knob is in different units -- arc step, scale, alpha -- so a sweep
plotted against its own knob cannot be compared with another method's. The measured
injection |h_new - h|, as a share of |h|, is the common axis. That is what this puts on
x, so the two manifold sweeps and the linear baseline sit on one plot and the question
becomes: at the same push, which method moves the property further and breaks less?

Top panel is the effect (Cohen's d against the no-injection control), bottom is how much
output survived. Each point carries the hyperparameter that produced it.

If the manifold sweeps trace the SAME curve as each other, magnitude is all that matters
and it does not matter whether you get there by stepping further along the arc or by
scaling the step. If they separate, the route matters too.

Reads the two CSVs the other scripts write, so run those first:
    python scripts/analysis/steering_ttest.py --all
    python scripts/analysis/injection_magnitude.py

Usage:
    python scripts/plots/plot_magnitude_response.py
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAG = "analysis/v1_all/test/plots/density_injection_magnitude.csv"
OUT = Path("analysis/v1_all/test/plots")


def series_for(d, layer, min_points=4):
    """Every ladder on this layer, as (name, colour, marker, frame, label-fn).

    A ladder is a set of runs that hold one manifold knob fixed and vary the other:
    arc-step sweeps at a fixed scale, and scale sweeps at a fixed arc step. Both are
    discovered rather than named, so new sweeps appear without editing this.

    Within a family the colour is sequential in the FIXED parameter (the ladders are
    ordered by it); between families the hue changes. Linear is its own hue.
    """
    man = d[(d.layer == layer) & (d.method == "manifold")]
    lin = d[(d.layer == layer) & (d.method == "linear")]
    out = []

    steps = sorted(v for v in man.strength.unique()
                   if man[man.strength == v].target.nunique() >= min_points)
    scales = sorted(v for v in man.target.unique()
                    if man[man.target == v].strength.nunique() >= min_points)

    def ramp(cmap, i, n):
        return plt.get_cmap(cmap)(0.35 + 0.5 * (i / max(n - 1, 1)))

    for i, sc in enumerate(steps):
        out.append((f"arc step swept, scale {sc:g}", ramp("Blues", i, len(steps)), "s",
                    man[man.strength == sc], lambda r: f"d{r.target:g}"))
    for i, dl in enumerate(scales):
        out.append((f"scale swept, arc step {dl:g}", ramp("Greens", i, len(scales)), "D",
                    man[man.target == dl], lambda r: f"s{r.strength:g}"))
    if len(lin) > 1:
        out.append(("linear (baseline)", "#D55E00", "o", lin, lambda r: f"α{r.strength:g}"))
    return [(n, c, m, f.sort_values("pct_of_h"), lab) for n, c, m, f, lab in out]


def pareto(d, series, layer, path):
    """Effect against validity: the trade every method actually offers.

    Magnitude is the knob, but nobody wants magnitude -- they want effect without
    breaking the output. Plotting those two directly makes the frontier the shape of the
    picture: up and to the RIGHT is better, and a method is dominated wherever another
    sits above and to the right of it. Same split as the magnitude figure.
    """
    lin = [t for t in series if t[0].startswith("linear")]
    cols = [("scale swept (arc step fixed)",
             [t for t in series if t[0].startswith("scale")] + lin),
            ("arc step swept (scale fixed)",
             [t for t in series if t[0].startswith("arc")] + lin)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (title, members) in zip(axes, cols):
        for name, colour, marker, f, lab in members:
            g = f.sort_values("valid_pct")
            ax.plot(g.valid_pct * 100, g.cohens_d, "-", color=colour, lw=2,
                    marker=marker, ms=7, label=name, zorder=3)
            last = None
            for _, r in g.iterrows():
                if last is not None and abs(r.valid_pct * 100 - last) < 3.5:
                    continue
                last = r.valid_pct * 100
                ax.annotate(lab(r), (r.valid_pct * 100, r.cohens_d),
                            textcoords="offset points", xytext=(0, 9), ha="center",
                            fontsize=7.5, color=colour)
        ax.axhline(0, color="#999", lw=1)
        ax.set_xlim(0, 100)
        ax.set_xlabel("valid output (%)")
        ax.set_title(title, fontsize=12, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.annotate("better \u2197", (0.97, 0.03), xycoords="axes fraction",
                    ha="right", fontsize=10, color="#777", style="italic")
    axes[0].set_ylabel("Cohen's $d$ vs the no-injection control")
    fig.suptitle(f"Layer {layer}, no space group: effect against how much output survived\n"
                 "up and to the RIGHT is better; a point is dominated if another sits "
                 "above and to its right", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag", default=MAG)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--min-points", type=int, default=4,
                    help="a ladder needs this many runs to be drawn as a series")
    args = ap.parse_args()

    d = pd.read_csv(args.mag)
    series = series_for(d, args.layer, args.min_points)
    if not series:
        raise SystemExit(f"no sweeps with more than one point at layer {args.layer}")

    # Two columns rather than one crowded panel: the two families answer different
    # questions (does scale matter at a fixed arc step / does arc step matter at a fixed
    # scale) and the linear baseline is repeated in both so each is self-contained.
    lin = [t for t in series if t[0].startswith("linear")]
    cols = [("scale swept (arc step fixed)",
             [t for t in series if t[0].startswith("scale")] + lin),
            ("arc step swept (scale fixed)",
             [t for t in series if t[0].startswith("arc")] + lin)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey="row")
    for col, (title, members) in enumerate(cols):
        ax_d, ax_v = axes[0][col], axes[1][col]
        for name, colour, marker, f, lab in members:
            for ax, c in ((ax_d, "cohens_d"), (ax_v, "valid_pct")):
                y = f[c] * (100 if c == "valid_pct" else 1)
                ax.plot(f.pct_of_h, y, "-", color=colour, lw=2, marker=marker, ms=6,
                        label=name if ax is ax_d else None, zorder=3)
            span = d.pct_of_h.max() - d.pct_of_h.min()
            last = None
            for _, r in f.iterrows():
                if last is not None and abs(r.pct_of_h - last) < span * 0.022:
                    continue
                last = r.pct_of_h
                ax_d.annotate(lab(r), (r.pct_of_h, r.cohens_d),
                              textcoords="offset points", xytext=(0, 9), ha="center",
                              fontsize=7.5, color=colour)
        ax_d.axhline(0, color="#999", lw=1)
        ax_d.set_title(title, fontsize=12, loc="left")
        ax_d.legend(frameon=False, fontsize=9, loc="upper left")
        ax_v.set_ylim(0, 100)
        ax_v.set_xlabel("injection magnitude  |$h_{new}-h$| as % of |$h$|")
        for ax in (ax_d, ax_v):
            ax.grid(alpha=0.25, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("Cohen's $d$ vs the no-injection control")
    axes[1][0].set_ylabel("valid output (%)")

    hnorm = d[d.layer == args.layer].h_norm.iloc[0]
    fig.suptitle(f"Layer {args.layer}, no space group: manifold sweeps against the linear "
                 f"baseline (repeated in both columns)\n"
                 f"x is the measured push, not each method's own knob, so the methods are "
                 f"comparable (median |$h$| = {hnorm:.1f}).  "
                 f"top row: effect.  bottom row: how much output survived",
                 fontsize=13)
    fig.tight_layout()
    path = OUT / f"density_magnitude_response_layer{args.layer}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    pareto(d, series, args.layer,
           OUT / f"density_pareto_layer{args.layer}.png")
    for name, _, _, f, _ in series:
        print(f"\n{name}")
        print(f[["label", "pct_of_h", "cohens_d", "valid_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
