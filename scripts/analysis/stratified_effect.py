#!/usr/bin/env python3
"""Does steering work uniformly across the density range, or only in part of it?

A single pooled Cohen's d asks whether the whole distribution shifted. It cannot say
whether a prompt whose true structure sits at 14 A^3/atom responds like one already at
32 -- and steering density UPWARD has obviously different headroom in those two cases.
This stratifies the same paired comparison by where the prompt STARTED.

Buckets are on the GROUND-TRUTH density of the prompt's source structure, width 1, not
on the alpha=0 generation. That matters: control and steered are both noisy draws from
the same prompt, so bucketing on the realised control value enriches low buckets for
prompts whose control happened to sample low, while the steered draw does not share that
noise. The difference would then trend downward across buckets from regression to the
mean alone. Ground truth is fixed, so it cannot do that.

Within each bucket the comparison is the same as the main table: paired on prompt id,
Cohen's d on log10 density (matching steering_ttest.py's log scale), and the difference
of means also reported in A^3/atom because that is the readable unit.

Usage:
    python scripts/analysis/stratified_effect.py \
        --runs steered_manifold_test_d2_residual_s9_k64_layer7_nosg \
               steered_test_alpha32.0_layer7_nosg
"""
import argparse
import importlib.util as _ilu
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
_spec = _ilu.spec_from_file_location(
    "ds", HERE.parent / "plots" / "plot_steering_distribution_shift.py")
_ds = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ds)

RESULTS_DIR = "density_atomic"
CONTROL = "steered_test_alpha0.0_layer7_nosg"
OUT = Path("analysis/v1_all/test/plots")
COLOR = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
MARKER = ["s", "o", "^", "D"]


def per_prompt(stem):
    """{id: mean density over that prompt's valid samples} for one run."""
    d = _ds.load_alpha(RESULTS_DIR, f"{stem}.parquet", "density_atomic",
                       relaxed=False, agg="mean", measure="model")
    return d.set_index("id")["value"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--control", default=CONTROL)
    ap.add_argument("--labels", default="density_atomic_v1.parquet")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--min-per-bucket", type=int, default=20,
                    help="drop buckets thinner than this; a d on 6 prompts is noise")
    ap.add_argument("--out-stem", default="density_stratified_effect")
    args = ap.parse_args()

    ctrl = per_prompt(args.control)
    truth = (pd.read_parquet(args.labels, columns=["id", "density_atomic"])
             .dropna().set_index("id")["density_atomic"])
    truth = truth[~truth.index.duplicated()]
    shared = ctrl.index.intersection(truth.index)
    print(f"control {args.control}: {len(ctrl):,} prompts, "
          f"{len(shared):,} with a ground-truth density")

    bucket = np.floor(truth[shared] / args.width) * args.width
    rows = []
    for stem in args.runs:
        s = per_prompt(stem)
        ids = shared.intersection(s.index)
        df = pd.DataFrame({"truth": truth[ids], "ctrl": ctrl[ids], "steer": s[ids],
                           "bucket": bucket[ids]})
        # log10 to match the main table's scale; the raw difference is reported too
        df["diff_log"] = np.log10(df.steer) - np.log10(df.ctrl)
        for b, g in df.groupby("bucket"):
            if len(g) < args.min_per_bucket:
                continue
            sd = g.diff_log.std(ddof=1)
            rows.append(dict(run=stem, bucket=b, n=len(g),
                             truth_mean=g.truth.mean(), ctrl_mean=g.ctrl.mean(),
                             steer_mean=g.steer.mean(),
                             mean_diff_A3=g.steer.mean() - g.ctrl.mean(),
                             cohens_d=g.diff_log.mean() / sd if sd else np.nan))
        print(f"  {stem}: {len(ids):,} paired prompts, "
              f"{sum(r['run'] == stem for r in rows)} buckets kept")

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / f"{args.out_stem}.csv", index=False, float_format="%.6g")

    fig, (ax_d, ax_m) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for i, stem in enumerate(args.runs):
        g = d[d.run == stem].sort_values("bucket")
        short = (stem.replace("steered_manifold_test_", "manifold ")
                     .replace("steered_test_", "linear ")
                     .replace("_k64_layer7_nosg", "").replace("_layer7_nosg", "")
                     .replace("_residual", ""))
        for ax, col in ((ax_d, "cohens_d"), (ax_m, "mean_diff_A3")):
            ax.plot(g.bucket, g[col], "-", color=COLOR[i % 4], lw=2,
                    marker=MARKER[i % 4], ms=6, label=short if ax is ax_d else None)
    # how many prompts each bucket rests on -- the tails are thin
    counts = d.groupby("bucket")["n"].max()
    for b, n in counts.items():
        ax_d.annotate(f"{n}", (b, ax_d.get_ylim()[0]), ha="center", va="bottom",
                      fontsize=6.5, color="#999")

    for ax in (ax_d, ax_m):
        ax.axhline(0, color="#999", lw=1)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax_d.set_ylabel("Cohen's $d$ within the bucket")
    ax_d.legend(frameon=False, fontsize=10, loc="upper right")
    ax_d.set_title("effect size, by where the prompt started "
                   "(grey = prompts in bucket)", fontsize=11, loc="left")
    ax_m.set_ylabel("mean(steered) - mean(control)   (Å³/atom)")
    ax_m.set_xlabel(f"ground-truth density of the source structure "
                    f"(Å³/atom, bucket width {args.width:g})")
    ax_m.set_title("difference of means, same buckets", fontsize=11, loc="left")

    fig.suptitle("Steering effect stratified by the prompt's true density\n"
                 "buckets are on ground truth, not on the control generation, so "
                 "regression to the mean cannot manufacture a trend", fontsize=13)
    fig.tight_layout()
    path = OUT / f"{args.out_stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {path}")
    with pd.option_context("display.width", 200):
        print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
