#!/usr/bin/env python3
"""
Is a steered distribution statistically different from the alpha=0 control?

The distribution-shift plot answers "does steering move the curve away from the ground
truth"; this answers the narrower question the plot cannot: for each alpha, is the
generated distribution distinguishable from what the SAME model produced with the hook
adding nothing (alpha=0)? Ground truth never enters -- alpha=0 is the only baseline.

WHY PAIRED
----------
Every alpha generates from the SAME prompts, so alpha and the control share ids. The
paired test differences out the composition effect (a prompt for a heavy oxide has a big
cell at every alpha), which is far and away the largest source of variance here. An
unpaired Welch test on the same numbers is reported alongside it as a sanity check --
if the two disagree badly, the pairing is carrying the result.

Values and their filtering are read exactly as plot_steering_distribution_shift.py reads
them (same loaders, same valid-only rule, same one-point-per-prompt mean, same log10 for
density_atomic), so the numbers here and the curves there describe the same samples.
Alphas are compared on the ids common to ALL plotted alphas, so every row is the same
population of prompts.

WHAT THE COLUMNS MEAN
---------------------
  n              paired prompts (identical for every row by construction)
  mean_diff      mean(alpha) - mean(alpha=0), in the units of the x axis (log10 units
                 for density_atomic)
  cohens_d       mean_diff / sd(diff) -- paired effect size. This is the number to read.
                 With n in the thousands a t-test calls a 0.2% shift "significant"; d
                 says whether that shift is large relative to the spread it moved in.
  t, p_paired    scipy ttest_rel
  p_holm         p_paired after Holm-Bonferroni across the alphas in this table
  p_wilcoxon     signed-rank, distribution-free. Band gaps pile up at 0 and volumes are
                 log-normal-ish, so the t-test's normality assumption is shaky; if
                 Wilcoxon and the t-test disagree, trust Wilcoxon.
  p_welch        unpaired Welch, ignores the pairing (sanity check)
  p_levene       Brown-Forsythe: are the SPREADS different? A steering vector can widen
                 the distribution without moving its mean, and the t-test is blind to that.

OUTPUT
------
analysis/v1_all/test/{prop}_steering_ttest.csv

Usage:
    python scripts/analysis/steering_ttest.py --property band_gap
    python scripts/analysis/steering_ttest.py --property density_atomic
    python scripts/analysis/steering_ttest.py --property band_gap --alphas -16 0 16 40 60
"""
import argparse
import importlib.util as _ilu
import os as _os
import sys as _sys

import numpy as np
import pandas as pd
from scipy.stats import levene, ttest_ind, ttest_rel, wilcoxon

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from utils import analysis_dir

# The loaders live in the plot script; load it by file path because `plots` is not a
# package and the filename is not importable as a module name from here.
_spec = _ilu.spec_from_file_location(
    "dist_shift",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                  "plots", "plot_steering_distribution_shift.py"))
_ds = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ds)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--property", default="band_gap", choices=sorted(_ds.PROPS))
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="Steering strengths to test (default: every run on disk)")
    ap.add_argument("--family", choices=["nosg", "sg"], default=None,
                    help="band_gap only: prompts without/with a space-group header")
    ap.add_argument("--relaxed", action="store_true",
                    help="Read the M3GNet-relaxed value instead of the raw generated one")
    ap.add_argument("--x-scale", choices=["auto", "linear", "log"], default="auto",
                    help="log tests the multiplicative shift; matches the plot by default")
    args = ap.parse_args()

    spec = _ds.PROPS[args.property]
    family = args.family or spec["default_family"]
    print(f"Property: {args.property}  ({spec['label']})")
    print(f"Value source: {'relaxed' if args.relaxed else 'raw generated CIF'}   "
          f"family: {family}")

    runs = _ds.discover_runs(spec["results_dir"], family)
    if not runs:
        raise SystemExit(f"No runs found for {args.property} (family={family}).")
    print(f"Runs on disk: alphas {sorted(runs)}")
    if args.alphas is not None:
        missing = [a for a in args.alphas if a not in runs]
        if missing:
            print(f"  ! requested but absent: {missing} -- skipping them")
        alphas = [a for a in args.alphas if a in runs]
    else:
        alphas = sorted(runs)
    if 0.0 not in alphas:
        raise SystemExit("alpha=0 is the baseline for this test and is not in the set.")
    if len(alphas) < 2:
        raise SystemExit("Need alpha=0 plus at least one steered run.")

    print("Loading runs ...")
    per_alpha = {}
    for a in alphas:
        print(f"  alpha {a:g}  [{runs[a]}]")
        # agg='mean' -- one point per prompt, so the pairing is well defined
        per_alpha[a] = _ds.load_alpha(spec["results_dir"], runs[a], spec["col"],
                                      args.relaxed, "mean", spec["measure"])

    common = sorted(set.intersection(*(set(d["id"]) for d in per_alpha.values())))
    print(f"\nPrompts common to all {len(alphas)} alphas: {len(common):,}")
    if len(common) < 10:
        raise SystemExit("Too few shared prompts to compare.")

    # One column per alpha, rows aligned on id -> every comparison is paired by row.
    wide = pd.DataFrame({"id": common}).set_index("id")
    for a in alphas:
        d = per_alpha[a].set_index("id")["value"]
        wide[a] = d.reindex(common)

    scale = spec["scale"] if args.x_scale == "auto" else args.x_scale
    if scale == "log" and (wide.to_numpy(float) <= 0).any():
        print("  ! non-positive values present -- testing on the linear scale")
        scale = "linear"
    if scale == "log":
        wide = np.log10(wide)
    unit = " [log10]" if scale == "log" else ""
    print(f"Testing on the {scale} scale{unit}")

    base = wide[0.0].to_numpy(float)
    rows = []
    for a in alphas:
        if a == 0.0:
            continue
        x = wide[a].to_numpy(float)
        diff = x - base
        t, p_t = ttest_rel(x, base)
        sd = diff.std(ddof=1)
        # Wilcoxon errors out when every pair is identical; that IS the answer, so say so.
        try:
            p_w = wilcoxon(x, base).pvalue
        except ValueError:
            p_w = 1.0
        rows.append(dict(
            alpha=a, n=len(diff),
            mean_alpha=float(x.mean()), mean_zero=float(base.mean()),
            mean_diff=float(diff.mean()), sd_diff=float(sd),
            median_diff=float(np.median(diff)),
            cohens_d=float(diff.mean() / sd) if sd else np.nan,
            t=float(t), p_paired=float(p_t),
            p_wilcoxon=float(p_w),
            p_welch=float(ttest_ind(x, base, equal_var=False).pvalue),
            sd_alpha=float(x.std(ddof=1)), sd_zero=float(base.std(ddof=1)),
            p_levene=float(levene(x, base, center="median").pvalue),
        ))

    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("p_wilcoxon"), "p_holm",
               holm(out["p_paired"].to_numpy(float)))

    cols = ["alpha", "n", "mean_zero", "mean_alpha", "mean_diff", "cohens_d",
            "t", "p_paired", "p_holm", "p_wilcoxon", "p_welch",
            "sd_zero", "sd_alpha", "p_levene"]
    print(f"\n=== paired t-test vs alpha=0{unit} ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print(out[cols].to_string(index=False))

    print("\nReading: p says 'the shift is not noise'; cohens_d says whether it matters.")
    for r in rows:
        if r["p_wilcoxon"] > 0.05:
            print(f"  alpha {r['alpha']:g}: indistinguishable from the control")
            continue
        d = abs(r["cohens_d"])
        size = ("negligible" if d < 0.2 else "small" if d < 0.5 else
                "medium" if d < 0.8 else "large")
        print(f"  alpha {r['alpha']:g}: different (d={r['cohens_d']:+.3f}, {size})")

    dest = analysis_dir("v1_all", None, "test")
    tag = "_relaxed" if args.relaxed else ""
    path = dest / f"{args.property}_steering_ttest{tag}.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
