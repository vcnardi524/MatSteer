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
Each alpha is paired against alpha 0 on the ids those two share. Every comparison is
therefore still perfectly paired, but a prompt lost by one alpha does not remove it
from the other comparisons -- so n differs by row, on purpose.

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
import re
import os as _os
import sys as _sys

import numpy as np
import pandas as pd
from scipy.stats import levene, ttest_ind, ttest_rel, wilcoxon

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from utils import analysis_dir, write_results_table, RESULT_UNITS, RESULT_COLUMNS

# Every method's stem grammar lives in the loader module (`_ds`, imported below), so it
# is written once and both the plot and this table agree on what a filename means.
# `family` (sg vs nosg) is part of the identity: the two are different prompt sets and a
# run may only be paired against a control of its own.
# Methods, in table order. The manifold variants are separate entries because five
# manifold runs share delta 15 and would otherwise collide on one (target, strength) key.
METHODS = ("linear", "pca_centroid", "pca_local",
           "manifold", "manifold_project", "manifold_project_nomu")


def run_meta(stem: str) -> dict:
    """{method, target, layer, family} for a run stem.

    `target` means the property value aimed at for pca runs and the arc step `delta` for
    manifold runs; it is NaN for linear, which has no per-sweep parameter.
    """
    stem = stem[:-len(".parquet")] if stem.endswith(".parquet") else stem
    family = "nosg" if stem.endswith("_nosg") else "sg"
    kind = _ds.kind_of(stem)
    tgt = _ds.sweep_target(stem, kind)
    lm = re.search(r"_layer(\d+)", stem)
    return dict(method=kind, target=float("nan") if tgt is None else float(tgt),
                layer=int(lm.group(1)) if lm else 14, family=family)

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


def analyse(prop: str, method: str, relaxed: bool, family: str = None,
            x_scale: str = "auto", verbose: bool = True,
            target: float = None, layer: int = None, agg: str = "mean") -> pd.DataFrame:
    """Paired stats for every run of one (property, method, source), in the canonical
    schema. Returns an empty frame when the family has no alpha=0 control to pair on.

    This is the one place the paired comparison is computed. Both the per-property
    t-test CSV and the combined steering_runs.csv come out of it, so a change to how a
    run is scored cannot land in one table and not the other.
    """
    spec = _ds.PROPS[prop]
    family = family or spec["default_family"]
    runs = _ds.discover_runs(spec["results_dir"], family, method, target, layer)
    if not runs:
        return pd.DataFrame()

    per = {}
    for a, stem in runs.items():
        try:
            per[a] = _ds.load_alpha(spec["results_dir"], stem, spec["col"],
                                    relaxed, agg, spec["measure"])
        except (SystemExit, FileNotFoundError):
            continue                      # not scored on this source yet
    per = {a: d for a, d in per.items() if not d.empty}
    if not per:
        return pd.DataFrame()
    # A family with no alpha=0 run of its own still gets listed -- population and value,
    # with the paired statistics left empty. Pairing it against another family's control
    # would compare two different prompt sets, and dropping it would hide that the run
    # exists. Only band_gap's sg family is in this state: it was never generated at 0.
    if 0.0 not in per and verbose:
        print(f"  ! {prop}/{method}/{family or 'all'}: no control in this family -- "
              f"listing runs without paired statistics")

    series = {a: d.set_index("id")["value"] for a, d in per.items()}
    scale = spec["scale"] if x_scale == "auto" else x_scale
    if scale == "log" and any((s.to_numpy(float) <= 0).any() for s in series.values()):
        scale = "linear"
    if scale == "log":
        series = {a: np.log10(s) for a, s in series.items()}
    ref = series.get(0.0)
    ctrl_median = float(ref.median()) if ref is not None else np.nan
    source = "relaxed" if relaxed else "raw"

    rows = []
    for a in sorted(series):
        s_a, meta = series[a], run_meta(runs[a])
        r = dict(property=prop, source=source, agg=agg,
                 run=runs[a][:-len(".parquet")],
                 strength=a, unit=RESULT_UNITS[prop], control_median=ctrl_median,
                 median=float(s_a.median()),
                 valid_pct=float(per[a].attrs.get("valid_frac", np.nan)),
                 samples_per_prompt=float(per[a].attrs.get("samples_per_prompt", np.nan)),
                 n_prompts=len(s_a), **meta)
        if a == 0.0:
            r["method"] = "linear"        # the no-injection control belongs to no method
        idx = (ref.index.intersection(s_a.index) if ref is not None else pd.Index([]))
        r["n_paired"] = len(idx) if a != 0.0 else np.nan
        if a != 0.0 and len(idx) >= 10:
            base, x = ref.loc[idx].to_numpy(float), s_a.loc[idx].to_numpy(float)
            diff = x - base
            sd = diff.std(ddof=1)
            try:
                p_w = wilcoxon(x, base).pvalue
            except ValueError:
                p_w = 1.0                 # every pair identical: that IS the answer
            # how much of the way to the target it got, on the value scale
            # `target` is a property value for pca runs but an ARC STEP for manifold
            # runs, so "how far toward the target did it move" is undefined there --
            # leave it NaN rather than scoring delta as though it were A^3/atom.
            target_is_property = np.isfinite(meta["target"]) and \
                not meta["method"].startswith("manifold")
            need = ((np.log10(meta["target"]) if scale == "log" else meta["target"])
                    - ctrl_median) if target_is_property else np.nan
            r.update(mean_diff=float(diff.mean()), sd_diff=float(sd),
                     median_diff=float(np.median(diff)),
                     frac_of_target_move=float(diff.mean() / need) if need else np.nan,
                     cohens_d=float(diff.mean() / sd) if sd else np.nan,
                     t=float(ttest_rel(x, base).statistic),
                     p_paired=float(ttest_rel(x, base).pvalue), p_wilcoxon=float(p_w),
                     p_welch=float(ttest_ind(x, base, equal_var=False).pvalue),
                     sd_zero=float(base.std(ddof=1)), sd_alpha=float(x.std(ddof=1)),
                     p_levene=float(levene(x, base, center="median").pvalue))
        rows.append(r)

    out = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS + [
        "samples_per_prompt",
        "sd_diff", "median_diff", "t", "p_welch", "sd_zero", "sd_alpha", "p_levene"])
    m = out["p_paired"].notna()
    if m.any():
        out.loc[m, "p_holm"] = holm(out.loc[m, "p_paired"].to_numpy(float))
    return out


def build_all(out_path=None) -> pd.DataFrame:
    """Every property x method x source, in one canonical table."""
    frames = []
    for prop in sorted(_ds.PROPS):
        for method in METHODS:
            for relaxed in (False, True):
                # Split every property by family, not just band_gap. density gained a
                # nosg arm at layer 7, and family=None disables the filter in
                # discover_runs -- so sg and nosg runs collided on `strength` and the
                # nosg ones won on sort order, dropping the layer-14 linear baselines.
                for family in ("sg", "nosg"):
                    fam = family
                    # a run is keyed by (layer, target, strength), so each sweep is
                    # discovered and paired separately
                    sweeps = _ds.discover_sweeps(_ds.PROPS[prop]["results_dir"], method)
                    for lay, tgt in (sweeps or [(None, None)]):
                        # mean over the 3 samples, and best-of-3. Both are paired the
                        # same way; agg is part of a row's identity, not a variant of it.
                        for agg in ("mean", "max"):
                            f = analyse(prop, method, relaxed, family, verbose=False,
                                        target=tgt, layer=lay, agg=agg)
                            if not f.empty:
                                frames.append(f)
    key = ["property", "source", "agg", "family", "method", "layer", "target", "strength"]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=key)
    df = df.sort_values(key, na_position="first").reset_index(drop=True)
    path = write_results_table(df, out_path or
                               analysis_dir("v1_all", None, "test") / "steering_runs.csv")
    print(f"Wrote {path}  ({len(df)} rows)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", choices=["mean", "max"], default="mean",
                    help="How the 3 samples per prompt collapse to one value: their mean, "
                         "or the best of the 3. Ignored by --all, which writes both.")
    ap.add_argument("--all", action="store_true",
                    help="Every property x method x source into one canonical table, "
                         "analysis/v1_all/test/steering_runs.csv, instead of a single "
                         "per-property file. Same statistics either way.")
    ap.add_argument("--method", choices=["linear", "pca_centroid", "pca_local"],
                    default="linear",
                    help="Which steering family to test (default: linear)")
    ap.add_argument("--target", type=float, default=None,
                    help="pca methods only: which target sweep to test. A run is keyed "
                         "by (layer, target, strength) -- two targets swept over the "
                         "same t, or one target swept at two layers, are different runs.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Which layer's sweep to test; see --target")
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

    if args.all:
        build_all()
        return

    out = analyse(args.property, args.method, args.relaxed, args.family,
                  args.x_scale, target=args.target, layer=args.layer, agg=args.agg)
    if out.empty:
        raise SystemExit(f"No runs found for {args.property} (method={args.method}).")

    unit = " [log10]" if out["unit"].iloc[0].startswith("log10") else ""
    steered = out[out["strength"] != 0.0]
    print(f"Property: {args.property}  method: {args.method}  "
          f"source: {'relaxed' if args.relaxed else 'raw generated CIF'}")
    print(f"control covers {int(out.loc[out.strength == 0, 'n_prompts'].iloc[0]):,} "
          f"prompts; each run is paired against it on the prompts they share")

    cols = ["strength", "n_paired", "control_median", "median", "mean_diff", "cohens_d",
            "t", "p_paired", "p_holm", "p_wilcoxon", "p_welch", "p_levene"]
    print(f"\n=== paired t-test vs the no-injection control{unit} ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print(steered[cols].to_string(index=False))

    print("\nReading: p says 'the shift is not noise'; cohens_d says whether it matters.")
    for _, r in steered.iterrows():
        if not np.isfinite(r["p_wilcoxon"]):
            continue
        if r["p_wilcoxon"] > 0.05:
            print(f"  {r['strength']:g}: indistinguishable from the control")
            continue
        d = abs(r["cohens_d"])
        size = ("negligible" if d < 0.2 else "small" if d < 0.5 else
                "medium" if d < 0.8 else "large")
        print(f"  {r['strength']:g}: different (d={r['cohens_d']:+.3f}, {size})")

    # The sweep is part of the filename for the same reason it is part of the identity:
    # two targets, or two layers, are different runs and must not share a file.
    dest = analysis_dir("v1_all", None, "test")
    tag = "_relaxed" if args.relaxed else ""
    meth = "" if args.method == "linear" else f"_{args.method}"
    sweep = (f"_target{args.target:g}" if args.target is not None else "") + \
            (f"_layer{args.layer}" if args.layer is not None else "")
    path = write_results_table(
        out, dest / f"{args.property}{meth}{sweep}_steering_ttest{tag}.csv")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
