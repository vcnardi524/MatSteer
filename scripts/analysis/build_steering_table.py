#!/usr/bin/env python3
"""
Every steering run, both properties, both methods, in one table.

Replaces the four ad-hoc CSVs this analysis grew (pca_centroid_vs_linear_*,
pca_centroid_density_targets, pca_local_vs_centroid_*, pca_centroid_bandgap_raw_vs_*),
which each keyed the run differently and disagreed about what `median` meant. The
schema is utils.RESULT_COLUMNS; see there for what the columns are.

One row per (property, method, layer, target, strength, source). Runs are discovered on
disk, so a new sweep appears here as soon as it has predictions and validation.

Values are read exactly as the plots and t-tests read them: valid samples only, one
point per prompt (the mean of its valid samples), log10 for density_atomic. Each run is
paired against its own method-appropriate control on the prompts the two share -- the
alpha=0 run, which is the t=0 control too since neither method injects anything at 0.

Usage:
    python scripts/analysis/build_steering_table.py
    python scripts/analysis/build_steering_table.py --out analysis/v1_all/test/steering_runs.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import analysis_dir, write_results_table, RESULT_UNITS

# results dir, prediction column base, and the alpha=0 control for each property
PROPS = {
    "density_atomic": dict(results_dir="density_atomic", base="density_atomic",
                           controls={"sg": "steered_test_alpha0.0_layer14"}, log10=True),
    # Only the nosg family was ever generated at alpha=0, so sg runs have no control to
    # pair against. They are still listed, with the paired statistics left empty --
    # pairing them against the nosg control would compare two different prompt sets.
    "band_gap":       dict(results_dir="bandgap", base="predicted_bandgap_ev",
                           controls={"nosg": "steered_test_clean_alpha0.0_layer14_nosg"},
                           log10=False),
}

# What the filename tells us. The pca methods carry target/t/k, the linear ones alpha.
RUN_RE = [
    ("pca_local", re.compile(r"^steered_pcalocal_\w+?_target([\d.]+)_t([\d.]+)_k\d+_nb\d+_layer(\d+)")),
    ("pca_centroid", re.compile(r"^steered_pca_\w+?_target([\d.]+)_t([\d.]+)_k\d+_layer(\d+)")),
]
LINEAR_RE = re.compile(r"^steered_test_(?:clean_)?alpha(-?[\d.]+)_layer(\d+)")


def parse_run(stem: str):
    """(method, target, strength, layer, family) from a run stem, or None."""
    family = "nosg" if stem.endswith("_nosg") else "sg"
    for method, rx in RUN_RE:
        m = rx.match(stem)
        if m:
            return method, float(m.group(1)), float(m.group(2)), int(m.group(3)), family
    m = LINEAR_RE.match(stem)
    if m:
        return "linear", np.nan, float(m.group(1)), int(m.group(2)), family
    return None


def load_run(results_dir: str, stem: str, col: str, log10: bool):
    """(per-prompt series, valid fraction) for one run, or (None, nan) if not scored."""
    pred_path = f"steering_results/{results_dir}/property_predictions/{stem}.parquet"
    val_path = f"steering_results/{results_dir}/validation/{stem}.parquet"
    if not (os.path.exists(pred_path) and os.path.exists(val_path)):
        return None, np.nan
    pred = pd.read_parquet(pred_path)
    if col not in pred.columns:
        return None, np.nan
    val = pd.read_parquet(val_path, columns=["id", "sample", "is_valid"])
    df = pred.merge(val, on=["id", "sample"], how="left")
    valid_pct = float(df["is_valid"].fillna(False).mean())
    df = df[df["is_valid"].fillna(False) & df[col].notna()]
    if df.empty:
        return None, valid_pct
    s = df.groupby("id")[col].mean()
    return (np.log10(s) if log10 else s), valid_pct


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order, m, adj, running = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Default: analysis/v1_all/test/steering_runs.csv")
    args = ap.parse_args()

    rows = []
    for prop, spec in PROPS.items():
        for source, col in [("raw", f"{spec['base']}_raw"), ("relaxed", spec["base"])]:
            controls, ctrl_medians = {}, {}
            for fam, cstem in spec["controls"].items():
                c, cvalid = load_run(spec["results_dir"], cstem, col, spec["log10"])
                if c is None:
                    print(f"  ! {prop}/{source}: control {cstem} has no {col}")
                    continue
                controls[fam], ctrl_medians[fam] = c, float(c.median())
                rows.append(dict(property=prop, method="linear", layer=14, family=fam,
                                 target=np.nan, strength=0.0, source=source, run=cstem,
                                 valid_pct=cvalid, n_prompts=len(c), n_paired=np.nan,
                                 unit=RESULT_UNITS[prop], control_median=ctrl_medians[fam],
                                 median=ctrl_medians[fam], mean_diff=np.nan,
                                 frac_of_target_move=np.nan, cohens_d=np.nan,
                                 p_paired=np.nan, p_wilcoxon=np.nan))
            if not controls:
                continue

            for f in sorted(glob.glob(
                    f"steering_results/{spec['results_dir']}/property_predictions/*.parquet")):
                stem = os.path.basename(f)[:-len(".parquet")]
                if stem in spec["controls"].values() or stem == "testset_baseline":
                    continue
                parsed = parse_run(stem)
                if parsed is None:
                    continue
                method, target, strength, layer, family = parsed
                s, valid_pct = load_run(spec["results_dir"], stem, col, spec["log10"])
                if s is None:
                    continue
                control = controls.get(family)
                ctrl_median = ctrl_medians.get(family, np.nan)
                idx = (control.index.intersection(s.index) if control is not None
                       else pd.Index([]))
                r = dict(property=prop, method=method, layer=layer, family=family,
                         target=target, strength=strength, source=source, run=stem,
                         valid_pct=valid_pct, n_prompts=len(s), n_paired=len(idx),
                         unit=RESULT_UNITS[prop], control_median=ctrl_median,
                         median=float(s.median()))
                if len(idx) >= 10:
                    diff = (s.loc[idx] - control.loc[idx]).to_numpy()
                    sd = diff.std(ddof=1)
                    # how much of the way to the target the run actually got; the target
                    # is in the property's own units, so convert it to the value scale
                    need = ((np.log10(target) if spec["log10"] else target) - ctrl_median
                            if np.isfinite(target) else np.nan)
                    r.update(mean_diff=float(diff.mean()),
                             frac_of_target_move=float(diff.mean() / need) if need else np.nan,
                             cohens_d=float(diff.mean() / sd) if sd else np.nan,
                             p_paired=float(ttest_rel(s.loc[idx], control.loc[idx]).pvalue),
                             p_wilcoxon=float(wilcoxon(diff).pvalue))
                rows.append(r)

    df = pd.DataFrame(rows)
    # Holm across the steered runs within each property+source family, so the correction
    # covers the comparisons that were actually made together.
    df["p_holm"] = np.nan
    for _, g in df.groupby(["property", "source", "family"]):
        m = g["p_paired"].notna()
        if m.any():
            df.loc[g.index[m], "p_holm"] = holm(g.loc[m, "p_paired"].to_numpy())

    df = df.sort_values(["property", "source", "family", "method", "layer",
                         "target", "strength"],
                        na_position="first").reset_index(drop=True)
    out = args.out or (analysis_dir("v1_all", None, "test") / "steering_runs.csv")
    write_results_table(df, out)
    print(df[["property", "source", "family", "method", "layer", "target", "strength",
              "valid_pct", "n_paired", "median", "mean_diff", "cohens_d",
              "p_holm"]].to_string(index=False))
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
