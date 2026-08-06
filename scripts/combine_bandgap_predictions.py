#!/usr/bin/env python3
"""
Combine per-run band-gap prediction files into one wide table for easy access.

Each per-run file in steering_results/bandgap_predictions/ has columns
    id, sample, predicted_bandgap_ev_raw, predicted_bandgap_ev
This script pivots them into ONE table keyed on (id, sample), with a pair of
columns per run so raw (unrelaxed) and relaxed gaps are easy to grab side by side:

    bandgap_raw_<run>   from the raw generated CIF   (unrelaxed)
    bandgap_<run>       from the M3GNet-relaxed CIF  (relaxed)

matching the source convention where raw carries the _raw marker and relaxed is
unmarked (predicted_bandgap_ev_raw vs. predicted_bandgap_ev).

The <run> label is derived from the file stem, e.g.
    steered_test_clean_alpha40.0_layer14        -> alpha40
    steered_test_clean_alpha40.0_layer14_nosg   -> alpha40_nosg
    testset_baseline                            -> baseline

Runs are outer-joined on (id, sample), so a run that didn't cover a given
prompt/sample simply has NaN in its columns.

Output (default): steering_results/bandgap_all.parquet

Usage:
    python scripts/combine_bandgap_predictions.py
    python scripts/combine_bandgap_predictions.py --exclude-baseline
"""
import argparse
import re
from pathlib import Path
from functools import reduce

import pandas as pd

DEFAULT_RESULTS = "steering_results"


def run_label(stem: str) -> str:
    """Short run tag from a file stem, e.g. steered_test_clean_alpha40.0_layer14_nosg -> alpha40_nosg."""
    if stem == "testset_baseline":
        return "baseline"
    s = stem
    if s.startswith("steered_test_"):
        s = s[len("steered_test_"):]
    s = s.replace("clean_", "").replace("_layer14", "")
    s = re.sub(r"(alpha-?\d+)\.0(?=_|$)", r"\1", s)  # alpha40.0 -> alpha40, alpha-16.0 -> alpha-16
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS,
                        help="Base results dir; defaults derive <results-dir>/property_predictions "
                             "and <results-dir>/property_all.parquet")
    parser.add_argument("--in-dir", default=None,
                        help="Dir of per-run prediction parquets (default: "
                             "<results-dir>/property_predictions)")
    parser.add_argument("--out", default=None,
                        help="Combined output parquet (default: <results-dir>/property_all.parquet)")
    parser.add_argument("--exclude-baseline", action="store_true",
                        help="Skip testset_baseline.parquet (the un-steered baseline)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else Path(args.results_dir) / "property_predictions"
    out_path = Path(args.out) if args.out else Path(args.results_dir) / "property_all.parquet"
    out_resolved = out_path.resolve()

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {in_dir}")

    frames = []
    for f in files:
        if f.resolve() == out_resolved:
            continue  # never fold the output back into itself
        if args.exclude_baseline and f.stem == "testset_baseline":
            continue

        df = pd.read_parquet(f)
        if not {"id", "sample"}.issubset(df.columns):
            print(f"  SKIP {f.name}: missing id/sample")
            continue

        label = run_label(f.stem)
        # mirror the source convention: raw carries the _raw marker, relaxed is unmarked
        raw_col, rel_col = f"bandgap_raw_{label}", f"bandgap_{label}"
        keep = df[["id", "sample"]].copy()
        # coerce: the relaxed col is often object/None until computed
        keep[raw_col] = pd.to_numeric(df.get("predicted_bandgap_ev_raw"), errors="coerce")
        keep[rel_col] = pd.to_numeric(df.get("predicted_bandgap_ev"), errors="coerce")

        print(f"  {f.stem}: {len(keep):,} rows -> {label} "
              f"(raw={int(keep[raw_col].notna().sum()):,}, "
              f"relaxed={int(keep[rel_col].notna().sum()):,})")
        frames.append(keep)

    if not frames:
        raise SystemExit("Nothing to combine.")

    combined = reduce(
        lambda l, r: l.merge(r, on=["id", "sample"], how="outer"),
        frames,
    ).sort_values(["id", "sample"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"\nCombined {len(frames)} runs -> {out_path}  "
          f"({len(combined):,} rows, {len(combined.columns)} cols)")


if __name__ == "__main__":
    main()
