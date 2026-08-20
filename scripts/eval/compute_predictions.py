#!/usr/bin/env python3
"""
Compute a property for steered/relaxed CIFs, dispatching to the right predictor by
--property (see scripts/predictors.py REGISTRY). Geometric properties are measured
directly; model-based ones (band_gap) load their model once.

Reads ONE CIF file (generated_cifs/<stem> -> cif_steered, or relaxed/<stem> ->
cif_relaxed), restricts to valid (id, sample) via the matching validation file, and
writes <results-dir>/property_predictions/<stem>.parquet with columns:
    id, sample, <base>_raw (from cif_steered), <base> (from cif_relaxed)
where <base> is the predictor's output_base. Raw and relaxed runs share a stem, so
they accumulate into the two columns. Resumable + checkpointed.

For band_gap, output_base is 'predicted_bandgap_ev', so the columns match the old
predict_bandgap.py exactly (combine_bandgap_predictions.py keeps working).

Usage:
    # geometric (no model)
    python scripts/eval/compute_predictions.py --property density_atomic \
        --input steering_results/density_atomic/generated_cifs/steered_test_alpha40.0_layer14.parquet \
        --results-dir steering_results/density_atomic

    # model-based
    python scripts/eval/compute_predictions.py --property band_gap \
        --input steering_results/bandgap/generated_cifs/<stem>.parquet \
        --results-dir steering_results/bandgap
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
from pymatgen.core import Structure

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from predictors import REGISTRY

from utils import postprocess

CHECKPOINT_EVERY = 1000
_parse_errors: list[str] = []


def parse_cif(cif: str):
    try:
        return Structure.from_str(postprocess(cif, "generated"), fmt="cif")
    except Exception as e:
        _parse_errors.append(f"{type(e).__name__}: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--property", required=True, choices=sorted(REGISTRY),
                   help="Which property to compute (dispatches via predictors.REGISTRY)")
    p.add_argument("--input", required=True,
                   help="One CIF file: generated_cifs/<stem> (cif_steered) or "
                        "relaxed/<stem> (cif_relaxed)")
    p.add_argument("--validation", default=None,
                   help="Validation parquet with is_valid (default: "
                        "<results-dir>/validation/<input_stem>.parquet). Invalid CIFs skipped.")
    p.add_argument("--out", default=None,
                   help="Output path (default: <results-dir>/property_predictions/<input_stem>.parquet)")
    p.add_argument("--source", default=None, choices=["cif_relaxed", "cif_steered"],
                   help="CIF column to compute from (default: auto-detect from input)")
    p.add_argument("--results-dir", default="steering_results",
                   help="Base results dir; <results-dir>/{property_predictions,validation}")
    p.add_argument("--fidelity", type=int, default=0, help="band_gap only: DFT fidelity (0=PBE)")
    args = p.parse_args()

    predictor = REGISTRY[args.property](args)
    base = predictor.output_base
    out_col_for = {"cif_steered": f"{base}_raw", "cif_relaxed": base}
    pred_cols = [f"{base}_raw", base]

    in_path = Path(args.input)
    out_dir = Path(args.results_dir) / "property_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{in_path.stem}.parquet"

    print(f"Loading {in_path} ...")
    src = pd.read_parquet(in_path)
    for c in ("id", "sample"):
        if c not in src.columns:
            raise ValueError(f"input missing required column '{c}'")

    if args.source:
        src_col = args.source
    elif "cif_relaxed" in src.columns:
        src_col = "cif_relaxed"
    elif "cif_steered" in src.columns:
        src_col = "cif_steered"
    else:
        raise ValueError("input has neither cif_relaxed nor cif_steered")
    out_col = out_col_for[src_col]

    val_path = Path(args.validation) if args.validation else \
        Path(args.results_dir) / "validation" / f"{in_path.stem}.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"validation flags not found: {val_path} "
                                "(run validate_steered_cifs.py first, or pass --validation)")
    val = pd.read_parquet(val_path, columns=["id", "sample", "is_valid"])
    valid_keys = set(zip(val.loc[val["is_valid"] == True, "id"],
                         val.loc[val["is_valid"] == True, "sample"]))
    print(f"  {len(valid_keys):,} valid (id, sample) from {val_path.name}")

    if out_path.exists():
        pred = pd.read_parquet(out_path)
        print(f"  resuming from {out_path} ({len(pred):,} rows)")
    else:
        pred = src[["id", "sample"]].copy()
    for c in pred_cols:
        if c not in pred.columns:
            pred[c] = None
    pred = pred[["id", "sample"] + pred_cols].reset_index(drop=True)

    pos = {k: i for i, k in enumerate(zip(pred["id"], pred["sample"]))}
    cif_map = dict(zip(zip(src["id"], src["sample"]), src[src_col]))
    todo = [k for k in pos
            if k in valid_keys and pd.isna(pred.at[pos[k], out_col])
            and k in cif_map and pd.notna(cif_map[k])]
    print(f"  {len(pred):,} rows, {len(todo):,} to compute ({src_col} -> {out_col}, valid only)")

    print(f"Setting up '{args.property}' predictor ...")
    predictor.setup()

    ns = npf = nf = 0
    for k in tqdm(todo):
        s = parse_cif(cif_map[k])
        if s is None:
            npf += 1
            continue
        try:
            pred.at[pos[k], out_col] = float(predictor.predict(s))
            ns += 1
        except Exception as e:
            if nf < 5:
                print(f"  COMPUTE FAIL id={k[0]}: {type(e).__name__}: {e}", flush=True)
            nf += 1
        if ns % CHECKPOINT_EVERY == 0 and ns > 0:
            pred.to_parquet(out_path, index=False)

    pred.to_parquet(out_path, index=False)
    print(f"\nSaved {len(pred):,} rows -> {out_path}")
    print(f"Success: {ns:,}  Parse fails: {npf:,}  Other fails: {nf:,}")
    if _parse_errors:
        print("First parse errors:", _parse_errors[:2])

    v = pred[out_col].dropna().astype(float)
    if len(v):
        print(f"\n{out_col} (n={len(v):,}): mean={v.mean():.4g} median={v.median():.4g} "
              f"std={v.std():.4g} min={v.min():.4g} max={v.max():.4g}")


if __name__ == "__main__":
    main()
