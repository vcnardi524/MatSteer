#!/usr/bin/env python3
"""
Predict band gap for steered/relaxed CIF structures using the MEGNet
multi-fidelity model, and write a standalone predictions file.

Takes ONE CIF file at a time (raw or relaxed), reads it read-only, and writes a
separate predictions file:

    steering_results/bandgap_predictions/<input_stem>.parquet

with columns:  id, sample, predicted_bandgap_ev_raw, predicted_bandgap_ev
  predicted_bandgap_ev_raw : predicted from cif_steered  (unrelaxed, generated_cifs/)
  predicted_bandgap_ev     : predicted from cif_relaxed  (relaxed, relaxed/)

The source column is auto-detected from the input (cif_relaxed if present, else
cif_steered); override with --source. Only valid CIFs are predicted: is_valid is
read from the matching validation file (steering_results/validation/<stem>.parquet)
and invalid (id, sample) rows are skipped.

Raw and relaxed runs share the same stem, so they write the same output file and
accumulate into the two columns. You can predict the unrelaxed gaps from
generated_cifs/ while relaxation is still producing relaxed/, then predict the
relaxed gaps later.

The model is multi-fidelity with 4 DFT methods:
  0 = PBE (GGA)   <- used here, matches our NOMAD/MP data (99.96% GGA)
  1 = GLLB-SC
  2 = HSE
  3 = SCAN

Usage:
    # unrelaxed gaps (can run while relaxation is still going)
    python scripts/predict_bandgap.py \
        --input steering_results/generated_cifs/steered_test_clean_alpha16.0_layer14.parquet

    # relaxed gaps (once relaxation is done)
    python scripts/predict_bandgap.py \
        --input steering_results/relaxed/steered_test_clean_alpha16.0_layer14.parquet
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

import matgl
from pymatgen.core import Structure

# PBE fidelity index (matches GGA which is 99.96% of our training data)
FIDELITY = 0
MODEL_NAME = "MEGNet-BandGap-mfi-MP-2019.4.1"
CHECKPOINT_EVERY = 1000

# source CIF column -> output prediction column
OUT_COL = {"cif_steered": "predicted_bandgap_ev_raw", "cif_relaxed": "predicted_bandgap_ev"}
PRED_COLS = ["predicted_bandgap_ev_raw", "predicted_bandgap_ev"]

_parse_errors: list[str] = []


def parse_cif(cif: str) -> Structure | None:
    try:
        return Structure.from_str(cif, fmt="cif")
    except Exception as e:
        _parse_errors.append(f"{type(e).__name__}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="One CIF file: generated_cifs/<stem> (cif_steered) "
                             "or relaxed/<stem> (cif_relaxed)")
    parser.add_argument("--validation", default=None,
                        help="Validation parquet with is_valid (default: "
                             "steering_results/validation/<input_stem>.parquet). "
                             "Invalid CIFs are skipped.")
    parser.add_argument("--out", default=None,
                        help="Output path (default: steering_results/bandgap_predictions/<input_stem>.parquet)")
    parser.add_argument("--fidelity", type=int, default=FIDELITY,
                        help="DFT fidelity: 0=PBE, 1=GLLB-SC, 2=HSE, 3=SCAN")
    parser.add_argument("--source", default=None,
                        choices=["cif_relaxed", "cif_steered"],
                        help="CIF column to predict from (default: auto-detect from input)")
    parser.add_argument("--results-dir", default="steering_results",
                        help="Base results dir; <results-dir>/{property_predictions,validation}")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.results_dir) / "property_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{in_path.stem}.parquet"

    print(f"Loading {in_path} ...")
    src = pd.read_parquet(in_path)
    for c in ("id", "sample"):
        if c not in src.columns:
            raise ValueError(f"input missing required column '{c}'")

    # pick the CIF column: explicit --source, else auto-detect from the file
    if args.source:
        src_col = args.source
    elif "cif_relaxed" in src.columns:
        src_col = "cif_relaxed"
    elif "cif_steered" in src.columns:
        src_col = "cif_steered"
    else:
        raise ValueError("input has neither cif_relaxed nor cif_steered")
    if src_col not in src.columns:
        raise ValueError(f"input missing chosen source column '{src_col}'")
    out_col = OUT_COL[src_col]

    # restrict to valid CIFs via the validation flags (skip invalid)
    val_path = Path(args.validation) if args.validation else \
        Path(args.results_dir) / "validation" / f"{in_path.stem}.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"validation flags not found: {val_path} "
                                "(run validate_steered_cifs.py first, or pass --validation)")
    val = pd.read_parquet(val_path, columns=["id", "sample", "is_valid"])
    valid_keys = set(zip(val.loc[val["is_valid"] == True, "id"],
                         val.loc[val["is_valid"] == True, "sample"]))
    print(f"  {len(valid_keys):,} valid (id, sample) from {val_path.name}")

    # build (or resume) the predictions frame, keyed by id+sample
    if out_path.exists():
        pred = pd.read_parquet(out_path)
        print(f"  resuming from {out_path} ({len(pred):,} rows)")
    else:
        pred = src[["id", "sample"]].copy()
    for c in PRED_COLS:
        if c not in pred.columns:
            pred[c] = None
    pred = pred[["id", "sample"] + PRED_COLS].reset_index(drop=True)

    # position lookup: (id, sample) -> row index in `pred`
    pos = {k: i for i, k in enumerate(zip(pred["id"], pred["sample"]))}
    cif_map = dict(zip(zip(src["id"], src["sample"]), src[src_col]))
    todo = [k for k in pos
            if k in valid_keys and pd.isna(pred.at[pos[k], out_col])
            and k in cif_map and pd.notna(cif_map[k])]
    print(f"  {len(pred):,} rows, {len(todo):,} to predict ({src_col} -> {out_col}, valid only)")

    print(f"Loading MEGNet BandGap model ({MODEL_NAME}) ...")
    model = matgl.load_model(MODEL_NAME)
    state_attr = torch.tensor([args.fidelity])
    print(f"  Loaded. Using fidelity={args.fidelity} (PBE/GGA)")

    n_success = n_parse_fail = n_fail = 0
    for k in tqdm(todo):
        struct = parse_cif(cif_map[k])
        if struct is None:
            if n_parse_fail < 3:
                print(f"  PARSE FAIL id={k[0]} sample={k[1]}", flush=True)
            n_parse_fail += 1
            continue
        try:
            bg = model.predict_structure(structure=struct, state_attr=state_attr)
            pred.at[pos[k], out_col] = float(bg)
            n_success += 1
        except Exception as e:
            if n_fail < 5:
                print(f"  PREDICT FAIL id={k[0]}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1

        if n_success % CHECKPOINT_EVERY == 0 and n_success > 0:
            pred.to_parquet(out_path, index=False)
            print(f"  Checkpoint: {n_success:,} predicted", flush=True)

    pred.to_parquet(out_path, index=False)
    print(f"\nSaved {len(pred):,} rows -> {out_path}")
    print(f"Success: {n_success:,}  Parse fails: {n_parse_fail:,}  Other fails: {n_fail:,}")
    if _parse_errors:
        print("First parse errors:")
        for e in _parse_errors[:3]:
            print(f"  {e}")

    bg = pred[out_col].dropna().astype(float)
    if len(bg):
        print(f"\n{out_col} (eV, n={len(bg):,}):")
        print(f"  mean={bg.mean():.3f}  median={bg.median():.3f}  std={bg.std():.3f}  "
              f"min={bg.min():.3f}  max={bg.max():.3f}  >0: {(bg > 0).mean():.1%}")


if __name__ == "__main__":
    main()
