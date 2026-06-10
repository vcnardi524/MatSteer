#!/usr/bin/env python3
"""
Predict band gap for relaxed CIF structures using MEGNet multi-fidelity model.

Reads a validation parquet that has a cif_relaxed column (output of
relax_steered_cifs.py) and adds a predicted_bandgap_ev column in place.

The model is multi-fidelity with 4 DFT methods:
  0 = PBE (GGA)   ← used here, matches our NOMAD/MP data (99.96% GGA)
  1 = GLLB-SC
  2 = HSE
  3 = SCAN

Usage:
    python scripts/predict_bandgap.py \
        --input validation/steered_test_p5_alpha1.0_layer14.parquet

Output:
    same parquet with added column: predicted_bandgap_ev
"""
import argparse
import warnings
from pathlib import Path

import numpy
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

import matgl
from pymatgen.core import Structure

# PBE fidelity index (matches GGA which is 99.96% of our training data)
FIDELITY = 0


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
                        help="Validation parquet")
    parser.add_argument("--out", default=None,
                        help="Output parquet path (default: same as input)")
    parser.add_argument("--fidelity", type=int, default=FIDELITY,
                        help="DFT fidelity: 0=PBE, 1=GLLB-SC, 2=HSE, 3=SCAN")
    parser.add_argument("--source", default="cif_relaxed",
                        choices=["cif_relaxed", "cif_steered"],
                        help="CIF column to predict from (default: cif_relaxed)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path

    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)

    src_col = args.source
    out_col = "predicted_bandgap_ev" if src_col == "cif_relaxed" else "predicted_bandgap_ev_raw"

    if src_col not in df.columns:
        raise ValueError(f"No {src_col} column in parquet")

    if out_col not in df.columns:
        df[out_col] = None

    to_predict = df[df[src_col].notna() & df[out_col].isna()]
    print(f"  {len(df):,} total rows, {len(to_predict):,} to predict (source: {src_col})")

    print("Loading MEGNet BandGap model ...")
    model = matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
    state_attr = torch.tensor([args.fidelity])
    print(f"  Loaded. Using fidelity={args.fidelity} (PBE/GGA)")

    n_fail = 0
    n_parse_fail = 0
    n_success = 0
    CHECKPOINT_EVERY = 1000

    for idx, row in tqdm(to_predict.iterrows(), total=len(to_predict)):
        struct = parse_cif(row[src_col])
        if struct is None:
            if n_parse_fail < 3:
                print(f"  PARSE FAIL idx={idx}", flush=True)
            n_parse_fail += 1
            n_fail += 1
            continue
        try:
            bg = model.predict_structure(structure=struct, state_attr=state_attr)
            df.at[idx, out_col] = float(bg)
            n_success += 1
        except Exception as e:
            if n_fail < 5:
                print(f"  PREDICT FAIL idx={idx}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1

        if n_success % CHECKPOINT_EVERY == 0 and n_success > 0:
            df.to_parquet(out_path, index=False)
            print(f"  Checkpoint: {n_success:,} predicted so far", flush=True)

    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Success: {n_success:,}  Parse fails: {n_parse_fail:,}  Other fails: {n_fail - n_parse_fail:,}")
    if _parse_errors:
        print(f"First parse errors:")
        for e in _parse_errors[:3]:
            print(f"  {e}")

    bg = df[out_col].dropna().astype(float)
    print(f"\nBand gap stats (eV):")
    print(f"  Mean:   {bg.mean():.3f}")
    print(f"  Median: {bg.median():.3f}")
    print(f"  Std:    {bg.std():.3f}")
    print(f"  Min:    {bg.min():.3f}")
    print(f"  Max:    {bg.max():.3f}")
    print(f"  > 0:    {(bg > 0).sum():,}  ({(bg > 0).mean():.1%})")


if __name__ == "__main__":
    main()
