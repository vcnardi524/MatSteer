#!/usr/bin/env python3
"""
Predict band gap for original (pre-steering) test set CIFs using MEGNet.

Reads CrystaLLM/cifs_v1_test.pkl.gz, predicts band gap for each structure,
and saves results to steering_results/bandgap_predictions/testset_baseline.parquet with columns:
  id, predicted_bandgap_ev_raw

Usage:
    python scripts/eval/predict_bandgap_testset.py
"""
import argparse
import gzip
import pickle
import warnings
from pathlib import Path

import numpy
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

import matgl
from pymatgen.core import Structure

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import postprocess

FIDELITY = 0  # PBE/GGA
CHECKPOINT_EVERY = 1000


def parse_cif(cif: str) -> Structure | None:
    try:
        return Structure.from_str(postprocess(cif, "testset"), fmt="cif")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="CrystaLLM/cifs_v1_test.pkl.gz")
    parser.add_argument("--out", default=None,
                        help="Default: <results-dir>/property_predictions/testset_baseline.parquet")
    parser.add_argument("--results-dir", default="steering_results")
    parser.add_argument("--fidelity", type=int, default=FIDELITY)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else \
        Path(args.results_dir) / "property_predictions" / "testset_baseline.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    with gzip.open(args.input, "rb") as f:
        test_data = pickle.load(f)
    print(f"  {len(test_data):,} test structures")

    # Resume from existing output if present
    if out_path.exists():
        df = pd.read_parquet(out_path)
        done_ids = set(df[df["predicted_bandgap_ev_raw"].notna()]["id"])
        print(f"  {len(done_ids):,} already predicted, resuming")
    else:
        df = pd.DataFrame({"id": [d[0] for d in test_data]})
        df["predicted_bandgap_ev_raw"] = None
        done_ids = set()

    to_predict = [(id_, cif) for id_, cif in test_data if id_ not in done_ids]
    print(f"  {len(to_predict):,} left to predict")

    print("Loading MEGNet BandGap model ...")
    model = matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
    state_attr = torch.tensor([args.fidelity])
    print(f"  Loaded. Fidelity={args.fidelity} (PBE/GGA)")

    n_success = 0
    n_fail = 0
    n_parse_fail = 0

    id_to_idx = {row["id"]: idx for idx, row in df.iterrows()}

    for id_, cif in tqdm(to_predict):
        struct = parse_cif(cif)
        if struct is None:
            if n_parse_fail < 3:
                print(f"  PARSE FAIL id={id_}", flush=True)
            n_parse_fail += 1
            n_fail += 1
            continue
        try:
            bg = model.predict_structure(structure=struct, state_attr=state_attr)
            df.at[id_to_idx[id_], "predicted_bandgap_ev_raw"] = float(bg)
            n_success += 1
        except Exception as e:
            if n_fail < 5:
                print(f"  FAIL id={id_}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1

        if n_success % CHECKPOINT_EVERY == 0 and n_success > 0:
            df.to_parquet(out_path, index=False)
            print(f"  Checkpoint: {n_success:,} predicted", flush=True)

    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Success: {n_success:,}  Parse fails: {n_parse_fail:,}  Other fails: {n_fail - n_parse_fail:,}")

    bg = df["predicted_bandgap_ev_raw"].dropna().astype(float)
    print(f"\nTest set band gap stats (eV):")
    print(f"  Mean:   {bg.mean():.3f}")
    print(f"  Median: {bg.median():.3f}")
    print(f"  Std:    {bg.std():.3f}")
    print(f"  > 0:    {(bg > 0).sum():,}  ({(bg > 0).mean():.1%})")


if __name__ == "__main__":
    main()
