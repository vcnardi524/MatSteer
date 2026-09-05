#!/usr/bin/env python3
"""
Relax steered CIF structures using M3GNet universal potential.

Reads validity flags from a validation parquet (output of
validate_steered_cifs.py) and the raw CIFs from the matching generated_cifs
parquet, joins them on (id, sample), skips invalid CIFs, relaxes each valid
structure with M3GNet-PES, and writes a relaxed CIF store. Resumes from a
previous partial output; checkpoints periodically.

WHAT CONTROLS SPEED AND ACCURACY
--------------------------------
Three flags, and they interact. `Relaxer.relax(struct, fmax=..., steps=...)` runs an ASE
optimiser until the largest force falls below `--fmax` OR it has taken `--steps` steps,
whichever comes first; `--timeout` then caps the wall clock independently of both.

  --fmax     eV/Å, default 0.05. The convergence criterion, and the only one of the
             three that sets the QUALITY of a converged structure. Lower is a tighter
             relaxation and costs more steps -- roughly, halving fmax adds steps until
             the forces are that much smaller. 0.05 is the usual materials-informatics
             default and matches what MP-style workflows accept. Raising it to 0.1 is
             noticeably faster and gives structures that are still near equilibrium;
             dropping to 0.01 mostly buys precision the downstream property models
             cannot resolve.
  --steps    default 200 here, 500 in experiments_configs/relax_steered.conf. A ceiling,
             not a target: most structures converge well inside it and stop early. It
             only binds on hard cases, and a structure that hits the ceiling is returned
             UNCONVERGED with no error -- it is counted as a success. Raising it trades
             time for fewer silently-unconverged rows.
  --timeout  seconds per structure, default 120, 60 in the conf. A SIGALRM wall-clock
             cap that abandons the structure and counts it under "Timed out". This is
             what bounds the tail: a few pathological cells would otherwise run for
             minutes each. With --steps 500 and --timeout 60 the timeout is usually the
             binding constraint, so raising steps alone changes little.

The model itself is the fourth lever and is not a flag: M3GNet-PES-MatPES-PBE-2025.2,
loaded at line ~108. A different potential changes both cost and the forces being
minimised.

Observed rate on this cluster: ~3.5 s/structure at fmax 0.05, steps 500, timeout 60
(2,515 structures in 2h46m; 2,819 in 2h19m). Budget from that, not from --steps.

Read the tail line -- "Success / Failed / Timed out" -- before trusting a run: a high
timeout count means the settings were too tight for that arm, and those rows carry no
relaxed CIF, so they silently drop out of any downstream join.

Usage:
    python scripts/eval/relax_steered_cifs.py \
        --input steering_results/validation/steered_test_clean_alpha16.0_layer14.parquet

Output:
    steering_results/relaxed/<input_stem>.parquet with columns: id, sample, cif_relaxed
"""
import argparse
import signal
import warnings
from pathlib import Path

import numpy
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

import matgl
from matgl.ext.ase import Relaxer
from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp as _SymmOp
if not hasattr(_SymmOp, "as_xyz_string"):
    _SymmOp.as_xyz_string = _SymmOp.as_xyz_str

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import postprocess


def parse_cif(cif: str) -> Structure | None:
    try:
        processed = postprocess(cif, "relax")
        return Structure.from_str(processed, fmt="cif")
    except Exception:
        return None


def structure_to_cif(struct: Structure) -> str:
    return struct.to(fmt="cif")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Validation parquet with is_valid column")
    parser.add_argument("--cif-source", default=None,
                        help="Parquet with cif_steered (default: "
                             "steering_results/generated_cifs/<input_stem>.parquet)")
    parser.add_argument("--out", default=None,
                        help="Output parquet path (default: steering_results/relaxed/<stem>.parquet)")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="Force convergence criterion, eV/Å. Sets the quality of a "
                             "converged structure; lower is tighter and slower. See the "
                             "module docstring.")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max optimiser steps. A ceiling, not a target -- a structure "
                             "that hits it is returned UNCONVERGED and still counted a "
                             "success.")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-structure wall-clock cap in seconds (0 = none). Bounds "
                             "the tail; usually the binding constraint, not --steps.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--results-dir", default="steering_results",
                        help="Base results dir; <results-dir>/{relaxed,generated_cifs}")
    args = parser.parse_args()

    in_path = Path(args.input)          # validation parquet (holds is_valid)
    out_dir = Path(args.results_dir) / "relaxed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{in_path.stem}.parquet"
    cif_path = Path(args.cif_source) if args.cif_source else \
        Path(args.results_dir) / "generated_cifs" / f"{in_path.stem}.parquet"

    # validation flags + raw CIFs, joined on (id, sample)
    print(f"Loading validation flags {in_path} ...")
    val = pd.read_parquet(in_path)
    if "is_valid" not in val.columns:
        raise ValueError("input validation parquet must have an is_valid column")
    print(f"Loading raw CIFs {cif_path} ...")
    cifs = pd.read_parquet(cif_path, columns=["id", "sample", "cif_steered"])
    df = val.merge(cifs, on=["id", "sample"], how="left")

    # resume: pull already-relaxed CIFs from a previous output
    if out_path.exists():
        prev = pd.read_parquet(out_path, columns=["id", "sample", "cif_relaxed"])
        df = df.merge(prev, on=["id", "sample"], how="left")
    else:
        df["cif_relaxed"] = None
    n_done = int(df["cif_relaxed"].notna().sum())
    print(f"  {len(df):,} rows, {n_done:,} already relaxed")

    to_relax = df[(df["is_valid"] == True)
                  & df["cif_relaxed"].isna()
                  & df["cif_steered"].notna()].reset_index()
    print(f"  {len(to_relax):,} valid CIFs left to relax")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading M3GNet potential ...")
    pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    pot = pot.to(device)
    relaxer = Relaxer(potential=pot)
    print("  Loaded.")

    n_success = 0
    n_fail = 0
    n_timeout = 0

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    for i, row in tqdm(to_relax.iterrows(), total=len(to_relax)):
        orig_idx = row["index"]
        struct = parse_cif(row["cif_steered"])
        if struct is None:
            if n_fail < 5:
                print(f"  PARSE FAIL idx={orig_idx}", flush=True)
            n_fail += 1
            continue

        try:
            if args.timeout > 0:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(args.timeout)
            result = relaxer.relax(struct, fmax=args.fmax, steps=args.steps)
            if args.timeout > 0:
                signal.alarm(0)
            df.at[orig_idx, "cif_relaxed"] = structure_to_cif(result["final_structure"])
            n_success += 1
        except TimeoutError:
            if args.timeout > 0:
                signal.alarm(0)
            n_timeout += 1
            continue
        except Exception as e:
            if args.timeout > 0:
                signal.alarm(0)
            if n_fail < 5:
                print(f"  FAIL idx={orig_idx}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            continue

        if n_success % args.checkpoint_every == 0:
            df[["id", "sample", "cif_relaxed"]].to_parquet(out_path, index=False)

    # relaxed CIF store: id/sample/cif_relaxed only (no flags, no raw CIFs)
    df[["id", "sample", "cif_relaxed"]].to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Success: {n_success:,}  Failed: {n_fail:,}  Timed out: {n_timeout:,}")
    print(f"Total relaxed: {df['cif_relaxed'].notna().sum():,}")


if __name__ == "__main__":
    main()
