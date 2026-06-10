#!/usr/bin/env python3
"""
Relax steered CIF structures using M3GNet universal potential.

Reads a validation parquet (output of validate_steered_cifs.py), filters to
valid CIFs, relaxes each structure with M3GNet-PES, and adds a cif_relaxed
column back to the same parquet. Checkpoints every 100 structures.

Usage:
    python scripts/relax_steered_cifs.py \
        --input validation/steered_test_p5_alpha1.0_layer14.parquet

Output:
    same parquet with added column: cif_relaxed
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

# Load crystallm._utils directly by file path to avoid triggering
# crystallm/__init__.py (which needs full pymatgen, not just pymatgen-core).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "cryst_utils", "CrystaLLM/crystallm/_utils.py")
_cryst_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cryst_utils)
extract_space_group_symbol = _cryst_utils.extract_space_group_symbol
replace_symmetry_operators = _cryst_utils.replace_symmetry_operators
remove_atom_props_block = _cryst_utils.remove_atom_props_block


def postprocess(cif: str) -> str:
    sg = extract_space_group_symbol(cif)
    if sg is not None and sg != "P 1":
        cif = replace_symmetry_operators(cif, sg)
    cif = remove_atom_props_block(cif)
    return cif


def parse_cif(cif: str) -> Structure | None:
    try:
        processed = postprocess(cif)
        return Structure.from_str(processed, fmt="cif")
    except Exception:
        return None


def structure_to_cif(struct: Structure) -> str:
    return struct.to(fmt="cif")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Validation parquet with is_valid column")
    parser.add_argument("--out", default=None,
                        help="Output parquet path (default: relaxed/<stem>.parquet)")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="Force convergence criterion for relaxation (eV/Å)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max relaxation steps")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-structure timeout in seconds (0 = no timeout)")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path

    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)

    # initialise column if not present
    if "cif_relaxed" not in df.columns:
        df["cif_relaxed"] = None

    # resume: skip rows already relaxed
    done = set(
        tuple(r) for r in df[df["cif_relaxed"].notna()][["id", "sample"]].values
    )
    print(f"  {len(df):,} total rows, {len(done):,} already relaxed")

    to_relax = df[df["is_valid"] == True] if "is_valid" in df.columns else df
    to_relax = to_relax[to_relax["cif_relaxed"].isna()].reset_index()
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
            df.to_parquet(out_path, index=False)

    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Success: {n_success:,}  Failed: {n_fail:,}  Timed out: {n_timeout:,}")
    print(f"Total relaxed: {df['cif_relaxed'].notna().sum():,}")


if __name__ == "__main__":
    main()
