#!/usr/bin/env python3
"""
Validate steered CIFs from a generated_cifs parquet.

Runs two tiers of checks:
  1. is_sensible  — fast regex check on cell parameters (no structure parsing)
  2. is_valid     — full chemical validity: formula, multiplicities, bond lengths, space group

Output parquet columns:
  id, sample, is_sensible, is_valid, bond_length_score,
  space_group_consistent, atom_site_consistent, error

Usage:
    python validate_steered_cifs.py --input generated_cifs/steered_test_p5_alpha1.0_layer14.parquet
    python validate_steered_cifs.py --input generated_cifs/steered_test_p5_alpha1.0_layer14.parquet --workers 8
"""
import argparse
import multiprocessing as mp
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy  # must import before pandas to avoid inspect shadowing issue
import pandas as _pd_preload  # noqa: F401

import sys
sys.path.insert(0, "CrystaLLM")

# CrystaLLM uses pymatgen APIs that were renamed in newer versions. Patch before importing.
from pymatgen.io.cif import CifParser as _CifParser
if not hasattr(_CifParser, "from_string"):
    _CifParser.from_string = classmethod(lambda cls, s, **kw: cls.from_str(s, **kw))

from pymatgen.core.operations import SymmOp as _SymmOp
if not hasattr(_SymmOp, "as_xyz_string"):
    _SymmOp.as_xyz_string = _SymmOp.as_xyz_str

from crystallm import (
    CIFTokenizer,
    bond_length_reasonableness_score,
    extract_space_group_symbol,
    is_atom_site_multiplicity_consistent,
    is_sensible,
    is_space_group_consistent,
    is_valid,
    replace_symmetry_operators,
)


def eval_one(args):
    idx, cif = args
    result = {
        "idx": idx,
        "is_sensible": False,
        "is_valid": False,
        "bond_length_score": None,
        "space_group_consistent": None,
        "atom_site_consistent": None,
        "gen_len": None,
        "error": None,
    }
    try:
        tokenizer = CIFTokenizer()
        result["gen_len"] = len(tokenizer.tokenize_cif(cif))

        if not is_sensible(cif):
            return result

        result["is_sensible"] = True

        # Replace generated symmetry operators with canonical ones for the stated
        # space group before checking consistency — same as evaluate_cifs.py does.
        sg = extract_space_group_symbol(cif)
        if sg is not None and sg != "P 1":
            cif = replace_symmetry_operators(cif, sg)

        result["atom_site_consistent"] = is_atom_site_multiplicity_consistent(cif)
        result["space_group_consistent"] = is_space_group_consistent(cif)
        result["bond_length_score"] = bond_length_reasonableness_score(cif)
        result["is_valid"] = is_valid(cif, bond_length_acceptability_cutoff=1.0)

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to steered parquet file")
    parser.add_argument("--out", default=None, help="Output parquet path (default: validation_<input_stem>.parquet)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path("validation")
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{in_path.stem}.parquet"

    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} CIFs ({df['id'].nunique():,} unique prompts)")

    tasks = list(enumerate(df["cif_steered"].tolist()))

    print(f"Running validation with {args.workers} workers ...")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(eval_one, tasks, chunksize=50), total=len(tasks)))

    results_df = pd.DataFrame(results).set_index("idx")
    out_df = df.copy()
    out_df["is_sensible"]           = results_df["is_sensible"].values
    out_df["is_valid"]               = results_df["is_valid"].values
    out_df["bond_length_score"]      = results_df["bond_length_score"].values
    out_df["space_group_consistent"] = results_df["space_group_consistent"].values
    out_df["atom_site_consistent"]   = results_df["atom_site_consistent"].values
    out_df["gen_len"]                = results_df["gen_len"].values
    out_df["error"]                  = results_df["error"].values

    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    n = len(out_df)
    n_sensible = out_df["is_sensible"].sum()
    n_valid    = out_df["is_valid"].sum()
    n_errors   = out_df["error"].notna().sum()

    sg  = out_df["space_group_consistent"].sum()
    ams = out_df["atom_site_consistent"].sum()
    bl  = out_df["bond_length_score"].dropna()
    gl  = out_df["gen_len"].dropna()

    # prompt-level aggregation: at least one sample per id passes
    prompt = out_df.groupby("id").agg(
        any_valid   =("is_valid",   "any"),
        any_sensible=("is_sensible","any"),
    )
    n_prompts        = len(prompt)
    n_prompts_valid  = int(prompt["any_valid"].sum())
    n_prompts_sensible = int(prompt["any_sensible"].sum())

    print(f"\n{'='*50}")
    print(f"Results for: {in_path.name}")
    print(f"{'='*50}")
    print(f"--- Sample level ({n:,} CIFs) ---")
    print(f"Sensible:                 {n_sensible:>8,}  ({n_sensible/n:.1%})")
    print(f"Valid:                    {n_valid:>8,}  ({n_valid/n:.1%})")
    print(f"Space group consistent:   {sg:>8,}  ({sg/n:.1%})")
    print(f"Atom site consistent:     {ams:>8,}  ({ams/n:.1%})")
    print(f"Avg bond length score:    {bl.mean():.4f} ± {bl.std():.4f}")
    print(f"Avg token length:         {gl.mean():.1f} ± {gl.std():.1f}")
    print(f"Errors (pymatgen):        {n_errors:>8,}  ({n_errors/n:.1%})")
    print(f"\n--- Prompt level ({n_prompts:,} unique IDs, at least 1 sample passes) ---")
    print(f"At least 1 sensible:      {n_prompts_sensible:>8,}  ({n_prompts_sensible/n_prompts:.1%})")
    print(f"At least 1 valid:         {n_prompts_valid:>8,}  ({n_prompts_valid/n_prompts:.1%})")


if __name__ == "__main__":
    main()
