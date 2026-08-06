#!/usr/bin/env python3
"""
Check uniqueness and novelty of steered CIFs.

Uniqueness: no two generated structures are the same (within the steered output).
Novelty:    a unique structure does not exist in the training set.

Follows the same approach as CrystaLLM/bin/check_valid_unique_novel.py but
reads validity flags from our validation parquet (so only valid CIFs are
checked) instead of a .tar.gz of .cif files.

Steps:
  1. Load training CIFs → build formula -> [cif_strings] index
  2. Load validation flags, join cif_steered from generated_cifs on (id, sample),
     filter to is_valid == True
  3. Postprocess each valid CIF (replace symmetry operators, remove atom props)
  4. Deduplicate within generated set using StructureMatcher -> unique set
  5. For each unique structure, check against training set -> novel set
  6. Save results parquet + print summary

Output: steering_results/validation/novelty_<input_stem>.parquet
  columns: id, sample, ...(validation flag cols)..., is_unique, is_novel
  (flags only — CIF strings are never written here)

Usage:
    python scripts/novelty_steered_cifs.py \\
        --input steering_results/validation/steered_test_clean_alpha16.0_layer14.parquet \\
        --base CrystaLLM/cifs_v1_train.pkl.gz
"""
import argparse
import gzip
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, "CrystaLLM")

from pymatgen.io.cif import CifParser as _CifParser
if not hasattr(_CifParser, "from_string"):
    _CifParser.from_string = classmethod(lambda cls, s, **kw: cls.from_str(s, **kw))

from pymatgen.core.operations import SymmOp as _SymmOp
if not hasattr(_SymmOp, "as_xyz_string"):
    _SymmOp.as_xyz_string = _SymmOp.as_xyz_str

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

from crystallm import extract_data_formula
sys.path.insert(0, "CrystaLLM/bin")
from postprocess import postprocess


def build_base_index(base_path: str) -> dict:
    """Load training CIFs and index by reduced formula."""
    print(f"Loading base CIFs from {base_path} ...")
    with gzip.open(base_path, "rb") as f:
        base_cifs = pickle.load(f)
    print(f"  {len(base_cifs):,} base CIFs")

    index = {}
    for _, cif in tqdm(base_cifs, desc="Indexing base CIFs"):
        try:
            formula = Composition(extract_data_formula(cif)).reduced_formula
            cif = re.sub(r"^[ \t]+|[ \t]+$", "", cif, flags=re.MULTILINE).replace("  ", " ")
            if formula not in index:
                index[formula] = []
            index[formula].append(cif)
        except Exception:
            pass

    print(f"  {len(index):,} unique reduced formulas in base")
    return index


def parse_structure(cif: str) -> Structure | None:
    try:
        processed = postprocess(cif, "generated")
        return Structure.from_str(processed, fmt="cif")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Validation parquet (output of validate_steered_cifs.py)")
    parser.add_argument("--base", default="CrystaLLM/cifs_v1_train.pkl.gz",
                        help="Training CIFs pkl.gz for novelty check")
    parser.add_argument("--cif-source", default=None,
                        help="Parquet with cif_steered (default: "
                             "steering_results/generated_cifs/<input_stem>.parquet)")
    parser.add_argument("--out", default=None,
                        help="Output parquet path "
                             "(default: steering_results/validation/novelty_<input_stem>.parquet)")
    parser.add_argument("--ltol",       type=float, default=0.2)
    parser.add_argument("--stol",       type=float, default=0.3)
    parser.add_argument("--angle-tol",  type=float, default=5.0)
    parser.add_argument("--results-dir", default="steering_results",
                        help="Base results dir; <results-dir>/{validation,generated_cifs}")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.results_dir) / "validation"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"novelty_{in_path.stem}.parquet"

    # --- load validation flags, then join raw CIFs from generated_cifs ---
    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)
    df = df.drop(columns=[c for c in ("cif_steered", "cif_relaxed", "cif_original")
                          if c in df.columns])
    cif_path = Path(args.cif_source) if args.cif_source else \
        Path(args.results_dir) / "generated_cifs" / f"{in_path.stem}.parquet"
    print(f"Loading CIFs from {cif_path} ...")
    cifs = pd.read_parquet(cif_path, columns=["id", "sample", "cif_steered"])
    df = df.merge(cifs, on=["id", "sample"], how="left")

    valid_df = df[df["is_valid"] == True].copy()   # keep original df index labels for write-back
    print(f"  {len(df):,} total CIFs, {len(valid_df):,} valid")

    # initialise output columns on full df
    df["is_unique"] = False
    df["is_novel"]  = False

    if len(valid_df) == 0:
        print("No valid CIFs — nothing to check.")
        df.drop(columns=["cif_steered"]).to_parquet(out_path, index=False)
        return

    # --- build training index ---
    base_index = build_base_index(args.base)

    matcher = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)

    # --- parse valid structures ---
    print("Parsing valid structures ...")
    structs = []
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        structs.append(parse_structure(row["cif_steered"]))

    valid_df["_struct"] = structs

    # drop rows where parsing failed (index labels stay tied to the full df)
    parsed = valid_df[valid_df["_struct"].notna()].copy()
    print(f"  {len(parsed):,} structures parsed successfully")

    # --- uniqueness: deduplicate within generated set ---
    print("Checking uniqueness ...")
    # formula -> [(struct, df_label)]; df_label is the original index into `df`
    unique_by_formula: dict[str, list[tuple]] = {}

    for label, row in tqdm(parsed.iterrows(), total=len(parsed)):
        struct = row["_struct"]
        formula = struct.composition.reduced_formula

        if formula not in unique_by_formula:
            unique_by_formula[formula] = [(struct, label)]
        else:
            is_dup = any(
                matcher.fit(struct, existing)
                for existing, _ in unique_by_formula[formula]
            )
            if not is_dup:
                unique_by_formula[formula].append((struct, label))

    unique_structs = [(s, label) for lst in unique_by_formula.values() for s, label in lst]
    print(f"  {len(unique_structs):,} unique structures out of {len(parsed):,} valid")

    # mark unique directly on the full df (labels preserved from the merge)
    for _, label in unique_structs:
        df.loc[label, "is_unique"] = True

    # --- novelty: check each unique structure against training set ---
    print("Checking novelty ...")
    novel_by_composition = 0

    for struct, label in tqdm(unique_structs):
        formula = struct.composition.reduced_formula

        if formula not in base_index:
            df.loc[label, "is_novel"] = True
            novel_by_composition += 1
        else:
            is_matched = False
            for base_cif in base_index[formula]:
                try:
                    base_processed = postprocess(base_cif, "base")
                    base_struct = Structure.from_str(base_processed, fmt="cif")
                    if matcher.fit(struct, base_struct):
                        is_matched = True
                        break
                except Exception:
                    continue
            if not is_matched:
                df.loc[label, "is_novel"] = True

    # flags only — never write CIF strings into the validation file
    df.drop(columns=["cif_steered"]).to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    n        = len(df)
    n_valid  = int(df["is_valid"].sum())
    n_unique = int(df["is_unique"].sum())
    n_novel  = int(df["is_novel"].sum())

    print(f"\n{'='*50}")
    print(f"Results for: {in_path.name}")
    print(f"{'='*50}")
    print(f"Total CIFs:    {n:>8,}")
    print(f"Valid:         {n_valid:>8,}  ({n_valid/n:.1%})")
    print(f"Unique:        {n_unique:>8,}  ({n_unique/n_valid:.1%} of valid)")
    print(f"Novel:         {n_novel:>8,}  ({n_novel/n_unique:.1%} of unique)  [{novel_by_composition} by composition alone]")


if __name__ == "__main__":
    main()
