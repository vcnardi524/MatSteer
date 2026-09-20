#!/usr/bin/env python3
"""
Validate steered CIFs from a steering_results/generated_cifs parquet.

Runs two tiers of checks:
  1. is_sensible  — fast regex check on cell parameters (no structure parsing)
  2. is_valid     — full chemical validity: formula, multiplicities, bond lengths, space group

Output parquet columns:
  id, sample, is_sensible, is_valid, bond_length_score,
  space_group_consistent, atom_site_consistent, error

Usage:
    python validate_steered_cifs.py --input steering_results/generated_cifs/steered_test_clean_alpha16.0_layer14.parquet
    python validate_steered_cifs.py --input steering_results/generated_cifs/steered_test_clean_alpha16.0_layer14.parquet --workers 8
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
    is_formula_consistent,
    is_sensible,
    is_space_group_consistent,
    is_valid,
)

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import restore_symmetry_operators


def detected_space_group(cif_str, symprec=0.1):
    """The space group SpacegroupAnalyzer finds in the coordinates, or None.

    symprec 0.1 matches what is_space_group_consistent uses internally, so the two
    columns describe the same detection. Returns None rather than raising: this is a
    diagnostic, and a structure too broken to analyse has already failed elsewhere.
    """
    try:
        from pymatgen.core.structure import Structure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Structure.from_str(cif_str, fmt="cif").get_space_group_info(
                symprec=symprec)[0]
    except Exception:
        return None


def eval_one(args):
    """(idx, cif) or (idx, cif, check_space_group) -> one row of flags.

    check_space_group=False drops the space-group term from is_valid, leaving a
    THREE-check bar: formula, atom site multiplicity, bond length. That is the right bar
    for a model that does not state a space group. llamat2-cif never does, so its CIFs
    are written in P 1 and is_space_group_consistent -- stated against detected -- fails
    for every structure that has real symmetry, measuring the decoder rather than the
    model. `space_group_detected` is still recorded, because what symmetry the generated
    coordinates actually carry is worth knowing; it just does not gate validity.

    Keep it True for crystallm, which states its own symbol and can genuinely fail.
    """
    idx, cif = args[0], args[1]
    check_space_group = args[2] if len(args) > 2 else True
    result = {
        "idx": idx,
        "is_sensible": False,
        "is_valid": False,
        "bond_length_score": None,
        "space_group_consistent": None,
        "space_group_detected": None,
        "atom_site_consistent": None,
        "formula_consistent": None,
        "gen_len": None,
        "error": None,
    }
    try:
        # An empty CIF means generation or decoding produced nothing usable. It must
        # short-circuit: crystallm's is_sensible only range-checks the cell lengths and
        # angles it FINDS, so a string with none passes vacuously -- is_sensible("") is
        # True. Without this every failed llamat decode would be counted as sensible.
        if not cif or not cif.strip():
            result["error"] = "empty CIF (generation or decoding produced nothing)"
            return result

        tokenizer = CIFTokenizer()
        result["gen_len"] = len(tokenizer.tokenize_cif(cif))

        if not is_sensible(cif):
            return result

        result["is_sensible"] = True

        # Replace generated symmetry operators with canonical ones for the stated
        # space group before checking consistency — same as evaluate_cifs.py does.
        # Via utils so indented CIFs are handled and a failed swap raises: without it
        # every non-P1 structure reads as the asymmetric unit alone and fails the
        # space-group consistency check.
        cif = restore_symmetry_operators(cif, extract_space_group_symbol(cif))

        result["atom_site_consistent"] = is_atom_site_multiplicity_consistent(cif)
        # The fourth is_valid term, and the only one that was never recorded. It is the
        # one that bites hardest on decoded CIFs, so a run where is_valid is low and the
        # other three are ~100% is answered by this column instead of by re-deriving it.
        result["formula_consistent"] = is_formula_consistent(cif)
        result["bond_length_score"] = bond_length_reasonableness_score(cif)

        # What symmetry the coordinates actually carry, recorded either way. For a model
        # that writes P 1 this is the only informative symmetry number there is.
        result["space_group_detected"] = detected_space_group(cif)

        if check_space_group:
            result["space_group_consistent"] = is_space_group_consistent(cif)
            result["is_valid"] = is_valid(cif, bond_length_acceptability_cutoff=1.0)
        else:
            # The same three terms crystallm's is_valid ANDs, minus the space-group one.
            # Recomputed here rather than calling is_valid, which cannot be told to skip
            # a check. Left as None above so it is absent, not False, in the flags table.
            result["is_valid"] = bool(result["formula_consistent"]
                                      and result["atom_site_consistent"]
                                      and result["bond_length_score"] >= 1.0)

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to steered parquet file")
    parser.add_argument("--out", default=None, help="Output parquet path (default: validation_<input_stem>.parquet)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="The generation cap these runs used. With it, any sample "
                             "whose n_new_tokens reached the cap is marked truncated and "
                             "is_valid False: it was cut off mid-structure, so the cell "
                             "is complete but most of its atoms were never written. "
                             "Needs the n_new_tokens column -- crystallm runs do not have "
                             "one and are unaffected. Backfill older llamat runs with "
                             "scripts/data/backfill_token_counts.py.")
    parser.add_argument("--space-group-check", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include is_space_group_consistent in is_valid. ON for "
                             "crystallm, which states its own symbol and can genuinely "
                             "fail it. Pass --no-space-group-check for a model that does "
                             "not state one (llamat2-cif): its CIFs are P 1, so the term "
                             "fails for every structure with real symmetry and measures "
                             "the decoder, not the model. space_group_detected is "
                             "recorded either way.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--results-dir", default="steering_results",
                        help="Base results dir; output goes to <results-dir>/validation")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.results_dir) / "validation"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{in_path.stem}.parquet"

    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} CIFs ({df['id'].nunique():,} unique prompts)")

    tasks = [(i, c, args.space_group_check)
             for i, c in enumerate(df["cif_steered"].tolist())]

    print(f"Running validation with {args.workers} workers ...")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(eval_one, tasks, chunksize=50), total=len(tasks)))

    results_df = pd.DataFrame(results).set_index("idx")
    # flags only, keyed by id/sample — the raw CIF strings stay in
    # steering_results/generated_cifs and are never copied into the flag files.
    out_df = df[["id", "sample"]].copy()
    out_df["is_sensible"]           = results_df["is_sensible"].values
    out_df["is_valid"]               = results_df["is_valid"].values
    out_df["bond_length_score"]      = results_df["bond_length_score"].values
    out_df["space_group_consistent"] = results_df["space_group_consistent"].values
    out_df["atom_site_consistent"]   = results_df["atom_site_consistent"].values
    out_df["formula_consistent"]     = results_df["formula_consistent"].values
    out_df["space_group_detected"]   = results_df["space_group_detected"].values

    # Truncation, folded into is_valid. It cannot be detected from the CIF -- a cut-off
    # structure is internally consistent, so formula and multiplicity both pass at 100%.
    # bond_length catches ~90% of them incidentally, because what is left is too sparse
    # to bond sensibly, but that leaves a tenth of them scoring valid while describing a
    # crystal the model never finished writing. n_new_tokens settles it exactly: nothing
    # stops naturally one token short of the cap, so reaching it means cut off.
    if args.max_new_tokens and "n_new_tokens" in df.columns:
        trunc = df["n_new_tokens"].fillna(0).astype(int) >= args.max_new_tokens
        out_df["truncated"] = trunc.values
        out_df.loc[trunc.values, "is_valid"] = False
    elif args.max_new_tokens:
        print("  ! --max-new-tokens given but the input has no n_new_tokens column -- "
              "no truncation check applied")
    out_df["gen_len"]                = results_df["gen_len"].values
    out_df["error"]                  = results_df["error"].values
    # Carried through from generation, so decode failures are countable here without
    # re-reading the generations. Absent for models whose output already is a CIF.
    if "decode_reason" in df.columns:
        out_df["decode_reason"] = df["decode_reason"].values

    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    n = len(out_df)
    n_sensible = out_df["is_sensible"].sum()
    n_valid    = out_df["is_valid"].sum()
    n_errors   = out_df["error"].notna().sum()
    if "decode_reason" in out_df.columns:
        failed = out_df["decode_reason"].fillna("").str.len().gt(0)
        print(f"decode: {len(out_df) - failed.sum():,}/{len(out_df):,} produced a CIF")
        for reason, k in out_df.loc[failed, "decode_reason"].value_counts().head(5).items():
            print(f"    {k:>6,}  {reason}")

    sg  = out_df["space_group_consistent"].sum()
    ams = out_df["atom_site_consistent"].sum()
    fc  = out_df["formula_consistent"].sum()
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
    if out_df["space_group_consistent"].notna().any():
        print(f"Space group consistent:   {sg:>8,}  ({sg/n:.1%})")
    else:
        print(f"Space group consistent:   {'not checked':>16}  "
              f"(--no-space-group-check; see space_group_detected)")
    print(f"Atom site consistent:     {ams:>8,}  ({ams/n:.1%})")
    print(f"Formula consistent:       {fc:>8,}  ({fc/n:.1%})")
    if "truncated" in out_df.columns:
        t = out_df["truncated"].sum()
        print(f"Truncated (hit the cap):  {t:>8,}  ({t/n:.1%})  -> forced invalid")
    det = out_df["space_group_detected"].dropna()
    if len(det):
        p1 = (det == "P1").sum()
        print(f"Space groups detected:    {det.nunique():>8,} distinct, "
              f"{(len(det)-p1)/len(det):.1%} above P1")
    print(f"Avg bond length score:    {bl.mean():.4f} ± {bl.std():.4f}")
    print(f"Avg token length:         {gl.mean():.1f} ± {gl.std():.1f}")
    print(f"Errors (pymatgen):        {n_errors:>8,}  ({n_errors/n:.1%})")
    print(f"\n--- Prompt level ({n_prompts:,} unique IDs, at least 1 sample passes) ---")
    print(f"At least 1 sensible:      {n_prompts_sensible:>8,}  ({n_prompts_sensible/n_prompts:.1%})")
    print(f"At least 1 valid:         {n_prompts_valid:>8,}  ({n_prompts_valid/n_prompts:.1%})")


if __name__ == "__main__":
    main()
