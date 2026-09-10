#!/usr/bin/env python3
"""Place an already-predicted formation energy on the Materials Project convex hull.

WHY THIS IS SEPARATE FROM compute_predictions.py
------------------------------------------------
`EnergyAboveHullPredictor` does both halves in one process -- MEGNet for E_form, then the
MP hull -- and no virtualenv here can run both:

    venv              matgl     mp_api    pymatgen.entries.compatibility
    crystallm_venv    BROKEN    yes       yes
    megnet_venv       yes       yes       BROKEN   (Species.from_string, SpeciesLike)
    relax_venv        yes       no        BROKEN

megnet_venv's pymatgen is too old for `pymatgen.entries.compatibility`, which MP entries
need in order to deserialise; crystallm_venv's lightning/torch pairing cannot import
matgl. So the pipeline is split at the natural seam:

    compute_predictions.py --property formation_energy   (megnet_venv, matgl)
    compute_hull.py                                      (crystallm_venv, mp_api)

This script needs no model. It reads the E_form column another job already wrote, takes
the composition from the CIF, and does the hull arithmetic -- which is exactly what
HullLookup in scripts/predictors.py already implements and what
scripts/eval/validate_hull_predictor.py verified to 3.7e-15 eV/atom.

Adds `energy_above_hull_raw` (from formation_energy_per_atom_raw) and/or
`energy_above_hull` (from formation_energy_per_atom) to the SAME predictions parquet,
preserving every column it does not own.

Usage:
    python scripts/eval/compute_hull.py \
        --input steering_results/energy_above_hull/property_predictions/<stem>.parquet \
        --results-dir steering_results/energy_above_hull
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from predictors import HullLookup
from utils import postprocess, steering_path

PAIRS = [("formation_energy_per_atom_raw", "energy_above_hull_raw", "cif_steered"),
         ("formation_energy_per_atom",     "energy_above_hull",     "cif_relaxed")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="property_predictions parquet holding a formation_energy column")
    ap.add_argument("--results-dir", default="steering_results")
    ap.add_argument("--api-key-file", default="api_keys.json")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N rows per column (smoke test). "
                         "The file is still written, so use a copy.")
    args = ap.parse_args()

    in_path = Path(args.input)
    stem = in_path.stem
    pred = pd.read_parquet(in_path)
    print(f"{stem}: {len(pred):,} rows, columns {[c for c in pred.columns if c not in ('id','sample')]}")

    from pymatgen.core import Structure
    hull = HullLookup(key_file=args.api_key_file)
    hull.setup()

    for e_col, out_col, cif_col in PAIRS:
        if e_col not in pred.columns:
            print(f"  skip {out_col}: no {e_col} column")
            continue
        # the CIF gives the composition; it lives in a different store, joined on (id, sample)
        sub = "generated_cifs" if cif_col == "cif_steered" else "relaxed"
        try:
            cifs = pd.read_parquet(steering_path(Path(args.results_dir).name, sub,
                                                 f"{stem}.parquet"),
                                   columns=["id", "sample", cif_col])
        except FileNotFoundError:
            print(f"  skip {out_col}: no {sub}/{stem}.parquet")
            continue
        df = pred[["id", "sample", e_col]].merge(cifs, on=["id", "sample"], how="left")
        todo = df[df[e_col].notna() & df[cif_col].notna()]
        if args.limit:
            todo = todo.head(args.limit)
        print(f"  {out_col}: {len(todo):,} rows with both an E_form and a CIF")

        vals = {}
        for n, (_, r) in enumerate(tqdm(todo.iterrows(), total=len(todo)), 1):
            try:
                s = Structure.from_str(postprocess(r[cif_col], "hull"), fmt="cif")
                vals[(r["id"], r["sample"])] = hull.e_above_hull(s.composition, float(r[e_col]))
            except Exception:
                vals[(r["id"], r["sample"])] = float("nan")
        pred[out_col] = [vals.get((i, s)) for i, s in zip(pred["id"], pred["sample"])]
        got = pred[out_col].notna().sum()
        print(f"    {got:,} scored; median {pred[out_col].median():.4f} eV/atom; "
              f"{(pred[out_col] < 0).sum()} negative (should be 0 -- hull distance is >= 0)")

    pred.to_parquet(in_path, index=False)
    print(f"\nSaved {in_path}")


if __name__ == "__main__":
    main()
