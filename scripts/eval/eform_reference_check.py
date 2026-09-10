#!/usr/bin/env python3
"""Is MEGNet's formation energy sane on REAL structures, or is the whole scale off?

Formation energy feeds the convex hull, and e_above_hull is a small difference of two
E_form numbers, so a systematic bias in E_form propagates straight into every hull result.
On the alpha=0 control's generated structures, 46% of predicted E_form came out POSITIVE
with a median of -0.037 eV/atom. Real materials are mostly well below zero, so that is
either (a) generated structures genuinely being poor, or (b) the predictor/parse being
wrong. Those have opposite consequences and nothing in the pipeline distinguishes them.

This runs the identical predictor and the identical parse over the ORIGINAL test
structures -- the same 1,000 prompts the steering runs were built from. Those are real,
experimentally-reported materials, so their E_form distribution is the control:

  reference E_form mostly NEGATIVE  -> the predictor is fine, generated structures are bad
  reference E_form also near ZERO   -> the predictor or the parse is wrong, and every
                                       hull number computed so far is suspect

Where a prompt is also in metadata_mp, MP's own DFT formation_energy_per_atom is compared
directly, which is the harder check: absolute agreement, not just a plausible shape.

Usage:
    megnet_venv/bin/python scripts/eval/eform_reference_check.py
"""
import argparse
import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import postprocess

PKL = "CrystaLLM/cifs_v1_test_sample1000.pkl.gz"
OUT = Path("analysis/v1_all/test/eform_reference_check.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=PKL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare", default="steering_results/baseline/property_predictions/"
                                         "steered_test_alpha0.0_layer0_nosg.parquet")
    args = ap.parse_args()

    import matgl
    from pymatgen.core import Structure
    model = matgl.load_model("MEGNet-Eform-MP-2018.6.1")

    with gzip.open(args.pkl, "rb") as f:
        data = pickle.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"{len(data):,} reference structures from {args.pkl}")

    rows = []
    for id_, cif in tqdm(data):
        try:
            # postprocess before parsing, exactly as compute_predictions.py does. It is a
            # no-op on already-indented reference CIFs, so this is the same code path.
            s = Structure.from_str(postprocess(cif, "reference"), fmt="cif")
            rows.append({"id": id_, "formula": s.composition.reduced_formula,
                         "n_elements": len(s.composition.elements),
                         "e_form_pred": float(model.predict_structure(s))})
        except Exception as e:
            rows.append({"id": id_, "formula": None, "n_elements": None,
                         "e_form_pred": np.nan})
    ref = pd.DataFrame(rows)
    ok = ref.e_form_pred.dropna()
    print(f"\n=== MEGNet E_form on REAL test structures (n={len(ok):,}) ===")
    print(f"  median {ok.median():+.4f}   mean {ok.mean():+.4f}   %negative {100*(ok<0).mean():.1f}%")
    print(f"  quantiles 5/25/50/75/95: {np.percentile(ok,[5,25,50,75,95]).round(3)}")

    if Path(args.compare).exists():
        g = pd.read_parquet(args.compare)
        for col, lab in (("formation_energy_per_atom_raw", "generated, raw"),
                         ("formation_energy_per_atom", "generated, relaxed")):
            if col in g.columns:
                s = g[col].dropna()
                print(f"\n=== {lab} (n={len(s):,}) ===")
                print(f"  median {s.median():+.4f}   mean {s.mean():+.4f}   "
                      f"%negative {100*(s<0).mean():.1f}%")

    # the hard check: absolute agreement with MP's own DFT value where we have it
    try:
        mp = pd.read_parquet("metadata_mp.parquet",
                             columns=["material_id", "formation_energy_per_atom"])
        mp = mp.rename(columns={"material_id": "id", "formation_energy_per_atom": "e_form_dft"})
        j = ref.merge(mp, on="id", how="inner").dropna(subset=["e_form_pred", "e_form_dft"])
        if len(j):
            err = j.e_form_pred - j.e_form_dft
            print(f"\n=== vs MP DFT, on the {len(j)} prompts MP also has ===")
            print(f"  MAE {err.abs().mean():.4f}   bias {err.mean():+.4f}   "
                  f"corr {j.e_form_pred.corr(j.e_form_dft):+.3f} eV/atom")
            print(f"  DFT   median {j.e_form_dft.median():+.4f}  %negative {100*(j.e_form_dft<0).mean():.0f}%")
            print(f"  MEGNet median {j.e_form_pred.median():+.4f}  %negative {100*(j.e_form_pred<0).mean():.0f}%")
    except Exception as e:
        print(f"\n(MP comparison skipped: {type(e).__name__}: {e})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(OUT, index=False, float_format="%.6g")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
