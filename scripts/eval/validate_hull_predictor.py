#!/usr/bin/env python3
"""Does the hull machinery reproduce MP's own energy_above_hull?

Feed MP's own formation energy into our hull lookup and compare to MP's stored
e_above_hull. MEGNet is not involved, so any disagreement is a bug in the hull code
rather than model error. Run this before trusting any e_above_hull number.

TWO TRAPS, both hit on the first attempt:

1. `formula_pretty` is NOT a unique key. 46% of metadata_mp rows share a
   (chemsys, formula_pretty) with at least one other material -- 22 different materials
   are called LiFe(PO3)4. Join on material_id.

2. metadata_mp MIXES THERMO TYPES. Most rows are GGA/GGA+U, but some are pure r2SCAN,
   whose energies live on a different scale and whose e_above_hull is measured against a
   different hull. mp-2912291 (r2SCAN) has a WORSE formation energy than mp-25977
   (GGA_GGA+U) at the same composition yet a smaller stored e_above_hull, which is
   impossible within one scheme. Comparing an r2SCAN row to a GGA/GGA+U hull produces a
   ~0.3 eV/atom "error" that is not an error.

So ground truth is fetched LIVE from MP for one named thermo_type, and both sides of the
comparison come from that same scheme.

    python scripts/eval/validate_hull_predictor.py --n-materials 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from predictors import HullLookup                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="metadata_mp.parquet")
    ap.add_argument("--n-systems", type=int, default=8,
                    help="distinct chemical systems to draw materials from")
    ap.add_argument("--per-system", type=int, default=40,
                    help="materials sampled per system")
    ap.add_argument("--thermo-type", default="GGA_GGA+U",
                    help="both the hull and the ground truth use this scheme")
    ap.add_argument("--out", default="analysis/hull_predictor_validation.csv")
    args = ap.parse_args()

    mp = pd.read_parquet(args.metadata,
                         columns=["material_id", "chemsys", "formula_pretty"]).dropna()
    counts = mp.chemsys.value_counts()
    systems = counts.head(args.n_systems).index.tolist()
    rng = np.random.default_rng(0)

    import sys as _s
    import pymatgen.entries.compatibility as _c
    _s.modules.setdefault("pymatgen.analysis.compatibility", _c)
    from mp_api.client import MPRester
    from pymatgen.core import Composition

    hull = HullLookup(thermo_types=(args.thermo_type,))
    hull.setup()
    key = hull.api_key

    rows = []
    with MPRester(key) as mpr:
        for i, cs in enumerate(systems, 1):
            sub = mp[mp.chemsys == cs]
            take = sub.iloc[rng.choice(len(sub), min(args.per_system, len(sub)),
                                       replace=False)]
            mids = [m.replace("MP_", "") for m in take.material_id]
            # ground truth from the SAME scheme the hull is built in
            docs = mpr.materials.thermo.search(
                material_ids=mids, thermo_types=[args.thermo_type],
                fields=["material_id", "thermo_type", "composition",
                        "formation_energy_per_atom", "energy_above_hull"])
            pd_form = hull.diagram(cs.split("-"))
            if pd_form is None:
                print(f"  [{i}/{len(systems)}] {cs:<16} NO MP DIAGRAM")
                continue
            for d in docs:
                if str(d.thermo_type) != args.thermo_type:
                    continue
                c = Composition({str(k): v for k, v in d.composition.items()})
                ours = hull.e_above_hull(c, d.formation_energy_per_atom)
                rows.append(dict(chemsys=cs, material_id=str(d.material_id),
                                 formula=c.reduced_formula,
                                 mp_truth=d.energy_above_hull, ours=ours,
                                 err=ours - d.energy_above_hull))
            e = np.abs([r["err"] for r in rows if r["chemsys"] == cs])
            print(f"  [{i}/{len(systems)}] {cs:<16} n={len(e):<4} "
                  f"MAE {np.nanmean(e):.2e}  max {np.nanmax(e):.2e}")

    d = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False, float_format="%.8g")
    ok = d.err.abs()
    print(f"\noverall  n={len(d):,}  MAE {ok.mean():.3e}  median {ok.median():.3e}  "
          f"max {ok.max():.3e} eV/atom")
    print(f"  within 1e-6 eV/atom: {100*(ok < 1e-6).mean():.2f}%")
    print(f"\nSaved {args.out}")
    print("VERDICT:", "hull arithmetic is exact" if ok.max() < 1e-6
          else "MISMATCH -- do not trust e_above_hull until this is explained")


if __name__ == "__main__":
    main()
