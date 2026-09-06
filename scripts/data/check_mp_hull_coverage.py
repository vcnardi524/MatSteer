#!/usr/bin/env python3
"""Can Materials Project supply a convex hull for every chemical system we steer into?

e_above_hull needs a phase diagram for the structure's chemical system. If MP has no
entries for a system, that prompt yields no value -- and if a chemically biased subset
drops out, the surviving results are biased too. This measures it directly rather than
inferring it from how much of MP happens to sit in our own corpus (which badly
underestimated coverage: 18.8% inferred vs 100% on a 40-system sample).

Distinguishes THREE outcomes, which HullLookup deliberately collapses to None:
    ok        MP returned entries and a PhaseDiagram was built
    empty     MP genuinely has nothing for this system
    error     the API call or the diagram construction failed

Conflating the last two would report a transient network failure as missing chemistry.
Only `empty` is a real coverage gap; `error` should be retried.

Resumable -- rerunning skips systems already in the output.

    python scripts/data/check_mp_hull_coverage.py
"""
import argparse
import gzip
import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EL = re.compile(r"([A-Z][a-z]?)(\d*)")
FORMULA = re.compile(r"_chemical_formula_sum\s+'([^']+)'")


def systems_for(ids: set, pkl: str) -> dict:
    with gzip.open(pkl, "rb") as f:
        cifs = pickle.load(f)
    out = {}
    for i, c in cifs:
        if i not in ids:
            continue
        m = FORMULA.search(c)
        if m:
            out[i] = frozenset(e for e, _ in EL.findall(m.group(1).replace(" ", "")) if e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/test_sample1000_ids.csv")
    ap.add_argument("--pkl", default="CrystaLLM/cifs_v1_prep.pkl.gz")
    ap.add_argument("--out", default="analysis/mp_hull_coverage.csv")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls")
    args = ap.parse_args()

    import pymatgen.entries.compatibility as _c
    sys.modules.setdefault("pymatgen.analysis.compatibility", _c)
    from predictors import HullLookup

    want = set(pd.read_csv(args.ids)["id"])
    sysmap = systems_for(want, args.pkl)
    uniq = sorted({s for s in sysmap.values()}, key=lambda s: sorted(s))
    n_prompt = pd.Series([len(s) for s in sysmap.values()])
    print(f"{len(sysmap):,} prompts -> {len(uniq):,} distinct chemical systems "
          f"({n_prompt.value_counts().sort_index().to_dict()} elements each)\n")

    out = Path(args.out)
    done = {}
    if out.exists():
        prev = pd.read_csv(out)
        done = dict(zip(prev.chemsys, prev.status))
        print(f"  resuming: {len(done):,} systems already checked")

    hull = HullLookup()
    hull.setup()
    rows = []
    for i, s in enumerate(uniq, 1):
        cs = "-".join(sorted(s))
        if cs in done:
            continue
        status, n_entries = "ok", 0
        try:
            entries = hull._mpr.get_entries_in_chemsys(
                elements=sorted(s),
                additional_criteria={"thermo_types": hull.thermo_types})
            if not entries:
                status = "empty"
            else:
                n_entries = len(entries)
                if hull.diagram(s) is None:
                    status = "error"
        except Exception as e:
            status, n_entries = "error", 0
            if sum(r["status"] == "error" for r in rows) < 3:
                print(f"    {cs}: {type(e).__name__}")
        rows.append(dict(chemsys=cs, n_elements=len(s), n_entries=n_entries,
                         status=status,
                         n_prompts=sum(1 for v in sysmap.values() if v == s)))
        if i % 50 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            k = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
            print(f"  [{i}/{len(uniq)}] {k}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    d = pd.concat([pd.read_csv(out), pd.DataFrame(rows)]) if out.exists() and done \
        else pd.DataFrame(rows)
    d = d.drop_duplicates("chemsys")
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)

    print(f"\n{'status':<10}{'systems':>9}{'prompts':>10}")
    for st, g in d.groupby("status"):
        print(f"  {st:<8}{len(g):>9,}{int(g.n_prompts.sum()):>10,}")
    ok = d[d.status == "ok"]
    print(f"\ncoverage: {len(ok):,}/{len(d):,} systems "
          f"({100*len(ok)/len(d):.1f}%), {int(ok.n_prompts.sum()):,} of "
          f"{int(d.n_prompts.sum()):,} prompts ({100*ok.n_prompts.sum()/d.n_prompts.sum():.1f}%)")
    print(f"  MP entries per system: median {ok.n_entries.median():.0f}, "
          f"min {ok.n_entries.min():.0f}, max {ok.n_entries.max():.0f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
