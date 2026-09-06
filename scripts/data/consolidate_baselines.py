#!/usr/bin/env python3
"""Move the no-injection controls into one shared steering_results/baseline/ tree.

At alpha=0 the hook adds exactly zero, so the generated CIFs cannot depend on which
property is being steered. They were nonetheless regenerated per property:
steered_test_alpha0.0_layer14.parquet exists five times over, byte-identical each time
(md5 5e6723a661fe), and its validation five times over (5c4e442679). Every new property
would mean another 2 GPU-hours to reproduce a file we already have.

WHAT IS SHARED AND WHAT IS NOT

    generated_cifs   shared -- alpha=0 output is property-independent
    validation       shared -- "does this CIF parse" has no property in it
    relaxed          shared -- M3GNet relaxation is property-independent, and this is
                               the expensive one: relax the baseline once, not per property
    property_predictions   NOT shared -- density_atomic vs band_gap vs energy_above_hull
                               are different numbers for the same structure. Stays put.

A baseline is keyed on the PROMPT SET, not on the property or the layer. alpha=0 means no
injection at any layer, so layer14 and layer7 controls on the same prompts are the same
experiment -- but sg and nosg prompts are NOT, and neither are different --n-samples. The
filename already carries those, so identical names are safe to merge and different names
are kept apart.

Verifies byte-identity before removing anything, and refuses to merge files that differ.

    python scripts/data/consolidate_baselines.py --dry-run
    python scripts/data/consolidate_baselines.py
"""
import argparse
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("steering_results")
BASELINE = ROOT / "baseline"
SHARED = ("generated_cifs", "validation", "relaxed")
CONTROL = re.compile(r"alpha-?0(\.0)?_")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # stem -> subdir -> [paths]
    found = defaultdict(lambda: defaultdict(list))
    for prop_dir in sorted(ROOT.iterdir()):
        if not prop_dir.is_dir() or prop_dir.name == "baseline":
            continue
        for sub in SHARED:
            d = prop_dir / sub
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.parquet")):
                if CONTROL.search(f.name):
                    found[f.name][sub].append(f)

    if not found:
        print("No alpha=0 controls found.")
        return

    total_freed = 0
    plan = []
    for stem, subs in sorted(found.items()):
        print(f"\n{stem}")
        for sub, paths in sorted(subs.items()):
            digests = {md5(p): p for p in paths}
            size = paths[0].stat().st_size
            if len(digests) > 1:
                print(f"  {sub:<16} {len(paths)} copies but {len(digests)} DISTINCT "
                      f"contents -- NOT merging")
                for h, p in digests.items():
                    print(f"      {h[:10]}  {p}")
                continue
            keep = paths[0]
            dup = len(paths) - 1
            total_freed += dup * size
            print(f"  {sub:<16} {len(paths)} identical copies ({size/1e6:.1f} MB each) "
                  f"-> baseline/{sub}/, freeing {dup * size / 1e6:.1f} MB")
            plan.append((keep, BASELINE / sub / stem, paths))

    print(f"\ntotal freed: {total_freed / 1e6:.1f} MB across {len(plan)} files")
    if args.dry_run:
        print("--dry-run: nothing moved")
        return

    for keep, dest, paths in plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(keep, dest)
        assert md5(dest) == md5(keep), f"copy mismatch for {dest}"
        for p in paths:
            p.unlink()
    print(f"Moved {len(plan)} files into {BASELINE}/ and removed the per-property copies.")
    print("property_predictions were left in place -- they are property-specific.")


if __name__ == "__main__":
    main()
