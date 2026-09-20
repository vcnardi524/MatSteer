#!/usr/bin/env python3
"""Re-decode llamat2-cif generations from their stored raw_output, in place.

WHY THIS CAN EXIST AT ALL. Decoding happens at GENERATION time, so the decode policy is
baked into whatever generated_cifs holds. But `raw_output` keeps the model's text
verbatim, and crystal_string_to_cif is deterministic, so a policy change is a CPU pass
over the existing parquets rather than a regeneration on the GPU. Verified: re-decoding
an untouched file reproduces its cif_steered byte-for-byte.

USE IT WHEN the decode policy changes. It changed on 2026-09-20, from
`symprec=0.1` (which wrote the conventional cell -- a generated 6.2/6.2/6.2 four-atom
cell stored as 7.112/7.429/10.157 with sixteen atoms, on 3.5% of structures) to the
authors' faithful P 1 output, which preserves the generated cell exactly.

Anything downstream of generation is invalidated by a rewrite, because validation,
relaxation and predictions all join on the CIF. Delete those and recompute; this script
refuses to touch a file whose flags already exist unless --force is given.

Usage:
    python scripts/data/redecode_llamat_cifs.py steering_results/llamat2_cif/*/generated_cifs/*.parquet
    python scripts/data/redecode_llamat_cifs.py --dry-run <files>
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llamat_prompts


def redecode(path, dry_run=False, force=False):
    df = pd.read_parquet(path)
    if "raw_output" not in df.columns:
        return f"skipped -- no raw_output column (not a decoding model's run)"

    flags = path.replace("/generated_cifs/", "/validation/")
    if os.path.exists(flags) and not force and not dry_run:
        return (f"REFUSED -- {os.path.basename(flags)} exists and would be stale. "
                f"Delete the downstream files, or pass --force")

    new_cif, new_reason, changed = [], [], 0
    for raw, old in zip(df["raw_output"], df["cif_steered"]):
        cif, reason = llamat_prompts.crystal_string_to_cif(raw)
        cif = cif or ""
        changed += (cif != old)
        new_cif.append(cif)
        new_reason.append(reason)

    if not dry_run:
        df["cif_steered"] = new_cif
        df["decode_reason"] = new_reason
        df.to_parquet(path, index=False)

    n_decoded = sum(1 for c in new_cif if c)
    return (f"{len(df):>6,} rows  {changed:>6,} changed  "
            f"{n_decoded:>6,} decode to a CIF" + ("  [dry run]" if dry_run else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="generated_cifs parquets")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="rewrite even when downstream flags exist (they become stale)")
    args = ap.parse_args()

    for path in args.files:
        print(f"{os.path.basename(path):<52} {redecode(path, args.dry_run, args.force)}",
              flush=True)


if __name__ == "__main__":
    main()
