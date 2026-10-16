#!/usr/bin/env python3
"""Rebuild the 1,000-prompt evaluation subset every steering run generates from.

Every arm in `steering_runs.csv` -- and the alpha=0 control they are all paired against
-- generates from the same 1,000 test structures. That is what makes the paired t-test
valid: each arm and the control share prompt ids, so prompt-to-prompt variance (by far
the largest source here) differences out. Change the subset and nothing new can be
compared with anything already measured.

The file itself lives at CrystaLLM/cifs_v1_test_sample1000.pkl.gz, inside the submodule
where *.pkl* is gitignored -- so it exists on one machine and is not recoverable if lost.
This script plus the tracked id list at data/test_sample1000_ids.csv make it
reproducible.

The original was drawn ad hoc in July 2026 with no script. The draw was recovered
afterwards by search: `random.Random(42).sample(range(len(source)), 1000)` reproduces it
exactly, ids and order. That is the default here, and --verify checks it still holds.

    # regenerate the canonical subset and its id list
    python scripts/data/make_test_sample.py

    # check the file on disk is still the one every result was computed against
    python scripts/data/make_test_sample.py --verify

    # rebuild from the tracked id list instead of re-drawing (survives a source reorder)
    python scripts/data/make_test_sample.py --from-ids data/test_sample1000_ids.csv
"""
import argparse
import gzip
import hashlib
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

SOURCE = "CrystaLLM/cifs_v1_test.pkl.gz"
OUT_PKL = "CrystaLLM/cifs_v1_test_sample1000.pkl.gz"
OUT_IDS = "data/test_sample1000_ids.csv"
SEED = 42
N = 1000


def load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def fingerprint(ids):
    """Order-independent id-set hash, so a methods section can name the subset."""
    return hashlib.sha1(",".join(sorted(ids)).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out-pkl", default=OUT_PKL)
    ap.add_argument("--out-ids", default=OUT_IDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--from-ids", default=None,
                    help="Rebuild from this id list rather than re-drawing. Exact even "
                         "if the source pkl is ever reordered.")
    ap.add_argument("--verify", action="store_true",
                    help="Compare against the existing pkl and exit non-zero on a "
                         "mismatch. Writes nothing.")
    args = ap.parse_args()

    full = load(args.source)
    full_ids = [i for i, _ in full]
    print(f"source {args.source}: {len(full_ids):,} structures")

    if args.from_ids:
        want = pd.read_csv(args.from_ids)["id"].tolist()
        by_id = dict(full)
        missing = [i for i in want if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} ids are not in the source, e.g. {missing[:3]}")
        sample = [(i, by_id[i]) for i in want]
        print(f"rebuilt {len(sample):,} entries from {args.from_ids}")
    else:
        idx = random.Random(args.seed).sample(range(len(full_ids)), args.n)
        sample = [full[j] for j in idx]
        print(f"drew {len(sample):,} with random.Random({args.seed}).sample")

    ids = [i for i, _ in sample]
    print(f"  fingerprint (sha1 of sorted ids): {fingerprint(ids)}")

    existing = Path(args.out_pkl)
    matches = None
    if existing.exists():
        old = [i for i, _ in load(args.out_pkl)]
        same_set, same_order = set(old) == set(ids), old == ids
        matches = same_order
        print(f"  vs the file on disk: same ids={same_set}  same order={same_order}")
        if args.verify:
            if not same_order:
                print("\nMISMATCH -- the subset on disk is not what this script produces.\n"
                      "Every number in steering_runs.csv was computed against the file on\n"
                      "disk, so do NOT overwrite it without re-running everything.",
                      file=sys.stderr)
                return 1
            print("\nVerified: the subset on disk is reproducible from this script.")
            return 0
    elif args.verify:
        print(f"\n{args.out_pkl} does not exist -- nothing to verify.", file=sys.stderr)
        return 1

    Path(args.out_ids).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids}).to_csv(args.out_ids, index=False)
    print(f"Wrote {args.out_ids}  ({len(ids):,} ids, tracked in git)")

    # Never rewrite a subset that is already correct. Every result in
    # steering_runs.csv was generated from the bytes on disk; touching them buys
    # nothing and risks silently changing what "the control" means.
    if matches:
        print(f"Left {args.out_pkl} alone -- already identical.")
        return 0
    if matches is False:
        print(f"\nREFUSING to overwrite {args.out_pkl}: it differs from this draw, and\n"
              "every existing result was computed against it. Delete it deliberately\n"
              "first if you really mean to replace the evaluation set.", file=sys.stderr)
        return 1
    with gzip.open(args.out_pkl, "wb") as f:
        pickle.dump(sample, f)
    print(f"Wrote {args.out_pkl}  (gitignored; rebuild with this script)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
