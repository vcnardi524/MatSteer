#!/usr/bin/env python3
"""
get_preparsed_metadata_mp.py — Fetch MP summary entries and store raw results JSON.

Columns per row:
  - id:         MP_<material_id>
  - results:    raw JSON string of doc.model_dump()
  - quantities: JSON string of the flat field-path list

Usage:
    python get_preparsed_metadata_mp.py \
        --out preparsed_metadata_mp.parquet \
        [--batch-size 1000] \
        [--limit 5000]
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester


def _flatten_keys(obj, prefix="") -> list:
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.append(full)
            keys.extend(_flatten_keys(v, full))
    elif isinstance(obj, list) and obj:
        keys.extend(_flatten_keys(obj[0], prefix))
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="preparsed_metadata_mp.parquet")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY environment variable not set")

    out_path = Path(args.out)

    done_ids = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["id"])
        done_ids = set(existing["id"].values)
        print(f"Resuming — {len(done_ids):,} already fetched")

    print("Fetching all material IDs from MP...")
    with MPRester(api_key) as mpr:
        all_ids = [
            doc.material_id
            for doc in mpr.materials.summary.search(fields=["material_id"])
        ]

    print(f"Total MP entries: {len(all_ids):,}")

    api_ids = [id_ for id_ in all_ids if f"MP_{id_}" not in done_ids]
    if args.limit:
        api_ids = api_ids[:args.limit]
    total = len(api_ids)
    print(f"Fetching {total:,} remaining entries, batch size {args.batch_size} ...")

    pending_rows = []
    completed = 0
    batches_done = 0
    start_time = time.time()

    with MPRester(api_key) as mpr:
        for i in range(0, total, args.batch_size):
            batch_ids = api_ids[i: i + args.batch_size]
            try:
                docs = mpr.materials.summary.search(material_ids=batch_ids)
                for doc in docs:
                    results = doc.model_dump()
                    mid = str(results.get("material_id", batch_ids[0]))
                    quantities = _flatten_keys(results)
                    pending_rows.append({
                        "id": f"MP_{mid}",
                        "results": json.dumps(results, default=str),
                        "quantities": json.dumps(quantities),
                    })
            except Exception as e:
                print(f"  WARNING: batch {i//args.batch_size} failed: {e}", flush=True)
                for mid in batch_ids:
                    pending_rows.append({
                        "id": f"MP_{mid}",
                        "results": "{}",
                        "quantities": "[]",
                    })

            completed += len(batch_ids)
            batches_done += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed * 3600
            remaining = (total - completed) / (completed / elapsed) if completed else 0
            print(
                f"  {completed + len(done_ids):,} / {total + len(done_ids):,}"
                f"  ({rate:,.0f}/hr, ~{remaining/3600:.1f}h left)",
                flush=True
            )

            if batches_done % args.checkpoint_every == 0:
                chunk = pd.DataFrame(pending_rows)
                if out_path.exists():
                    chunk = pd.concat([pd.read_parquet(out_path), chunk], ignore_index=True)
                chunk.to_parquet(out_path, index=False)
                pending_rows = []

    if pending_rows:
        chunk = pd.DataFrame(pending_rows)
        if out_path.exists():
            chunk = pd.concat([pd.read_parquet(out_path), chunk], ignore_index=True)
        chunk.to_parquet(out_path, index=False)

    df = pd.read_parquet(out_path)
    print(f"\nDone. {len(df):,} rows saved to {out_path}")
    print(df.dtypes)


if __name__ == "__main__":
    main()
