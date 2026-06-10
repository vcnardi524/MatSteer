#!/usr/bin/env python3
"""
get_preparsed_metadata.py — Fetch NOMAD entries and store raw results JSON.

Columns per row:
  - id:         NOMAD_<entry_id>
  - results:    raw JSON string of archive["results"]
  - quantities: JSON string of the flat field-path list (structure index)

Usage:
    python get_preparsed_metadata.py --pkl CrystaLLM/cifs_v1_prep.pkl.gz \
                                     --out preparsed_metadata_nomad.parquet \
                                     [--batch-size 200] \
                                     [--workers 5] \
                                     [--limit 5000]
"""

import argparse
import gzip
import io
import json
import pickle
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

NOMAD_BASE = "https://nomad-lab.eu/prod/v1/api/v1"

REQUIRED = {
    "results": "*",
}


def load_nomad_ids(pkl_path: str, limit: int = None) -> list:
    print(f"Loading {pkl_path} ...")
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if limit:
        data = data[:limit]
    ids = [id_ for id_, _ in data if id_.startswith("NOMAD_")]
    print(f"Found {len(ids):,} NOMAD entries")
    return ids


def _flatten_keys(obj, prefix="") -> list:
    """Recursively flatten a nested dict into dot-separated field paths."""
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.append(full)
            keys.extend(_flatten_keys(v, full))
    elif isinstance(obj, list) and obj:
        keys.extend(_flatten_keys(obj[0], prefix))
    return keys


def fetch_batch(api_ids: list, retries: int = 3, backoff: float = 5.0) -> list:
    payload = {
        "query": {"entry_id:any": api_ids},
        "required": REQUIRED,
    }
    session = requests.Session()
    for attempt in range(1, retries + 1):
        try:
            r = session.post(
                f"{NOMAD_BASE}/entries/archive/download/query",
                json=payload,
                timeout=300,
            )
            if r.status_code != 200:
                print(f"  WARNING: batch returned {r.status_code} (attempt {attempt}/{retries})", flush=True)
                if attempt < retries:
                    time.sleep(backoff * attempt)
                continue

            rows = []
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for name in zf.namelist():
                    if name == "manifest.json":
                        continue
                    with zf.open(name) as jf:
                        data = json.load(jf)
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    entry_id = data.get("entry_id", name.split("/")[-1].replace(".json", ""))
                    archive = data.get("archive", {})
                    results = archive.get("results", {})
                    quantities = archive.get("quantities", _flatten_keys(results, "results"))
                    rows.append({
                        "id": f"NOMAD_{entry_id}",
                        "results": json.dumps(results),
                        "quantities": json.dumps(quantities),
                    })
            return rows

        except Exception as e:
            print(f"  WARNING: error on attempt {attempt}/{retries}: {e}", flush=True)
            if attempt < retries:
                time.sleep(backoff * attempt)

    print(f"  ERROR: giving up on batch of {len(api_ids)} entries", flush=True)
    return [{"id": f"NOMAD_{id_}", "results": "{}", "quantities": "[]"} for id_ in api_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out", default="preparsed_metadata_nomad.parquet")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)

    done_ids = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["id"])
        done_ids = set(existing["id"].values)
        print(f"Resuming — {len(done_ids):,} already fetched")

    nomad_ids = load_nomad_ids(args.pkl, args.limit)
    api_ids = [id_.replace("NOMAD_", "") for id_ in nomad_ids if id_ not in done_ids]
    total = len(api_ids)
    print(f"Fetching {total:,} entries with {args.workers} workers, batch size {args.batch_size} ...")

    batches = [api_ids[i: i + args.batch_size] for i in range(0, total, args.batch_size)]

    lock = threading.Lock()
    pending_rows = []
    completed_batches = 0
    completed_entries = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            rows = future.result()
            batch_size = len(futures[future])

            with lock:
                pending_rows.extend(rows)
                completed_batches += 1
                completed_entries += batch_size

                elapsed = time.time() - start_time
                rate = completed_entries / elapsed * 3600
                remaining = (total - completed_entries) / (completed_entries / elapsed) if completed_entries else 0
                print(
                    f"  {completed_entries + len(done_ids):,} / {total + len(done_ids):,}"
                    f"  ({rate:,.0f}/hr, ~{remaining/3600:.1f}h left)",
                    flush=True
                )

                if completed_batches % args.checkpoint_every == 0:
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
