#!/usr/bin/env python3
"""
Consolidate the per-layer checkpoint parquets written by extract_cif_embeddings.py
into a single parquet per layer. Skips layers that already have a consolidated file.

Reads  embeddings/<dataset>/<variant>/cif_layer{N}/checkpoint_*.parquet
Writes embeddings/<dataset>/<variant>/cif_layer{N}.parquet

load_embeddings() prefers the single file when it exists, so consolidating changes
nothing for callers -- it just replaces ~358 opens per layer with one.

Usage:
    python consolidate_embeddings.py                          # v1_all/full, all layers
    python consolidate_embeddings.py --variant nosym
    python consolidate_embeddings.py --dataset v1_mp --layers 0,5,14
"""
import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import DEFAULT_DATASET, DEFAULT_VARIANT, embeddings_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        help="full or nosym (see utils.VARIANTS)")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices. Default: every cif_layer* dir found")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing consolidated files")
    args = parser.parse_args()

    base = Path("embeddings") / args.dataset / args.variant
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = sorted(int(d.name.replace("cif_layer", ""))
                        for d in base.iterdir()
                        if d.is_dir() and d.name.startswith("cif_layer"))
    print(f"Consolidating {base}/ layers: {layers}")

    for layer in layers:
        out_path, ckpt_dir = embeddings_paths(layer, args.dataset, args.variant)

        if out_path.exists() and not args.force:
            print(f"Layer {layer}: already consolidated ({out_path}), skipping")
            continue

        files = sorted(ckpt_dir.glob("checkpoint_*.parquet"))
        if not files:
            print(f"Layer {layer}: no checkpoint files found, skipping")
            continue

        # Streamed rather than pd.concat'ed: a layer is ~13.6 GB, so materializing every
        # checkpoint and then concatenating peaks near 2x that. Writing chunk by chunk
        # holds one checkpoint at a time, so memory stays flat no matter how big the layer.
        print(f"Layer {layer}: combining {len(files)} files -> {out_path} ...")
        seen = set()
        kept = dropped = 0
        writer = None
        tmp_path = out_path.with_suffix(".parquet.partial")
        try:
            for f in files:
                table = pq.read_table(f)
                keep = []
                for cif_id in table.column("id").to_pylist():
                    is_new = cif_id not in seen
                    seen.add(cif_id)
                    keep.append(is_new)
                dropped += keep.count(False)
                table = table.filter(pa.array(keep))
                if table.num_rows == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema)
                writer.write_table(table)
                kept += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        # Only becomes the real file once fully written, so an interrupted run cannot
        # leave a truncated cif_layer{N}.parquet that load_embeddings would happily read.
        os.replace(tmp_path, out_path)
        print(f"  -> {kept:,} rows saved (dropped {dropped:,} duplicates)")

    print("Done.")


if __name__ == "__main__":
    main()
