#!/usr/bin/env python3
"""
Consolidate checkpoint parquet files for each layer into a single parquet.
Skips layers that already have a consolidated file.

Usage:
    python consolidate_embeddings.py [--layers 0,1,2,...] [--force]
"""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices. Default: all dirs in embeddings/")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing consolidated files")
    args = parser.parse_args()

    emb_dir = Path("embeddings")

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = sorted(
            int(d.name.replace("cif_layer", ""))
            for d in emb_dir.iterdir()
            if d.is_dir() and d.name.startswith("cif_layer")
        )

    print(f"Layers to consolidate: {layers}")

    for layer in layers:
        out_path = emb_dir / f"cif_layer{layer}.parquet"
        ckpt_dir = emb_dir / f"cif_layer{layer}"

        if out_path.exists() and not args.force:
            print(f"Layer {layer}: already consolidated ({out_path}), skipping")
            continue

        files = sorted(ckpt_dir.glob("checkpoint_*.parquet"))
        if not files:
            print(f"Layer {layer}: no checkpoint files found, skipping")
            continue

        print(f"Layer {layer}: combining {len(files)} files -> {out_path} ...")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        before = len(df)
        df = df.drop_duplicates(subset=["id"])
        after = len(df)
        df.to_parquet(out_path, index=False)
        print(f"  -> {after:,} rows saved (dropped {before - after:,} duplicates)")

    print("Done.")

if __name__ == "__main__":
    main()
