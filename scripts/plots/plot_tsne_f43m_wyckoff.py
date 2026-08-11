#!/usr/bin/env python3
"""
Zoom-in diagnostic: t-SNE / PCA of CrystaLLM layer-14 embeddings for materials
in a SINGLE space group (default F-43m), coloured by anonymous Wyckoff signature
(the "prototype" within that space group).

Tests whether sub-clusters seen inside one space-group colour in the global
t-SNE correspond to distinct Wyckoff-occupation patterns (zincblende `a c`,
half-Heusler `a b c`, ...).

Usage:
    python scripts/plots/plot_tsne_f43m_wyckoff.py [--sg F-43m] [--layer 14]
                                             [--n-samples 10000] [--top-n 8]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
RANDOM_SEED = 42
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PREPARSED = "preparsed_metadata_nomad.parquet"


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import load_embeddings, add_partition_args, filter_partition, analysis_dir


def find_wyckoff_sets(results: dict):
    topo = results.get("material", {}).get("topology")
    if not isinstance(topo, list):
        return None
    best = None
    for t in topo:
        if not isinstance(t, dict):
            continue
        sym = t.get("symmetry")
        if isinstance(sym, dict) and sym.get("wyckoff_sets"):
            if t.get("label") == "conventional cell":
                return sym["wyckoff_sets"]
            best = best or sym["wyckoff_sets"]
    return best


def signature(ws):
    letters = [s.get("wyckoff_letter") for s in ws
               if isinstance(s, dict) and s.get("wyckoff_letter") is not None]
    if not letters:
        return None
    lc = Counter(letters)
    return " ".join(f"{n}{l}" if n > 1 else l for l, n in sorted(lc.items()))


def extract_signatures(id_set: set) -> dict:
    """Stream preparsed metadata; return {id -> signature} for ids in id_set."""
    pf = pq.ParquetFile(PREPARSED)
    out = {}
    for batch in pf.iter_batches(batch_size=4000, columns=["id", "results"]):
        ids = batch.column("id").to_pylist()
        res = batch.column("results").to_pylist()
        for id_, r in zip(ids, res):
            if id_ not in id_set:
                continue
            try:
                ws = find_wyckoff_sets(json.loads(r))
            except (json.JSONDecodeError, TypeError):
                ws = None
            out[id_] = signature(ws) if ws else None
        if len(out) >= len(id_set):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg", default="F-43m")
    ap.add_argument("--layer", type=int, default=14)
    add_partition_args(ap)   # --dataset / --variant / --partition
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--tsne-perplexity", type=float, default=30)
    args = ap.parse_args()
    np.random.seed(RANDOM_SEED)

    print(f"Space group: {args.sg}  layer {args.layer}")
    meta = pd.read_parquet("metadata.parquet", columns=["id", "space_group_symbol"])
    ids = meta.loc[meta["space_group_symbol"] == args.sg, "id"]
    print(f"  {args.sg} materials: {len(ids):,}")

    emb = load_embeddings(args.layer, dataset=args.dataset, variant=args.variant)
    emb = filter_partition(emb, args.partition)
    df = emb[emb["id"].isin(set(ids))].reset_index(drop=True)
    print(f"  with embeddings: {len(df):,}")

    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  sampled: {len(df):,}")

    print("Extracting Wyckoff signatures ...")
    sig_map = extract_signatures(set(df["id"]))
    df["signature"] = df["id"].map(sig_map).fillna("unknown")

    counts = df["signature"].value_counts()
    print(f"\n  distinct signatures in sample: {df['signature'].nunique()}")
    print("  top signatures:")
    print(counts.head(15).to_string())

    top = counts.head(args.top_n).index.tolist()
    df["label"] = df["signature"].where(df["signature"].isin(top), other="Other")
    ordered = top + (["Other"] if (df["label"] == "Other").any() else [])
    label2int = {l: i for i, l in enumerate(ordered)}
    cmap = plt.get_cmap("tab10", len(ordered))

    X = np.vstack(df["embedding"].values)
    print("\nRunning PCA ...")
    X_pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)
    print("Running t-SNE ...")
    X_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                  random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)

    out_dir = analysis_dir(args.dataset, args.variant, args.partition,
                           subdir=f"plots/layer{args.layer}")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, coords, method in [(axes[0], X_pca, "PCA"), (axes[1], X_tsne, "t-SNE")]:
        for lbl in ordered:
            m = df["label"] == lbl
            color = "lightgray" if lbl == "Other" else cmap(label2int[lbl])
            ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6,
                       color=color, label=f"{lbl} ({m.sum()})")
        ax.set_title(f"{method} — {args.sg} embeddings by Wyckoff signature")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9)
    sg_safe = args.sg.replace("/", "_").replace("-", "m")
    out = out_dir / f"tsne_pca_{sg_safe}_wyckoff.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
