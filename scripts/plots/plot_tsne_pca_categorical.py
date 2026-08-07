#!/usr/bin/env python3
"""
PCA and t-SNE plots of CrystaLLM embeddings coloured by a categorical property
(point_group or space_group_symbol).

Usage:
    python plot_tsne_pca_categorical.py [--layer 14] [--n-samples 10000]
                                        [--property point_group]
                                        [--top-n 20]
"""
import argparse
import numpy as np

RANDOM_SEED = 42
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import load_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--dataset", default="v1_all",
                        help="Embeddings subdir under embeddings/ (v1_all or v1_mp)")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--property", type=str, default="point_group",
                        choices=["point_group", "space_group_symbol"])
    parser.add_argument("--top-n", type=int, default=20,
                        help="Keep top-N most common labels; rest become 'Other'")
    parser.add_argument("--tsne-perplexity", type=float, default=30)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    print("Loading metadata ...")
    meta = pd.read_parquet("metadata.parquet", columns=["id", args.property])
    meta = meta[meta[args.property].notna()].reset_index(drop=True)
    print(f"  Entries with {args.property}: {len(meta):,}")

    emb = load_embeddings(args.layer, dataset=args.dataset)
    print(f"  Embeddings: {len(emb):,}")

    df = emb.merge(meta, on="id", how="inner")
    print(f"  After join: {len(df):,}")

    # collapse rare labels into "Other"
    top_labels = df[args.property].value_counts().head(args.top_n).index.tolist()
    df["label"] = df[args.property].where(df[args.property].isin(top_labels), other="Other")

    # random sample (categorical — no value-based binning)
    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Sampled: {len(df):,}")

    # build label→int mapping; put "Other" last
    ordered = top_labels + (["Other"] if "Other" in df["label"].values else [])
    label2int = {lbl: i for i, lbl in enumerate(ordered)}
    codes = df["label"].map(label2int).values
    n_colors = len(ordered)

    # pick a colormap with enough distinct colors
    if n_colors <= 20:
        cmap = plt.get_cmap("tab20", n_colors)
    else:
        cmap = plt.get_cmap("gist_ncar", n_colors)

    colors = [cmap(label2int[lbl]) for lbl in df["label"]]

    X = np.vstack(df["embedding"].values)
    prop_title = args.property.replace("_", " ").title()

    print("Running PCA ...")
    X_pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

    print("Running t-SNE ...")
    X_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                  random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)

    out_dir = Path(f"plots/layer{args.layer}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, coords, method in [
        (axes[0], X_pca,  "PCA"),
        (axes[1], X_tsne, "t-SNE"),
    ]:
        for lbl in ordered:
            mask = df["label"].values == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(label2int[lbl])],
                       s=6, alpha=0.6, linewidths=0, label=lbl)
        # perplexity is a t-SNE hyperparameter only; it means nothing for PCA
        detail = f" (perplexity={args.tsne_perplexity:g})" if method == "t-SNE" else ""
        ax.set_title(f"{method} — Layer {args.layer}{detail}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    # shared legend outside the plots
    handles = [mpatches.Patch(color=cmap(label2int[lbl]), label=lbl) for lbl in ordered]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=7, ncol=1, framealpha=0.8)

    plt.suptitle(f"{prop_title} — CrystaLLM Layer {args.layer} (n={len(df):,})", y=1.01)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    fname = args.property.replace("_", "")
    # perplexity in the name only when non-default, so existing p=30 plots keep their paths
    suffix = "" if args.tsne_perplexity == 30 else f"_p{args.tsne_perplexity:g}"
    out = out_dir / f"{fname}_layer{args.layer}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
