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
import matplotlib.lines as mlines
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import load_embeddings, add_partition_args, filter_partition, analysis_dir, display_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    add_partition_args(parser)   # --dataset / --variant / --partition
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--property", type=str, default="point_group",
                        choices=["point_group", "space_group_symbol"])
    parser.add_argument("--top-n", type=int, default=20,
                        help="Keep top-N most common labels; rest become 'Other'")
    parser.add_argument("--tsne-perplexity", type=float, default=30)
    parser.add_argument("--labels", default=None,
                        help="Parquet with id + the label column. Default: "
                             "symmetry_v1_mp.parquet for v1_mp, else metadata.parquet")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    # metadata.parquet (NOMAD) is the only shipped file carrying symmetry columns and it
    # does not cover v1_mp -- metadata_mp.parquet has none in any of its 56. For v1_mp the
    # labels come from symmetry_v1_mp.parquet, derived from the CIF text by
    # scripts/data/build_symmetry_mp.py, whose id is already MP_-prefixed so it joins
    # straight onto the embedding ids.
    labels = args.labels or ("symmetry_v1_mp.parquet" if args.dataset == "v1_mp"
                             else "metadata.parquet")
    print(f"Loading labels from {labels} ...")
    meta = pd.read_parquet(labels, columns=["id", args.property])
    meta = meta[meta[args.property].notna()].reset_index(drop=True)
    print(f"  Entries with {args.property}: {len(meta):,}")

    emb = load_embeddings(args.layer, dataset=args.dataset, variant=args.variant,
                          model=args.model)
    print(f"  Embeddings: {len(emb):,}")
    emb = filter_partition(emb, args.partition, model=args.model)

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

    # COLOUR x MARKER, not 21 hues. The previous version fell back to gist_ncar -- a
    # rainbow colormap -- once there were more than 20 categories, which put Pm-3m and
    # Immm 0.125 apart in RGB and P4/mmm and P6_3/mmc 0.200 apart. Anything under ~0.25
    # is not reliably distinguishable, so two different labels in two different clusters
    # looked like one label scattered across the plot, i.e. like the labels were wrong.
    #
    # Okabe-Ito is colourblind-safe; 8 hues x 3 markers gives 24 series that stay
    # distinguishable, and the marker carries the distinction when the hue repeats.
    OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                 "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    MARKERS = ["o", "^", "s"]
    if n_colors > len(OKABE_ITO) * len(MARKERS):
        print(f"  ! {n_colors} categories exceeds the {len(OKABE_ITO)*len(MARKERS)} "
              f"distinguishable combinations -- lower --top-n")
    style = {lbl: (OKABE_ITO[i % len(OKABE_ITO)], MARKERS[i // len(OKABE_ITO)])
             for i, lbl in enumerate(ordered)}
    style["Other"] = ("#BBBBBB", ".")          # never competes with a real class

    X = np.vstack(df["embedding"].values)
    prop_title = args.property.replace("_", " ").title()

    print("Running PCA ...")
    X_pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

    print("Running t-SNE ...")
    X_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                  random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)

    out_dir = analysis_dir(args.dataset, args.variant, args.partition,
                           subdir=f"plots/layer{args.layer}", model=args.model)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, coords, method in [
        (axes[0], X_pca,  "PCA"),
        (axes[1], X_tsne, "t-SNE"),
    ]:
        for lbl in ordered:
            mask = df["label"].values == lbl
            colour, marker = style[lbl]
            ax.scatter(coords[mask, 0], coords[mask, 1], c=colour, marker=marker,
                       s=7 if lbl != "Other" else 4,
                       alpha=0.75 if lbl != "Other" else 0.25,
                       linewidths=0, label=lbl)
        # perplexity is a t-SNE hyperparameter only; it means nothing for PCA
        detail = f" (perplexity={args.tsne_perplexity:g})" if method == "t-SNE" else ""
        ax.set_title(f"{method} — Layer {args.layer}{detail}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    # shared legend outside the plots
    # Line2D, not Patch: the marker is half the identity now, so the legend has to show it
    handles = [mlines.Line2D([], [], color=style[lbl][0], marker=style[lbl][1],
                             linestyle="none", markersize=5, label=lbl)
               for lbl in ordered]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=7, ncol=1, framealpha=0.8)

    plt.suptitle(f"{prop_title} — {display_name(args.model)} layer {args.layer} "
                 f"({args.dataset}/{args.variant}/{args.partition}, n={len(df):,})", y=1.01)
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
