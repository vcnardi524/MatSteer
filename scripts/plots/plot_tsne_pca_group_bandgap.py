#!/usr/bin/env python3
"""
PCA and t-SNE of CrystaLLM embeddings for materials in ONE symmetry group
(a single space group or a single point group), coloured by band gap.

BAND GAP SOURCE — read this before trusting any output
------------------------------------------------------
The gap plotted here is `metadata.parquet`'s **`dos_electronic.band_gap`**, in eV,
which comes from NOMAD's `results.properties.electronic.dos_electronic.band_gap[0].value`
(see scripts/data/add_nomad_bandgap_columns.py). It is byte-for-byte identical to
`electronic.band_gap` in this file (same 582,596 non-null rows, correlation 1.0), so
either name selects the same numbers; `--bg-col` can switch between them.

It is NOT `energy_lowest_unoccupied - energy_highest_occupied`. Those two columns are
raw Joules (~1e-19) and their difference is the corrupt LUMO-HOMO gap that
compute_bandgap_steering.py / analyze_steering_norms.py used. This script never reads
them.

WHY THE SAMPLING IS ENRICHED BY DEFAULT
---------------------------------------
Inside any one symmetry group the gap distribution is ~99% metals (F-43m: 1,322 of
96,447 entries above 0.05 eV). A uniform random sample therefore renders as a solid
block of the colormap's zero end with a handful of stray coloured dots. So the default
`--sample-mode enriched` takes EVERY non-metal in the group plus a random metal fill up
to --n-samples, which makes the gradient visible. That over-represents non-metals
relative to the population — use `--sample-mode random` for a proportional sample.

The colour scale is capped (default: 99th percentile of the plotted gaps) because a
single 18 eV outlier would otherwise flatten everything else to one colour.

OUTPUT
------
analysis/<dataset>/<variant>/<partition>/plots/layer{N}/bandgap_tsne_pca_{sg|pg}_{group}.png — a 2x2 grid:
    PC1-2 and PC3-4 coloured by gap, t-SNE coloured by gap, and t-SNE with a binary
    metal / non-metal colouring (same t-SNE embedding, just recoloured — the binary
    panel is the one that stays readable when the gaps are this zero-inflated).

Usage:
    python scripts/plots/plot_tsne_pca_group_bandgap.py --sg F-43m
    python scripts/plots/plot_tsne_pca_group_bandgap.py --pg m-3m --layer 14
    python scripts/plots/plot_tsne_pca_group_bandgap.py --sg R-3m --vmax 2 --n-samples 8000
    python scripts/plots/plot_tsne_pca_group_bandgap.py --list-groups
"""
import argparse
from pathlib import Path

import numpy as np
RANDOM_SEED = 42
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py
from utils import load_embeddings, add_partition_args, filter_partition, analysis_dir

METADATA_PATH = "metadata.parquet"
# The clean gap in eV. NOT the LUMO-HOMO difference -- see the module docstring.
DEFAULT_BG_COL = "dos_electronic.band_gap"
CMAP = "plasma"


def load_group(group_col: str, group: str, layer: int, dataset: str,
               bg_col: str, variant: str, partition: str) -> pd.DataFrame:
    """Embeddings for one symmetry group, joined to the clean band gap (eV)."""
    print(f"Band-gap column: {bg_col}  (eV, clean -- not LUMO-HOMO)")
    meta = pd.read_parquet(METADATA_PATH, columns=["id", group_col, bg_col])
    in_group = meta[group_col] == group
    print(f"{group_col} == {group!r}: {in_group.sum():,} materials in metadata")
    if not in_group.any():
        raise SystemExit(f"No materials with {group_col} == {group!r}. "
                         f"Run with --list-groups to see the available values.")

    meta = meta[in_group].dropna(subset=[bg_col]).reset_index(drop=True)
    meta["band_gap_ev"] = meta[bg_col].astype(float)
    print(f"  with a non-null {bg_col}: {len(meta):,}")
    if meta.empty:
        raise SystemExit(f"Every {group} entry has a null {bg_col} -- nothing to plot.")

    emb = load_embeddings(layer, dataset=dataset, variant=variant)
    emb = filter_partition(emb, partition)
    df = emb.merge(meta[["id", "band_gap_ev"]], on="id", how="inner")
    print(f"  with layer-{layer} embeddings: {len(df):,}")
    if len(df) < 10:
        raise SystemExit(f"Only {len(df)} points after the join -- too few to embed.")
    return df


def sample(df: pd.DataFrame, mode: str, n_samples: int, metal_max: float) -> pd.DataFrame:
    """Down-sample to n_samples, either proportionally or enriched for non-metals."""
    nonmetals = df[df["band_gap_ev"] > metal_max]
    metals = df[df["band_gap_ev"] <= metal_max]
    print(f"  metals (gap <= {metal_max} eV): {len(metals):,}   "
          f"non-metals: {len(nonmetals):,} ({len(nonmetals) / len(df):.2%})")

    if len(df) <= n_samples:
        print(f"  using all {len(df):,} points (below --n-samples)")
        out = df
    elif mode == "random":
        out = df.sample(n=n_samples, random_state=RANDOM_SEED)
        print(f"  random sample: {len(out):,} "
              f"({(out['band_gap_ev'] > metal_max).sum():,} non-metals)")
    else:  # enriched: keep every non-metal, fill the rest with metals
        keep_nm = (nonmetals if len(nonmetals) <= n_samples
                   else nonmetals.sample(n=n_samples, random_state=RANDOM_SEED))
        n_fill = min(max(0, n_samples - len(keep_nm)), len(metals))
        out = pd.concat([keep_nm, metals.sample(n=n_fill, random_state=RANDOM_SEED)])
        print(f"  enriched sample: {len(out):,} "
              f"({len(keep_nm):,} non-metals + {n_fill:,} metals) -- non-metals are "
              f"over-represented vs. the population")

    # Sort by gap so the highest-gap points are drawn last instead of being buried
    # under the metal fill.
    return out.sort_values("band_gap_ev").reset_index(drop=True)


def neighbour_purity(coords: np.ndarray, is_nm: np.ndarray, k: int = 10) -> tuple[float, float]:
    """(observed, chance) fraction of non-metal among the k nearest neighbours of
    non-metals, in the given 2-D layout. Observed >> chance means the gap is
    organizing the map rather than being scattered through it."""
    if is_nm.sum() < 2:
        return float("nan"), float("nan")
    k = min(k, len(coords) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords[is_nm])
    return float(is_nm[idx[:, 1:]].mean()), float(is_nm.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sg", help="space_group_symbol to restrict to, e.g. F-43m")
    g.add_argument("--pg", help="point_group to restrict to, e.g. m-3m")
    ap.add_argument("--layer", type=int, default=14)
    add_partition_args(ap)   # --dataset / --variant / --partition
    #   note: v1_mp has no symmetry metadata -- use v1_all here
    ap.add_argument("--bg-col", default=DEFAULT_BG_COL,
                    choices=["dos_electronic.band_gap", "electronic.band_gap"],
                    help="Clean band-gap column in eV (the two are identical)")
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--sample-mode", choices=["enriched", "random"], default="enriched")
    ap.add_argument("--metal-max", type=float, default=0.05,
                    help="Gap at or below this is a metal (repo convention: 0.05 eV)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Colour-scale cap in eV. Default: 99th percentile of the "
                         "plotted gaps, so one outlier can't flatten the scale.")
    ap.add_argument("--tsne-perplexity", type=float, default=30)
    ap.add_argument("--list-groups", action="store_true",
                    help="Print the space groups / point groups that have band-gap "
                         "data, with counts, and exit")
    args = ap.parse_args()
    np.random.seed(RANDOM_SEED)

    if args.list_groups:
        meta = pd.read_parquet(METADATA_PATH,
                               columns=["space_group_symbol", "point_group", args.bg_col])
        meta = meta.dropna(subset=[args.bg_col])
        for col in ["space_group_symbol", "point_group"]:
            counts = meta.groupby(col)[args.bg_col].agg(
                n="size", non_metals=lambda s: int((s > args.metal_max).sum()),
                max_gap="max").sort_values("n", ascending=False)
            print(f"\n=== {col} ({len(counts)} values with band-gap data) ===")
            print(counts.head(40).to_string())
        return

    if not (args.sg or args.pg):
        ap.error("pass --sg <space group> or --pg <point group> (or --list-groups)")
    group_col, group = (("space_group_symbol", args.sg) if args.sg
                        else ("point_group", args.pg))

    df = load_group(group_col, group, args.layer, args.dataset, args.bg_col,
                    args.variant, args.partition)
    df = sample(df, args.sample_mode, args.n_samples, args.metal_max)

    X = np.vstack(df["embedding"].values)
    y = df["band_gap_ev"].values
    is_nm = y > args.metal_max
    vmax = args.vmax if args.vmax is not None else max(float(np.percentile(y, 99)), 1e-3)
    print(f"  gap range {y.min():.4f}-{y.max():.4f} eV, colour capped at {vmax:.3f} eV")

    print("Running PCA (4 components) ...")
    n_pc = min(4, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_pc, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print("  explained variance:", " ".join(f"PC{i+1}={evr[i]:.3f}" for i in range(n_pc)))
    for i in range(n_pc):
        rho = spearmanr(X_pca[:, i], y).statistic
        print(f"  Spearman(PC{i+1}, gap) = {rho:+.3f}")

    print("Running t-SNE ...")
    perp = min(args.tsne_perplexity, max(5.0, (len(df) - 1) / 3))
    X_tsne = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_SEED,
                  n_jobs=-1).fit_transform(X)

    obs, chance = neighbour_purity(X_tsne, is_nm)
    purity = (f"non-metal 10-NN purity in t-SNE: {obs:.2f} vs {chance:.2f} chance"
              if np.isfinite(obs) else "non-metal 10-NN purity: n/a (<2 non-metals)")
    print(f"  {purity}")

    fig, axes = plt.subplots(2, 2, figsize=(17, 15))
    panels = [
        (axes[0, 0], X_pca[:, 0], X_pca[:, min(1, n_pc - 1)],
         f"PCA PC1 vs PC2 (var {evr[:2].sum():.1%})", "PC1", "PC2"),
        (axes[0, 1], X_pca[:, min(2, n_pc - 1)], X_pca[:, min(3, n_pc - 1)],
         f"PCA PC3 vs PC4 (var {evr[2:4].sum():.1%})", "PC3", "PC4"),
        (axes[1, 0], X_tsne[:, 0], X_tsne[:, 1], "t-SNE", "Component 1", "Component 2"),
    ]
    for ax, cx, cy, title, xlab, ylab in panels:
        sc = ax.scatter(cx, cy, c=y, cmap=CMAP, s=8, alpha=0.75, linewidths=0,
                        vmin=0, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=f"Band gap (eV, capped at {vmax:.2f})")
        ax.set_title(f"{title} — layer {args.layer}")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    # Binary panel: the gap is so zero-inflated that metal vs non-metal is the only
    # contrast that survives; same t-SNE coordinates, recoloured.
    ax = axes[1, 1]
    ax.scatter(X_tsne[~is_nm, 0], X_tsne[~is_nm, 1], s=6, alpha=0.4, linewidths=0,
               color="lightgray", label=f"metal, gap <= {args.metal_max} eV ({(~is_nm).sum():,})")
    ax.scatter(X_tsne[is_nm, 0], X_tsne[is_nm, 1], s=10, alpha=0.85, linewidths=0,
               color="crimson", label=f"non-metal ({is_nm.sum():,})")
    ax.set_title(f"t-SNE, metal vs non-metal — {purity}")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    ax.legend(markerscale=2, fontsize=9, loc="best", framealpha=0.9)

    fig.suptitle(
        f"{group_col} = {group} — CrystaLLM layer {args.layer} embeddings by band gap\n"
        f"n={len(df):,} ({args.sample_mode} sample), gap = {args.bg_col} in eV",
        y=1.00, fontsize=13)
    fig.tight_layout()

    out_dir = analysis_dir(args.dataset, args.variant, args.partition,
                           subdir=f"plots/layer{args.layer}")
    tag = "sg" if args.sg else "pg"
    safe = group.replace("/", "_").replace("-", "m")
    out = out_dir / f"bandgap_tsne_pca_{tag}_{safe}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
