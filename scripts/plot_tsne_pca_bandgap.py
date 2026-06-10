#!/usr/bin/env python3
"""
PCA and t-SNE plots of CrystaLLM embeddings coloured by band gap (eV).
Samples 5000 points uniformly across the band gap range (not the dataset).

Usage:
    python plot_tsne_pca_bandgap.py [--layer 14] [--n-samples 5000] [--bins 50]
"""
import argparse
import glob
import numpy as np

RANDOM_SEED = 42
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_embeddings(layer: int) -> pd.DataFrame:
    # prefer consolidated parquet, fall back to checkpoint files
    single = Path(f"embeddings/cif_layer{layer}.parquet")
    if single.exists():
        print(f"Loading {single} ...")
        return pd.read_parquet(single, columns=["id", "embedding"])

    ckpt_dir = Path(f"embeddings/cif_layer{layer}")
    files = sorted(ckpt_dir.glob("checkpoint_*.parquet")) + sorted(ckpt_dir.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No embedding files found for layer {layer}")
    print(f"Loading {len(files)} checkpoint files from {ckpt_dir} ...")
    return pd.concat([pd.read_parquet(f, columns=["id", "embedding"]) for f in files], ignore_index=True)


def uniform_sample_by_value(values: np.ndarray, n: int, bins: int) -> np.ndarray:
    """Return indices sampled uniformly across the value range."""
    bin_edges = np.linspace(values.min(), values.max(), bins + 1)
    bin_idx = np.digitize(values, bin_edges, right=True)
    bin_idx = np.clip(bin_idx, 1, bins)

    per_bin = max(1, n // bins)
    selected = []
    for b in range(1, bins + 1):
        idx = np.where(bin_idx == b)[0]
        if len(idx) == 0:
            continue
        chosen = np.random.choice(idx, size=min(per_bin, len(idx)), replace=False)
        selected.append(chosen)

    selected = np.concatenate(selected)
    if len(selected) > n:
        selected = np.random.choice(selected, size=n, replace=False)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--tsne-perplexity", type=float, default=30)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    # Load metadata — CLEAN NOMAD band_gap.value (eV), not the corrupt LUMO-HOMO diff
    print("Loading metadata ...")
    BG_COL = "dos_electronic.band_gap"
    meta = pd.read_parquet("metadata.parquet", columns=["id", BG_COL])
    meta = meta.dropna(subset=[BG_COL]).reset_index(drop=True)
    meta["band_gap_ev"] = meta[BG_COL].astype(float)
    print(f"  Entries with clean band gap: {len(meta):,}")

    # Load embeddings and join
    emb = load_embeddings(args.layer)
    print(f"  Embeddings: {len(emb):,}")

    df = emb.merge(meta, on="id", how="inner")
    print(f"  After join: {len(df):,}")

    metals = df[df["band_gap_ev"] <= 0.05].reset_index(drop=True)
    nonmetals = df[df["band_gap_ev"] > 0.05].reset_index(drop=True)
    print(f"  Metals (gap ≤ 0.05 eV):    {len(metals):,}")
    print(f"  Non-metals (gap > 0.05 eV): {len(nonmetals):,}")

    def uniform_sample(df_, n, bins):
        idx = uniform_sample_by_value(df_["band_gap_ev"].values, n, bins)
        return df_.iloc[idx].reset_index(drop=True)

    metals    = uniform_sample(metals,    args.n_samples, args.bins)
    nonmetals = uniform_sample(nonmetals, args.n_samples, args.bins)

    out_dir = Path(f"plots/layer{args.layer}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = "plasma"

    for group, label, fname_tag in [
        (metals,    "Metals (gap ≤ 0.05 eV)",      "metals"),
        (nonmetals, "Non-metals (gap > 0.05 eV)",  "nonmetals"),
    ]:
        X = np.vstack(group["embedding"].values)
        y = group["band_gap_ev"].values

        print(f"\nProcessing {label} (n={len(group):,}) ...")
        print(f"  Band gap range: {y.min():.3f} – {y.max():.3f} eV")

        print("  Running PCA ...")
        X_pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

        print("  Running t-SNE ...")
        X_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                      random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, coords, method in [
            (axes[0], X_pca,  "PCA"),
            (axes[1], X_tsne, "t-SNE"),
        ]:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap=cmap,
                            s=8, alpha=0.7, linewidths=0)
            plt.colorbar(sc, ax=ax, label="Band gap (eV)")
            ax.set_title(f"{method} — Layer {args.layer}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        plt.suptitle(f"{label} — CrystaLLM Layer {args.layer} (n={len(group):,})", y=1.01)
        plt.tight_layout()
        out = out_dir / f"bandgap_{fname_tag}_layer{args.layer}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


if __name__ == "__main__":
    main()
