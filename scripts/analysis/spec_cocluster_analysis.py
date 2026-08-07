#!/usr/bin/env python3
"""
  1. Load CIF embeddings (cif_layer14.parquet) intersected with metadata_prep.parquet
  2. Run Spectral Co-Clustering
  3. Build feature sets and save cluster metadata
  4. Sparsify embeddings by cluster-assigned dimensions
  5. Compute subspace bases (SVD) and pairwise epsilon = ||P_k P_i||_op
  6. Point group, space group, and space-group+Wyckoff-letter enrichment per cluster
  7. Cluster uniformity check vs random baseline
  8. Save diagnostics and plots to OUTPUT_DIR

Usage:
    python spec_cocluster_analysis.py <N_CLUSTERS> <run_name> <layer>
    python spec_cocluster_analysis.py 50 v1 14
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.preprocessing import StandardScaler

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import load_labeled_embeddings

# -------------------------------
# Configuration
# -------------------------------
N_CLUSTERS = int(sys.argv[1])
name = sys.argv[2]
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 14
DATASET = sys.argv[4] if len(sys.argv) > 4 else "v1_all"  # embeddings subdir (v1_all or v1_mp)
METADATA_PATH = "./metadata.parquet"
RANDOM_SEED = 1
OUTPUT_DIR = f"./cocluster_results/{name}/layer{LAYER}/{N_CLUSTERS}_clusters"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_histogram(data, xlabel, title, path, bins=50):
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    np.random.seed(RANDOM_SEED)

    # -------------------------------
    # Load and intersect datasets
    # -------------------------------
    df = load_labeled_embeddings(LAYER, dataset=DATASET, metadata_path=METADATA_PATH)

    # Space group + occupied Wyckoff letters. A letter is only meaningful relative
    # to its space group ("c" is a different orbit in Pnma than in P6_3/mmc), so the
    # pair is the finer-grained structural label. NaN in either part -> NaN.
    if "space_group_symbol" in df.columns and "wyckoff_letters" in df.columns:
        both = df["space_group_symbol"].notna() & df["wyckoff_letters"].notna()
        df["sg_wyckoff"] = np.where(
            both, df["space_group_symbol"].astype(str) + " | " + df["wyckoff_letters"].astype(str), None
        )
        print(f"  sg_wyckoff: {int(both.sum()):,} populated, "
              f"{int((~both).sum()):,} NaN, {df['sg_wyckoff'].nunique():,} distinct")

    X = np.vstack(df["embedding"].values)
    n_samples, n_features = X.shape
    print(f"Data shape: {X.shape}")

    # -------------------------------
    # Normalize (non-negative + L2)
    # -------------------------------
    X_pos = np.abs(X)
    row_norms = np.linalg.norm(X_pos, axis=1, keepdims=True) + 1e-8
    X_pos = X_pos / row_norms

    # -------------------------------
    # Spectral Co-Clustering
    # -------------------------------
    print("Running Spectral Co-Clustering...")
    coclustering = SpectralCoclustering(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    coclustering.fit(X_pos)

    sample_labels = coclustering.row_labels_
    feature_labels = coclustering.column_labels_
    print("Co-clustering finished")

    # -------------------------------
    # Build feature sets
    # -------------------------------
    cluster_features = [[] for _ in range(N_CLUSTERS)]
    for f, k in enumerate(feature_labels):
        cluster_features[k].append(int(f))
    cluster_features = [np.array(fset, dtype=int).tolist() for fset in cluster_features]

    # Save clustered samples (drop raw embeddings — rejoin via id if needed later)
    df["cluster"] = sample_labels
    clustered_path = os.path.join(OUTPUT_DIR, "clustered_embeddings.parquet")
    df.drop(columns=["embedding"]).to_parquet(clustered_path)
    print(f"Saved clustered samples to {clustered_path}")

    # Save cluster metadata
    cluster_metadata = {
        "n_clusters": int(N_CLUSTERS),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "cluster_features": cluster_features,
    }
    meta_path = os.path.join(OUTPUT_DIR, "cluster_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(cluster_metadata, f, indent=2)
    print(f"Saved cluster metadata to {meta_path}")

    sample_counts = np.bincount(sample_labels, minlength=N_CLUSTERS)
    feature_counts = np.array([len(fset) for fset in cluster_features])

    print("\nClustering Summary:")
    print(f"  Samples:              {n_samples:,}")
    print(f"  Features (emb dim):   {n_features}")
    print(f"  Clusters:             {N_CLUSTERS}")
    print(f"  Avg samples/cluster:  {sample_counts.mean():.1f}")
    print(f"  Avg features/cluster: {feature_counts.mean():.1f}")

    # -------------------------------
    # Cluster size histogram
    # -------------------------------
    nonzero_counts = sample_counts[sample_counts > 0]
    plot_histogram(
        nonzero_counts,
        xlabel="Samples per cluster",
        title=f"Cluster sizes ({name}, K={N_CLUSTERS})",
        path=os.path.join(OUTPUT_DIR, "cluster_sizes_hist.png"),
    )
    print("Saved cluster sizes histogram")

    # -------------------------------
    # Sparsify by cluster-assigned dims
    # -------------------------------
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    _cached = all(os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in
                  ["embeddings_sparse.npy", "singular_value_stats.csv", "pairwise_epsilons.csv"])
    if _cached:
        print("\nSkipping sparsification, SVD, and epsilons — cached outputs found.")
    if not _cached:
        print("\nSparsifying embeddings by cluster-assigned dimensions...")

        X_sparse = np.zeros_like(X_std)
        for k in range(N_CLUSTERS):
            cluster_indices = np.where(sample_labels == k)[0]
            if len(cluster_indices) == 0:
                continue
            assigned_dims = np.array(cluster_features[k], dtype=int)
            if assigned_dims.size == 0:
                continue
            X_sparse[cluster_indices[:, None], assigned_dims] = X_std[cluster_indices[:, None], assigned_dims]
            if (k + 1) % 10 == 0:
                print(f"  Processed {k + 1}/{N_CLUSTERS} clusters")

        np.save(os.path.join(OUTPUT_DIR, "embeddings_sparse.npy"), X_sparse)
        print(f"Saved sparsified embeddings ({X_sparse.shape})")

        # -------------------------------
        # Subspace bases via SVD
        # -------------------------------
        print("\nComputing subspace bases...")
        unique_labels = np.unique(sample_labels)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        remapped = np.array([mapping[l] for l in sample_labels])

        bases = []
        sv_stats = []
        for cid in range(len(unique_labels)):
            mask = remapped == cid
            cluster_vecs = X_sparse[mask].T  # (D, n_k)
            if cluster_vecs.shape[1] == 0:
                bases.append(None)
                continue
            U, S, _ = np.linalg.svd(cluster_vecs, full_matrices=False)
            r = np.sum(S > 1e-10)
            bases.append(U[:, :r])
            sv_ratio = S / (S.sum() + 1e-12)
            sv_stats.append((cid, int(mask.sum()), float(S[0]), float(sv_ratio[0])))

        sv_df = pd.DataFrame(sv_stats, columns=["cluster_id", "n_samples", "largest_sv", "top_sv_fraction"])
        sv_df.to_csv(os.path.join(OUTPUT_DIR, "singular_value_stats.csv"), index=False)

        non_empty = sv_df[sv_df["n_samples"] > 0]
        print(f"  Non-empty clusters:           {len(non_empty)}")
        print(f"  Avg top singular fraction:    {non_empty['top_sv_fraction'].mean():.4f}")
        print(f"  Median top singular fraction: {non_empty['top_sv_fraction'].median():.4f}")

        plot_histogram(
            non_empty["top_sv_fraction"],
            xlabel="Top singular value fraction",
            title=f"Dominant subspace direction strength ({name}, K={N_CLUSTERS})",
            path=os.path.join(OUTPUT_DIR, "top_sv_fraction_hist.png"),
        )

        # -------------------------------
        # Pairwise projector overlaps (epsilon)
        # -------------------------------
        print("\nComputing pairwise projector overlaps...")
        projectors = [None if b is None else (b @ b.T) for b in bases]

        eps_records = []
        eps_vals = []
        for i in range(len(bases)):
            for k in range(i + 1, len(bases)):
                if projectors[i] is None or projectors[k] is None:
                    continue
                val = float(np.linalg.norm(projectors[k] @ projectors[i], 2))
                eps_records.append((i, k, val))
                eps_vals.append(val)

        eps_vals = np.array(eps_vals) if eps_vals else np.array([])
        pd.DataFrame(eps_records, columns=["cluster_i", "cluster_k", "epsilon"]).to_csv(
            os.path.join(OUTPUT_DIR, "pairwise_epsilons.csv"), index=False
        )
        print(f"Saved pairwise epsilons ({len(eps_records):,} pairs)")

        if eps_vals.size > 0:
            print("\nSubspace incoherence (epsilon = ||P_k P_i||_op):")
            print(f"  Max:    {eps_vals.max():.6f}")
            print(f"  Mean:   {eps_vals.mean():.6f}")
            print(f"  Median: {np.median(eps_vals):.6f}")
            print(f"  Std:    {eps_vals.std():.6f}")
        plot_histogram(
            eps_vals,
            xlabel=r"$\|P_k P_i\|_{\mathrm{op}}$",
            title=f"Subspace overlap epsilon ({name}, K={N_CLUSTERS})",
            path=os.path.join(OUTPUT_DIR, "epsilons_hist.png"),
        )

    # -------------------------------
    # Point group & space group enrichment
    # -------------------------------
    import seaborn as sns

    for label_col, label_name in [("point_group", "Point group"),
                                  ("space_group_symbol", "Space group"),
                                  ("sg_wyckoff", "Space group + Wyckoff letters")]:
        if label_col not in df.columns:
            continue
        print(f"\n{'='*50}")
        print(f"{label_name} Analysis")
        print(f"{'='*50}")

        valid = df[df[label_col].notna() & (df[label_col] != "")]

        # -------------------------------
        # Global statistics
        # -------------------------------
        n_unique = valid[label_col].nunique()
        label_cluster_counts = valid.groupby(label_col)["cluster"].nunique()
        n_single = (label_cluster_counts == 1).sum()

        print(f"\nGlobal {label_name} Summary:")
        print(f"  Total unique {label_name.lower()}s in dataset: {n_unique}")
        print(f"  Appearing in exactly one cluster: {n_single} ({n_single/n_unique:.3f})")

        # -------------------------------
        # Unique count per cluster + listing
        # -------------------------------
        unique_per_cluster = valid.groupby("cluster")[label_col].nunique()
        values_per_cluster = valid.groupby("cluster")[label_col].apply(
            lambda x: sorted(set(x.dropna()))
        )

        counts_list = []
        print(f"\nUnique {label_name.lower()}s per cluster:")
        for cid, vals in values_per_cluster.items():
            counts_list.append(len(vals))
            print(f"\n  Cluster {cid}: {len(vals)} unique {label_name.lower()}s")
            for v in vals:
                print(f"    {v}")

        avg_unique = sum(counts_list) / len(counts_list)
        print(f"\nSummary:")
        print(f"  Average unique {label_name.lower()}s per cluster: {avg_unique:.2f}")

        # Bar plot: unique count per cluster
        plt.figure(figsize=(12, 6))
        sns.barplot(x=unique_per_cluster.index, y=unique_per_cluster.values, color="steelblue")
        plt.xlabel("Cluster ID")
        plt.ylabel(f"Number of unique {label_name.lower()}s")
        plt.title(f"Unique {label_name.lower()}s per cluster ({name}, K={N_CLUSTERS})")
        plt.xticks(rotation=90)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{label_col}_unique_per_cluster.png"), dpi=150)
        plt.close()
        print(f"Saved unique {label_name.lower()}s per cluster plot")

        # -------------------------------
        # Cluster span distribution
        # -------------------------------
        print(f"\nAnalyzing {label_name.lower()} distribution across clusters...")
        print(f"  {label_name} -> number of clusters it appears in (sample):")
        print(label_cluster_counts.head(10).to_string())

        cluster_span_counts = label_cluster_counts.value_counts().sort_index()
        print(f"\n  Number of {label_name.lower()}s per number of clusters they appear in:")
        print(cluster_span_counts.to_string())

        plt.figure(figsize=(10, 6))
        plt.bar(cluster_span_counts.index, cluster_span_counts.values,
                color="steelblue", alpha=0.8, edgecolor="black")
        plt.xlabel(f"Number of clusters {label_name.lower()} appears in")
        plt.ylabel(f"Number of {label_name.lower()}s")
        plt.title(f"{label_name} cluster span ({name}, K={N_CLUSTERS})")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{label_col}_cluster_span.png"), dpi=150)
        plt.close()

        # -------------------------------
        # Dominant label enrichment per cluster
        # -------------------------------
        enrichment = []
        for cid, group in df.groupby("cluster"):
            counts = group[label_col].value_counts(dropna=True)
            n_labeled = int(counts.sum())
            if len(counts) == 0:
                enrichment.append((cid, "unknown", 0.0, 0, len(group)))
                continue
            # denominator is the LABELLED members, not len(group): value_counts drops
            # NaN, so dividing by the full cluster size would deflate the fraction in
            # proportion to missing labels (~28% of materials have no Wyckoff letters).
            enrichment.append((cid, counts.index[0], counts.iloc[0] / n_labeled,
                               n_labeled, len(group)))

        enrich_df = pd.DataFrame(enrichment, columns=["cluster_id", f"dominant_{label_col}",
                                                     "fraction", "n_labeled", "cluster_size"])
        enrich_df.to_csv(os.path.join(OUTPUT_DIR, f"{label_col}_enrichment.csv"), index=False)
        print(f"\n  Avg dominant {label_name} fraction: {enrich_df['fraction'].mean():.3f}"
              f"  (over labelled members only)")
        cov = enrich_df["n_labeled"].sum() / enrich_df["cluster_size"].sum()
        print(f"  Label coverage: {enrich_df['n_labeled'].sum():,}/{enrich_df['cluster_size'].sum():,}"
              f" ({cov:.1%});  clusters with <10 labelled: "
              f"{int((enrich_df['n_labeled'] < 10).sum())}")

        plot_histogram(
            enrich_df["fraction"],
            xlabel=f"Fraction of dominant {label_name.lower()}",
            title=f"{label_name} enrichment per cluster ({name}, K={N_CLUSTERS})",
            path=os.path.join(OUTPUT_DIR, f"{label_col}_enrichment_hist.png"),
        )

    # -------------------------------
    # Cluster uniformity vs random baseline
    # -------------------------------
    sample_counts_nonzero = sample_counts[sample_counts > 0]
    uniformity = sample_counts_nonzero.std() / sample_counts_nonzero.mean()
    print(f"\nCluster size uniformity (std/mean): {uniformity:.3f}")

    print("Running random baseline...")
    X_shuffled = np.abs(X_std.copy())
    np.random.shuffle(X_shuffled)
    X_shuffled /= np.linalg.norm(X_shuffled, axis=1, keepdims=True) + 1e-8
    cc_rand = SpectralCoclustering(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    cc_rand.fit(X_shuffled)
    rand_counts = np.bincount(cc_rand.row_labels_, minlength=N_CLUSTERS)
    rand_nonzero = rand_counts[rand_counts > 0]
    rand_uniformity = rand_nonzero.std() / rand_nonzero.mean()
    print(f"Random baseline uniformity (std/mean): {rand_uniformity:.3f}")
    if rand_uniformity > uniformity:
        print("Real clusters are less uniform than random — meaningful signal present")
    else:
        print("Warning: real clusters similar to random — patterns may be weak")

    print(f"\nDone. All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
