
#!/usr/bin/env python3
"""
  1. Run Spectral Co-Clustering 
  2. Build feature sets and save cluster metadata
  3. Sparsify steering vectors by cluster-assigned dimensions
  4. Compute subspace bases and pairwise epsilon = ||P_k P_i||_op
  5. Save diagnostics and plots to `OUTPUT_DIR`
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem 
import sys

# -------------------------------
# Configuration
# -------------------------------
name = sys.argv[2]
DATA_PATH = f"./steering_vectors/steering_vectors_all_{name}.parquet"
N_CLUSTERS = int(sys.argv[1])
RANDOM_SEED = 1
OUTPUT_DIR = f"./subspace_clustering_results_coclustering_for_figs_{name}/{N_CLUSTERS}_clusters"
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

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    X = np.vstack(df["steering_vector"].values)
    n_samples, n_features = X.shape
    print(f"Data shape: {X.shape}")

    # -------------------------------
    # Make matrix non-negative,  also normalize
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

    print("Co-clustering complete")

    # -------------------------------
    # Build feature sets
    # -------------------------------
    cluster_features = [[] for _ in range(N_CLUSTERS)]
    for f, k in enumerate(feature_labels):
        cluster_features[k].append(int(f))
    cluster_features = [np.array(fset, dtype=int).tolist() for fset in cluster_features]

    # Save clustered samples
    df["cluster"] = sample_labels
    clustered_path = os.path.join(OUTPUT_DIR, "clustered_steering_vectors_coclustering.parquet")
    df.to_parquet(clustered_path)
    print(f"✓ Saved clustered samples to {clustered_path}")

    # Save cluster metadata
    cluster_metadata = {
        "n_clusters": int(N_CLUSTERS),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "cluster_features": cluster_features,
    }
    meta_path = os.path.join(OUTPUT_DIR, "cluster_metadata_coclustering.json")
    with open(meta_path, "w") as f:
        json.dump(cluster_metadata, f, indent=2)
    print(f"✓ Saved cluster metadata to {meta_path}")

    # Diagnostics
    feature_counts = np.array([len(fset) for fset in cluster_features])
    sample_counts = np.bincount(sample_labels, minlength=N_CLUSTERS)

    print("\nClustering Summary:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Clusters: {N_CLUSTERS}")
    print(f"  Avg samples/cluster: {sample_counts.mean():.1f}")
    print(f"  Avg features/cluster: {feature_counts.mean():.1f}")

    # -------------------------------
    # Cluster size histogram
    # -------------------------------
    counts = np.bincount(sample_labels.astype(int), minlength=N_CLUSTERS)
    nonzero_counts = counts[counts > 0]
    print(f"\nCluster size stats (non-empty): min={nonzero_counts.min()}, max={nonzero_counts.max()}, mean={nonzero_counts.mean():.1f}")
    plot_histogram(
        nonzero_counts,
        xlabel="Samples per cluster",
        title=f"Cluster sizes distribution ({name})",
        path=os.path.join(OUTPUT_DIR, "cluster_sizes_hist.png"),
    )
    print("saved cluster sizes histogram as cluster_sizes_hist.png")

    # -------------------------------
    # Sparsify by cluster (keep only assigned dims per sample)
    # -------------------------------
    print("\nSparsifying vectors by cluster assigned dimensions...")

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    X_sparse = np.zeros_like(X_std)
    for k in range(N_CLUSTERS):
        cluster_mask = df['cluster'] == k
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        assigned_dims = np.array(cluster_features[k], dtype=int)
        if assigned_dims.size == 0:
            continue
        X_sparse[cluster_indices[:, None], assigned_dims] = X_std[cluster_indices[:, None], assigned_dims]
        if (k + 1) % 50 == 0:
            print(f"Processed {k + 1}/{N_CLUSTERS} clusters")

    print(f"\nSparsified data shape: {X_sparse.shape}")

    # Save sparse vectors into dataframe and files
    df['steering_vector_sparse'] = [X_sparse[i] for i in range(len(df))]
    sparse_parquet = os.path.join(OUTPUT_DIR, "clustered_steering_vectors_sparse_cocluster.parquet")
    df.to_parquet(sparse_parquet)
    np.save(os.path.join(OUTPUT_DIR, "steering_vectors_sparse_cocluster.npy"), X_sparse)
    print(f"Saved sparsified vectors to {sparse_parquet} and numpy array")

    # -------------------------------
    # Compute subspace bases and projector overlaps (epsilon)
    # -------------------------------
    print("\nComputing subspace bases and projector overlaps...")
    vectors = X_sparse
    labels = np.array(df['cluster'].values)

    unique_labels = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    remapped = np.array([mapping[l] for l in labels])

    bases = []
    for cid in range(len(unique_labels)):
        mask = remapped == cid
        cluster_vecs = vectors[mask].T  # D x n_k
        if cluster_vecs.shape[1] == 0:
            bases.append(None)
            continue

        # SVD on (D x n_k) matrix
        U, S, _ = np.linalg.svd(cluster_vecs, full_matrices=False)
        # Take only significant singular vectors
        r = np.sum(S > 1e-10)
        U_r = U[:, :r]
        bases.append(U_r)
        print(f"  Cluster {cid}: samples={cluster_vecs.shape[1]}, rank={r}, basis_shape={U_r.shape}")

    # -------------------------------
    # Count unique scaffolds per cluster
    # -------------------------------

    def get_scaffold(smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return ""
    
    print("\nComputing Murcko scaffolds...")
    df['scaffold'] = df['smile'].apply(get_scaffold)

    # -------------------------------
    # Global scaffold statistics
    # -------------------------------

    # Remove empty scaffolds
    valid_scaffolds = df[df['scaffold'] != ""]

    # 1. Total number of unique scaffolds in the dataset
    n_unique_scaffolds = valid_scaffolds['scaffold'].nunique()

    # 2. For each scaffold, count how many clusters it appears in
    scaffold_cluster_counts = (
        valid_scaffolds
        .groupby('scaffold')['cluster']
        .nunique()
    )

    # 3. Scaffolds that appear in exactly one cluster
    n_single_cluster_scaffolds = (scaffold_cluster_counts == 1).sum()

    print("\nGlobal Scaffold Summary:")
    print(f"  Total unique scaffolds in dataset: {n_unique_scaffolds}")
    print(f"  Scaffolds appearing in exactly one cluster: {n_single_cluster_scaffolds}")
    print(f"  Fraction single-cluster scaffolds: "
        f"{n_single_cluster_scaffolds / n_unique_scaffolds:.3f}")


    scaffold_counts_per_cluster = df.groupby('cluster')['scaffold'].nunique()

    print("\nUnique scaffolds per cluster:")

    # group by cluster and collect unique scaffold SMILES
    scaffolds_per_cluster = (
        df.groupby('cluster')['scaffold']
        .apply(lambda x: sorted(set(x.dropna())))
    )

    scaffold_counts = []

    for cluster_id, scaffolds in scaffolds_per_cluster.items():
        count = len(scaffolds)
        scaffold_counts.append(count)

        print(f"\nCluster {cluster_id}: {count} unique scaffolds")
        for s in scaffolds:
            print(f"    {s}")

    # ---- summary statistic ----
    avg_scaffolds = sum(scaffold_counts) / len(scaffold_counts)

    print("\nSummary:")
    print(f"  Average number of unique scaffolds per cluster: {avg_scaffolds:.2f}")

    for cluster_id, scaffolds in scaffolds_per_cluster.items():
        print(f"\nCluster {cluster_id}: {len(scaffolds)} unique scaffolds")
        for s in scaffolds:
            print(f"    {s}")
    # print("\nNumber of unique scaffolds per cluster:")
    # for cluster_id, count in scaffold_counts_per_cluster.items():
    #     print(f"  Cluster {cluster_id}: {count} unique scaffolds")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # -------------------------------
    # Plot unique scaffolds per cluster
    # -------------------------------
    plt.figure(figsize=(12, 6))
    sns.barplot(x=scaffold_counts_per_cluster.index, y=scaffold_counts_per_cluster.values, color="steelblue")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of unique scaffolds")
    plt.title(f"Unique scaffolds per cluster ({name})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(OUTPUT_DIR, "unique_scaffolds_per_cluster.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Saved plot of unique scaffolds per cluster to {plot_path}")

    # -------------------------------
    # Analyze scaffold distribution across clusters
    # -------------------------------
    print("\nAnalyzing scaffold distribution across clusters...")

    # Map scaffold -> set of clusters it appears in
    scaffold_clusters = df.groupby('scaffold')['cluster'].unique()

    # Count number of clusters per scaffold
    scaffold_cluster_counts = scaffold_clusters.apply(len)

    print("Scaffold -> number of clusters it appears in (sample):")
    print(scaffold_cluster_counts.head(10))

    # Count how many scaffolds appear in exactly N clusters
    cluster_span_counts = scaffold_cluster_counts.value_counts().sort_index()
    print("\nNumber of scaffolds per number of clusters they appear in:")
    print(cluster_span_counts)

    # -------------------------------
    # Bar plot: all scaffolds
    # -------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_span_counts.index, y=cluster_span_counts.values, color="steelblue")
    plt.xlabel("Number of clusters scaffold appears in")
    plt.ylabel("Number of scaffolds")
    plt.title(f"Scaffold Distribution Across Clusters ({name})")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.grid(alpha=0.3, axis='y')
    plot_path_all = os.path.join(OUTPUT_DIR, "scaffold_cluster_distribution_all.png")
    plt.savefig(plot_path_all, dpi=150)
    plt.close()
    print(f"✓ Saved scaffold cluster distribution plot (all scaffolds) to {plot_path_all}")

    # -------------------------------
    # Bar plot: scaffolds in multiple clusters
    # -------------------------------
    multi_cluster_scaffolds = scaffold_cluster_counts[scaffold_cluster_counts > 1]

    if not multi_cluster_scaffolds.empty:
        multi_cluster_counts = multi_cluster_scaffolds.value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=multi_cluster_counts.index, y=multi_cluster_counts.values, color="salmon")
        plt.xlabel("Number of clusters scaffold appears in")
        plt.ylabel("Number of scaffolds")
        plt.title(f"Scaffolds Occurring in Multiple Clusters ({name})")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.grid(alpha=0.3, axis='y')
        plot_path_multi = os.path.join(OUTPUT_DIR, "scaffold_cluster_distribution_multi.png")
        plt.savefig(plot_path_multi, dpi=150)
        plt.close()
        print(f"✓ Saved scaffold cluster distribution plot (multi-cluster scaffolds) to {plot_path_multi}")
    else:
        print("No scaffolds appear in multiple clusters.")


        # -------------------------------
    # Check cluster singular values / rank for meaningfulness
    # -------------------------------
    print("\nChecking cluster singular values for meaningfulness...")

    singular_value_stats = []
    for cid, U_r in enumerate(bases):
        if U_r is None:
            continue
        mask = remapped == cid
        cluster_vecs = vectors[mask].T
        _, S, _ = np.linalg.svd(cluster_vecs, full_matrices=False)
        sv_ratio = S / S.sum()  # fraction of total variance
        top_sv_frac = sv_ratio[0] if len(sv_ratio) > 0 else 0
        n_k = mask.sum()
        singular_value_stats.append((cid, n_k, S[0], top_sv_frac))


    sv_df = pd.DataFrame(
        singular_value_stats,
        columns=["cluster_id", "n_samples", "largest_singular_value", "top_singular_fraction"]
    )

    # -------------------------------
    # Strongest direction statistics
    # -------------------------------

    # Total clusters requested
    total_clusters = N_CLUSTERS

    # Non-empty clusters (have at least one sample AND a valid SVD)
    non_empty_clusters = sv_df[sv_df["n_samples"] > 0]

    n_non_empty = len(non_empty_clusters)

    avg_strongest_direction = non_empty_clusters["top_singular_fraction"].mean()
    median_strongest_direction = non_empty_clusters["top_singular_fraction"].median()

    print("\n=== Strongest Direction Summary ===")
    print(f"Total clusters requested        : {total_clusters}")
    print(f"Non-empty clusters              : {n_non_empty}")
    print(f"Empty clusters                  : {total_clusters - n_non_empty}")
    print(f"Average top singular fraction   : {avg_strongest_direction:.4f}")
    print(f"Median top singular fraction    : {median_strongest_direction:.4f}")


    print("\nTop singular value fraction per cluster (sample):")
    print(sv_df.head(10))

    plt.figure(figsize=(10, 6))
    plt.hist(sv_df["top_singular_fraction"], bins=50, color="purple", alpha=0.7, edgecolor="black")
    plt.xlabel("Top singular value fraction")
    plt.ylabel("Number of clusters")
    plt.title(f"Strength of strongest subspace direction per cluster ({name})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_top_singular_fraction.png"), dpi=150)
    plt.close()
    print("✓ Saved histogram of top singular value fractions per cluster")

    # -------------------------------
    # Cluster size uniformity check
    # -------------------------------
    sample_counts_nonzero = sample_counts[sample_counts > 0]
    uniformity_metric = sample_counts_nonzero.std() / sample_counts_nonzero.mean()
    print(f"\nCluster size uniformity metric (std/mean): {uniformity_metric:.3f}")
    if uniformity_metric < 0.2:
        print("Warning: clusters are very uniform in size → may be less meaningful")
    else:
        print("Cluster sizes show natural variation → likely meaningful")

    # -------------------------------
    # Optional: compare to random baseline
    # -------------------------------
    print("\nOptional random baseline comparison...")
    np.random.seed(RANDOM_SEED)
    X_shuffled = np.copy(X_std)
    np.random.shuffle(X_shuffled)  # shuffle rows

    coclustering_random = SpectralCoclustering(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    coclustering_random.fit(np.abs(X_shuffled))
    random_labels = coclustering_random.row_labels_

    random_sample_counts = np.bincount(random_labels, minlength=N_CLUSTERS)
    random_nonzero = random_sample_counts[random_sample_counts > 0]
    random_uniformity = random_nonzero.std() / random_nonzero.mean()
    print(f"Random baseline cluster uniformity (std/mean): {random_uniformity:.3f}")
    if random_uniformity > uniformity_metric:
        print("Your real clusters are less uniform than random → meaningful signal present")
    else:
        print("Warning: real clusters are similar to random → patterns may be weak")

        # -------------------------------
    # Scaffold enrichment per cluster
    # -------------------------------
    print("\nComputing scaffold enrichment per cluster...")

    enrichment_scores = []
    for cluster_id, scaffolds in scaffolds_per_cluster.items():
        n_scaffolds = len(scaffolds)
        cluster_mask = df['cluster'] == cluster_id
        n_samples_in_cluster = cluster_mask.sum()

        if n_samples_in_cluster == 0 or n_scaffolds == 0:
            enrichment_scores.append((cluster_id, 0.0))
            continue

        # Count frequency of each scaffold in this cluster
        scaffold_counts = df.loc[cluster_mask, 'scaffold'].value_counts()
        max_fraction = scaffold_counts.iloc[0] / n_samples_in_cluster  # fraction of dominant scaffold
        enrichment_scores.append((cluster_id, max_fraction))

    enrichment_df = pd.DataFrame(enrichment_scores, columns=["cluster_id", "dominant_scaffold_fraction"])
    print("\nSample scaffold enrichment per cluster:")
    print(enrichment_df.head(10))

    # Save scaffold enrichment
    enrichment_path = os.path.join(OUTPUT_DIR, "scaffold_enrichment_per_cluster.csv")
    enrichment_df.to_csv(enrichment_path, index=False)
    print(f"✓ Saved scaffold enrichment per cluster to {enrichment_path}")

    # Plot histogram of scaffold enrichment
    plt.figure(figsize=(10, 6))
    plt.hist(enrichment_df["dominant_scaffold_fraction"], bins=50, color="green", alpha=0.7, edgecolor="black")
    plt.xlabel("Fraction of cluster in dominant scaffold")
    plt.ylabel("Number of clusters")
    plt.title(f"Scaffold Enrichment Across Clusters ({name})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scaffold_enrichment_hist.png"), dpi=150)
    plt.close()
    print("✓ Saved histogram of scaffold enrichment per cluster")



    # -------------------------------
    # Compute fraction cut for real clusters
    # -------------------------------
    # scaffold x cluster adjacency matrix
    scaffold_cluster_table = (
        df[df['scaffold'] != ""]
        .groupby(['scaffold', 'cluster'])
        .size()
        .unstack(fill_value=0)
    )

    A = scaffold_cluster_table.values  # shape: [n_scaffolds, n_clusters]
    scaffold_names = scaffold_cluster_table.index.tolist()
    cluster_ids = scaffold_cluster_table.columns.tolist()

    # scaffold -> dominant cluster
    scaffold_to_cluster = (
        df[df['scaffold'] != ""]
        .groupby('scaffold')['cluster']
        .agg(lambda x: x.value_counts().idxmax())
    )
    scaffold_labels = scaffold_to_cluster.loc[scaffold_names].values
    cluster_labels = np.array(cluster_ids)

    def count_cut_edges(A, scaffold_labels, cluster_labels):
        cut_weight = 0
        total_weight = A.sum()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] == 0:
                    continue
                if scaffold_labels[i] != cluster_labels[j]:
                    cut_weight += A[i, j]
        return cut_weight, cut_weight / total_weight

    cut_w_real, cut_frac_real = count_cut_edges(A, scaffold_labels, cluster_labels)

    print("Spectral co-clustering:")
    print(f"  Cut edge weight     : {cut_w_real}")
    print(f"  Fraction cut edges  : {cut_frac_real:.4f}")

    # -------------------------------
    # Random cluster baseline using shuffled cluster labels
    # -------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    random_scaffold_labels = rng.permutation(scaffold_labels)

    cut_w_rand, cut_frac_rand = count_cut_edges(
        A,
        random_scaffold_labels,
        cluster_labels
    )


    print("\nRandom cluster baseline:")
    print(f"  Cut edge weight     : {cut_w_rand}")
    print(f"  Fraction cut edges  : {cut_frac_rand:.4f}")

    # # scaffold x cluster adjacency matrix
    # scaffold_cluster_table = (
    #     df[df['scaffold'] != ""]
    #     .groupby(['scaffold', 'cluster'])
    #     .size()
    #     .unstack(fill_value=0)
    # )

    # A = scaffold_cluster_table.values   # shape: [n_scaffolds, n_clusters]
    # scaffold_names = scaffold_cluster_table.index.tolist()
    # cluster_ids = scaffold_cluster_table.columns.tolist()

    # # scaffold -> dominant cluster
    # scaffold_to_cluster = (
    #     df[df['scaffold'] != ""]
    #     .groupby('scaffold')['cluster']
    #     .agg(lambda x: x.value_counts().idxmax())
    # )

    # scaffold_labels = scaffold_to_cluster.loc[scaffold_names].values
    # cluster_labels = np.array(cluster_ids)


    # def count_cut_edges(A, scaffold_labels, cluster_labels):
    #     cut_weight = 0
    #     total_weight = A.sum()

    #     for i in range(A.shape[0]):
    #         for j in range(A.shape[1]):
    #             if A[i, j] == 0:
    #                 continue
    #             if scaffold_labels[i] != cluster_labels[j]:
    #                 cut_weight += A[i, j]

    #     return cut_weight, cut_weight / total_weight

    # cut_w, cut_frac = count_cut_edges(A, scaffold_labels, cluster_labels)

    # print("Spectral co-clustering:")
    # print(f"  Cut edge weight     : {cut_w}")
    # print(f"  Fraction cut edges  : {cut_frac:.4f}")

    # def random_cut_baseline(A, cluster_labels, n_trials=100, seed=0):
    #     rng = np.random.default_rng(seed)
    #     cut_fracs = []

    #     for _ in range(n_trials):
    #         random_scaffold_labels = rng.permutation(scaffold_labels)
    #         _, frac = count_cut_edges(A, random_scaffold_labels, cluster_labels)
    #         cut_fracs.append(frac)

    #     return np.mean(cut_fracs), np.std(cut_fracs)

    # rand_mean, rand_std = random_cut_baseline(A, cluster_labels)

    # print("\nRandom baseline:")
    # print(f"  Mean cut fraction : {rand_mean:.4f}")
    # print(f"  Std              : {rand_std:.4f}")


    # ============================================================
    # Within-cluster chemical similarity evaluation
    # (Tanimoto + SMILES edit distance)
    # ============================================================

    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    import editdistance

    # -------------------------------
    # Helper: average Tanimoto similarity
    # -------------------------------
    def average_tanimoto(smiles_list, radius=2, nBits=2048):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list if isinstance(s, str)]
        mols = [m for m in mols if m is not None]

        if len(mols) < 2:
            return np.nan

        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
            for m in mols
        ]

        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

        return np.mean(sims)


    # -------------------------------
    # Helper: average SMILES edit distance
    # -------------------------------
    def average_edit_distance(smiles_list):
        smiles_list = [s for s in smiles_list if isinstance(s, str)]
        if len(smiles_list) < 2:
            return np.nan

        dists = []
        for i in range(len(smiles_list)):
            for j in range(i + 1, len(smiles_list)):
                dists.append(editdistance.eval(smiles_list[i], smiles_list[j]))

        return np.mean(dists)


    # ------------------------------------------------------------
    # IMPORTANT: restrict to same molecules used for clustering
    # ------------------------------------------------------------
    df_eval = df.copy()  # df already has correct sample_labels length


    # ============================================================
    # Real spectral co-clustering metrics
    # ============================================================
    # tanimoto_real = []
    # editdist_real = []

    # for cluster_id, group in df_eval.groupby("cluster"):
    #     smiles = group["smile"].dropna().tolist()
    #     tanimoto_real.append(average_tanimoto(smiles))
    #     editdist_real.append(average_edit_distance(smiles))

    # real_tan_mean = np.nanmean(tanimoto_real)
    # real_ed_mean = np.nanmean(editdist_real)

    # print("\nSpectral co-clustering:")
    # print(f"  Average within-cluster Tanimoto similarity : {real_tan_mean:.4f}")
    # print(f"  Average within-cluster SMILES edit distance: {real_ed_mean:.2f}")


    # ============================================================
    # Random baseline (same random assignment as cut experiment)
    # ============================================================
    df_rand = df.copy()

    scaffold_to_indices = df.groupby('scaffold').indices
    df_rand['cluster'] = -1

    for indices in scaffold_to_indices.values():
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            df_rand.at[idx, 'cluster'] = i % N_CLUSTERS

    # ============================================================
    # Random assignment scaffold statistics
    # ============================================================

    # Compute scaffold counts per cluster in random assignment
    scaffold_counts_rand = df_rand.groupby('cluster')['scaffold'].nunique()

    # Average number of unique scaffolds per cluster
    avg_scaffolds_rand = scaffold_counts_rand.mean()

    print("\nRandom Cluster Scaffold Summary:")
    print(f"  Average number of unique scaffolds per cluster: {avg_scaffolds_rand:.2f}")

    # # Print unique scaffolds per cluster
    # scaffolds_per_rand_cluster = (
    #     df_rand.groupby('cluster')['scaffold']
    #     .apply(lambda x: sorted(set(x.dropna())))
    # )

    # for cluster_id, scaffolds in scaffolds_per_rand_cluster.items():
    #     print(f"\nRandom Cluster {cluster_id}: {len(scaffolds)} unique scaffolds")
    #     for s in scaffolds:
    #         print(f"    {s}")



    # # tanimoto_rand = []
    # # editdist_rand = []

    # # for cluster_id, group in df_rand.groupby("cluster"):
    # #     smiles = group["smile"].dropna().tolist()
    # #     tanimoto_rand.append(average_tanimoto(smiles))
    # #     editdist_rand.append(average_edit_distance(smiles))

    # # rand_tan_mean = np.nanmean(tanimoto_rand)
    # # rand_ed_mean = np.nanmean(editdist_rand)

    # # print("\nRandom cluster baseline:")
    # # print(f"  Average within-cluster Tanimoto similarity : {rand_tan_mean:.4f}")
    # # print(f"  Average within-cluster SMILES edit distance: {rand_ed_mean:.2f}")


    # from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    # def average_bleu(smiles_list, n_gram=4):
    #     """
    #     Compute average BLEU score across all pairs in smiles_list.
    #     Uses cumulative BLEU-n (default n=4) with smoothing.
    #     """
    #     smiles_list = [s for s in smiles_list if isinstance(s, str)]
    #     if len(smiles_list) < 2:
    #         return np.nan

    #     smoothie = SmoothingFunction().method1
    #     scores = []
    #     for i in range(len(smiles_list)):
    #         ref_tokens = list(smiles_list[i])
    #         for j in range(i + 1, len(smiles_list)):
    #             cand_tokens = list(smiles_list[j])
    #             score = sentence_bleu(
    #                 [ref_tokens],  # reference
    #                 cand_tokens,   # candidate
    #                 weights=[1/n_gram]*n_gram,  # cumulative BLEU-n
    #                 smoothing_function=smoothie
    #             )
    #             scores.append(score)
    #     return np.mean(scores)
    
    # # ============================================================
    # # Real spectral co-clustering metrics
    # # ============================================================
    # tanimoto_real = []
    # editdist_real = []
    # bleu_real = []

    # for cluster_id, group in df_eval.groupby("cluster"):
    #     smiles = group["smile"].dropna().tolist()
    #     # tanimoto_real.append(average_tanimoto(smiles))
    #     # editdist_real.append(average_edit_distance(smiles))
    #     bleu_real.append(average_bleu(smiles))

    # # real_tan_mean = np.nanmean(tanimoto_real)
    # # real_ed_mean = np.nanmean(editdist_real)
    # real_bleu_mean = np.nanmean(bleu_real)

    # print("\nSpectral co-clustering:")
    # # print(f"  Average within-cluster Tanimoto similarity : {real_tan_mean:.4f}")
    # # print(f"  Average within-cluster SMILES edit distance: {real_ed_mean:.2f}")
    # print(f"  Average within-cluster BLEU similarity    : {real_bleu_mean:.4f}")

    # # ============================================================
    # # Random baseline (same random assignment as cut experiment)
    # # ============================================================
    # tanimoto_rand = []
    # editdist_rand = []
    # bleu_rand = []

    # for cluster_id, group in df_rand.groupby("cluster"):
    #     smiles = group["smile"].dropna().tolist()
    #     # tanimoto_rand.append(average_tanimoto(smiles))
    #     # editdist_rand.append(average_edit_distance(smiles))
    #     bleu_rand.append(average_bleu(smiles))

    # # rand_tan_mean = np.nanmean(tanimoto_rand)
    # # rand_ed_mean = np.nanmean(editdist_rand)
    # rand_bleu_mean = np.nanmean(bleu_rand)

    # print("\nRandom cluster baseline:")
    # # print(f"  Average within-cluster Tanimoto similarity : {rand_tan_mean:.4f}")
    # # print(f"  Average within-cluster SMILES edit distance: {rand_ed_mean:.2f}")
    # print(f"  Average within-cluster BLEU similarity    : {rand_bleu_mean:.4f}")




    # Projectors
    projectors = [None if b is None else (b @ b.T) for b in bases]

    # All unique pairs i < k
    eps_records = []
    eps_vals = []
    n_bases = len(bases)
    for i in range(n_bases):
        for k in range(i + 1, n_bases):
            if projectors[i] is None or projectors[k] is None:
                continue
            Pi = projectors[i]
            Pk = projectors[k]
            val = np.linalg.norm(Pk @ Pi, 2)
            eps_records.append((i, k, float(val)))
            eps_vals.append(val)
            print(val)

    eps_vals = np.array(eps_vals) if len(eps_vals) else np.array([])
    eps_csv = os.path.join(OUTPUT_DIR, "pairwise_epsilons.csv")
    pd.DataFrame(eps_records, columns=["cluster_i", "cluster_k", "epsilon"]).to_csv(eps_csv, index=False)
    print(f"✓ Saved pairwise epsilons to {eps_csv}")

    if eps_vals.size > 0:
        print("\nSubspace incoherence (epsilon):")
        print(f"  Max:    {eps_vals.max():.6f}")
        print(f"  Mean:   {eps_vals.mean():.6f}")
        print(f"  Median: {np.median(eps_vals):.6f}")
        print(f"  Std:    {eps_vals.std():.6f}")

        plot_histogram(eps_vals, xlabel=r"$\\|P_k P_i\\|_{\\mathrm{op}}$",
                       title="Subspace Overlap ε", path=os.path.join(OUTPUT_DIR, "epsilons_hist.png"))
        



if __name__ == '__main__':
    main()
    
    