"""
Unified cross-validation script for steering vector evaluation.
Supports SA (Synthetic Accessibility), QED (Quantitative Estimate of Druglikeness), and LogP metrics.
"""
import os
import argparse
from collections import Counter
import time
import csv

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralCoclustering


# Load thresholds from the summary file
thresholds_file = 'thresholds_summary.csv'
thresholds = {}
with open(thresholds_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_name = row['file']
        metric = row['metric']
        if file_name not in thresholds:
            thresholds[file_name] = {}
        if metric not in thresholds[file_name]:
            thresholds[file_name][metric] = {}
        thresholds[file_name][metric]['q_low'] = float(row['bottom_q'])
        thresholds[file_name][metric]['q_high'] = float(row['top_q'])

# Update METRIC_CONFIG dynamically based on input file
def get_metric_config(input_file, metric):
    file_name = os.path.basename(input_file)
    if file_name in thresholds and metric in thresholds[file_name]:
        print("Using thresholds from summary file for", file_name, metric)
        return {
            'q_low': thresholds[file_name][metric]['q_low'],
            'q_high': thresholds[file_name][metric]['q_high'],
            'coefficient': 1.0,
            'col_name': metric,
            'steer_direction': 'decrease' if metric == 'sa' else 'increase',
        }
    else:
        print(f"Warning: Thresholds for file '{file_name}' and metric '{metric}' not found. Using default.")
        

# Model-specific input file paths
MODEL_PATHS = {
    'chEMBL': 'steering_vectors_all_chembl.parquet',
    'ZINC': 'steering_vectors_all_zinc.parquet',
    'qm9': 'steering_vectors_all_qm9.parquet',
    'moses': 'steering_vectors_all_moses.parquet',
    }

MIN_CLUSTER_POINTS = 1

def get_metric_column(df, metric, input_file):
    """Automatically detect the metric column in the dataframe."""
    metric_col = get_metric_config(input_file, metric)['col_name']
    # Try variations of the column name
    for variant in [metric_col, metric_col.upper(), metric_col.capitalize()]:
        if variant in df.columns:
            return variant
    raise ValueError(f"Could not find {metric} column in dataframe. Available columns: {df.columns.tolist()}")


def load_data(file_path, metric):
    """Load parquet file with steering vectors, metric values, and SMILES.
    
    Returns: vectors (n, d), metric_values (n,), smiles (n,), dataframe
    """
    df = pd.read_parquet(file_path)
    
    # Get metric column
    metric_col = get_metric_column(df, metric, file_path)
    
    # Get SMILES column
    smiles_col = 'SMILES' if 'SMILES' in df.columns else ('smile' if 'smile' in df.columns else None)
    if smiles_col is None:
        raise ValueError("Could not find SMILES column in dataframe")
    
    vectors = np.vstack(df['steering_vector'].values)
    metric_values = df[metric_col].values.astype(float)
    smiles = df[smiles_col].values
    
    return vectors, metric_values, smiles, df


def split_train_test_clusters(selected_clusters, train_frac=0.7, random_state=42):
    """Split clusters into train and test sets."""
    rng = np.random.RandomState(random_state)
    n = len(selected_clusters)
    n_train = int(np.floor(train_frac * n))
    perm = rng.permutation(selected_clusters)
    train = list(perm[:n_train])
    test = list(perm[n_train:])
    return train, test


def compute_cluster_steer(vectors, metric_values, q_low, q_high, min_points=MIN_CLUSTER_POINTS):
    """Compute per-cluster steering vector using centroids of low and high metric regions.

    Returns the raw normalized direction (high_centroid - low_centroid) / ||.||
    without applying any metric-specific sign or coefficient. Sign and scaling are
    applied at apply_steer time to keep responsibilities separated.
    """
    mask_low = metric_values <= q_low
    mask_high = metric_values >= q_high

    if np.sum(mask_low) < min_points or np.sum(mask_high) < min_points:
        return None

    low_centroid = np.mean(vectors[mask_low], axis=0)
    high_centroid = np.mean(vectors[mask_high], axis=0)

    steer = high_centroid - low_centroid
    features = np.where(steer != 0)[0]
    print("steer", features)

    norm = np.linalg.norm(steer)

    if norm == 0:
        return None
    return steer / norm

def select_clusters(cluster_labels, n_select, random_state=42):
    """Select the top n_select largest clusters by size."""
    sizes = Counter(cluster_labels)
    eligible = [(c, s) for c, s in sizes.items()]
    eligible.sort(key=lambda x: x[1], reverse=True)
    chosen = [c for c, s in eligible[:n_select]]
    print(f"Selected clusters: {chosen}")
    return chosen

def aggregate_steers_linear(steer_list):
    """
    Aggregate local steering vectors by summing them (linear combination) and normalizing.
    Assumes local steers occupy disjoint dimensions.
    """
    if not steer_list:
        return None
   
    agg = np.sum(np.vstack(steer_list), axis=0)
    norm = np.linalg.norm(agg)

    return agg / norm if norm > 0 else None

def cosine_similarity(u, v):
    # """Compute cosine similarity between two vectors."""
    # print("U:", u)
    # print("V:", v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)

# import numpy as np

def compare_sparse_dense_steering(
    X_sparse,
    X_dense,
    metric_values, 
    q_low, 
    q_high,
    eps=1e-8,
):

    v_sparse = compute_cluster_steer(X_sparse, metric_values=metric_values, q_low=q_low, q_high=q_high)
    
    features = np.where(X_sparse[0] != 0)[0]
    if v_sparse is None:
        return {"cosine_similarity": None}

    U, S, Vt = np.linalg.svd(X_dense, full_matrices=False)

    P = Vt.T @ Vt # projector onto sparse subspace

    X_proj = P @ X_sparse.T # project dense steer onto sparse subspace

    v_proj = compute_cluster_steer(X_proj.T, metric_values=metric_values, q_low=q_low, q_high=q_high)

    cos = cosine_similarity(v_sparse[features], v_proj[features])
    
    return {"cosine_similarity": cos}


def main():
    script_start = time.time()
    
    parser = argparse.ArgumentParser(
        description="Unified cross-validation for steering vector evaluation"
    )
    parser.add_argument(
        '--metric',
        choices=['sa', 'qed', 'logp'],
        required=True,
        help='Metric to optimize for'
    )
    parser.add_argument(
        '--model',
        choices=list(MODEL_PATHS.keys()),
        help='Language model to use (determines input file path). Use --input to override.'
    )
    parser.add_argument(
        '--input',
        help='Input parquet file with steering vectors (overrides model default)'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=300,
        help='Number of clusters'
    )
   
    parser.add_argument(
        '--outdir',
        default='./benchmarking',
        help='Output directory for results'
    )
    parser.add_argument(
        '--select_clusters',
        type=int,
        default=300,
        help='Number of clusters to select for analysis (default: same as n_clusters)'
    )

    parser.add_argument(
        '--random_cluster',
        action='store_true',
        help='Use random cluster assignments instead of clustering'
    )
    
    args = parser.parse_args()

    # Determine input file based on model argument
    if args.input:
        input_file = args.input
        model_name = 'custom'  # Use 'custom' for user-specified input
    elif args.model:
        input_file = MODEL_PATHS[args.model]
        model_name = args.model
        print(f"Using model '{args.model}' with default path: {input_file}")
    else:
        raise ValueError("Must specify either --model or --input")
    
    # Set default for select_clusters if not provided
    if args.select_clusters is None:
        args.select_clusters = args.n_clusters

    steeringvecsfolder = os.path.join(args.outdir, f'steering_vectors_{model_name}')

    # Create output directories
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, f'localsteers_{model_name}'), exist_ok=True)
    os.makedirs(steeringvecsfolder, exist_ok=True)
    
    metric = args.metric
    print(f"Loading data from {input_file}...")
    vectors, metric_values, smiles, df = load_data(input_file, metric)
    print(f"Loaded {len(vectors)} samples")

    config = get_metric_config(input_file, metric)
    q_low = config['q_low']
    q_high = config['q_high']    

    if args.random_cluster:
        print("Using random cluster assignments with random feature sets (overlap allowed)...")
        n_samples, d = vectors.shape
        rng = np.random.RandomState(42)
        cluster_labels = rng.randint(0, args.n_clusters, size=n_samples)
        feature_labels = {}
        for k in range(args.n_clusters):
            idx = np.where(cluster_labels == k)[0]
            if len(idx) == 0:
                continue
            Xk = vectors[idx]
            Xk = Xk - Xk.mean(axis=0, keepdims=True)
            # feature importance = variance per dimension
            feature_scores = np.var(Xk, axis=0)
            # top-k features
            k_feat = getattr(args, "k_features", 20)
            top_idx = np.argsort(feature_scores)[-k_feat:]
            mask = np.zeros(Xk.shape[1], dtype=bool)
            mask[top_idx] = True
            feature_labels[k] = mask

    elif os.path.exists(os.path.join(args.outdir, f'cluster_labels_{model_name}.npy')) and os.path.exists(os.path.join(args.outdir, f'feature_labels_{model_name}.npy')):
        print(f"Loading existing cluster labels for model '{model_name}'...")
        cluster_labels = np.load(os.path.join(args.outdir, f'cluster_labels_{model_name}.npy'))
        feature_labels = np.load(os.path.join(args.outdir, f'feature_labels_{model_name}.npy'), allow_pickle=True)
        print("Cluster labels loaded.")
    else:
        print(f"Clustering SMILES...")
        print("Running Spectral Co-Clustering...")
        coclustering = SpectralCoclustering(n_clusters=args.n_clusters, random_state=1)
        # Preprocessing: Make matrix non-negative and normalize
        arr = np.abs(vectors)
        row_norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        arr = arr / row_norms

        coclustering.fit(arr)

        cluster_labels = coclustering.row_labels_
        feature_labels = coclustering.column_labels_

        print("Co-clustering complete")

        # Save cluster labels to npy
        np.save(os.path.join(args.outdir, f'cluster_labels_{model_name}.npy'), cluster_labels)
        np.save(os.path.join(args.outdir, f'feature_labels_{model_name}.npy'), feature_labels)
        print(f"✓ Saved cluster labels to {os.path.join(args.outdir, f'cluster_labels_{model_name}.npy')}")

    cluster_features = [[] for _ in range(args.n_clusters)]

    if isinstance(feature_labels, dict):
        # random cluster case
        for k, mask in feature_labels.items():
            cluster_features[k] = np.where(mask)[0].tolist()
    else:
        # coclustering case
        for f, k in enumerate(feature_labels):
            cluster_features[k].append(f)

    print("\nSparsifying vectors by cluster assigned dimensions...")

    X_sparse = np.zeros_like(vectors)
    for k in range(args.n_clusters):
        cluster_mask = cluster_labels == k
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        
        assigned_dims = cluster_features[k] if k < len(cluster_features) else []  # Handle missing keys by checking index bounds
        if not assigned_dims:  # Check if assigned_dims is empty
            continue
        assigned_dims = np.array(assigned_dims, dtype=int)
        X_sparse[np.ix_(cluster_indices, assigned_dims)] = vectors[np.ix_(cluster_indices, assigned_dims)]
        if (k + 1) % 50 == 0:
            print(f"Processed {k + 1}/{args.n_clusters} clusters")
        
    origvectors = vectors #original vectors
    vectors = X_sparse
   
    
    # make sure all the vectors have non-zero dimensions in different dimesions

    # Select and split clusters: Right now all clusters are in train set and test set is empty. 
    selected = select_clusters(cluster_labels, n_select=args.select_clusters)
    train_clusters, test_clusters = split_train_test_clusters(selected, train_frac=1)
    print(f"Selected {len(selected)} clusters -> train: {len(train_clusters)}, test: {len(test_clusters)}")
    print(f"Split thresholds: low<= {q_low:.3f}, high>= {q_high:.3f}")

    # Compute per-cluster steering vectors (training clusters)
    steer_list = []
    orig_steer_list = []
    weights = []
    cluster_vecs = []
    train_idxs = []
    sims = []
    off_diag_fractions = []

    for cid in train_clusters:
        mask_c = cluster_labels == cid
        if not np.any(mask_c):
            continue

        metric_c = metric_values[mask_c]
        vecs_c = vectors[mask_c]
        origvecs_c = origvectors[mask_c]


        print("numNonZero dimensions in cluster", cid, ":", np.count_nonzero(vecs_c[0]))
        stats = compare_sparse_dense_steering(vecs_c, origvecs_c, metric_c, q_low, q_high, subspace_rank=np.count_nonzero(vecs_c[0]))
        print("cosine similarity between sparse local steer and projected dense steer for cluster", cid, ":", stats["cosine_similarity"])
        sims.append(stats["cosine_similarity"])
        steer = compute_cluster_steer(vecs_c, metric_c, q_low, q_high)
        if steer is not None:
            steer_list.append(steer)
            weights.append(len(metric_c))
            cluster_vecs.append(vecs_c)
            train_idxs.extend(np.nonzero(mask_c)[0].tolist())
            np.save(os.path.join(args.outdir, f'localsteers_{model_name}/local_steer_{cid}.npy'), steer)
        
    print("AVERAGE COSINE SIMILARITY BETWEEN SPARSE AND DENSE LOCAL STEER:", np.mean([ims for ims in sims if ims is not None]))

    print(f"Computed steering vectors for {len(steer_list)} training clusters")

    global_steer = aggregate_steers_linear(steer_list)


    if global_steer is None:
        raise RuntimeError("Could not compute a global steering vector (no valid cluster steers)")

    # Save global steering vector
    steer_vec_path = os.path.join(steeringvecsfolder, f'global_steer_{model_name}_{metric}.npy')
    np.save(steer_vec_path, global_steer)
    print(f"Saved global steering vector to {steer_vec_path}")

    # Save cosine similarity statistics to file
    stats_path = os.path.join(args.outdir, f'cosine_similarity_stats_{model_name}_{metric}.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Cross-Validation Results: {model_name.upper()} Model - {metric.upper()} Metric\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Model:            {model_name}\n")
        f.write(f"  Metric:           {metric}\n")
        f.write(f"  Input File:       {input_file}\n")
        f.write(f"  N Clusters:       {args.n_clusters}\n")
        f.write(f"  Selected Clusters:{args.select_clusters}\n")
        f.write(f"  Train Clusters:   {len(train_clusters)}\n")
        f.write(f"  Test Clusters:    {len(test_clusters)}\n")
        f.write(f"  Q Low:            {q_low}\n")
        f.write(f"  Q High:           {q_high}\n")
        f.write(f" sparse/dense sim: {np.mean([i for i in sims if i is not None])}\n")
    
    print(f"Saved cosine similarity statistics to {stats_path}")

    total_time = time.time() - script_start
    print(f"\n✓ Cross-validation complete for {metric.upper()} ({model_name})")
    print(f"  Results saved to: {args.outdir}")
    print(f"  Plots saved to: {args.plot_dir}")
    print(f"\n[TOTAL TIME]: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")


if __name__ == "__main__":
    main()
