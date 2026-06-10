#!/usr/bin/env python3
import argparse
import numpy as np
import scipy
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--vectors', required=True)
# parser.add_argument('--cluster-file', required=True)
parser.add_argument('--sample-pairs', type=int, default=None)
args = parser.parse_args()

# Load data
df = pd.read_parquet(args.vectors)
vectors = np.vstack(df['steering_vector_sparse'].values)
labels = np.array(df['cluster'].values)
print(f"Loaded vectors (already sparse)")
# labels = np.load(args.cluster_file)

print(f"Loaded {len(vectors)} vectors ({vectors.shape[1]}D), "
      f"{len(np.unique(labels))} clusters")

# Remap cluster labels
unique_labels = np.unique(labels)
cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_labels)}
remapped_labels = np.array([cluster_mapping[l] for l in labels])

# Compute orthonormal bases for each subspace
bases = []
for cid in range(len(unique_labels)):
    mask = remapped_labels == cid
    cluster_vecs = vectors[mask].T  # shape: D x n_k = 4096 x n_samples
    n_dims_total = cluster_vecs.shape[0]
    n_samples = cluster_vecs.shape[1]
    
    # Count nonzero dimensions (dimensions that have at least one nonzero value)
    nonzero_dims_count = np.count_nonzero(np.any(cluster_vecs != 0, axis=1))
    
    print(f"Computing basis for cluster {cid}: "
          f"samples={n_samples}, total_dims={n_dims_total}, nonzero_dims={nonzero_dims_count}")

    # # --- CENTERING ---
    # cluster_mean = cluster_vecs.mean(axis=1, keepdims=True)
    # cluster_vecs_centered = cluster_vecs - cluster_mean

    # --- SVD ---
    U, S, _ = np.linalg.svd(cluster_vecs, full_matrices=False)
    print(S)
    # Take only significant singular vectors
    r = np.sum(S > 1e-10)
    U_r = U[:, :r]
    print(f"  Cluster {cid}: rank={r}, basis_shape={U_r.shape}")
    bases.append(U_r)

print(f"Computed {len(bases)} subspace bases")

# Compute projector matrices
projectors = [Uk @ Uk.T for Uk in bases]
print(f"Computed {len(projectors)} projector matrices")
print(f"Each projector has shape {projectors[0].shape}")

# Generate (i, k) pairs with i != k
all_pairs = [(i, k) for k in range(len(bases))
                      for i in range(len(bases)) if i != k and i < k]

if args.sample_pairs and args.sample_pairs < len(all_pairs):
    np.random.seed(2)
    idx = np.random.choice(len(all_pairs), args.sample_pairs, replace=False)
    pairs = [all_pairs[j] for j in idx]
    print(f"Computing epsilon on {len(pairs)} sampled pairs "
          f"(out of {len(all_pairs)})")
else:
    pairs = all_pairs
    print(f"Computing epsilon on all {len(pairs)} pairs")

# Compute epsilon = max ||P_k P_i||_op
epsilon_vals = []
for i, k in pairs:
    Pk = projectors[k]
    Pi = projectors[i]
    eigenval = np.linalg.norm(Pk @ Pi, 2)  # largest singular value
    epsilon_vals.append(eigenval)
    print(eigenval)

epsilon_vals = np.array(epsilon_vals)

print("\nSubspace incoherence (epsilon):")
print(f"  Max:    {epsilon_vals.max():.6f}")
print(f"  Mean:   {epsilon_vals.mean():.6f}")
print(f"  Median: {np.median(epsilon_vals):.6f}")
print(f"  Std:    {epsilon_vals.std():.6f}")
