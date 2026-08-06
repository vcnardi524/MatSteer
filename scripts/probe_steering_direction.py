#!/usr/bin/env python3
"""
Is band gap LINEARLY decodable from the layer-N embedding, and does our
metal->insulator steering direction point along that axis?

Unlike unsupervised PCA/t-SNE (dominated by composition variance), this is the
question that determines whether steering can work. We:

  1. Split metals (gap<=low) vs insulators (gap>=high) into train/test.
  2. Build the steering direction on TRAIN (mean insulator - mean metal).
  3. On held-out TEST, project onto that direction -> AUC (how well OUR vector
     separates classes on unseen data).
  4. Fit a logistic-regression probe on TRAIN -> TEST AUC (best linear separability).
  5. Histogram of the test projection by class.

Output: analysis/steering_probe_layer{N}.png  + printed AUCs.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BG_COL = "dos_electronic.band_gap"
SEED = 42


from utils import load_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--dataset", default="v1_all",
                    help="Embeddings subdir under embeddings/ (v1_all or v1_mp)")
    ap.add_argument("--low", type=float, default=0.05)
    ap.add_argument("--high", type=float, default=1.0)
    args = ap.parse_args()

    meta = pd.read_parquet("metadata.parquet", columns=["id", BG_COL]).dropna(subset=[BG_COL])
    meta["gap"] = meta[BG_COL].astype(float)
    emb = load_embeddings(args.layer, dataset=args.dataset)
    df = emb.merge(meta[["id", "gap"]], on="id", how="inner")

    metal = df[df["gap"] <= args.low]
    ins = df[df["gap"] >= args.high]
    print(f"metals={len(metal):,}  insulators={len(ins):,}")

    # balance: downsample metals to match insulators so AUC isn't trivial
    metal = metal.sample(n=min(len(metal), 5 * len(ins)), random_state=SEED)
    X = np.vstack(pd.concat([metal, ins])["embedding"].values).astype(np.float32)
    yb = np.r_[np.zeros(len(metal)), np.ones(len(ins))]   # 1 = insulator

    Xtr, Xte, ytr, yte = train_test_split(X, yb, test_size=0.3,
                                          random_state=SEED, stratify=yb)

    # (3) OUR steering direction, built on TRAIN only
    steer = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    steer /= np.linalg.norm(steer)
    proj_te = Xte @ steer
    auc_steer = roc_auc_score(yte, proj_te)

    # (4) best linear probe (logistic regression on standardized features)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    auc_probe = roc_auc_score(yte, clf.decision_function(sc.transform(Xte)))

    print(f"\nHeld-out TEST AUC (metal vs insulator):")
    print(f"  steering-vector projection : {auc_steer:.3f}")
    print(f"  logistic-regression probe  : {auc_probe:.3f}")
    print("  (0.5 = no signal, 1.0 = perfect separation)")

    # (5) histogram of test projection by class
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(proj_te[yte == 0], bins=60, alpha=0.6, label=f"metals (n={int((yte==0).sum())})",
            color="navy", density=True)
    ax.hist(proj_te[yte == 1], bins=60, alpha=0.6, label=f"insulators (n={int((yte==1).sum())})",
            color="orange", density=True)
    ax.set_xlabel("projection onto steering vector"); ax.set_ylabel("density")
    ax.set_title(f"Layer {args.layer}: held-out separation along steering direction\n"
                 f"steering AUC={auc_steer:.3f}  |  best linear probe AUC={auc_probe:.3f}")
    ax.legend()
    out = Path("analysis"); out.mkdir(exist_ok=True)
    png = out / f"steering_probe_layer{args.layer}.png"
    plt.tight_layout(); plt.savefig(png, dpi=140)
    print(f"\nSaved {png}")


if __name__ == "__main__":
    main()
