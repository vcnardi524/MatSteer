#!/usr/bin/env python3
"""
Does steering push the residual stream off the data manifold, and is that why the
crystals stop being valid?

Validity collapses with steering strength the same way for every property and target
(96% -> 95% -> 82% -> 13% -> 0% across t), which looks like damage caused by the SIZE of
the displacement rather than its direction. The hypothesis this tests: the steered
activation leaves the region real activations occupy, the model is then running on an
input unlike anything it was trained on, and the CIF grammar falls apart as a result.

The reference manifold is the class bank -- 100,000 real training activations already
projected into the PCA subspace (`compute_centroid_target.py --save-bank`). For a
teacher-forced prompt this reports, in subspace coordinates:

  d_nn      distance to the nearest real activation in the bank
  d_knn     mean distance to the 32 nearest, which is steadier than a single neighbour
  mahal     Mahalanobis distance under the bank's own covariance -- how many standard
            deviations out the point sits, in the bank's shape

Read the clean row as the scale: it is what a real prompt's activation looks like by
these measures. If the steered rows climb far past it exactly where validity collapses,
the hypothesis has support and a manifold-constrained step is worth building. If they
climb smoothly while validity falls off a cliff, the two are not the same phenomenon.

Usage:
    python scripts/analysis/manifold_distance.py --target 30 --ts 0 0.25 0.5 0.75 1 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM", "bin"))
from crystallm import CIFTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "embeddings"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "steering"))
from make_prompts import PATTERN_COMP_SG, extract_prompt
from extract_cif_embeddings import load_model, load_cifs
from compute_centroid_target import load_pca
from utils import analysis_dir

TEST_PKL = "CrystaLLM/cifs_v1_test_sample1000.pkl.gz"
# validity of the matching generation runs, so the two curves can be read together
KNOWN_VALID = {0.0: 0.963, 0.25: 0.961, 0.5: 0.950, 0.75: 0.825, 1.0: 0.125, 2.0: 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CrystaLLM/crystallm_v1_large")
    ap.add_argument("--property", default="density_atomic")
    ap.add_argument("--target", type=float, default=30.0)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--ts", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    ap.add_argument("--knn", type=int, default=32)
    ap.add_argument("--n-prompts", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.model, device)
    tokenizer = CIFTokenizer()

    mean, comps = load_pca(args.layer, args.k)
    stem = f"layer{args.layer}_k{args.k}_target{args.target:g}"
    bank_path = (f"steering_vectors/pca_centroid/{args.property}/{stem}_bank.parquet")
    Z = np.vstack(pd.read_parquet(bank_path)["coord"].to_numpy()).astype(np.float32)
    print(f"reference manifold: {Z.shape[0]:,} real training activations x {Z.shape[1]} dims")

    mu = torch.tensor(mean, device=device)
    W = torch.tensor(comps, device=device)
    bank = torch.tensor(Z, device=device)
    centroid = bank.mean(0)
    # Mahalanobis under the bank's own covariance: how far out the point sits in the
    # shape the real activations actually have, not in raw euclidean distance.
    cov = torch.tensor(np.cov(Z.T) + 1e-4 * np.eye(Z.shape[1]), dtype=torch.float32,
                       device=device)
    prec = torch.linalg.inv(cov)

    data = load_cifs(TEST_PKL)
    prompts = []
    for _, cif in data:
        try:
            prompts.append(extract_prompt(cif, PATTERN_COMP_SG))
        except Exception:
            pass
        if len(prompts) >= args.n_prompts:
            break
    print(f"{len(prompts)} prompts, layer {args.layer}, target {args.target:g}\n")

    rows = []
    for t in args.ts:
        acc = []
        for p in prompts:
            ids = torch.tensor(tokenizer.encode(tokenizer.tokenize_cif(p)),
                               device=device).unsqueeze(0)
            got = {}

            def grab(module, inp, out):
                got["h"] = (out[0] if isinstance(out, tuple) else out).detach().float()

            hd = model.transformer.h[args.layer].register_forward_hook(grab)
            with torch.no_grad():
                model(ids)
            hd.remove()

            z = (got["h"][0] - mu) @ W.T                 # (T, k) clean coordinates
            z_new = z + t * (centroid - z)               # the steered coordinates
            d = torch.cdist(z_new, bank)                 # (T, bank)
            nn = d.min(dim=1).values
            knn = d.topk(args.knn, dim=1, largest=False).values.mean(dim=1)
            delta = z_new - centroid
            mahal = ((delta @ prec) * delta).sum(-1).clamp_min(0).sqrt()
            acc.append(dict(d_nn=float(nn.mean()), d_knn=float(knn.mean()),
                            mahal=float(mahal.mean())))
        m = pd.DataFrame(acc).mean()
        rows.append(dict(t=t, d_nn=m.d_nn, d_knn=m.d_knn, mahal=m.mahal,
                         valid_pct=KNOWN_VALID.get(t, np.nan)))
        r = rows[-1]
        print(f"  t={t:<5} d_nn {r['d_nn']:7.2f}   d_knn {r['d_knn']:7.2f}   "
              f"mahal {r['mahal']:7.2f}   valid {r['valid_pct']:.1%}"
              if np.isfinite(r["valid_pct"]) else
              f"  t={t:<5} d_nn {r['d_nn']:7.2f}   d_knn {r['d_knn']:7.2f}   "
              f"mahal {r['mahal']:7.2f}", flush=True)

    df = pd.DataFrame(rows)
    base = df.loc[df["t"] == 0.0]
    if not base.empty:
        df["d_knn_vs_clean"] = df["d_knn"] / float(base["d_knn"].iloc[0])
        print("\nd_knn relative to the unsteered activation:")
        for _, r in df.iterrows():
            print(f"  t={r['t']:<5} {r['d_knn_vs_clean']:.3f}x")
    out = (analysis_dir("v1_all", None, "test")
           / f"manifold_distance_{args.property}_layer{args.layer}_target{args.target:g}.csv")
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
