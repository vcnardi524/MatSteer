#!/usr/bin/env python3
"""
Diagnose steering strength at a given layer.

Cross-references three norms that determine whether activation steering can
actually affect generation:

  1. Per-token residual-stream norm   (live model forward; what the hook adds to)
  2. Mean-pooled embedding norm        (extracted embeddings/<dataset>/cif_layer{N}.parquet)
  3. Raw mean-difference vector norm   (per band-gap percentile, pre-normalization)

Also quantifies the zero-gap (metal) pile-up that distorts percentile selection.

Writes a tidy CSV summarizing all findings.

Usage (analysis only, any env with pandas/numpy):
    python scripts/analyze_steering_norms.py --layer 14 --residual-norm 164

Usage (also re-measure per-token residual norm; needs crystallm_venv + model):
    python scripts/analyze_steering_norms.py --layer 14 --with-model \
        --model CrystaLLM/crystallm_v1_large --pkl CrystaLLM/cifs_v1_test.pkl.gz

Output: analysis/steering_norm_analysis_layer{N}.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

J_TO_EV = 6.2415e18
PERCENTILES = [5, 10, 15, 20, 25]
ZERO_EPS = [0.001, 0.01, 0.1, 0.5]  # eV windows around 0 to call "metal"


from utils import load_embeddings


def measure_residual_norm(model_dir: str, pkl: str, layer: int, n_cifs: int) -> float:
    """Per-token L2 norm of layer-`layer` hidden states, averaged over n_cifs CIFs."""
    import gzip, pickle
    import torch
    sys.path.insert(0, "CrystaLLM")
    sys.path.insert(0, "scripts")
    from crystallm import CIFTokenizer
    from extract_cif_embeddings import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(model_dir, dev)
    tok = CIFTokenizer()

    with gzip.open(pkl, "rb") as f:
        data = pickle.load(f)

    captured = {}
    model.transformer.h[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", o))

    per_cif = []
    for id_, cif in data[:n_cifs]:
        ids = tok.encode(tok.tokenize_cif(cif))[:config.block_size]
        x = torch.tensor(ids, device=dev).unsqueeze(0)
        with torch.no_grad():
            model(x)
        h = captured["h"][0]                 # (T, D)
        per_cif.append(h.norm(dim=-1).mean().item())
    return float(np.mean(per_cif))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--dataset", default="v1_all",
                    help="Embeddings subdir under embeddings/ (v1_all or v1_mp)")
    ap.add_argument("--residual-norm", type=float, default=None,
                    help="Per-token residual norm (skip --with-model and supply directly)")
    ap.add_argument("--with-model", action="store_true",
                    help="Re-measure per-token residual norm (needs crystallm_venv + model)")
    ap.add_argument("--model", default="CrystaLLM/crystallm_v1_large")
    ap.add_argument("--pkl", default="CrystaLLM/cifs_v1_test.pkl.gz")
    ap.add_argument("--n-cifs", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else \
        Path(f"analysis/steering_norm_analysis_layer{args.layer}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. per-token residual norm
    residual_norm = args.residual_norm
    if args.with_model:
        print("Measuring per-token residual norm via model forward ...")
        residual_norm = measure_residual_norm(args.model, args.pkl, args.layer, args.n_cifs)
        print(f"  residual per-token norm = {residual_norm:.2f}")

    # 2. pooled embedding norms
    print(f"Loading layer-{args.layer} embeddings ...")
    emb = load_embeddings(args.layer, dataset=args.dataset)
    X = np.vstack(emb["embedding"].values).astype(np.float32)
    pooled = np.linalg.norm(X, axis=1)
    print(f"  {len(emb):,} embeddings, dim={X.shape[1]}, "
          f"pooled norm mean={pooled.mean():.2f} median={np.median(pooled):.2f}")

    # 3. band gaps + zero-gap pile-up
    print("Loading metadata band gaps ...")
    meta = pd.read_parquet("metadata.parquet",
                           columns=["id", "energy_lowest_unoccupied", "energy_highest_occupied"])
    meta = meta.dropna(subset=["energy_lowest_unoccupied", "energy_highest_occupied"]).copy()
    meta["gap"] = (meta["energy_lowest_unoccupied"] - meta["energy_highest_occupied"]) * J_TO_EV

    df = emb.merge(meta[["id", "gap"]], on="id", how="inner")
    Xj = np.vstack(df["embedding"].values).astype(np.float32)
    gaps = df["gap"].values
    n = len(gaps)
    print(f"  joined embeddings+gap: {n:,}")

    pileup = {eps: float(np.mean(np.abs(gaps) <= eps)) for eps in ZERO_EPS}
    for eps in ZERO_EPS:
        print(f"  |gap| <= {eps} eV: {pileup[eps]*100:.1f}%")

    # 4. per-percentile raw mean-diff vector
    denom = residual_norm if residual_norm else pooled.mean()
    rows = []
    for p in PERCENTILES:
        lo = float(np.percentile(gaps, p))
        hi = float(np.percentile(gaps, 100 - p))
        m_lo, m_hi = gaps <= lo, gaps >= hi
        raw = Xj[m_hi].mean(0) - Xj[m_lo].mean(0)
        raw_norm = float(np.linalg.norm(raw))
        rows.append({
            "layer": args.layer,
            "percentile": p,
            "thresh_low_ev": lo,
            "thresh_high_ev": hi,
            "n_low": int(m_lo.sum()),
            "n_high": int(m_hi.sum()),
            "raw_meandiff_norm": raw_norm,
            "residual_per_token_norm": residual_norm,
            "pooled_emb_norm_mean": float(pooled.mean()),
            "raw_pct_of_residual": 100.0 * raw_norm / denom,
            "alpha_for_10pct_residual": (0.10 * denom),  # since steering vec is unit-norm
            "alpha_for_25pct_residual": (0.25 * denom),
            "low_in_zero_pileup": bool(abs(lo) <= 0.1),
            "high_in_zero_pileup": bool(abs(hi) <= 0.1),
            **{f"pct_metals_within_{eps}eV": pileup[eps] for eps in ZERO_EPS},
        })
        print(f"  p={p}: raw_norm={raw_norm:.3f} ({100*raw_norm/denom:.2f}% of residual)  "
              f"low<= {lo:.4f} (n={m_lo.sum():,})  high>= {hi:.4f} (n={m_hi.sum():,})")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved findings -> {out_path}")
    if residual_norm:
        print(f"alpha=1.0 is {100.0/denom:.2f}% of residual (unit-norm steering vector).")
        print(f"For 10% effect use alpha~{0.10*denom:.0f}; for 25% use alpha~{0.25*denom:.0f}.")


if __name__ == "__main__":
    main()
