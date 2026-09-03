#!/usr/bin/env python3
"""How far does each steering run actually displace the hidden state?

Effect sizes across methods only mean something next to the size of the intervention
that produced them. "alpha 40" and "scale 6" and "t 0.5" are knobs in different units;
the comparable quantity is |h_new - h| at the layer the hook runs on, as a fraction of
|h| itself.

Three of the four methods are data-dependent -- pca_centroid moves a fraction of the way
to a fixed centroid FROM WHERE THE TOKEN IS, and manifold's step is the curve's own
geometry -- so this cannot be computed from the config. It has to be measured on real
per-token states.

Method: run real test CIFs through the model, capture every token's hidden state at each
layer, then apply each run's ACTUAL hook (imported from steer_generate_cif.py, not
reimplemented) and measure the displacement. Using the hook builders directly is the
point: a reimplementation could drift from what generation does.

pca_local is excluded. Its centroid is recomputed per prompt from a neighbour search, so
it has no single displacement -- it would need the generation loop, not a forward pass.

Usage:
    python scripts/analysis/injection_magnitude.py --n-cifs 25
"""
import argparse
import importlib.util as _ilu
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                       # scripts/
sys.path.insert(0, str(HERE.parent / "steering"))          # its neighbours
sys.path.insert(0, str(HERE.parent.parent / "CrystaLLM"))
sys.path.insert(0, str(HERE.parent / "embeddings"))

from extract_cif_embeddings import load_model, load_cifs   # noqa: E402
from compute_centroid_target import load_pca               # noqa: E402
from manifold import Manifold                              # noqa: E402
from crystallm import CIFTokenizer                         # noqa: E402

# the hooks themselves, so the measured transform is the one generation applies
_spec = _ilu.spec_from_file_location("sgc", HERE.parent / "steering" / "steer_generate_cif.py")
_sgc = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_sgc)

CSV = "analysis/v1_all/test/steering_runs.csv"
OUT = Path("analysis/v1_all/test/plots")
COLOR = {"linear": "#D55E00", "manifold": "#0072B2", "pca_centroid": "#009E73"}
MARKER = {"linear": "o", "manifold": "s", "pca_centroid": "^"}


def capture(model, tokenizer, cifs, layers, device):
    """{layer: (N, 1024) real per-token hidden states} from a forward pass."""
    grabbed = {l: [] for l in layers}
    handles = []
    for l in layers:
        def make(l):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                grabbed[l].append(h.detach().float().squeeze(0).cpu())
            return hook
        handles.append(model.transformer.h[l].register_forward_hook(make(l)))
    with torch.no_grad():
        for _, cif in cifs:
            ids = tokenizer.encode(tokenizer.tokenize_cif(cif))[:1024]
            model(torch.tensor(ids, device=device).unsqueeze(0))
    for h in handles:
        h.remove()
    return {l: torch.cat(v) for l, v in grabbed.items()}


def hook_for(row, device):
    """The real hook for one table row, or None if this method is not measurable here."""
    m, layer = row.method, int(row.layer)
    if m == "linear":
        sv = Path(f"steering_vectors/density_atomic/layer{layer}.parquet")
        vec = np.asarray(pd.read_parquet(sv).iloc[0]["steering_vector"], np.float32)
        return _sgc.linear_hook(vec, float(row.strength), device)
    mean, comps = load_pca(layer, 64)
    if m == "pca_centroid":
        cen = Path("steering_vectors/pca_centroid/density_atomic/"
                   f"layer{layer}_k64_target{row.target:g}.parquet")
        if not cen.exists():
            return None
        c = np.asarray(pd.read_parquet(cen).iloc[0]["centroid_pca"], np.float32)
        return _sgc.pca_centroid_hook(mean, comps, c, float(row.strength), device)
    if m == "manifold":
        man = Path("steering_vectors/manifolds/"
                   f"density_atomic_layer{layer}_k64_w1_max40.parquet")
        if not man.exists():
            return None
        return _sgc.manifold_hook(mean, comps, Manifold.load(str(man)),
                                  float(row.target), device, float(row.strength),
                                  "residual")
    return None


def label(r):
    if r.method == "linear":
        return f"linear a{r.strength:g}"
    if r.method == "manifold":
        return f"manifold d{r.target:g} s{r.strength:g}"
    return f"pca_cent {r.target:g}/t{r.strength:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CrystaLLM/crystallm_v1_large")
    ap.add_argument("--pkl", default="CrystaLLM/cifs_v1_test_sample1000.pkl.gz")
    ap.add_argument("--n-cifs", type=int, default=25)
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()

    runs = pd.read_csv(args.csv)
    runs = runs[(runs.property == "density_atomic") & (runs["agg"] == "mean")
                & (runs.source == "raw") & (runs.strength != 0)
                & (runs.method != "pca_local")].drop_duplicates("run")
    layers = sorted(runs.layer.unique())
    print(f"{len(runs)} runs across layers {layers}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.model, device)
    tokenizer = CIFTokenizer()
    cifs = load_cifs(args.pkl)[:args.n_cifs]
    states = capture(model, tokenizer, cifs, layers, device)
    for l in layers:
        print(f"  layer {l}: {states[l].shape[0]:,} token states, "
              f"median |h| = {states[l].norm(dim=1).median():.2f}")

    rows = []
    for _, r in runs.iterrows():
        hook = hook_for(r, device)
        if hook is None:
            print(f"  ! {r.run}: no artifact, skipped")
            continue
        h = states[int(r.layer)].to(device)
        with torch.no_grad():
            inj = (hook(None, None, h.unsqueeze(0)).squeeze(0) - h).norm(dim=1)
        hn = h.norm(dim=1)
        rows.append(dict(run=r.run, label=label(r), method=r.method, layer=int(r.layer),
                         target=r.target, strength=r.strength,
                         valid_pct=r.valid_pct, cohens_d=r.cohens_d,
                         h_norm=float(hn.median()),
                         injection=float(inj.median()),
                         pct_of_h=float((inj / hn).median() * 100)))
    d = pd.DataFrame(rows).sort_values(["layer", "method", "injection"])
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "density_injection_magnitude.csv", index=False, float_format="%.6g")
    print("\n" + d[["label", "layer", "injection", "pct_of_h", "valid_pct",
                    "cohens_d"]].to_string(index=False))

    fig, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 6.4),
                             squeeze=False)
    for ax, lay in zip(axes[0], layers):
        s = d[d.layer == lay].sort_values("pct_of_h")
        ax.barh(range(len(s)), s.pct_of_h,
                color=[COLOR[m] for m in s.method], height=0.72)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.label, fontsize=9)
        for i, (p, v) in enumerate(zip(s.pct_of_h, s.valid_pct)):
            ax.text(p + 1.5, i, f"{v*100:.0f}% valid", va="center", fontsize=8.5,
                    color="#555")
        ax.set_xlabel("injection as % of $|h|$")
        ax.set_title(f"layer {lay}   (median $|h|$ = {s.h_norm.iloc[0]:.1f})")
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.set_xlim(0, max(d.pct_of_h) * 1.32)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("How hard each density intervention actually pushes\n"
                 f"median |h_new - h| over {states[layers[0]].shape[0]:,} real per-token "
                 "states, as a share of the state's own norm", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "density_injection_magnitude.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print(f"\nSaved {OUT / 'density_injection_magnitude.png'}")


if __name__ == "__main__":
    main()
