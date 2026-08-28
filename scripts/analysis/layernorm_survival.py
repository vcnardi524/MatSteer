#!/usr/bin/env python3
"""
Does the LayerNorm downstream of the injection erase the steering vector?

Every steering result in this repo is null, and one mundane explanation would
invalidate all of them: CrystaLLM is pre-norm (`x = x + attn(ln_1(x))`), so a vector
added to the residual stream at layer L is normalised before anything reads it. Two
parts of an injection are removed for free:

  the uniform component   LayerNorm subtracts the per-token mean, so whatever part of
                          the injection lies along the all-ones direction is annihilated
  the scale               LayerNorm divides by the per-token std, so adding a vector
                          that inflates the std shrinks everything, injection included

This measures how much of the change actually survives to the point where it can affect
a token: after block L+1's ln_1, after ln_f, and in the logits themselves.

The number to read is `rel_delta` -- ||steered - clean|| / ||clean|| at each stage. If it
holds roughly constant from the injection through to the logits, LayerNorm is not the
problem and the nulls stand. If it collapses at the first norm, the sweeps never tested
what they were meant to.

Usage:
    python scripts/analysis/layernorm_survival.py --method linear --alpha 40
    python scripts/analysis/layernorm_survival.py --method pca_centroid --target 30 --t 0.5
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "embeddings"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "steering"))
from make_prompts import PATTERN_COMP_SG, extract_prompt
from extract_cif_embeddings import load_model, load_cifs
from compute_centroid_target import load_pca
from utils import analysis_dir

TEST_PKL = "CrystaLLM/cifs_v1_test_sample1000.pkl.gz"


def capture(model, x, layer, inject=None):
    """Run one forward pass, returning the tensors the injection has to survive."""
    got, handles = {}, []

    def save(name):
        def hook(module, inp, out):
            got[name] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return hook

    def steer(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + inject.to(h.dtype)
        got["injected"] = h.detach().float()
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handles.append(model.transformer.h[layer].register_forward_hook(
        steer if inject is not None else save("h_inject")))
    handles.append(model.transformer.h[layer + 1].ln_1.register_forward_hook(save("ln_1")))
    handles.append(model.transformer.h[layer + 1].register_forward_hook(save("h_next")))
    handles.append(model.transformer.ln_f.register_forward_hook(save("ln_f")))
    with torch.no_grad():
        logits, _ = model(x)
    for h in handles:
        h.remove()
    if inject is not None:
        got["h_inject"] = got.pop("injected")
    got["logits"] = logits.detach().float()
    return got


def rel_delta(a, b):
    """||a - b|| / ||b||, averaged over token positions."""
    d = (a - b).flatten(0, -2).norm(dim=-1)
    n = b.flatten(0, -2).norm(dim=-1)
    return float((d / n.clamp_min(1e-9)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CrystaLLM/crystallm_v1_large")
    ap.add_argument("--method", choices=("linear", "pca_centroid"), default="linear")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=40.0, help="[linear]")
    ap.add_argument("--steering-property", default="density_atomic")
    ap.add_argument("--target", type=float, default=30.0, help="[pca_centroid]")
    ap.add_argument("--t", type=float, default=0.5, help="[pca_centroid]")
    ap.add_argument("--k", type=int, default=64, help="[pca_centroid]")
    ap.add_argument("--n-prompts", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.model, device)
    tokenizer = CIFTokenizer()

    # Build the same injection the generator would apply. For pca_centroid the vector
    # depends on the hidden state, so it is built per prompt inside the loop.
    if args.method == "linear":
        sv = pd.read_parquet(
            f"steering_vectors/{args.steering_property}/layer{args.layer}.parquet").iloc[0]
        vec = torch.tensor(np.array(sv["steering_vector"], dtype=np.float32) * args.alpha,
                           device=device).view(1, 1, -1)
        label = f"linear alpha={args.alpha:g}"
    else:
        mean, comps = load_pca(args.layer, args.k)
        cen = pd.read_parquet(
            f"steering_vectors/pca_centroid/{args.steering_property}/"
            f"layer{args.layer}_k{args.k}_target{args.target:g}.parquet").iloc[0]
        mu = torch.tensor(mean, device=device)
        W = torch.tensor(comps, device=device)
        c = torch.tensor(np.asarray(cen["centroid_pca"], dtype=np.float32), device=device)
        label = f"pca_centroid target={args.target:g} t={args.t:g} k={args.k}"

    data = load_cifs(TEST_PKL)
    prompts = []
    for id_, cif in data:
        try:
            prompts.append(extract_prompt(cif, PATTERN_COMP_SG))
        except Exception:
            pass
        if len(prompts) >= args.n_prompts:
            break
    print(f"{label}   layer {args.layer}   {len(prompts)} prompts\n")

    rows = []
    for p in prompts:
        tok = tokenizer.encode(tokenizer.tokenize_cif(p))
        x = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)
        clean = capture(model, x, args.layer)

        if args.method == "linear":
            inject = vec
        else:
            h = clean["h_inject"]
            inject = (args.t * (c - (h - mu) @ W.T) @ W)
        steered = capture(model, x, args.layer, inject=inject)

        d = inject.flatten(0, -2) if inject.ndim == 3 else inject.view(1, -1)
        h0 = clean["h_inject"].flatten(0, -2)
        # LayerNorm subtracts the per-token mean, so the component of the injection along
        # the all-ones direction is removed outright, before any rescaling.
        uniform = (d.mean(-1, keepdim=True).expand_as(d)).norm(dim=-1) / d.norm(dim=-1).clamp_min(1e-9)
        rows.append(dict(
            inject_norm=float(d.norm(dim=-1).mean()),
            resid_norm=float(h0.norm(dim=-1).mean()),
            uniform_frac=float(uniform.mean()),
            at_injection=rel_delta(steered["h_inject"], clean["h_inject"]),
            after_ln_1=rel_delta(steered["ln_1"], clean["ln_1"]),
            after_block=rel_delta(steered["h_next"], clean["h_next"]),
            after_ln_f=rel_delta(steered["ln_f"], clean["ln_f"]),
            logits=rel_delta(steered["logits"], clean["logits"]),
        ))

    df = pd.DataFrame(rows)
    m = df.mean()
    print(f"injection norm {m.inject_norm:8.3f}   residual norm {m.resid_norm:8.3f}   "
          f"ratio {m.inject_norm / m.resid_norm:.3f}")
    print(f"fraction of the injection along the all-ones direction (LayerNorm removes "
          f"this outright): {m.uniform_frac:.4f}\n")
    print("relative change ||steered - clean|| / ||clean||, mean over tokens and prompts")
    print(f"  at the injection (layer {args.layer} output)   {m.at_injection:.4f}")
    print(f"  after block {args.layer+1} ln_1                     {m.after_ln_1:.4f}")
    print(f"  after block {args.layer+1} (residual)               {m.after_block:.4f}")
    print(f"  after ln_f                            {m.after_ln_f:.4f}")
    print(f"  in the logits                         {m.logits:.4f}")
    survival = m.after_ln_1 / m.at_injection if m.at_injection else float("nan")
    print(f"\nsurvival through the first LayerNorm: {survival:.1%}")
    print("  >90% means LayerNorm is not erasing the injection and the nulls stand;"
          "\n  a collapse means the sweeps never tested what they were meant to.")

    out = analysis_dir("v1_all", None, "test") / (
        f"layernorm_survival_{args.method}_layer{args.layer}.csv")
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
