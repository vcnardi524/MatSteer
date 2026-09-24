#!/usr/bin/env python3
"""The injection-magnitude table, computed rather than measured.

injection_magnitude.py gets |h_new - h| by running the model and applying each run's real
hook. That is the honest way when the injection is not knowable in closed form. For the
two methods here it is:

    linear    |alpha * v|            = |alpha|, the vector being unit-norm
    manifold  |scale * (dec(u+d) - dec(u)) @ W| = scale * |dec(u+d) - dec(u)|,
              the PCA components being orthonormal

so the GPU pass buys nothing the arithmetic does not already give, and this runs on CPU in
seconds. |h| is supplied per layer from a one-off measurement on answer tokens.

Emits the same schema plot_magnitude_response.py reads, so that script works unchanged.
The p25/p75 columns it uses for error bars are left empty: those describe token-to-token
spread in the measured injection, which a closed-form value does not have. Do not read
their absence as zero spread.

Usage:
    python scripts/analysis/injection_magnitude_analytic.py \
        --model llamat2_cif --property density_atomic
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/
from utils import analysis_dir, steering_vectors_dir, MODELS, DEFAULT_MODEL
from manifold import Manifold

# Per-token |h| on ANSWER tokens -- the states the hook modifies. Prompt tokens run larger
# and would understate every fraction computed from them.
HIDDEN_NORM = {"llamat2_cif": {8: 6.4, 12: 8.7, 24: 22.2},
               "crystallm":   {7: 127.1, 14: 165.8}}
# The bucket width each sweep used: the wider, better-conditioned fit of the two.
SWEEP_WIDTH = {"band_gap": "w0.5_rc", "density_atomic": "w3",
               "formation_energy_per_atom": "w0.5"}


def step_norm(model, prop, layer, delta, cache={}):
    key = (model, prop, layer)
    if key not in cache:
        w = SWEEP_WIDTH.get(prop)
        p = steering_vectors_dir(model, "manifolds") / f"{prop}_layer{layer}_k32_{w}.parquet"
        cache[key] = Manifold.load(p) if w and p.exists() else None
    m = cache[key]
    if m is None:
        return None
    u = torch.linspace(0, float(m.length), 60).unsqueeze(-1)
    return float((m.decode(u + delta) - m.decode(u)).norm(dim=-1).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--property", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = analysis_dir("v1_all", None, "test", model=args.model)
    frames = []
    for suffix, method in (("", "linear"), ("_manifold", "manifold")):
        p = base / f"{args.property}{suffix}_steering_ttest.csv"
        if p.exists():
            f = pd.read_csv(p)
            frames.append(f[(f.method == method) & f.cohens_d.notna()])
    if not frames:
        raise SystemExit(f"no t-test CSVs for {args.property} under {base}")
    d = pd.concat(frames, ignore_index=True)

    norms = HIDDEN_NORM.get(args.model, {})
    inj, hn, lab = [], [], []
    for _, r in d.iterrows():
        h = norms.get(int(r.layer), np.nan)
        hn.append(h)
        if r.method == "linear":
            inj.append(abs(r.strength))
            lab.append(f"linear a{r.strength:g}")
        else:
            s = step_norm(args.model, args.property, int(r.layer), float(r.target))
            inj.append(np.nan if s is None else abs(r.strength * s))
            lab.append(f"d{r.target:g} s{r.strength:g}")
    d["h_norm"] = hn
    d["injection"] = inj
    d["pct_of_h"] = 100 * d.injection / d.h_norm
    d["label"] = lab
    d["pct_p25"] = np.nan
    d["pct_p75"] = np.nan

    cols = ["run", "label", "method", "layer", "target", "strength", "valid_pct",
            "cohens_d", "h_norm", "injection", "pct_of_h", "pct_p25", "pct_p75"]
    d = d.dropna(subset=["pct_of_h"])[cols]
    out = args.out or (analysis_dir("v1_all", None, "test", subdir="plots",
                                    model=args.model)
                       / f"{args.property}_injection_magnitude.csv")
    d.to_csv(out, index=False)
    print(f"{len(d)} runs -> {out}")
    print(d[["label", "method", "layer", "pct_of_h", "cohens_d", "valid_pct"]]
          .sort_values(["method", "layer", "pct_of_h"]).to_string(index=False))


if __name__ == "__main__":
    main()
