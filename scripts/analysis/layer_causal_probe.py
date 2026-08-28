#!/usr/bin/env python3
"""
Which layer, if any, has causal control over the number the model writes?

Steering at layer 14 changes the logits by 13-23% (layernorm_survival.py) and moves
neither band gap nor volume per atom. That leaves the question of whether any *other*
layer carries a usable handle, and answering it by generating a full sweep per layer
would cost days of GPU. This answers it with forward passes only.

HOW
---
CrystaLLM writes numbers digit by digit -- `_cell_volume   160.9257` tokenizes as
['_cell_volume', ' ', '1', '6', '0', '.', ...]. So the model's belief about the volume
is a distribution over digit tokens at known positions. Teacher-force a real CIF, inject
at layer L, and compare the next-token distribution at those positions against the clean
one. No sampling, no generation, one forward pass per layer per CIF.

WHAT IT REPORTS
---------------
  d_log10_volume    change in the model's implied log10 of the volume it is about to
                    write. Positive means the intervention pushes the number UP, which
                    is what steering toward high volume/atom is supposed to do. Built as
                    (change in expected integer-digit count) + (change in E[log10 of the
                    leading digit]), because log10(value) ~= (n_int_digits - 1) +
                    log10(leading digit). The raw leading digit is NOT usable on its own:
                    90 -> 100 raises the value and drops the leading digit.
  kl_target         mean KL(steered || clean) at the volume's digit positions
  kl_other          the same at every other position
  selectivity       kl_target / kl_other. A layer with
                    causal control over the property should disturb the tokens that
                    carry it MORE than the rest of the CIF. ~1.0 would mean the injection
                    is undifferentiated noise. It is NOT the whole story: selectivity says
                    the injection lands on the right tokens, d_log10_volume says whether
                    it moves them anywhere. High selectivity with no magnitude shift means
                    the distribution is being scrambled, not steered.

Injection strength is matched across layers as a fraction of that layer's own residual
norm (--frac), since the residual stream grows through depth and a fixed alpha would
mean something different at each layer.

Usage:
    python scripts/analysis/layer_causal_probe.py --property density_atomic
    python scripts/analysis/layer_causal_probe.py --property density_atomic --frac 0.25
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM", "bin"))
from crystallm import CIFTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "embeddings"))
from extract_cif_embeddings import load_model, load_cifs
from utils import analysis_dir, write_results_table  # noqa: F401  (analysis_dir only)

TEST_PKL = "CrystaLLM/cifs_v1_test_sample1000.pkl.gz"
DIGITS = [str(d) for d in range(10)]


def volume_digit_positions(tokens: list):
    """(all digit positions, integer-part digit positions) for `_cell_volume`'s value.

    Positions index the token list; the model *predicts* token i+1 from position i, so
    the caller shifts by one. The integer part is separated out because that is what
    carries magnitude: a number gains an order of magnitude by gaining an integer digit.
    """
    try:
        start = tokens.index("_cell_volume")
    except ValueError:
        return [], []
    allpos, intpos, i, seen_dot = [], [], start + 1, False
    while i < len(tokens) and tokens[i] in (" ", "\t"):
        i += 1
    while i < len(tokens) and (tokens[i] in DIGITS or tokens[i] == "."):
        if tokens[i] == ".":
            seen_dot = True
        else:
            allpos.append(i)
            if not seen_dot:
                intpos.append(i)
        i += 1
    return allpos, intpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CrystaLLM/crystallm_v1_large")
    ap.add_argument("--property", default="density_atomic",
                    help="steering_vectors/<property>/layer{N}.parquet must exist per layer")
    ap.add_argument("--method", choices=("linear", "pca_centroid"), default="linear",
                    help="Which injection to probe. pca_centroid needs --target/--t and "
                         "a basis + centroid; its displacement depends on the hidden "
                         "state, so --frac is ignored and t sets the strength.")
    ap.add_argument("--target", type=float, default=30.0, help="[pca_centroid]")
    ap.add_argument("--t", type=float, default=0.5, help="[pca_centroid]")
    ap.add_argument("--k", type=int, default=64, help="[pca_centroid]")
    ap.add_argument("--frac", type=float, default=0.21,
                    help="Injection norm as a fraction of the layer's own residual norm. "
                         "0.21 matches alpha=40 at layer 14.")
    ap.add_argument("--n-cifs", type=int, default=60)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.model, device)
    tokenizer = CIFTokenizer()
    digit_ids = torch.tensor([tokenizer.token_to_id[d] for d in DIGITS], device=device)
    # log10 of each digit, for the leading-digit term of the magnitude estimate;
    # digit 0 never leads a number, so its weight is irrelevant and set to 0.
    log_digit = torch.tensor([0.0] + [np.log10(d) for d in range(1, 10)],
                             device=device, dtype=torch.float32)

    vecs, pca = {}, {}
    if args.method == "linear":
        for L in range(config.n_layer):
            p = f"steering_vectors/{args.property}/layer{L}.parquet"
            if os.path.exists(p):
                v = np.array(pd.read_parquet(p).iloc[0]["steering_vector"], dtype=np.float32)
                vecs[L] = torch.tensor(v, device=device)
        print(f"linear vectors found for layers: {sorted(vecs)}\n")
    else:
        for L in range(config.n_layer):
            base = f"steering_vectors/pca_centroid/pca_layer{L}_k{args.k}.parquet"
            cen = (f"steering_vectors/pca_centroid/{args.property}/"
                   f"layer{L}_k{args.k}_target{args.target:g}.parquet")
            if os.path.exists(base) and os.path.exists(cen):
                b = pd.read_parquet(base).iloc[0]
                c = pd.read_parquet(cen).iloc[0]
                pca[L] = (torch.tensor(np.asarray(b["mean"], dtype=np.float32), device=device),
                          torch.tensor(np.asarray(b["components"], dtype=np.float32)
                                       .reshape(int(b["k"]), -1), device=device),
                          torch.tensor(np.asarray(c["centroid_pca"], dtype=np.float32),
                                       device=device))
                vecs[L] = None
        print(f"pca_centroid target={args.target:g} t={args.t:g}: layers {sorted(vecs)}\n")
    if not vecs:
        raise SystemExit(f"No injection available for method={args.method}")

    data = load_cifs(TEST_PKL)
    cifs = []
    for _, cif in data:
        toks = tokenizer.tokenize_cif(cif)
        pos, intpos = volume_digit_positions(toks)
        if len(pos) >= 2 and len(intpos) >= 1 and len(toks) < config.block_size:
            cifs.append((toks, pos, intpos))
        if len(cifs) >= args.n_cifs:
            break
    print(f"{len(cifs)} CIFs with a readable _cell_volume\n")

    rows = []
    for L, vec in sorted(vecs.items()):
        acc = []
        for toks, pos, intpos in cifs:
            ids = torch.tensor(tokenizer.encode(toks), device=device).unsqueeze(0)
            # positions that PREDICT a volume digit
            tgt = torch.tensor([p - 1 for p in pos], device=device)

            hidden = {}

            def grab(module, inp, out):
                hidden["h"] = (out[0] if isinstance(out, tuple) else out).detach()

            h_handle = model.transformer.h[L].register_forward_hook(grab)
            with torch.no_grad():
                # targets=ids only to make the model run lm_head over EVERY position;
                # without it _model.py returns logits for the last token alone.
                clean, _ = model(ids, targets=ids)
            h_handle.remove()
            if args.method == "linear":
                # match the injection to this layer's own scale, so depth is not confounded
                scale = args.frac * hidden["h"].float().norm(dim=-1).mean()
                inject = (vec * scale).view(1, 1, -1)
            else:
                # the pca displacement depends on the hidden state, exactly as the
                # generator computes it; t alone sets the strength
                mu, W, c = pca[L]
                h = hidden["h"].float()
                inject = args.t * (c - (h - mu) @ W.T) @ W

            def steer(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + inject.to(h.dtype)
                return (h,) + out[1:] if isinstance(out, tuple) else h

            s_handle = model.transformer.h[L].register_forward_hook(steer)
            with torch.no_grad():
                steered, _ = model(ids, targets=ids)
            s_handle.remove()

            lp_c = F.log_softmax(clean[0].float(), -1)
            lp_s = F.log_softmax(steered[0].float(), -1)
            kl = (lp_s.exp() * (lp_s - lp_c)).sum(-1)          # KL(steered || clean)

            mask = torch.zeros(lp_c.shape[0], dtype=torch.bool, device=device)
            mask[tgt] = True

            # MAGNITUDE, not the raw leading digit. A volume going 90 -> 100 raises the
            # value while dropping the leading digit from 9 to 1, so E[leading digit] is
            # not monotone in magnitude and its sign cannot be read. What is monotone:
            #   log10(value) ~= (number of integer digits - 1) + log10(leading digit)
            # Both terms are read off the model's own distributions.
            first = tgt[0]
            pc = lp_c[first, digit_ids].exp(); pc = pc / pc.sum()
            ps = lp_s[first, digit_ids].exp(); ps = ps / ps.sum()
            d_log_lead = float((ps * log_digit).sum() - (pc * log_digit).sum())

            # At each integer-digit position, P(the number keeps going) rather than
            # ending. More integer digits = an order of magnitude larger. Summed over
            # positions this is the change in expected integer-digit count.
            ipos = torch.tensor([p - 1 for p in intpos], device=device)
            cont_c = lp_c[ipos][:, digit_ids].exp().sum(-1)
            cont_s = lp_s[ipos][:, digit_ids].exp().sum(-1)
            d_n_digits = float((cont_s - cont_c).sum())

            acc.append(dict(
                d_log10_volume=d_n_digits + d_log_lead,
                d_n_int_digits=d_n_digits,
                d_log_leading=d_log_lead,
                kl_target=float(kl[mask].mean()),
                kl_other=float(kl[~mask].mean()),
            ))
        m = pd.DataFrame(acc).mean()
        rows.append(dict(layer=L, method=args.method, n_cifs=len(cifs),
                         inject_frac=args.frac if args.method == "linear" else np.nan,
                         t=args.t if args.method != "linear" else np.nan,
                         d_log10_volume=m.d_log10_volume,
                         d_n_int_digits=m.d_n_int_digits, d_log_leading=m.d_log_leading,
                         kl_target=m.kl_target, kl_other=m.kl_other,
                         selectivity=m.kl_target / m.kl_other if m.kl_other else np.nan))
        r = rows[-1]
        print(f"  layer {L:>2}  d_log10_vol {r['d_log10_volume']:+.4f}  "
              f"(n_digits {r['d_n_int_digits']:+.4f}, lead {r['d_log_leading']:+.4f})   "
              f"KL tgt {r['kl_target']:.3f} oth {r['kl_other']:.3f}   "
              f"sel {r['selectivity']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    tag = ("linear" if args.method == "linear"
           else f"pca_centroid_target{args.target:g}_t{args.t:g}")
    out = (analysis_dir("v1_all", None, "test")
           / f"layer_causal_probe_{args.property}_{tag}.csv")
    df.to_csv(out, index=False, float_format="%.6g")
    print("\nRead the two columns together. High selectivity with d_log10_volume ~ 0 is\n"
          "the injection SCRAMBLING the property's tokens rather than TRANSLATING them:\n"
          "it lands where the number is written and changes the distribution there\n"
          "without moving the magnitude. A layer worth a real generation sweep needs\n"
          "d_log10_volume of the right sign and a usable size, not just selectivity.")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
