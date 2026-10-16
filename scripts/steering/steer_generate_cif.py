#!/usr/bin/env python3
"""
Steered CIF generation, for any registered model.

Two axes, kept separate. `--model` picks a BACKEND from scripts/backends.py (the weights,
the tokenizer, and where the transformer blocks live) and, with it, a PROMPT SOURCE --
what the model is asked and how its answer becomes a CIF:

  crystallm    CifPrefixPrompts. The prompt is the head of a real CIF, which the model
               continues, so the output already IS a CIF and to_cif is the identity.
               Every arm draws from the same 1,000 test structures, which is what makes
               the paired t-test valid.
  llamat2_cif  UnconditionalPrompts. One constant instruction; the model writes a crystal
               string, which is decoded into a CIF with pymatgen deriving the space group
               (llamat_prompts.crystal_string_to_cif). Raw output is kept beside the CIF.
               NOT pairable on id -- see --paired-seed and CLAUDE.md.

The steering methods themselves are architecture-agnostic: they take a bare hidden-state
tensor, and the hook goes on backend.blocks[layer]. It is applied to every token position
at every generation step (ChemSteer's inject_timestep='all'):

  linear        h <- h + alpha * v
                v is the normalized high-class minus low-class mean difference from
                compute_steering_vector.py. Assumes the property is a straight line in
                the residual stream.

  pca_centroid  z <- (h - mu) @ W.T ;  h <- h + t * (centroid_pca - z) @ W
                Project into the top-K PCA subspace of the training activations, move
                the coordinates a fraction t of the way to a target centroid, and map
                the change back. There is no low class: the centroid alone sets the
                destination, and everything outside the K principal directions passes
                through untouched, so the residual stream keeps supplying the context
                the centroid does not specify. t=0 is no steering, t=1 snaps the
                subspace coordinates onto the centroid.
                Needs compute_pca_basis.py and compute_centroid_target.py.

Outputs a single parquet with columns id, sample, cif_steered -- plus raw_output and
decode_reason when the model does not emit a CIF directly. Every method writes into the
same directory; the filename carries the method so they never collide.

Usage:
    python steer_generate_cif.py --model crystallm --ckpt-dir CrystaLLM/crystallm_v1_large \
        --pkl CrystaLLM/cifs_v1_test.pkl.gz --alpha 40 --layer 14 --n-samples 3 \
        --with-spacegroup --steering-property density_atomic

    python steer_generate_cif.py --model llamat2_cif --ckpt-dir models/llamat2_cif \
        --method linear --steering-property band_gap --layer 24 --alpha 16 \
        --n-prompts 1000 --n-samples 1 --temperature 0.01 --top-p 0.95 --paired-seed
"""
import argparse
import zlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CrystaLLM", "bin"))

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # scripts/ -> utils.py, predictors.py
from utils import steering_vectors_dir, MODELS, DEFAULT_MODEL
from backends import BACKENDS
import llamat_prompts
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "embeddings"))   # -> extract_cif_embeddings.py
from make_prompts import PATTERN_COMP, PATTERN_COMP_SG, extract_prompt
from extract_cif_embeddings import load_cifs
# sys.path[0] is this script's own dir, so its neighbour imports directly.
# load_pca is imported lazily, inside the three builders that need it. Its module chain
# reaches compute_pca_basis, which imports sklearn at module level -- and sklearn is not
# in llamat_venv. At top level that makes `--method linear`, which never touches a PCA
# basis, unimportable for llamat purely by association.
from manifold import Manifold

RANDOM_SEED = 42
CHECKPOINT_EVERY = 100  # write to parquet every N prompts


def rewrap(out, hidden):
    """Put a modified hidden state back into whatever Block.forward returned.

    With the KV cache it returns (hidden, present_kv); the cache passes through
    untouched.
    """
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


def linear_hook(steer_vec, alpha, device):
    vec = torch.tensor(steer_vec * alpha, dtype=torch.float32, device=device).view(1, 1, -1)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        return rewrap(out, h + vec.to(h.dtype))

    return hook


def pca_centroid_hook(mean, components, centroid_pca, t, device):
    mu = torch.tensor(mean, dtype=torch.float32, device=device)            # (1024,)
    W = torch.tensor(components, dtype=torch.float32, device=device)       # (k, 1024)
    c = torch.tensor(centroid_pca, dtype=torch.float32, device=device)     # (k,)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        z = (h.float() - mu) @ W.T                    # (B, T, k) subspace coordinates
        return rewrap(out, h + (t * (c - z) @ W).to(h.dtype))

    return hook


def local_centroid(z_prompt, bank, n_neighbours):
    """Centroid of the n_neighbours class members closest to this prompt.

    The global centroid averages the whole class, so every prompt is pulled toward the
    same point no matter where it starts. If the class does not sit in one place, that
    average is a point between the pieces rather than inside any of them.
    """
    d = torch.cdist(z_prompt.view(1, -1), bank).view(-1)
    idx = torch.topk(d, min(n_neighbours, bank.shape[0]), largest=False).indices
    return bank[idx].mean(0), float(d[idx].mean())


def manifold_hook(mean, components, manifold, delta, device, scale=1.0,
                  variant="residual"):
    """Move a hidden state along a curve fitted through the property's bucket centroids.

    Writing h = mu + z @ W + h_perp and z = decode(u) + r, the three variants differ only
    in what they keep:

      residual      h + scale * (decode(u+d) - decode(u)) @ W
                    keeps mu, h_perp and r. The injection is ONLY the curve step, whose
                    size is capped by the curve's own extent (19.47 end to end here), so
                    `scale` is what lets it reach the magnitudes the linear method uses.
      project       mu + z_new @ W
                    keeps mu and r, discards h_perp. Its injection is the curve step
                    MINUS h_perp, so it is dominated by the deletion, not the steering.
      project_nomu  z_new @ W
                    discards mu as well. Injection exceeds |h| itself, since |mu| alone
                    is larger than the median hidden state.

    Measured on real per-token states at layer 14 (|h| = 130.8), with delta=15:
    residual 6.80 (5.2% of |h|), project 94.60 (72.3%), project_nomu 161.99 (124%).
    For reference linear alpha=40 injects 30.6% and alpha=80 61.2%.
    """
    mu = torch.tensor(mean, dtype=torch.float32, device=device)          # (1024,)
    W = torch.tensor(components, dtype=torch.float32, device=device)     # (k, 1024)
    manifold = manifold.to(device)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        z = (h.float() - mu) @ W.T                     # (B, T, k)
        u, r = manifold.encode(z)                      # (B, T, 1), (B, T, k)
        z_new = manifold.decode(u + delta) + r
        if variant == "project":
            h_new = (mu + z_new @ W).to(h.dtype)
        elif variant == "project_nomu":
            h_new = (z_new @ W).to(h.dtype)
        else:
            h_new = h + (scale * ((z_new - z) @ W)).to(h.dtype)
        return rewrap(out, h_new)

    return hook


def generate(backend, prompt_str, args, hook=None):
    """(text, n_new_tokens) for one sample, hook attached for its lifetime only.

    n_new_tokens is what makes truncation detectable. A generation that used the whole
    --max-new-tokens budget was CUT OFF mid-structure, and nothing in the resulting CIF
    records that: the file is internally consistent, it just describes a smaller crystal
    than the model was writing. Counting tokens here is the only exact way to know, and
    it is free -- generate_ids already has them.
    """
    handle = backend.blocks[args.layer].register_forward_hook(hook) if hook else None
    try:
        y, n_prompt = backend.generate_ids(
            prompt_str, args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, use_cache=args.use_cache)
        n_new = int(y.shape[1]) - n_prompt
        text = backend.decode(y[0][n_prompt:].tolist()) if backend.strips_prompt \
            else backend.decode(y[0].tolist())
        return text, n_new
    finally:
        if handle:
            handle.remove()


# ---------------------------------------------------------------------------------
# Prompt sources -- what the model is asked, and how its output becomes a CIF
# ---------------------------------------------------------------------------------
#
# A separate axis from the backend, for the same reason the text builders are separate
# in extract_cif_embeddings.py: the weights decide HOW to run the model, the prompt
# source decides WHAT it is asked and what its answer means. Each one supplies
#
#   prompts() -> [(id, text)]
#   to_cif(raw) -> (cif or "", reason)
#
# `raw` is always kept alongside the CIF, so a decode failure can be diagnosed from the
# stored generation instead of being re-run.


class CifPrefixPrompts:
    """CrystaLLM: the head of a real CIF, which the model continues.

    Every arm draws from the SAME 1,000 test structures, which is what makes the paired
    t-test valid -- each arm and the alpha=0 control pair on `id`. See CLAUDE.md.
    """

    def __init__(self, pkl, with_spacegroup, n_prompts):
        # Part of the filename because it names the PROMPT SET, which is what a baseline
        # is keyed on (CLAUDE.md). Unchanged from before the refactor -- crystallm stems
        # are join keys across four stores, so they must not move.
        self.tag = "" if with_spacegroup else "_nosg"
        # Unchanged from before: inferred from the pkl filename (cifs_v1_test -> test).
        stem = Path(pkl).stem
        self.split = next((s for s in ("train", "test", "val") if s in stem), stem)
        pattern = PATTERN_COMP_SG if with_spacegroup else PATTERN_COMP
        self._prompts = []
        for id_, cif in load_cifs(pkl):
            try:
                self._prompts.append((id_, extract_prompt(cif, pattern)))
            except Exception:
                pass
            if n_prompts and len(self._prompts) >= n_prompts:
                break

    def prompts(self):
        return self._prompts

    decodes = False          # output already is a CIF; nothing to keep beside it

    def to_cif(self, raw):
        """Identity. CrystaLLM emits a CIF directly -- pymatgen never runs on this path,
        so every existing crystallm result is reproduced unchanged."""
        return raw, ""


class UnconditionalPrompts:
    """llamat2-cif: one constant instruction, and each sample is an independent draw.

    THIS BREAKS THE PAIRED t-TEST. There is no per-structure id to pair an arm against
    its control on, because there is no per-structure prompt -- the ids below are draw
    indices, not materials. Use --paired-seed so draw k of every arm starts from the
    same random state (common random numbers), which recovers most of the variance
    reduction; without it the arms must be compared as unpaired distributions.
    """

    tag = ""                 # no sg/nosg choice exists: the prompt names no space group
    split = "uncond"         # not a split at all -- these draws come from no corpus

    def __init__(self, n_prompts, system_index=0, wrapper="notebook"):
        if not n_prompts:
            raise SystemExit("--n-prompts is required for unconditional generation "
                             "(there is no corpus to size it from)")
        self.text = llamat_prompts.unconditional_prompt(system_index, wrapper)
        self.n = n_prompts

    def prompts(self):
        return [(f"draw{i:05d}", self.text) for i in range(self.n)]

    decodes = True           # always keep the raw crystal string beside the CIF

    def to_cif(self, raw):
        """Decode the crystal string. pymatgen derives the space group here, at
        generation time, so the CIF is self-consistent before validation ever reads it."""
        cif, reason = llamat_prompts.crystal_string_to_cif(raw)
        return (cif or ""), reason


class ConditionalPrompts:
    """llamat2-cif conditioned on composition and space group. PAIRABLE on id.

    The analogue of CifPrefixPrompts, and the reason it exists: llamat's unconditional
    prompt is one constant string, so there is no per-structure prompt to pair an arm
    against its control on. Conditioning on the structure's own formula, elements and
    space group gives 1,000 distinct prompts drawn from llamat's own test split, so the
    paired t-test applies exactly as it does for crystallm.

    WHY THE SPACE GROUP IS NOT OPTIONAL HERE. Training drew k = randint(0, 3) conditions:
    k == 0 was fully unconditional, and k >= 1 was formula + elements plus at least one
    of OPTIONAL_CONDITIONS. Composition ALONE never appeared, so a "nosg" analogue would
    be off-distribution. Of the three optional conditions, spacegroup.number is the only
    one that is not also a steering target -- conditioning on formation_energy_per_atom
    or e_above_hull while steering toward it would hand the model the answer. band_gap
    has a phrase but was never a training condition, so it cannot leak.

    Prompts come from the tracked CSV, not the llamat clone, so this runs on a fresh
    checkout. See scripts/data/make_llamat_test_sample.py.
    """

    decodes = True
    # NOT "_sg": crystallm already uses "" for its sg prompt set and "_nosg" for the
    # other, and this is a THIRD prompt set, not a member of that pair. It also keeps the
    # stem distinct from crystallm's control in the shared baseline/ tree.
    tag = "_cond"

    def __init__(self, csv_path, n_prompts, system_index=0, wrapper="notebook"):
        stem = Path(csv_path).stem
        self.split = next((s for s in ("train", "test", "val") if s in stem), stem)
        df = pd.read_csv(csv_path)
        if n_prompts:
            df = df.iloc[:n_prompts]
        system = llamat_prompts.GENERATION_SYSTEMS[system_index]
        wrap = llamat_prompts.WRAPPERS[wrapper]
        self._prompts = []
        for r in df.itertuples():
            conditions = {
                "pretty_formula":    r.pretty_formula,
                "elements":          llamat_prompts.elements_from_formula_sum(r.formula_sum),
                "spacegroup.number": int(r.spacegroup_number),
            }
            text = wrap(system, llamat_prompts.conditional_generation_input(conditions))
            self._prompts.append((r.id, text))

    def prompts(self):
        return self._prompts

    def to_cif(self, raw):
        cif, reason = llamat_prompts.crystal_string_to_cif(raw)
        return (cif or ""), reason


def build_prompts(args):
    if args.model == "crystallm":
        return CifPrefixPrompts(args.pkl, args.with_spacegroup, args.n_prompts)
    if args.prompt_csv:
        return ConditionalPrompts(args.prompt_csv, args.n_prompts,
                                  args.system_index, args.wrapper)
    return UnconditionalPrompts(args.n_prompts, args.system_index, args.wrapper)


def build_linear(args, device, backend=None):
    """(hook, filename stem suffix) for the mean-difference method."""
    sv_path = (steering_vectors_dir(args.model, args.steering_property)
               / f"layer{args.layer}.parquet")
    if args.alpha == 0 and args.alpha_rel is None:
        # The vector is multiplied by zero, so its contents cannot matter and requiring
        # the file to exist is an artificial constraint -- it would force a vector to be
        # fitted at the control's layer for no reason. The hook is still registered and
        # still runs, it just adds exactly zero, which is what makes this a true control.
        # One such run per PROMPT SET serves every property and every layer (CLAUDE.md).
        if backend is None:
            raise SystemExit("alpha 0 needs the backend to size the zero vector")
        print(f"Steering vector: none loaded -- alpha is 0, so the hook adds exactly "
              f"zero at layer {args.layer} ({backend.n_embd} dims)")
        print(f"Method=linear  alpha=0.0  layer={args.layer}")
        zero = np.zeros(backend.n_embd, dtype=np.float32)
        return linear_hook(zero, 0.0, device), "alpha0.0"
    if not sv_path.exists():
        # legacy flat location (pre per-property dirs)
        legacy = (steering_vectors_dir(args.model)
                  / f"{args.steering_property}_layer{args.layer}.parquet")
        if not legacy.exists():
            raise FileNotFoundError(f"No steering vector at {sv_path} or {legacy}")
        sv_path = legacy
    row = pd.read_parquet(sv_path).iloc[0]   # single clean low-vs-high vector
    steer_vec = np.array(row["steering_vector"], dtype=np.float32)
    lo = row.get("low_thresh", row.get("low_thresh_ev"))   # new / legacy column names
    hi = row.get("high_thresh", row.get("high_thresh_ev"))
    raw_norm = float(row["raw_norm"])
    print(f"Steering vector [{args.steering_property}] {sv_path}: low<={lo} "
          f"(n={int(row['n_low']):,}) vs high>={hi} (n={int(row['n_high']):,})  "
          f"raw_norm={raw_norm:.2f}")

    alpha = args.alpha
    if args.alpha_rel is not None:
        # The stored vector is unit-norm, so alpha IS the injected norm -- an ABSOLUTE
        # quantity, and hidden states are not the same size across models or layers.
        # Measured per-token on answer tokens: crystallm layer 14 has |h| = 165.8 while
        # llamat layer 24 has |h| = 22.2, so alpha 40 is 24% of the residual stream for
        # one and 180% for the other -- it overwrites rather than steers, and llamat
        # degenerates into repetition. raw_norm (the class-mean difference before
        # normalising) sits at ~10% of |h| in BOTH models at every layer measured, so it
        # is the portable unit: --alpha-rel 1 means "one class separation".
        # Rounded, and the ROUNDED value is what gets injected, so the stem and the
        # actual perturbation agree exactly. raw_norm is a float32 estimate, so
        # 2 x 0.5770869255065918 would otherwise put "alpha1.1541738510131836" in a
        # filename that four stores join on. 3 decimals shifts the injection by at most
        # 0.0005 -- far below the uncertainty in raw_norm itself.
        alpha = round(args.alpha_rel * raw_norm, 3)
        print(f"  --alpha-rel {args.alpha_rel:g} x raw_norm {raw_norm:.4f} "
              f"-> alpha {alpha}")
    print(f"Method=linear  alpha={alpha}  layer={args.layer}")
    # NOT :g -- that renders 8.0 as "8" and the stem is a join key across four stores
    # (CLAUDE.md: "Renaming one breaks the joins"). Existing runs are alpha8.0.
    return linear_hook(steer_vec, alpha, device), f"alpha{alpha}"


def build_pca_centroid(args, device):
    """(hook, filename stem suffix) for the PCA-subspace centroid method."""
    if args.target is None:
        raise SystemExit("--method pca_centroid needs --target")
    from compute_centroid_target import load_pca
    # args.model, NOT the default: the PCA basis lives under
    # steering_vectors/<model>/pca_centroid/, and a basis fitted on crystallm's 1024-dim
    # activations is not merely wrong for llamat2-cif, it is the wrong shape.
    mean, comps = load_pca(args.layer, args.k, args.model)
    pca_dir = steering_vectors_dir(args.model, "pca_centroid")
    cen_path = (pca_dir / args.steering_property /
                f"layer{args.layer}_k{args.k}_target{args.target:g}.parquet")
    if not cen_path.exists():
        raise FileNotFoundError(
            f"No centroid at {cen_path} — run compute_centroid_target.py --target "
            f"{args.target:g} --property {args.steering_property}")
    row = pd.read_parquet(cen_path).iloc[0]
    centroid_pca = np.asarray(row["centroid_pca"], dtype=np.float32)
    print(f"Centroid [{args.steering_property}] {cen_path}: target={row['target']:g}, "
          f"class n={int(row['class_size']):,} over [{row['class_lo']:.3f}, "
          f"{row['class_hi']:.3f}] (mean {row['class_mean']:.3f})")
    print(f"Method=pca_centroid  t={args.t}  k={args.k}  layer={args.layer}")
    return (pca_centroid_hook(mean, comps, centroid_pca, args.t, device),
            f"target{args.target:g}_t{args.t}_k{args.k}")


def build_pca_local(args, device):
    """(mean, components, bank) for the per-prompt local-centroid method."""
    if args.target is None:
        raise SystemExit("--method pca_local needs --target")
    from compute_centroid_target import load_pca
    # args.model, NOT the default: the PCA basis lives under
    # steering_vectors/<model>/pca_centroid/, and a basis fitted on crystallm's 1024-dim
    # activations is not merely wrong for llamat2-cif, it is the wrong shape.
    mean, comps = load_pca(args.layer, args.k, args.model)
    stem = f"layer{args.layer}_k{args.k}_target{args.target:g}"
    bank_path = (steering_vectors_dir(args.model, "pca_centroid")
                 / args.steering_property / f"{stem}_bank.parquet")
    if not bank_path.exists():
        raise FileNotFoundError(
            f"No class bank at {bank_path} -- rerun compute_centroid_target.py with "
            f"--save-bank")
    Z = np.vstack(pd.read_parquet(bank_path)["coord"].to_numpy()).astype(np.float32)
    print(f"Class bank [{args.steering_property}] {bank_path}: {Z.shape[0]:,} members "
          f"x {Z.shape[1]} dims")
    print(f"Method=pca_local  t={args.t}  k={args.k}  neighbours={args.neighbours}  "
          f"layer={args.layer}")
    return (torch.tensor(mean, device=device), torch.tensor(comps, device=device),
            torch.tensor(Z, device=device))


def build_manifold(args, device):
    """(hook, filename stem suffix) for the fitted-curve method."""
    if args.manifold is None:
        raise SystemExit("--method manifold needs --manifold <path to a fitted curve>")
    from compute_centroid_target import load_pca
    # args.model, NOT the default: the PCA basis lives under
    # steering_vectors/<model>/pca_centroid/, and a basis fitted on crystallm's 1024-dim
    # activations is not merely wrong for llamat2-cif, it is the wrong shape.
    mean, comps = load_pca(args.layer, args.k, args.model)
    m = Manifold.load(args.manifold)
    print(f"Manifold {args.manifold}: {m!r}")
    delta = args.delta
    if args.target is not None:
        # a target property value is more legible than a raw arc-length step; convert it
        # against the curve's own median position so the step means "move to 30"
        u_med = m.property_to_arc(float(m.prop[m.n_samples // 2]))
        delta = m.property_to_arc(args.target) - u_med
        print(f"  --target {args.target:g} -> delta {delta:+.3f} in arc length "
              f"(curve length {m.length:.2f})")
    print(f"Method=manifold  variant={args.variant}  delta={delta:+.3f}  "
          f"scale={args.scale:g}  layer={args.layer}  k={args.k}")
    tag = f"target{args.target:g}" if args.target is not None else f"d{delta:g}"
    tag += f"_{args.variant}"
    if args.variant == "residual" and args.scale != 1.0:
        tag += f"_s{args.scale:g}"
    return (manifold_hook(mean, comps, m, delta, device, args.scale, args.variant),
            f"{tag}_k{args.k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                 help="Path to the checkpoint directory holding the weights. Distinct from --model, which names the model in the embeddings tree.")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                        help="Registered model NAME. Picks the backend, and the "
                             "steering_vectors/<model>/ tree to read vectors from. "
                             "Distinct from --ckpt-dir, which is where the weights are.")
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"),
                        default="float16",
                        help="[llamat] weight dtype. bfloat16 needs sm_80; the V100s "
                             "here are sm_70, so float16. Ignored by crystallm.")
    parser.add_argument("--pkl", default=None,
                        help="[crystallm] corpus the CIF-prefix prompts are cut from. "
                             "Unused for unconditional generation, which has no corpus.")
    parser.add_argument("--method",
                        choices=("linear", "pca_centroid", "pca_local", "manifold"),
                        default="linear",
                        help="linear: alpha * mean-difference vector. "
                             "pca_centroid: interpolate toward one global target centroid "
                             "inside the top-K PCA subspace. "
                             "pca_local: same, but the centroid is recomputed per prompt "
                             "from the class members nearest that prompt. "
                             "manifold: slide along a curve fitted through the property's "
                             "bucket centroids, keeping the off-curve offset.")
    parser.add_argument("--manifold", default=None,
                        help="[manifold] path to a curve from fit_manifold.py")
    parser.add_argument("--variant", choices=("residual", "project", "project_nomu"),
                        default="residual",
                        help="[manifold] what to keep. residual: add the curve step to h, "
                             "keeping everything else (needs --scale to reach a useful "
                             "magnitude). project: replace the in-subspace part, dropping "
                             "h_perp. project_nomu: also drop mu.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="[manifold, residual variant] multiplies the curve step. The "
                             "step is capped by the curve's extent, so scale is what sets "
                             "the injection magnitude: ~6 matches linear alpha=40, ~12 "
                             "matches alpha=80.")
    parser.add_argument("--delta", type=float, default=0.0,
                        help="[manifold] step in ARC LENGTH along the curve. Same units "
                             "for every run, unlike a fraction. --target overrides it.")
    parser.add_argument("--neighbours", type=int, default=256,
                        help="[pca_local] class members averaged into each prompt's "
                             "local centroid")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="[linear] steering strength (positive = towards the high class). "
                             "An ABSOLUTE injected norm, so a value tuned on one model or "
                             "layer does not carry to another -- prefer --alpha-rel.")
    parser.add_argument("--alpha-rel", type=float, default=None,
                        help="[linear] steering strength as a multiple of this layer's "
                             "raw_norm (the class-mean difference before normalising). "
                             "Portable across models and layers, where --alpha is not: "
                             "raw_norm is ~10%% of the hidden-state norm everywhere "
                             "measured. Overrides --alpha.")
    parser.add_argument("--target", type=float, default=None,
                        help="[pca_centroid] target property value; picks the centroid file")
    parser.add_argument("--t", type=float, default=0.5,
                        help="[pca_centroid] interpolation fraction toward the centroid, "
                             "0 = none, 1 = snap onto it")
    parser.add_argument("--k", type=int, default=64,
                        help="[pca_centroid] size of the PCA subspace")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--steering-property", default="bandgap",
                        help="Property subdir under steering_vectors/ (linear) or "
                             "steering_vectors/pca_centroid/ (pca_centroid)")
    parser.add_argument("--n-prompts", type=int, default=0,
                        help="Number of prompts to use (0 = all)")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=None,
                        help="[llamat] nucleus sampling. The authors used "
                             "temperature=0.01, top_p=0.95 in the notebooks that "
                             "produced their published CIFs. Ignored by crystallm, "
                             "which samples with top_k only.")
    parser.add_argument("--prompt-csv", default=None,
                        help="[llamat] conditional prompts from this CSV (id, "
                             "pretty_formula, formula_sum, spacegroup_number), giving "
                             "one prompt per structure so arms pair on id. Omit for "
                             "unconditional generation, which cannot be paired. See "
                             "scripts/data/make_llamat_test_sample.py.")
    parser.add_argument("--system-index", type=int, default=0,
                        help="[llamat] which generation system prompt to use")
    parser.add_argument("--wrapper", choices=("notebook", "chatml"), default="notebook",
                        help="[llamat] prompt wrapper; notebook matches extraction")
    parser.add_argument("--paired-seed", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Seed the RNG per (id, sample) so draw k of every arm "
                             "starts from the same random state. Common random numbers: "
                             "it restores a defensible paired comparison when the prompt "
                             "is constant. OFF by default -- turning it on changes the "
                             "sampling stream, so existing crystallm results would not "
                             "reproduce byte-for-byte.")
    parser.add_argument("--with-spacegroup", action="store_true",
                        help="Include space group in prompt (recommended)")
    parser.add_argument("--results-dir", default="steering_results",
                        help="Base results dir; output goes to <results-dir>/generated_cifs "
                             "unless --out is given explicitly")
    parser.add_argument("--out", default=None,
                        help="Explicit output dir (overrides --results-dir/generated_cifs)")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Use KV-cached decoding (generate_cached). On by default; "
                             "verified byte-identical to uncached, ~1.9x faster at batch 1. "
                             "Pass --no-use-cache to fall back to the uncached path.")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backend = BACKENDS[args.model](args.ckpt_dir, device, args.torch_dtype)

    local = None
    if args.method == "linear":
        hook, run_tag = build_linear(args, device, backend)
    elif args.method == "pca_centroid":
        hook, run_tag = build_pca_centroid(args, device)
    elif args.method == "manifold":
        hook, run_tag = build_manifold(args, device)
    else:
        local = build_pca_local(args, device)
        hook = None                      # rebuilt per prompt, once its centroid is known
        run_tag = f"target{args.target:g}_t{args.t}_k{args.k}_nb{args.neighbours}"
    print(f"Model={args.model}  KV cache={'on' if args.use_cache else 'off'}")

    source = build_prompts(args)
    prompts = source.prompts()
    print(f"{len(prompts):,} prompts from {type(source).__name__}")

    split = source.split

    # The property is encoded by the output directory (per-property <results-dir>), so
    # the filename carries method/split/strength/layer. The method prefix keeps the two
    # methods' runs distinguishable in a shared directory and by stem downstream.
    out_dir = Path(args.out) if args.out else Path(args.results_dir) / "generated_cifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    sg_tag = source.tag
    prefix = {"linear": "steered", "pca_centroid": "steered_pca",
              "pca_local": "steered_pcalocal", "manifold": "steered_manifold"}[args.method]
    out_path = out_dir / f"{prefix}_{split}_{run_tag}_layer{args.layer}{sg_tag}.parquet"

    # resume: skip already-done ids
    done_ids = set()
    if out_path.exists():
        done_ids = set(pd.read_parquet(out_path, columns=["id"])["id"].tolist())
        print(f"Resuming — {len(done_ids):,} ids already done")

    pending = []

    for i, (id_, prompt) in enumerate(prompts):
        if id_ in done_ids:
            continue

        print(f"[{i+1}/{len(prompts)}] {id_}", flush=True)

        if local is not None:
            # This prompt's own neighbourhood of the class, not the class average.
            mean_t, comps_t, bank = local
            z = (backend.hidden_mean(prompt, args.layer) - mean_t) @ comps_t.T
            c_local, dist = local_centroid(z, bank, args.neighbours)
            hook = pca_centroid_hook(mean_t.cpu().numpy(), comps_t.cpu().numpy(),
                                     c_local.cpu().numpy(), args.t, device)
            if i % 100 == 0:
                print(f"    local centroid {dist:.2f} from prompt in subspace", flush=True)

        for j in range(args.n_samples):
            if args.paired_seed:
                # Common random numbers: the same (id, sample) draws the same random
                # stream in every arm, so arms differ by the hook and nothing else.
                torch.manual_seed(zlib.crc32(f"{id_}|{j}".encode()))
            raw, n_new = generate(backend, prompt, args, hook=hook)
            cif, reason = source.to_cif(raw)
            row = {
                "id":          id_,
                "sample":      j + 1,
                "cif_steered": cif,
            }
            if source.decodes:
                # Exact truncation signal. A run that used the whole budget was cut off
                # mid-structure, and the CIF cannot show that -- it is internally
                # consistent, just a smaller crystal than the model was writing. Only
                # recorded for decoding models, so crystallm keeps its three columns.
                row["n_new_tokens"] = n_new
                # Keyed on whether a decode step exists, NOT on whether the text changed:
                # a generation that decoded to nothing has raw == cif == "" and would
                # otherwise lose its reason. crystallm, where to_cif is the identity,
                # never sets these and keeps exactly the schema and size it had.
                row["raw_output"] = raw
                row["decode_reason"] = reason
            pending.append(row)

        if len(pending) >= CHECKPOINT_EVERY * args.n_samples:
            chunk = pd.DataFrame(pending)
            if out_path.exists():
                chunk = pd.concat([pd.read_parquet(out_path), chunk], ignore_index=True)
            chunk.to_parquet(out_path, index=False)
            pending = []
            print(f"  Checkpointed at prompt {i+1}", flush=True)

    if pending:
        chunk = pd.DataFrame(pending)
        if out_path.exists():
            chunk = pd.concat([pd.read_parquet(out_path), chunk], ignore_index=True)
        chunk.to_parquet(out_path, index=False)

    total = len(pd.read_parquet(out_path))
    print(f"\nDone. {total:,} rows saved to {out_path}")


if __name__ == "__main__":
    main()
