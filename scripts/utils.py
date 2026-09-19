"""Shared helpers for the steering/analysis scripts.

Kept import-light so any script can `from utils import ...` (scripts/ is on
sys.path when a script is run as `python scripts/<name>.py`).
"""
import os
import re
from pathlib import Path

import pandas as pd

from pymatgen.core.operations import SymmOp as _SymmOp
if not hasattr(_SymmOp, "as_xyz_string"):
    _SymmOp.as_xyz_string = _SymmOp.as_xyz_str

# Duplicate of CrystaLLM's bin/postprocess.py:postprocess. It cannot be imported:
# that module does `from crystallm import ...`, which runs crystallm/__init__.py and
# pulls in omegaconf plus a pymatgen version megnet_venv does not have -- and
# compute_predictions.py runs in megnet_venv. _utils.py is loaded by file path for the
# same reason, the same idiom relax_steered_cifs.py uses. Assumes repo root is the cwd.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("cryst_utils", "CrystaLLM/crystallm/_utils.py")
_cryst_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cryst_utils)
extract_space_group_symbol = _cryst_utils.extract_space_group_symbol
replace_symmetry_operators = _cryst_utils.replace_symmetry_operators
remove_atom_props_block = _cryst_utils.remove_atom_props_block


# CrystaLLM's replace_symmetry_operators finds the placeholder identity-operator block
# with a literal regex that assumes no indentation -- true of CIFs its own model writes,
# false of pymatgen's indented reference CIFs. On a miss re.sub returns the text
# unchanged and nothing raises, leaving the structure to parse as the asymmetric unit
# alone (wrong atom count and formula). Normalise the block first so the swap fires
# for both sources.
_IDENTITY_OPS = re.compile(
    r"loop_[ \t]*\n[ \t]*_symmetry_equiv_pos_site_id[ \t]*\n"
    r"[ \t]*_symmetry_equiv_pos_as_xyz[ \t]*\n[ \t]*1[ \t]+'x,\s*y,\s*z'")
_CANONICAL_OPS = ("loop_\n_symmetry_equiv_pos_site_id\n"
                  "_symmetry_equiv_pos_as_xyz\n1 'x, y, z'")
# P1 is written both with and without the space, and needs no expansion either way.
_NO_SYMMETRY = ("P 1", "P1")


def restore_symmetry_operators(cif: str, space_group_symbol: str) -> str:
    """replace_symmetry_operators, but it also works on indented CIFs.

    Raises if the substitution did not take, instead of returning the input unchanged.
    Call this anywhere the raw CrystaLLM function would otherwise be used.
    """
    if space_group_symbol is None or space_group_symbol in _NO_SYMMETRY:
        return cif                      # P1: the identity operator is the whole story
    out = replace_symmetry_operators(_IDENTITY_OPS.sub(_CANONICAL_OPS, cif),
                                     space_group_symbol)
    # A missed substitution is silent, which is how the reference CIFs went unexpanded
    # for so long. Treat leftover identity ops as the failure they are.
    if _IDENTITY_OPS.search(out):
        raise ValueError(f"symmetry operators for '{space_group_symbol}' were not "
                         f"substituted -- CIF still has identity only")
    return out


def postprocess(cif: str, fname: str) -> str:
    try:
        # replace the symmetry operators with the correct operators
        cif = restore_symmetry_operators(cif, extract_space_group_symbol(cif))

        # remove atom props
        cif = remove_atom_props_block(cif)
    except Exception as e:
        cif = "# WARNING: CrystaLLM could not post-process this file properly!\n" + cif
        print(f"error post-processing CIF file '{fname}': {e}")

    return cif


# --- scalar reads straight off the CIF text -----------------------------------
# Both `_cell_volume` and `_chemical_formula_sum` are written for the FULL cell, so
# these need no symmetry expansion and no pymatgen parse -- which is what makes them
# cheap enough to run over the whole 2M-structure corpus.
_CELL_VOLUME_RE = re.compile(r"_cell_volume\s+([-\d.eE]+)")
# Single-element formulas have no space, so pymatgen writes them unquoted
# (`_chemical_formula_sum   Mn4`). Match both forms or elemental structures drop out.
_FORMULA_SUM_RE = re.compile(r"_chemical_formula_sum\s+(?:'([^']+)'|(\S+))")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def cell_volume_from_text(cif: str) -> float:
    """The `_cell_volume` token, in A^3. NaN if absent or unparseable."""
    m = _CELL_VOLUME_RE.search(cif) if isinstance(cif, str) else None
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


def natoms_from_text(cif: str) -> float:
    """Atom count summed from `_chemical_formula_sum`. NaN if absent or empty."""
    m = _FORMULA_SUM_RE.search(cif) if isinstance(cif, str) else None
    if not m:
        return float("nan")
    n = sum(int(cnt or 1) for el, cnt in _ELEMENT_RE.findall(m.group(1) or m.group(2)) if el)
    return float(n) if n else float("nan")


def density_atomic_from_text(cif: str) -> float:
    """Volume per atom (A^3/atom) -- the same quantity as MP's `density_atomic`."""
    n = natoms_from_text(cif)
    return cell_volume_from_text(cif) / n if n == n else float("nan")


EMBEDDINGS_ROOT = Path("embeddings")
DEFAULT_DATASET = "v1_all"   # combined NOMAD+OQMD+MP corpus (cifs_v1_prep / tokens_v1_all)
DEFAULT_METADATA = "metadata.parquet"
DEFAULT_LABEL_COLS = ("point_group", "space_group_symbol", "structural_type",
                      "spin_polarized", "band_gap_ev", "wyckoff_letters",
                      "wyckoff_sites")

# Which CIF text the embeddings were extracted from. The symmetry label is written
# verbatim into every CIF (_symmetry_space_group_name_H-M and _symmetry_Int_Tables_number),
# so "full" embeddings cannot be used to ask whether the model *represents* symmetry --
# a probe just reads the copied token back. "nosym" strips those lines before the forward
# pass, so symmetry has to be inferred from the cell and coordinates.
#
# Variants are MODEL-SPECIFIC in meaning. full/nosym describe CIF text and belong to
# crystallm; crystal_uncond belongs to llamat2_cif and is not a CIF at all -- it is the
# compact crystal string the model generates, placed in the answer slot of the
# unconditional generation prompt (see scripts/llamat_prompts.py for why). The tuple is
# flat because <model> already separates the trees, so an unused pairing is just an
# empty directory rather than a collision.
DEFAULT_VARIANT = "full"
VARIANTS = ("full", "nosym", "crystal_uncond")

# Which slice of a model's own train/val/test split an analysis runs on. This matters
# because 89.6% of the structures with metadata are in CrystaLLM's training set and only
# 0.45% are in its test set -- results on "all" cannot distinguish learning from
# memorization. There is deliberately no default: pick one explicitly.
DATASETS = ("v1_all", "v1_mp")
PARTITIONS = ("all", "train", "val", "test", "not_heldout")
ANALYSIS_ROOT = Path("analysis")

# A SPLIT BELONGS TO A MODEL, not to the corpus. CrystaLLM's train/val/test says nothing
# about what LLaMat held out, so scoring llamat2_cif on CrystaLLM's "val" would evaluate
# it on rows it may well have trained on.
#
#   crystallm_splits_v1  flattened from CrystaLLM's own three split pickles.
#   llamat_splits_v1     the ONLY evidence we have about LLaMat: the 9,046 materials in
#                        llamat/src/cifs/crystal-text-llm/data/test.csv. It lists nothing
#                        else, deliberately -- we do not know what LLaMat trained on, only
#                        what it held out. `not_heldout` is defined by EXCLUSION, so with
#                        just those test rows present it resolves to "everything we have
#                        no evidence was held out", which is the honest fitting pool.
#                        `train` and `val` are empty for this model by design.
#
# Both files share the (id, split) schema. The base model and its CIF finetune share a
# split because the finetune inherited that corpus.
SPLIT_FILES = {
    "crystallm": "crystallm_splits_v1.parquet",
    "llamat2": "llamat_splits_v1.parquet",
    "llamat2_cif": "llamat_splits_v1.parquet",
}
SPLIT_INDEX_PATH = SPLIT_FILES["crystallm"]     # back-compat default

# Which CIF pickle a dataset's `data_` header formulas come from. The probe and the
# separability test both read formulas to build their composition controls, and both used
# to hardcode the v1_all pickle -- on a v1_mp run that silently joins to nothing.
DATASET_PKL = {
    "v1_all": "CrystaLLM/cifs_v1_prep.pkl.gz",
    "v1_mp": "CrystaLLM/cifs_v1_mp.pkl.gz",
}

# Which model produced the hidden states. This sits between <dataset> and <variant> in
# the embeddings tree, because the same corpus read by two models gives two unrelated
# sets of vectors -- same ids, same CIF text, different dimensionality and no shared
# basis. Without this level the second model would overwrite the first layer by layer.
#
# The models differ in more than weights, and the differences decide what is comparable:
#
#   crystallm  16 blocks, 1024-dim, its own CIF tokenizer (one token per CIF field).
#   llamat2    LLaMA-2 7B continued-pretrained on materials text (m3rg-iitd), so 32
#              blocks, 4096-dim, and a general BPE tokenizer that splits a number like
#              4.2317 across several tokens.
#   llamat2_cif  llamat2 further instruction-tuned on CIF tasks. Same architecture,
#              DIFFERENT WEIGHTS, so it gets its own name -- sharing llamat2's would
#              silently mix two models' vectors in one directory.
#
# So a layer index means different things in each ("layer 7" is 7/16 of the way through
# one and 7/32 of the other), and a steering vector, PCA basis or manifold fitted on one
# cannot be applied to the other. Nothing in this repo mixes two models in one artifact;
# the path keeps that honest. Add a new name here to register it.
DEFAULT_MODEL = "crystallm"
MODELS = ("crystallm", "llamat2", "llamat2_cif")

# Not a model: the analysis/ subtree for outputs that never load one. Corpus property
# histograms, Wyckoff counts, MP hull coverage -- identical whichever model runs, so they
# are written once rather than duplicated under every model. Accepted by analysis_dir()
# but NOT by add_partition_args, since no script can load "corpus" as weights.
CORPUS_DIR = "corpus"

# How each registered model is spelled in a figure title. MODELS holds filesystem-safe
# names; a plot hardcoding one model's name mislabels every other model's figure, which
# is worse than an ugly axis because a wrong label survives being pasted into a document.
DISPLAY_NAME = {"crystallm": "CrystaLLM", "llamat2": "LLaMat-2",
                "llamat2_cif": "LLaMat-2-CIF", CORPUS_DIR: "corpus"}


def display_name(model: str) -> str:
    return DISPLAY_NAME.get(model, model)


# One schema for every steering results table under analysis/<dataset>/<partition>/.
# These files accumulated four different shapes -- `median` meant A^3/atom in one and
# log10 in another, the run was keyed by `run` here and `method`+`t` there -- which made
# them impossible to read side by side. Every results table now leads with these columns,
# in this order, and may append extras after them.
#
#   identity     what was run. `strength` is alpha for linear, t for the pca methods,
#                so one column orders every method's sweep. `family` is the prompt form
#                -- sg or nosg -- and belongs to identity because the two are different
#                prompt sets and a run may only be paired against a control of its own.
#   population   valid_pct is of all generated samples; n_prompts is how many survived
#                to contribute a point; n_paired is how many the control also has.
#   value        `unit` names the scale, so a median is never ambiguous. mean_diff and
#                median are always in that unit.
#   stats        cohens_d first: with ~1000 paired prompts a 0.5% shift reaches
#                p=1e-20, so the p-values rank runs but only d says whether one matters.
RESULT_COLUMNS = [
    "property", "method", "layer", "family", "target", "strength", "source", "agg", "run",
    "valid_pct", "n_prompts", "n_paired",
    "unit", "control_median", "median", "mean_diff", "frac_of_target_move",
    "cohens_d", "p_paired", "p_holm", "p_wilcoxon",
]
RESULT_UNITS = {"band_gap": "eV", "density_atomic": "log10_A3_per_atom",
                "energy_above_hull": "eV_per_atom",
                "formation_energy_per_atom": "eV_per_atom"}


def write_results_table(df: pd.DataFrame, path) -> Path:
    """Write a steering results table in the canonical column order.

    Raises on a missing core column rather than writing a table that cannot be
    compared with the others. Extra columns are kept, after the core ones.
    """
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"results table is missing core columns: {missing}")
    extras = [c for c in df.columns if c not in RESULT_COLUMNS]
    path = Path(path)
    df[RESULT_COLUMNS + extras].to_csv(path, index=False, float_format="%.6g")
    return path


# --- steering_results layout -------------------------------------------------------
STEERING_ROOT = "steering_results"
BASELINE_DIR = "baseline"
# Subdirs a control keeps under baseline/ rather than duplicating per property. At
# alpha=0 the hook adds exactly zero, so the CIFs, whether they parse, and their M3GNet
# relaxation are all property-independent -- one copy serves everything.
#
# property_predictions is included even though its VALUES are property-specific: a single
# file accumulates density_atomic, band_gap, energy_above_hull... side by side, because
# compute_predictions.py preserves columns it does not own. Sharing this directory once
# caused two controls from different prompt sets to collide on (family, strength);
# plot_steering_distribution_shift.pick_control now disambiguates by prompt-set overlap,
# which is the real key, so the collision cannot recur.
SHARED_SUBDIRS = ("generated_cifs", "validation", "relaxed", "property_predictions",
                  "embeddings")


def steering_path(results_dir: str, sub: str, stem: str) -> str:
    """Path to one run's file, falling back to the shared baseline tree.

    Looks in the property's own tree first, then baseline/ -- but only for subdirs that
    are actually shared. Asking for property_predictions here raises rather than silently
    resolving to a file that belongs to another property.
    """
    own = os.path.join(STEERING_ROOT, results_dir, sub, stem)
    if os.path.exists(own) or sub not in SHARED_SUBDIRS:
        if sub not in SHARED_SUBDIRS and not os.path.exists(own):
            raise FileNotFoundError(
                f"{own} not found, and {sub!r} is not shared so there is no baseline "
                f"fallback (shared: {', '.join(SHARED_SUBDIRS)})")
        return own
    shared = os.path.join(STEERING_ROOT, BASELINE_DIR, sub, stem)
    return shared if os.path.exists(shared) else own


def analysis_dir(dataset: str = DEFAULT_DATASET, variant: str = DEFAULT_VARIANT,
                 partition: str = "all", subdir: str = None,
                 model: str = DEFAULT_MODEL) -> Path:
    """Output dir for an analysis run:
    analysis/<model>/<dataset>[/<variant>]/<partition>[/<subdir>].

    Created if missing. `subdir` is for scripts that nest further (e.g. "layer5").

    THE MODEL LEVEL COMES FIRST here, while embeddings/ puts it between dataset and
    variant. That asymmetry is deliberate: an analysis directory is a per-model
    deliverable, so reading one model's results should not mean walking two dataset
    trees. embeddings/ has the opposite pressure -- a layer's shards for one corpus are
    read together regardless of model.

    Pass model=CORPUS_DIR for outputs that never touch a model at all: corpus property
    histograms, Wyckoff/space-group counts, MP hull coverage. Those are byte-identical
    whichever model is loaded, so they live under analysis/corpus/ rather than being
    regenerated once per model.

    `variant=None` drops the variant level, for outputs not tied to one CIF text. It is
    INDEPENDENT of `model` and does not imply corpus: the steering tables pass
    variant=None -- they come from generation, not from reading a CIF variant -- and are
    entirely model-dependent. Inferring one from the other would file every steering
    result under corpus/.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    if model not in MODELS and model != CORPUS_DIR:
        raise ValueError(f"model must be one of {MODELS} or {CORPUS_DIR!r}, got {model!r}")
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS} or None, got {variant!r}")
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    path = ANALYSIS_ROOT / model / dataset
    if variant is not None:
        path = path / variant
    path = path / partition
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


STEERING_VECTORS_ROOT = Path("steering_vectors")


def steering_vectors_dir(model: str = DEFAULT_MODEL, sub: str = None) -> Path:
    """steering_vectors/<model>[/<sub>] -- linear vectors, PCA bases, fitted manifolds.

    Every artifact under here is FITTED ON one model's activations and is meaningless
    for another: the hidden sizes differ (crystallm 1024, llamat2_cif 4096), a layer
    index means a different depth, and a PCA basis or manifold lives in a subspace of
    the model it was built from. Without the model level a llamat2_cif
    pca_layer8_k32.parquet would overwrite crystallm's at the identical path.

    `sub` is "manifolds", "pca_centroid", or a property name -- the layout inside is
    unchanged, so only the root moved.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    path = STEERING_VECTORS_ROOT / model
    if sub:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_root(model: str = DEFAULT_MODEL) -> Path:
    """analysis/<model>/ -- for outputs that span datasets rather than sitting in one.

    The cross-dataset summaries (probe R^2 by layer, manifold curve overlays) and the
    corpus-wide data checks (MP hull coverage, metadata property coverage) used to sit
    loose at the top of analysis/. They still need a model level, or the crystallm and
    llamat2_cif versions of a summary collide on one filename.

    Pass model=CORPUS_DIR for the ones that never load a model.
    """
    if model not in MODELS and model != CORPUS_DIR:
        raise ValueError(f"model must be one of {MODELS} or {CORPUS_DIR!r}, got {model!r}")
    path = ANALYSIS_ROOT / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_path(model: str = DEFAULT_MODEL) -> str:
    """Which split file belongs to a model. See SPLIT_FILES for why this is per-model."""
    if model not in SPLIT_FILES:
        raise ValueError(f"no split registered for model {model!r}; "
                         f"known: {sorted(SPLIT_FILES)}")
    return SPLIT_FILES[model]


def load_split_index(path: str = None, model: str = DEFAULT_MODEL) -> pd.DataFrame:
    """[id, split] for one model's own train/val/test split.

    `path` overrides; otherwise the file is chosen by `model`. crystallm's is built by
    scripts/data/build_split_index.py from the three cifs_v1_*.pkl.gz files; llamat's by
    scripts/data/build_llamat_split.py from the crystal-text-llm test set.
    """
    path = path or split_path(model)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found -- run the matching builder in scripts/data/")
    return pd.read_parquet(path)


def partition_id_sets(partition: str, model: str = DEFAULT_MODEL):
    """(keep, drop) id sets for a partition. Exactly one is not None; "all" gives both None.

    train/val/test are defined by MEMBERSHIP, so they yield a `keep` set. not_heldout is
    defined by EXCLUSION -- everything that is not val or test -- and the ids it keeps
    include structures absent from the split index entirely (the ~96k unpartitioned MP
    CIFs). No keep-set can enumerate those, so it yields a `drop` set instead.

    Use this wherever ids are filtered while STREAMING and there is no labels frame to
    hand to filter_partition. Treating not_heldout as a keep-set silently collapses it to
    train: that bug shipped in five scripts at once and cost 96k structures per run.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if partition == "all":
        return None, None
    sp = load_split_index(model=model)
    if partition == "not_heldout":
        return None, set(sp.query("split in ['val', 'test']")["id"])
    return set(sp.query("split == @partition")["id"]), None


def filter_partition(df: pd.DataFrame, partition: str, verbose: bool = True,
                     model: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Restrict a frame with an `id` column to one CrystaLLM split.

    partition="all" is a no-op. Ids missing from the split index are treated as
    "unknown" and dropped by train/val/test: the MP corpus carries ~96k CIFs that
    belong to no partition. (Measured 2026-09-05: 77.5% of those have a reduced
    formula that appears somewhere in the corpus anyway, so they are mostly other
    polymorphs of seen compositions rather than unseen chemistry.)

    partition="not_heldout" keeps everything EXCEPT val and test -- train plus the
    unpartitioned remainder. It exists for fitting on v1_mp, where restricting to
    `train` throws away the ~96k unpartitioned structures for no benefit: none of them
    are evaluated against, so excluding them only costs sample size. Use it for fitting
    steering artifacts, never for scoring.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if partition == "all":
        return df
    if partition == "not_heldout":
        sp = load_split_index(model=model)
        drop = set(sp.query("split in ['val', 'test']")["id"])
        out = df[~df["id"].isin(drop)].reset_index(drop=True)
        if verbose:
            print(f"  partition=not_heldout: {len(out):,} of {len(df):,} rows kept "
                  f"({len(df) - len(out):,} val/test dropped)")
        if out.empty:
            raise SystemExit("No rows left after filtering to partition='not_heldout'.")
        return out
    keep = set(load_split_index(model=model).query("split == @partition")["id"])
    out = df[df["id"].isin(keep)].reset_index(drop=True)
    if verbose:
        print(f"  partition={partition}: {len(out):,} of {len(df):,} rows kept")
    if out.empty:
        raise SystemExit(f"No rows left after filtering to partition={partition!r}.")
    return out


def add_partition_args(parser):
    """Attach the --dataset / --model / --variant / --partition set to an argparse parser.

    --partition is required on purpose so no analysis silently runs on the model's
    own training data.

    --model defaults to crystallm, unlike --partition, because the default is right
    rather than merely convenient: every run that predates the model level was
    crystallm, so the old behaviour and the new default are the same thing. A wrong
    --partition silently answers a different question; a wrong --model just fails to
    find the files, and the path in the error names the model it looked under.
    """
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS),
                        help="which model's hidden states to read (see MODELS)")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS),
                        help="which CIF text the embeddings came from (see VARIANTS)")
    parser.add_argument("--partition", required=True, choices=list(PARTITIONS),
                        help="which slice of CrystaLLM's train/val/test split to analyse")
    return parser


def embeddings_paths(layer: int, dataset: str = DEFAULT_DATASET,
                     variant: str = DEFAULT_VARIANT, model: str = DEFAULT_MODEL):
    """Candidate (single_file, checkpoint_dir) under embeddings/<dataset>/<model>/<variant>/.

    `model` is validated against MODELS rather than passed through. An unregistered name
    would otherwise create a sibling tree -- embeddings/v1_all/llamat-2/full/ next to
    embeddings/v1_all/llamat2/full/ -- and the second run would report zero rows done and
    quietly re-extract everything into the typo.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    base = EMBEDDINGS_ROOT / dataset / model / variant
    return base / f"cif_layer{layer}.parquet", base / f"cif_layer{layer}"


def embedding_files(layer: int, dataset: str = DEFAULT_DATASET,
                    variant: str = DEFAULT_VARIANT, model: str = DEFAULT_MODEL) -> list:
    """The consolidated parquet for a layer if it exists, else the checkpoint shards.

    For streaming readers, which want paths rather than one concatenated frame. This was
    copied verbatim into manifold.py, compute_pca_basis.py and compute_steering_vector.py;
    those now re-export this one so the model level only had to be added once.
    """
    single, ckpt = embeddings_paths(layer, dataset, variant, model)
    if single.exists():
        return [single]
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No embeddings for layer {layer} (dataset={dataset}, model={model}, "
            f"variant={variant}): looked for {single} and {ckpt}/checkpoint_*.parquet")
    return files


def load_embeddings(layer: int, dataset: str = DEFAULT_DATASET,
                    columns=("id", "embedding"),
                    variant: str = DEFAULT_VARIANT,
                    model: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Mean-pooled embeddings for a layer from embeddings/<dataset>/<model>/<variant>/.

    Uses the single cif_layer{N}.parquet if present, else concatenates the
    checkpoint_*.parquet / batch_*.parquet shards in cif_layer{N}/. Pass
    columns=None to read every column. See VARIANTS for what `variant` means and
    MODELS for what `model` means.
    """
    cols = list(columns) if columns is not None else None
    single, ckpt = embeddings_paths(layer, dataset, variant, model)
    if single.exists():
        return pd.read_parquet(single, columns=cols)
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No embeddings for layer {layer} in dataset '{dataset}', model '{model}', "
            f"variant '{variant}': looked for {single} and {ckpt}/checkpoint_*.parquet")
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def load_labeled_embeddings(layer: int, dataset: str = DEFAULT_DATASET,
                            metadata_path: str = DEFAULT_METADATA,
                            label_cols=DEFAULT_LABEL_COLS,
                            verbose: bool = True,
                            variant: str = DEFAULT_VARIANT,
                            model: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Embeddings for a layer, inner-joined with metadata labels on `id`.

    Returns a frame of [id, embedding, *label_cols] restricted to ids present in
    both sources, in embedding order. Label columns absent from the metadata file
    are silently skipped (metadata.parquet and metadata_mp.parquet carry different
    ones). The embeddings cover NOMAD+OQMD+MP while metadata.parquet is NOMAD-only,
    so the intersection is the NOMAD subset.
    """
    if verbose:
        print("Loading embeddings...")
    emb_df = load_embeddings(layer, dataset=dataset, variant=variant, model=model)
    if verbose:
        print(f"  Embeddings: {len(emb_df):,} entries")
        print("Loading metadata...")

    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(metadata_path).schema_arrow.names)
    cols = [c for c in label_cols if c in available]
    meta_df = pd.read_parquet(metadata_path, columns=["id"] + cols)
    if verbose:
        print(f"  Metadata:   {len(meta_df):,} entries")

    common_ids = set(emb_df["id"]) & set(meta_df["id"])
    if verbose:
        print(f"  Intersection: {len(common_ids):,} entries")

    df = emb_df[emb_df["id"].isin(common_ids)].reset_index(drop=True)
    meta_df = meta_df[meta_df["id"].isin(common_ids)].set_index("id")
    for col in cols:
        df[col] = df["id"].map(meta_df[col])
    return df
