"""Shared helpers for the steering/analysis scripts.

Kept import-light so any script can `from utils import ...` (scripts/ is on
sys.path when a script is run as `python scripts/<name>.py`).
"""
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
                      "spin_polarized", "band_gap_ev", "wyckoff_letters")

# Which CIF text the embeddings were extracted from. The symmetry label is written
# verbatim into every CIF (_symmetry_space_group_name_H-M and _symmetry_Int_Tables_number),
# so "full" embeddings cannot be used to ask whether the model *represents* symmetry --
# a probe just reads the copied token back. "nosym" strips those lines before the forward
# pass, so symmetry has to be inferred from the cell and coordinates.
DEFAULT_VARIANT = "full"
VARIANTS = ("full", "nosym")

# Which slice of CrystaLLM's own train/val/test split an analysis runs on. This matters
# because 89.6% of the structures with metadata are in the model's training set and only
# 0.45% are in its test set -- results on "all" cannot distinguish learning from
# memorization. There is deliberately no default: pick one explicitly.
DATASETS = ("v1_all", "v1_mp")
PARTITIONS = ("all", "train", "val", "test")
SPLIT_INDEX_PATH = "splits_v1.parquet"
ANALYSIS_ROOT = Path("analysis")


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
RESULT_UNITS = {"band_gap": "eV", "density_atomic": "log10_A3_per_atom"}


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


def analysis_dir(dataset: str = DEFAULT_DATASET, variant: str = DEFAULT_VARIANT,
                 partition: str = "all", subdir: str = None) -> Path:
    """Output dir for an analysis run: analysis/<dataset>/<variant>/<partition>[/<subdir>].

    Created if missing. `subdir` is for scripts that nest further (e.g. "layer5").

    Pass variant=None for outputs that read metadata but never embeddings (the property
    histograms): they are identical whichever CIF variant was extracted, so they drop
    that level rather than write the same bytes under both full/ and nosym/.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS} or None, got {variant!r}")
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    path = ANALYSIS_ROOT / dataset
    if variant is not None:
        path = path / variant
    path = path / partition
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_split_index(path: str = SPLIT_INDEX_PATH) -> pd.DataFrame:
    """[id, split] for every CIF in CrystaLLM's train/val/test split.

    Built by scripts/data/build_split_index.py from the three cifs_v1_*.pkl.gz files.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found -- run: python scripts/data/build_split_index.py")
    return pd.read_parquet(path)


def filter_partition(df: pd.DataFrame, partition: str, verbose: bool = True) -> pd.DataFrame:
    """Restrict a frame with an `id` column to one CrystaLLM split.

    partition="all" is a no-op. Ids missing from the split index are treated as
    "unknown" and dropped by train/val/test: the MP corpus carries ~96k CIFs that
    dedup removed before the split was made, so they belong to no partition.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if partition == "all":
        return df
    keep = set(load_split_index().query("split == @partition")["id"])
    out = df[df["id"].isin(keep)].reset_index(drop=True)
    if verbose:
        print(f"  partition={partition}: {len(out):,} of {len(df):,} rows kept")
    if out.empty:
        raise SystemExit(f"No rows left after filtering to partition={partition!r}.")
    return out


def add_partition_args(parser):
    """Attach the --dataset / --variant / --partition trio to an argparse parser.

    --partition is required on purpose so no analysis silently runs on the model's
    own training data.
    """
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS),
                        help="which CIF text the embeddings came from (see VARIANTS)")
    parser.add_argument("--partition", required=True, choices=list(PARTITIONS),
                        help="which slice of CrystaLLM's train/val/test split to analyse")
    return parser


def embeddings_paths(layer: int, dataset: str = DEFAULT_DATASET,
                     variant: str = DEFAULT_VARIANT):
    """Candidate (single_file, checkpoint_dir) for a dataset+variant+layer under embeddings/."""
    base = EMBEDDINGS_ROOT / dataset / variant
    return base / f"cif_layer{layer}.parquet", base / f"cif_layer{layer}"


def load_embeddings(layer: int, dataset: str = DEFAULT_DATASET,
                    columns=("id", "embedding"),
                    variant: str = DEFAULT_VARIANT) -> pd.DataFrame:
    """Load mean-pooled embeddings for a layer from embeddings/<dataset>/<variant>/.

    Uses the single cif_layer{N}.parquet if present, else concatenates the
    checkpoint_*.parquet / batch_*.parquet shards in cif_layer{N}/. Pass
    columns=None to read every column. See VARIANTS for what `variant` means.
    """
    cols = list(columns) if columns is not None else None
    single, ckpt = embeddings_paths(layer, dataset, variant)
    if single.exists():
        return pd.read_parquet(single, columns=cols)
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No embeddings for layer {layer} in dataset '{dataset}', variant '{variant}': "
            f"looked for {single} and {ckpt}/checkpoint_*.parquet")
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def load_labeled_embeddings(layer: int, dataset: str = DEFAULT_DATASET,
                            metadata_path: str = DEFAULT_METADATA,
                            label_cols=DEFAULT_LABEL_COLS,
                            verbose: bool = True,
                            variant: str = DEFAULT_VARIANT) -> pd.DataFrame:
    """Embeddings for a layer, inner-joined with metadata labels on `id`.

    Returns a frame of [id, embedding, *label_cols] restricted to ids present in
    both sources, in embedding order. Label columns absent from the metadata file
    are silently skipped (metadata.parquet and metadata_mp.parquet carry different
    ones). The embeddings cover NOMAD+OQMD+MP while metadata.parquet is NOMAD-only,
    so the intersection is the NOMAD subset.
    """
    if verbose:
        print("Loading embeddings...")
    emb_df = load_embeddings(layer, dataset=dataset, variant=variant)
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
