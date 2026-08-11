"""Shared helpers for the steering/analysis scripts.

Kept import-light so any script can `from utils import ...` (scripts/ is on
sys.path when a script is run as `python scripts/<name>.py`).
"""
from pathlib import Path

import pandas as pd

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
