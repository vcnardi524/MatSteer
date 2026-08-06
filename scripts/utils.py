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


def embeddings_paths(layer: int, dataset: str = DEFAULT_DATASET):
    """Candidate (single_file, checkpoint_dir) for a dataset+layer under embeddings/."""
    base = EMBEDDINGS_ROOT / dataset
    return base / f"cif_layer{layer}.parquet", base / f"cif_layer{layer}"


def load_embeddings(layer: int, dataset: str = DEFAULT_DATASET,
                    columns=("id", "embedding")) -> pd.DataFrame:
    """Load mean-pooled embeddings for a layer from embeddings/<dataset>/.

    Uses the single cif_layer{N}.parquet if present, else concatenates the
    checkpoint_*.parquet / batch_*.parquet shards in cif_layer{N}/. Pass
    columns=None to read every column.
    """
    cols = list(columns) if columns is not None else None
    single, ckpt = embeddings_paths(layer, dataset)
    if single.exists():
        return pd.read_parquet(single, columns=cols)
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No embeddings for layer {layer} in dataset '{dataset}': "
            f"looked for {single} and {ckpt}/checkpoint_*.parquet")
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def load_labeled_embeddings(layer: int, dataset: str = DEFAULT_DATASET,
                            metadata_path: str = DEFAULT_METADATA,
                            label_cols=DEFAULT_LABEL_COLS,
                            verbose: bool = True) -> pd.DataFrame:
    """Embeddings for a layer, inner-joined with metadata labels on `id`.

    Returns a frame of [id, embedding, *label_cols] restricted to ids present in
    both sources, in embedding order. Label columns absent from the metadata file
    are silently skipped (metadata.parquet and metadata_mp.parquet carry different
    ones). The embeddings cover NOMAD+OQMD+MP while metadata.parquet is NOMAD-only,
    so the intersection is the NOMAD subset.
    """
    if verbose:
        print("Loading embeddings...")
    emb_df = load_embeddings(layer, dataset=dataset)
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
