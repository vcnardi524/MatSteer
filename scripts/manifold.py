"""A 1-D manifold through property-bucket centroids, and the pieces that build one.

Every steering method in this repo so far moves in a straight line -- `linear` adds a
fixed direction, `pca_centroid` interpolates toward a class centroid -- and both measure
as doing nothing. Two of our own results say why: the property path is ~2x longer than
the straight line through it (centroid_pca_plots.py), and aiming at the class centroid
lands at Mahalanobis 0, the empty middle of a 64-dimensional shell (manifold_distance.py).

So: fit the curve, and slide along it instead of cutting across it.

    m = Manifold.load(path)
    u, r = m.encode(z)                  # where on the curve, and the offset from it
    z_new = m.decode(u + delta) + r     # slide along, keep the offset

`encode` returns the residual as a VECTOR, not a distance. That is the whole point --
adding it back means the steered state stays exactly as far off the curve as it started,
so it stays in the shell where real activations live.

The manifold is stored as a dense polyline, not spline coefficients. Fitting uses scipy;
inference never does, because the steering hook runs at every token of every generation
step and a scipy call there would sync GPU->CPU thousands of times per CIF.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from utils import embeddings_paths


def embedding_files(layer: int, dataset: str, variant: str) -> list:
    """The consolidated parquet for a layer if it exists, else the checkpoint shards."""
    single, ckpt = embeddings_paths(layer, dataset, variant)
    if single.exists():
        return [single]
    files = sorted(ckpt.glob("checkpoint_*.parquet")) + sorted(ckpt.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No embeddings for layer {layer} at {single} or {ckpt}/")
    return files


def bucket_centroids(labels, prop, width, layer, dataset, variant, batch_size,
                     sample_ids=None):
    """Per-bucket (sum, count) by streaming the layer, plus the raw rows for sample_ids.

    Buckets are [x, x+width). The sample is drawn as ids up front rather than
    reservoir-sampled, so one pass builds the centroids and collects the individual
    points a plot draws behind them.
    """
    idx = np.floor(labels[prop].to_numpy() / width).astype(np.int64)
    bucket_of = dict(zip(labels["id"].to_numpy(), idx))
    want = set() if sample_ids is None else set(sample_ids)

    sums, counts, s_vec, s_id = {}, {}, [], []
    for path in embedding_files(layer, dataset, variant):
        for rb in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                    columns=["id", "embedding"]):
            df = rb.to_pandas()
            b = df["id"].map(bucket_of)
            df = df[b.notna()]
            if df.empty:
                continue
            b = b[b.notna()].to_numpy().astype(np.int64)
            X = np.vstack(df["embedding"].to_numpy()).astype(np.float64)
            for u in np.unique(b):
                m = b == u
                sums[u] = sums.get(u, 0) + X[m].sum(0)
                counts[u] = counts.get(u, 0) + int(m.sum())
            if want:
                m = df["id"].isin(want).to_numpy()
                if m.any():
                    s_vec.append(X[m])
                    s_id.extend(df["id"].to_numpy()[m])
    S = np.vstack(s_vec) if s_vec else np.empty((0, 1024))
    return sums, counts, S, s_id


class Manifold:
    """A curve in PCA subspace coordinates, stored as a dense polyline.

    samples  (M, k)  points along the curve, ordered
    arc      (M,)    cumulative arc length at each sample -- the intrinsic coordinate
    prop     (M,)    the property value the curve passes through at each sample

    Arc length rather than the property value is the intrinsic coordinate because the
    buckets are dense at the distribution's mode and sparse in its tails; parameterising
    by property would make a step mean a different distance in different places.
    """

    def __init__(self, samples, arc, prop, meta=None):
        self.samples = torch.as_tensor(np.asarray(samples), dtype=torch.float32)
        self.arc = torch.as_tensor(np.asarray(arc), dtype=torch.float32).flatten()
        self.prop = torch.as_tensor(np.asarray(prop), dtype=torch.float32).flatten()
        self.meta = dict(meta or {})
        if self.samples.shape[0] != self.arc.shape[0]:
            raise ValueError(f"{self.samples.shape[0]} samples but {self.arc.shape[0]} arc values")

    def __repr__(self):
        return (f"Manifold({self.n_samples} samples, k={self.k}, "
                f"arc 0..{float(self.arc[-1]):.2f}, "
                f"{self.meta.get('property', '?')} "
                f"{float(self.prop[0]):.3g}..{float(self.prop[-1]):.3g})")

    @property
    def k(self):
        return self.samples.shape[1]

    @property
    def n_samples(self):
        return self.samples.shape[0]

    @property
    def length(self):
        """Total arc length -- the scale a --delta step is measured in."""
        return float(self.arc[-1])

    def to(self, device):
        self.samples = self.samples.to(device)
        self.arc = self.arc.to(device)
        self.prop = self.prop.to(device)
        return self

    def encode(self, z):
        """(n, k) -> (u, residual). u is (n, 1) arc length; residual is (n, k), z - decode(u).

        Nearest sample rather than an exact nearest-point-on-segment: at the default
        density the spacing is far below the scatter of the data around the curve, so the
        difference is noise. Adding `residual` back reproduces z exactly either way.
        """
        flat = z.reshape(-1, self.k)
        idx = torch.cdist(flat, self.samples).argmin(dim=1)
        u = self.arc[idx].unsqueeze(-1)
        residual = flat - self.samples[idx]
        return u.reshape(*z.shape[:-1], 1), residual.reshape(z.shape)

    def decode(self, u):
        """(..., 1) arc length -> (..., k) point on the curve, linearly interpolated."""
        flat = u.reshape(-1).clamp(float(self.arc[0]), float(self.arc[-1]))
        hi = torch.searchsorted(self.arc, flat).clamp(1, self.n_samples - 1)
        lo = hi - 1
        span = (self.arc[hi] - self.arc[lo]).clamp_min(1e-9)
        w = ((flat - self.arc[lo]) / span).unsqueeze(-1)
        out = self.samples[lo] * (1 - w) + self.samples[hi] * w
        return out.reshape(*u.shape[:-1], self.k)

    def project(self, z):
        """The nearest point on the curve to each row of z."""
        u, _ = self.encode(z)
        return self.decode(u)

    def property_to_arc(self, value):
        """Arc length where the curve passes through a property value.

        For steering to a target: `delta = m.property_to_arc(30) - u`.
        """
        i = int(torch.argmin((self.prop - float(value)).abs()))
        return float(self.arc[i])

    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "arc": self.arc.cpu().numpy(),
            "prop": self.prop.cpu().numpy(),
            "point": list(self.samples.cpu().numpy()),
        })
        for key, val in self.meta.items():
            df[f"meta_{key}"] = val
        df.to_parquet(path, index=False)
        return path

    @classmethod
    def load(cls, path):
        df = pd.read_parquet(path)
        meta = {c[len("meta_"):]: df[c].iloc[0] for c in df.columns if c.startswith("meta_")}
        return cls(np.vstack(df["point"].to_numpy()), df["arc"].to_numpy(),
                   df["prop"].to_numpy(), meta)
