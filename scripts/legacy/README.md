# Legacy scripts — do not copy their approach

These three compute the band gap as

    (energy_lowest_unoccupied - energy_highest_occupied) * J_TO_EV

which is **wrong**. Those two columns are raw Joules straight out of NOMAD, and their
difference is a corrupt LUMO-HOMO gap, not a band gap. Ground truth is the official
value, `metadata.parquet:dos_electronic.band_gap` (eV, 582,596 non-null), taken from
`results.properties.electronic.dos_electronic.band_gap[0].value`. `electronic.band_gap`
holds the same numbers — identical non-null set, correlation 1.0.

| script | what it produced |
|---|---|
| `compute_bandgap_steering.py` | an early band-gap steering vector |
| `analyze_steering_norms.py` | injection-magnitude reference numbers, per percentile |
| `bandgap_percentile_stats.py` | the zero-gap pile-up statistics |

They are kept, not deleted, because published numbers came out of them and the record of
how those numbers were produced matters. Nothing imports them and no config runs them.

**Anything using a band gap today should read `dos_electronic.band_gap`.** The current
steering vectors come from `compute_steering_vector.py`, which does.

Two scripts that mention these columns are NOT legacy and stay where they are:
`data/parse_nomad_metadata.py` extracts the raw column when building the metadata, which
is its job, and `plots/plot_tsne_pca_group_bandgap.py` only names it in a docstring
warning against exactly this mistake.
