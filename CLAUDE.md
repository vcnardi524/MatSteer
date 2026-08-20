# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `README.md` first.** It documents the science, the directory layout, the
`embeddings/` and `analysis/` tree conventions, and the current results. This file
covers how to *run* things and the cross-file conventions that are easy to get wrong.
`tasks.md` tracks what is in progress.

## RULES
All the code should be simple and accurate. It must be human readable. Naming conventions must be intuitive to humans and consitent. When speaking you should be clear and avoid flowery language, simple and accurate communication is ALWAYS the most effective. 

## Running things

There is no build, no test suite, and no package. Every script is a standalone
`python scripts/<stage>/<name>.py` run from the repo root, and real work goes through
SLURM.

Scripts are invoked **by path, not as modules**. They reach `scripts/utils.py` and
`scripts/predictors.py` via `sys.path.insert(0, dirname(dirname(__file__)))`. There
are no `__init__.py` files, so `python -m scripts.analysis.foo` does not work, and
importing one script from another needs `importlib.util.spec_from_file_location`
(see the top of `scripts/analysis/steering_ttest.py`).

### Three virtualenvs, not interchangeable

| venv | Used for | Why separate |
|---|---|---|
| `CrystaLLM/crystallm_venv` | generation, embeddings, sklearn analysis | torch cu130 — **GPU generation only** |
| `relax_venv` | M3GNet-PES relaxation | torch 2.4.1+cu121, the only one that runs on the V100 (sm_70) |
| `megnet_venv` | MEGNet band-gap prediction | CPU |

`crystallm_venv`'s cu130 build **cannot run on the V100**. Anything touching M3GNet
must use `relax_venv`.

### SLURM

Submit scripts live in `slurms/` and are driven entirely by environment variables,
which each script documents in a header comment. They `cd` to the repo root and
activate the right venv themselves.

```bash
sbatch --export=ALL,INPUT=steering_results/bandgap/validation/<stem>.parquet \
       slurms/relax_steered_cifs.slurm
```

**`--export` splits on commas.** A value containing a comma silently truncates:
`--export=ALL,LABEL_COLS=point_group,space_group_symbol` parses as
`LABEL_COLS=point_group` plus a stray variable, and the job runs one label while
looking successful. Export in the submitting shell and pass a plain `--export=ALL`
instead. Scripts that take comma-separated values echo their resolved config at
startup for exactly this reason — check that line in the log before trusting a run.

Logs go to `logs/<name>_<jobid>.log` (or `.out`/`.err`), gitignored.

Some nodes are excluded in the SLURM headers for cause — `node11` advertises 192G but
has been seen with under 5G free, which OOM-killed a job. Keep the exclusions.

## Architecture

### Pipeline stages produce separate stores keyed on `(id, sample)`

The pipeline is `generate → validate → relax → predict`, and each stage writes its own
parquet keyed on `(id, sample)`. Downstream stages **join**, they do not carry data
forward:

- `steering_results/generated_cifs/` — `cif_steered`. The only home for raw CIFs.
- `steering_results/relaxed/` — `cif_relaxed`. The only home for relaxed CIFs.
- `steering_results/validation/` — **flags only, no CIF strings**.
- `steering_results/<property>/property_predictions/` — one file per source stem,
  accumulating `<base>_raw` (from the raw CIF) and `<base>` (from the relaxed one).

So novelty, relaxation, and prediction all read the flags file, join the CIF source on
`(id, sample)`, and process `is_valid == True` rows only. A new stage should follow
that shape rather than widening an existing file.

Files are matched across stores by **stem**, so a filename like
`steered_test_clean_alpha16.0_layer14.parquet` is a key, not a label. Renaming one
breaks the joins.

### Adding a property

`scripts/predictors.py` holds a `REGISTRY` of `name -> factory -> PropertyPredictor`.
Subclass `PropertyPredictor` (or reuse `GeometricPredictor` for a direct structural
read), set `output_base`, put heavy model loading in `setup()` so the geometric path
stays cheap to import, and register it. `compute_predictions.py --property <name>`
then drives the shared load/validate/checkpoint loop with no further changes.

### `utils.py` owns the directory conventions

`analysis_dir()` and `embeddings_paths()` build the `<dataset>/<variant>/<partition>`
paths described in the README. Use them rather than composing paths by hand, so the
two trees stay aligned.

`add_partition_args()` makes `--partition` required with no default, deliberately:
89.6% of labelled structures are in CrystaLLM's own training set and only 0.45% in its
test set, so a number computed over `all` cannot separate learning from memorisation.
`filter_partition()` does the filtering, reading the `splits_v1.parquet` cache of
CrystaLLM's three split pickles.

### CrystaLLM is a submodule

Pinned to the `kv-cache` branch. `utils.py` loads `CrystaLLM/crystallm/_utils.py`
directly by file path to reuse `replace_symmetry_operators`, `remove_atom_props_block`
and friends, so the checkout must be populated even for scripts that never generate.

`utils.py:postprocess()` restores symmetry operators on a generated CIF. It must run
before parsing, or structures with a non-`P 1` space group parse wrong. It is a no-op
on already-indented reference CIFs — that difference has bitten this repo before.

## Data conventions

Ground-truth band gap is `metadata.parquet:dos_electronic.band_gap` (eV). **Not**
`energy_lowest_unoccupied - energy_highest_occupied` — those are raw Joules and their
difference is a corrupt LUMO-HOMO gap. Three legacy scripts still use it and are named
in the README; do not copy their approach.

`utils.py:DEFAULT_LABEL_COLS` asks for `band_gap_ev`, which does not exist in
`metadata.parquet` and is **silently dropped**. `load_labeled_embeddings` therefore
never returns a gap — join it yourself.

Probes must be evaluated on the `by_formula` split, not `random`. Chemical formulas
repeat across the corpus, so about half of val shares a formula with train and can be
answered by memorisation. `by_formula` scores only rows whose formula never appeared in
training. The check that it works: `lookup_acc` collapses to exactly `majority_acc`.
Note the two splits also score **different populations** (majority shifts 0.17 → 0.31),
so the columns are not a like-for-like difficulty comparison.

## What is version-controlled

`.gitignore` excludes all `*.parquet`, `*.pkl*`, `embeddings/`, `steering_vectors/`,
and the results subdirectories — data stays local. But `analysis/` **CSVs and PNGs are
tracked** (~160 files), so a rerun that changes numbers shows up as a diff. Commit the
code and the outputs it produced together.
