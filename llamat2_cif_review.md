# LLaMat-2-CIF: extraction and first results — review

Written 2026-09-18. Covers everything done since the `geometry_steering` branch was at
`f23855c`. Commits `0691802` … `fb235e8`, all carrying a `Co-Authored-By: Claude` trailer.

Read the **Open questions** section last; it is the part that needs your judgement rather
than your sign-off.

---

## 1. What was built

### The embeddings tree gained a model level

`embeddings/<dataset>/<variant>/` → `embeddings/<dataset>/<model>/<variant>/`. Two models
reading one corpus give unrelated vectors on the same ids, so without this the second
overwrites the first file by file. 1.1 TB of existing crystallm data moved by rename.

A flag collision fell out of it: `--model` already meant "checkpoint path" in the
generation scripts. Resolved repo-wide — **`--model` is always the model name,
`--ckpt-dir` is always a filesystem path**. 9 scripts, 14 configs. A stale
`--model CrystaLLM/crystallm_v1_large` now fails loudly rather than doing something
subtly wrong. `runs.tsv` was left untouched; it is the historical record.

### Storage changed from float64 to float32

Both models compute in float32; the extra 4 bytes per dimension were zeros, an artifact of
`.tolist()` going through Python floats. **Verified bit-identical**, max absolute
difference exactly 0.0.

### The analysis tree gained a model level, and a corpus split

```
analysis/<model>/<dataset>/<variant>/<partition>/
analysis/corpus/<dataset>/<partition>/
```

The model comes *first* here while `embeddings/` keeps it third. That asymmetry is
deliberate and documented: an analysis directory is a per-model deliverable, whereas a
layer's shards are read together whichever model wrote them.

`analysis/corpus/` holds outputs that never load a model — corpus property histograms,
Wyckoff counts, MP hull coverage. They are byte-identical whichever model runs, so they
are written once.

**The trap avoided:** `variant=None` looks like it identifies those, and does not. The
steering tables also pass `variant=None` — they come from generation, not from reading a
CIF variant — and are entirely model-dependent. Coupling the two would have filed every
steering result under `corpus/`. All ten call sites were checked individually: four are
corpus, six are model-dependent.

1,426 tracked files migrated with `git mv`: **1,426 renames, zero deletions**. Verified by
regenerating `steering_runs.csv`, which reproduced all 698 migrated rows with
**max |Δ Cohen's d| = 0.00e+00** and added 42 formation-energy arms that had not been
rescored.

The real work was the **~30 places that hardcoded an analysis path** and bypassed
`analysis_dir()` entirely. Changing only `analysis_dir()` would have left those writing to
the old tree — a half-migrated state worse than either end.

### Splits became per-model

`splits_v1.parquet` → `crystallm_splits_v1.parquet`. `utils.SPLIT_FILES` maps each model
to its own; `load_split_index`, `partition_id_sets` and `filter_partition` all take a
`model`, defaulting to crystallm so nothing existing changed.

**Why this matters:** CrystaLLM's train/val/test is a property of CrystaLLM's training and
says nothing about what LLaMat held out. Scoring llamat2_cif against CrystaLLM's `val`
would evaluate it on rows it may well have trained on.

`llamat_splits_v1.parquet` lists **only** the 9,046 crystal-text-llm test ids, labelled
`test`. Nothing else, deliberately: we know what LLaMat held out, not what it trained on.
`not_heldout` is defined by exclusion, so it resolves to "everything we have no evidence
was held out" without overclaiming. Against the embeddings: **8,670 test + 132,839
not_heldout = 141,509**, disjoint and complete.

### Symmetry labels built from scratch

`metadata_mp.parquet` has **no symmetry column in any of its 56**. `symmetry_v1_mp.parquet`
is derived from the CIF text plus pymatgen: 154,879 rows, 100% mapped, 228 space groups and
**exactly the 32 crystallographic point groups**, with the expected distribution (P1,
P2_1/c, Fm-3m, P-1, Pnma).

---

## 2. What is actually embedded — the central design decision

**Not a CIF.** LLaMat-2-CIF reads CIFs but only ever *writes* a compact crystal string, so
a direction fitted on CIF-reading activations need not transfer to generation, which is
what we steer.

Each structure becomes the model's own **unconditional generation prompt** — no formula,
no formation energy, no hull, no space group — with its crystal string where the answer
goes:

```
[system message] input-Below is a description of a bulk material. Generate a
description of the lengths and angles of the lattice vectors ... output-
4.9 5.4 3.3          <- lattice lengths, 1 decimal
90 90 90             <- angles, int() TRUNCATED not rounded
V
0.00 0.50 0.50       <- one element line + one coord line per atom
...
```

**Only the answer tokens are pooled.** The 203-token prompt is identical for every
structure and attention is causal, so those positions hold bit-identical hidden states
across all 141,509 rows. Measured, not assumed: prompt positions differ by **0.0** between
two structures, answer positions by **145**.

The crystal-string encoder is the authors' own logic, verified **identical on 300/300 real
structures** against `get_crystal_string_nate` executed straight out of their notebook.
`scripts/llamat_prompts.py` vendors it because `llamat/` is an untracked clone.

**What is in the input:** 3 lattice lengths, 3 angles, fractional coordinates, and element
symbols. Nothing else. No band gap, no formation energy, no hull, no space group, no
formula string.

---

## 3. The checkpoint

`m3rg-iitd/llamat-2-cif`, public and ungated, 26 GB of safetensors in `models/llamat2_cif`
(gitignored). Four things found by reading its config before running anything:

| finding | consequence |
|---|---|
| context is **2048**, not 4096 | 8.6% of v1_mp does not fit |
| index names `.bin`, we downloaded `.safetensors` | would have failed at load; index rebuilt from the shards' own headers, 323 tensors, key set identical to the shipped one |
| tokenizer has 32,005 entries, embedding has 32,000 rows | the 5 Megatron-default tokens (`<CLS> <SEP> <EOD> <MASK> <PAD>`) cannot be embedded at all |
| no `<\|im_start\|>`/`<\|im_end\|>` in vocab, no chat template | the checkpoint was **not** ChatML-trained → `notebook` wrapper |

The vocab gap needed no defensive code in the end: those tokens never appear in our input
(highest id produced across 3,000 real sequences is **29,999**), so the batch filler is
simply `0`.

The wrapper conclusion is strong evidence, not proof — `--wrapper chatml` remains
available, and one generation test would settle it.

---

## 4. Results

### Linear probes — fit on `not_heldout`, scored on LLaMat's held-out `test`

`by_formula`: eval rows whose formula appears nowhere in the fit pool, so nothing can be
answered by memorising formula → value.

| property | composition baseline | layer 0 | peak layer | peak R² | gain over layer 0 |
|---|---|---|---|---|---|
| volume per atom | 0.514 | 0.934 | 12 | 0.988 | +0.054 |
| formation energy | 0.494 | 0.870 | 8 | 0.949 | +0.079 |
| **band gap** | 0.270 | 0.373 | **6** | **0.689** | **+0.316** |

Sanity check passes exactly for all three: under `by_formula`, `lookup` collapses to
`mean` to four decimals. That identity is what proves the formula split is real.

**Band gap is the result worth caring about.** It is a DFT label absent from the input, the
crystal string strips the space group entirely, and it is the only property where depth
adds substantially (+0.316 against +0.054 and +0.079). Density and formation energy are
mostly already present at layer 0 — unsurprising, since both are largely determined by the
lattice parameters and element symbols the input hands over directly.

**Read density's 0.99 with care.** The crystal string carries the lattice parameters and
one line per atom, so volume-per-atom is computable from the text by arithmetic. Not the
token-copy shortcut that made crystallm's density result hard to read, but still fully
determined by the input.

All three peak at **layers 4–12 of 32** and decline after.

Plot: `analysis/llamat2_cif/probe_r2_by_layer.png`

### Cosine separability by symmetry — `not_heldout`, n = 132,839

Centered, same-label minus different-label mean cosine:

| | layer 0 | layer 30 |
|---|---|---|
| point group (32 classes) | 0.226 | 0.302 |
| space group (228 symbols) | 0.446 | 0.524 |

Separability rises with depth, and the uncontrolled permutation null sits at **0.90–1.06**,
right where it should, which validates the closed form.

Plots: `analysis/llamat2_cif/v1_mp/crystal_uncond/not_heldout/symmetry_separability_*.png`

---

## 5. Bugs found, and what they cost

### One that silently lost data — mine

The extraction loop did `continue` when a whole batch was skipped, which jumped over the
checkpoint block — and that block is where **both** the embeddings and the skip log are
written. The corpus is sorted shortest-first, so the tail is entirely over-context
structures and every tail batch took that branch.

**3,584 structures vanished from both outputs at once**, absent from the embeddings *and*
from `skipped.csv`, so nothing recorded they had been dropped. The job exited 0.

Caught by counting rows afterwards. Fixed: the forward pass is guarded instead of the loop
body, `flush()` is called unconditionally after the loop, and the run now ends by asserting
`embedded + skipped == total`. Repaired: 147 were embeddable, 3,437 genuinely over-context.

A corroborating detail — the over-context skip rate went from 6.23% to **8.45%** after the
repair, matching the 8.13% predicted offline. The bug had been suppressing exactly those
records.

### Others

- **`llamat_venv` had no pymatgen.** `utils.py` imports it at module level, so the job
  would have crashed seconds in. Caught before it burned a GPU slot.
- **`8.1%` in an argparse help string** — argparse `%`-formats help text, so `--help`
  raised `TypeError`. Only failed on `--help`, so it could have sat unnoticed.
- **`.gitignore` had `embeddings/` unanchored**, matching `scripts/embeddings/` too — new
  code there was silently ignored. Anchored to `/embeddings/`.
- **Probe filed output under a hardcoded `"val"`**, which would have put llamat's *test*
  results under `val/`.
- **Two scripts took their formula pickle from a v1_all hardcode**, which on a v1_mp run
  joins to nothing.
- **My own completeness check compared against the wrong total** on a resumed run, and
  cried wolf with a negative shortfall. The data was right; the check was wrong.
- Leftover smoke-test data (24 structures with OQMD ids) was sitting in the **v1_mp** tree.

---

## 6. Caveats that travel with every number above

1. **No matched baseline.** crystallm was not re-probed on this split (your call), so these
   R² values have nothing comparable beside them. The existing crystallm probe CSVs used
   different splits, structures and corpora.
2. **8.6% of v1_mp is missing**, systematically the largest cells — dropped median ~142
   sites against ~26 kept — because they exceed the 2,048-token context. Any size-dependent
   result meets this objection first.
3. **Separability ran on `not_heldout`** (your call), i.e. structures LLaMat may have
   trained on. Those numbers cannot distinguish learned geometry from memorisation, and
   there is no held-out run to compare against. The probes do not have this problem.
4. **94% of the MP corpus is of unknown status.** LLaMat's test set is only 5.6% of
   `metadata_mp`; the rest is not known-trained-on, merely unaccounted for.

---

## 7. Open questions — these need your judgement

### The composition-controlled null is inflated, and I do not know why

For llamat2_cif the composition-controlled permutation null sits at **1.13–1.39**. On the
same script, crystallm's runs give **0.84–1.09**. So the honest reading is
"composition-controlled ratio 2.2 against a null of 1.3", not "against 1.0" — part of that
apparent effect is a design artifact, not signal.

The likely cause is visible in the data: **raw mean pairwise cosine reaches 0.9964**, so
these embeddings are extremely anisotropic. Centering helps but does not restore the
geometry the null assumes. I would not report the composition-controlled symmetry number
until this is understood.

### Whether the wrapper is right

Strong evidence for `notebook` (the ChatML marker tokens are absent from the vocabulary),
but one generation run through both would settle it empirically.

### Whether band gap at 0.69 is a real capability claim

It is held out from both the probe (by formula) and from LLaMat (by test.csv), and the
label is absent from the input. That is about as clean as this setup gets. What it lacks
is a baseline: no number says whether 0.69 is good.

---

## 8. Where things live

| | |
|---|---|
| embeddings | `embeddings/v1_mp/llamat2_cif/crystal_uncond/cif_layer{0,2,…,30}.parquet` |
| skip log | `…/crystal_uncond/skipped.csv` — 13,370 rows with reasons |
| probe results | `analysis/llamat2_cif/v1_mp/crystal_uncond/test/property_probe_*.csv` |
| separability | `analysis/llamat2_cif/v1_mp/crystal_uncond/not_heldout/symmetry_separability_*.csv` |
| probe plot | `analysis/llamat2_cif/probe_r2_by_layer.png` |
| weights | `models/llamat2_cif/` (gitignored, 26 GB) |
| splits | `crystallm_splits_v1.parquet`, `llamat_splits_v1.parquet` |
| symmetry labels | `symmetry_v1_mp.parquet` |
| new code | `scripts/llamat_prompts.py`, `scripts/embeddings/check_extraction.py`, `scripts/data/build_llamat_split.py`, `scripts/data/build_symmetry_mp.py` |
