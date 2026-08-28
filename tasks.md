# Current Tasks

Active experiments. See `README.md` for everything completed so far and the
overall pipeline.
## consider probing at the layers to see if they are actually encoding information

## pca_centroid steering (branch `geometry_steering`, started 2026-08-25)
Testing whether the property direction is curved rather than straight. Instead of
`h + alpha * (mean_high - mean_low)`, project the layer-14 hidden state into the top-K
PCA subspace of the training activations, interpolate a fraction `t` toward the centroid
of a target class, and map the change back:

    z <- (h - mu) @ W.T ;  h <- h + t * (centroid_pca - z) @ W

Only the K principal directions move; the rest of the residual stream passes through, so
it keeps supplying the context the centroid does not specify. There is no negative class
-- the target centroid alone defines the destination.

Scripts: `compute_pca_basis.py` (basis, once per layer) -> `compute_centroid_target.py`
(one centroid per target) -> `steer_generate_cif.py --method pca_centroid`.
Artifacts under `steering_vectors/pca_centroid/`; runs named `steered_pca_*` so they sit
alongside the linear runs without colliding.

First target: density_atomic 30 A^3/atom (train median is 19.10, so ~1.6x), class = the
100,000 train structures nearest that value, which spans [28.50, 31.50].

### Result of the first t sweep (2026-08-25): no better than linear
`analysis/v1_all/test/pca_centroid_vs_linear_density_atomic.csv`. 1,000 test prompts x 3
samples, paired per prompt against the alpha=0 control, valid samples only, log10 units.

    run       valid%   median   cohens_d
    alpha0     96.3%    19.26      --
    alpha40    94.9%    19.34    0.181
    alpha80    88.4%    19.56    0.189
    t=0.25     96.1%    19.33    0.140
    t=0.5      95.0%    19.39    0.151
    t=1.0      12.5%    19.46    0.053
    t=2.0       0.0%      --       --

Where the model stays coherent (t <= 0.5) the effect matches the linear method's, which
is to say it is small. Past that the model degrades instead of steering.

**The trap this sweep walked into, for the next person.** Measured over every *parseable*
CIF, t=1.0 looked like a large win: median 19.29 -> 25.05, d=0.677, ~3.7x the best linear
run. It was an artifact. Only 12.5% of those CIFs are chemically valid, and the failures
are malformed text ("integer modulo by zero" 524, "zip() argument 2 is longer than
argument 1" 256, "len(items)=15 is not a multiple of n=2" 213), not oversized cells --
so it is not validity unfairly penalising us for making cells bigger. Filtering to valid
rows drops d to 0.053 on 210 surviving prompts. **Always join the validation flags before
reading a density shift**; a steering strength that breaks the CIF grammar produces huge
apparent movement in anything computed from the parsed text.

The PCA subspace itself is not the problem: the target centroid sits 4.273 from the
global activation mean and 4.221 of that (98.8%) lies inside the top-64 directions, so
the projection keeps essentially all of the class signal. Top-64 explains 68.9% of the
variance overall (pc1 alone 14.2%).

### Results tables: one schema, one script
**Never write a results CSV by hand.** `scripts/analysis/steering_ttest.py` computes the
paired comparison and is the only thing that writes these files:

    python scripts/analysis/steering_ttest.py --all        # every run -> steering_runs.csv
    python scripts/analysis/steering_ttest.py --property density_atomic \
        --method pca_centroid --target 30 --layer 14       # one sweep -> its own file

`analyse()` is the single place a run is scored, so the combined table and the
per-property ones cannot disagree. Columns are `utils.RESULT_COLUMNS`;
`write_results_table()` raises rather than write a table missing any of them. `unit`
names the value scale (a median was A^3/atom in one old file, log10 in another, eV in a
third) and `strength` holds alpha for the linear method and t for the pca ones, so one
column orders every sweep. Read `cohens_d`, not the p-values.

**A run is keyed by (layer, target, strength), and the family is part of its identity.**
Any two of those alone collide, and the collision is silent -- one run overwrites the
other in the discovery dict. Found three ways while standardising:
- two targets swept over the same t (22.5 and 30 both at t=0.5)
- one target swept at two layers (30 at layer 9 and layer 14 -- layer 9 won, so the
  reported layer-14 numbers were briefly layer 9's)
- sg and nosg runs sharing a key, *and* sg runs being paired against the nosg control,
  which compares two different prompt sets

Only nosg was ever generated at alpha=0 for band gap, so sg runs are listed with
population and value and no paired statistics, rather than dropped.

### The full sweep (2026-08-26): eight arms, all null
Two properties x two targets x two layers, plus a local-centroid variant. CSVs in
`analysis/v1_all/test/`: `pca_centroid_density_targets.csv`,
`pca_local_vs_centroid_density.csv`, `pca_centroid_bandgap_raw_vs_relaxed.csv`.

Nothing moved. Read `cohens_d`, not `p_wilcoxon`: with ~980 paired prompts a 0.5% shift
lands at p=1e-20 while |d| stays at 0.14. Every arm sits at |d| <= 0.15.

**Three explanations were tested and ruled out.**

*Reach.* density 22.5 needs +0.068 log10 from the control median, density 30 needs
+0.192 -- 2.8x further. Both produced the same ~0.002 absolute move. The displacement
does not scale with where the centroid sits, so the target being far away is not the
limit. Head-to-head the two targets differ at p=0.21.

*Model degradation.* The cleanest nulls come at the model's best operating points:
band gap t=0.75 is 85.9% valid and p=0.55; density t=0.5 is 95% valid and d=0.12.

*Centroid choice.* `pca_local` rebuilds the centroid per prompt from the 256 class
members nearest that prompt (`--save-bank` on compute_centroid_target.py writes the
member coordinates; `--method pca_local` consumes them). Local centroids sit a median
10.63 from the global one -- 2.5x the global centroid's whole displacement from the
corpus mean -- and both methods degrade the model identically (96.0/95.2/81.0% valid vs
96.1/95.0/82.5%), so they are comparable at equal t and differ only in direction. Result:
no difference, p=0.65 at t=0.25 and p=0.17 at t=0.5.

**Relaxation erases the raw band-gap drift.** The +0.03 eV nudge visible in raw CIFs goes
to -0.001..-0.013 after M3GNet (all p >= 0.56, |d| <= 0.074). Steering perturbs the
written cell in a way MEGNet reads as slightly gapped; relaxation settles it back out.
The same thing happened with the linear method -- always check the relaxed column.

**What the bank actually showed.** The target class is not a place. Its centroid sits
4.22 from the corpus mean while members sit a median 13.43 from that centroid -- a
signal-to-spread ratio of 0.31. And the class matches a single Gaussian of the same
covariance (13.43 vs 14.02 median, near-identical tails), so it is one diffuse cloud, not
several lobes with an average falling in the gap. Structures at 30 A^3/atom are scattered
through the same region as everything else. "Interpolate toward the class" has nothing
to aim at, which is sufficient to explain every null without any claim about curvature.

Two genuinely different destinations (local vs global, 10.63 apart) with matched
displacement and matched model damage give the same non-result. That points upstream of
centroid choice: layer-14 activations may not carry a controllable handle on these
properties at all, only a decodable one.

### LayerNorm is not the explanation (2026-08-27)
`scripts/analysis/layernorm_survival.py`, 40 prompts, forward passes only.
CrystaLLM is pre-norm, so anything added at layer 14 is normalised by block 15's ln_1
before it is read. Two parts of an injection could vanish for free -- the component along
the all-ones direction (removed by mean subtraction) and the scale (divided out by the
per-token std). Neither happens.

    ||steered - clean|| / ||clean||   linear a=40   pca_centroid t=0.5
      at the injection                    0.271          0.292
      after block 15 ln_1                 0.295          0.254
      after block 15 (residual)           0.246          0.233
      after ln_f                          0.213          0.246
      in the logits                       0.229          0.133
    uniform component (LN removes it)     0.28%          0.48%

**The injection reaches the logits at 13-23% relative magnitude and the property still
does not move.** That closes off the last mundane explanation: the sweeps did apply what
they were meant to apply, and the nine nulls stand as measured. The model's output
distribution is being perturbed substantially -- just not along the property.

Worth noting for a later experiment: pca_centroid injects a *larger* vector than linear
(55.6 vs 40.0) but reaches the logits with a *smaller* relative change (0.133 vs 0.229).
The top-64 subspace has less influence on the output per unit norm than the raw
mean-difference direction does.

Caveat: measured on the prompt forward pass (short sequences), not mid-generation. The
normalisation mechanism is the same either way, but the residual norm grows with sequence
length, so the ratio during a long generation will differ.

### No layer translates the property in a single step (2026-08-27)
`scripts/analysis/layer_causal_probe.py`. CrystaLLM writes numbers digit by digit, so
the model's belief about the volume is a distribution over digit tokens at known
positions. Teacher-force a real CIF, inject at layer L, compare those distributions
against clean. Forward passes only -- 16 layers x 60 CIFs, minutes, where a generation
sweep per layer would be days. Needed a steering vector at every layer, which is why
`compute_steering_vector.py` now streams (verified against the stored layer-14 vector:
cosine 0.99999998).

    layer   d_log10_vol   n_digits   lead     selectivity
      0       +0.0066      0.0000   +0.0066     0.801
      4       -0.0566      0.0000   -0.0566     2.276   <- largest magnitude effect
      6       -0.0462      0.0000   -0.0462     3.094   <- peak selectivity
      9       -0.0196      0.0000   -0.0196     2.669
     14       -0.0000      0.0000   -0.0000     2.582   <- where every sweep was run
     15       +0.0001      0.0000   +0.0001     2.428

**The injection lands on the right tokens and does not move them.** Selectivity is
2.4-3.1 across layers 1-15: the volume digits are perturbed ~2.6x more than the rest of
the CIF, so this is not undifferentiated noise. But the magnitude does not shift. The
integer-digit count -- the term that carries orders of magnitude, since a volume goes
90 -> 900 by gaining a digit -- is **exactly zero at every layer**. All movement is in
the leading digit, at most 0.057 log10, and **negative at 14 of 16 layers** when the
vector points low -> high. Reaching target 30 needs +0.192.

So the injection SCRAMBLES the digit distribution rather than TRANSLATING it. That
explains high KL with an unmoved median, and it explains all nine null sweeps better
than anything before it.

**The probe agrees with generation, which validates it.** It predicts negative at layer
9 and the layer-9 sweep gave d = -0.084/-0.091 (wrong direction); it predicts ~0 at
layer 14 and those sweeps gave d ~ +0.11 (nothing). Two independent methods, same answer.

Note the layer-14 collapse to -0.0000 is not a rounding artifact of the fixed metric: the
earlier, confounded E[leading digit] version showed the same profile (large at layers
4-8, ~0 at 11-15). Only the sign and scale were unreadable there.

**Scope: the probe is teacher-forced**, so it measures a single-step effect from a
well-formed prefix and is blind to anything that compounds over a long generation. That
blind spot is where the one visible behavioural effect lives. pca_centroid t=1.0 really
did make the model write bigger volumes, and splitting by validity shows what that was:

    control (valid)             n=2,888  median 19.25 A^3/atom
    t=1.0 VALID                 n=  374  median 19.48
    t=1.0 parsed but INVALID    n=2,571  median 24.56

All of it is in the invalid structures. The probe agrees: at layer 14, t=1.0 selectivity
falls to 0.598 -- the injection disturbs non-volume tokens MORE than volume tokens
(KL 1.672 vs 0.999), the fingerprint of degradation, not targeting. At layer 9, t=1.0 the
digit count moves (-0.165) with KL 10.3: the grammar coming apart. A strong injection
changes the written number by pushing the model off-distribution over hundreds of
autoregressive steps, and ~87% of what it writes is then not a valid crystal.

**Still untested:** steering only the token positions where the property is written,
rather than all of them; larger K; whether a non-additive intervention (patching
activations from a high-volume structure) transfers the property where adding a
direction does not.

**Not done yet: the analysis path.** `plot_steering_distribution_shift.py:discover_runs`
keys runs on `alpha(-?[\d.]+)` in the stem, so `steered_pca_*` files are invisible to it
and to `steering_ttest.py`. Those two need a way to enumerate `t` runs before the method
can be measured against the linear one.

## consider different types of steering

## consider different generative models

## Known bugs / to fix

### relax resume trusts stale `relaxed/<stem>.parquet` (data contamination)
`relax_steered_cifs.py` resumes by merging any existing
`steering_results/relaxed/<stem>.parquet` on `(id, sample)` and skipping rows that
already have `cif_relaxed`. It **never checks that the reused `cif_relaxed` was
produced from the current `cif_steered`.** If a stem is regenerated (generation is
stochastic at temperature 1.0, so a re-run yields different structures for the same
prompt), the old relaxed structures no longer correspond to the new `cif_steered`,
but resume keeps them anyway → contaminated relaxed rows feed wrong structures into
`predict_bandgap.py`.

Hit on 2026-07-13: α16 with-SG and α40 with-SG were regenerated to full 30k on
Jul-12 but still had Jun-2 900-row relaxed files; the relax jobs resumed and reused
~850 stale relaxed structures each. Fixed immediately by deleting those two relaxed
files and resubmitting (see below), but the underlying resume logic is still unsafe.

**Fix idea:** guard the resume — only reuse a relaxed row if the source `cif_steered`
matches (e.g. store a hash of `cif_steered` alongside `cif_relaxed`, or refuse to
resume when the relaxed file is older than the generated_cifs file for that stem).

## Backlog / not started
- Predict alpha = 11/25 relaxed files (already relaxed in `steering_results/relaxed/`, no
  prediction columns yet) to fill in the full alpha sweep table.

## Reference / notes

### MP metadata has NO space-group / Wyckoff data — symmetry analyses can't use it
`preparsed_metadata_mp.parquet` and `metadata_mp.parquet` carry no space group,
Wyckoff, point group, or crystal system. For `metadata_mp.parquet` this was checked
against all 5,491 leaf columns (including the nested ones) — zero matches for
symmetry / space_group / wyckoff / point_group / crystal_system.

**Consequence:** anything keyed on symmetry — `symmetry_separability.py`,
`spec_cocluster_analysis.py`'s SG / point-group / sg_wyckoff enrichment — only works
on the NOMAD side (`metadata.parquet`, which does have `space_group_symbol`,
`point_group`, `wyckoff_letters`). Pointing them at `v1_mp` needs the space group
from somewhere else: the prep CIF header (`_symmetry_space_group_name_H-M`, which is
in the token stream the model reads, so it's arguably the better source for BOTH
datasets) or recomputed with `SpacegroupAnalyzer`. Not verified which of those is
cleanest — the CIF-header route was not tested.

Also note the MP id columns differ from the embedding ids: `metadata_mp.parquet` has
`id` = `MP_mp-bwbt` (a short hash-like code) and `material_id` = `MP_mp-32493`, while
`embeddings/v1_mp` ids look like `MP_mp-1217888` — so the join key is
**`material_id`**, not `id`.

### metadata_mp scalar cols are PRIMITIVE-cell, but CIFs/embeddings are CONVENTIONAL
`metadata_mp.parquet`'s `volume`, `nsites` (and any other extensive cell quantity) come
straight from MP's `SummaryDoc`, which describes the **primitive** cell. But the CIFs we
built — and therefore the `v1_mp` embeddings — are **conventional** cells (CifWriter
symprec standardizes; see below). So for centered lattices (I/F/C) the column is a
factor of 2/4 smaller than what the model ingested; only for P lattices do they match.
- Verified: mp-32502 `volume`=529.78 (primitive) but the tokenized CIF has
  `_cell_volume 1059.57` (conventional, 2x); `nsites`=36 vs conventional 72.
- **Implication for steering:** an extensive property like `volume` is a dirty target —
  the label describes a different cell than the embedding. If steering on "size", either
  recompute conventional volume (read `_cell_volume` from the CIF) or use an intensive,
  cell-invariant property (`volume/nsites`, `density`, `density_atomic`). Volume steering
  is shelved for now (2026-07-27).

### MP CIF pipeline: cells are already conventional; ~4% exceed block_size (deferred)
Building CIFs for the MP data (`scripts/data/build_mp_cifs_tar.py`, from `metadata_mp.parquet`'s
`structure` column → `CrystaLLM/cifs_v1_mp_orig.tar.gz` → `tar_to_pickle` → `preprocess`
→ `tokenize_cifs`):
- MP's `SummaryDoc.structure` is the **primitive** cell, but **`CifWriter(s, symprec=0.1)`
  runs its own symmetry analysis and emits the standardized CONVENTIONAL cell regardless**
  of the input cell. So the written CIFs are conventional and in-distribution — matching
  `cifs_v1_orig.tar.gz` (verified: mp-10164 → 16 atoms = orig; 400/400 atom-count and
  399/400 space-group match vs the orig tar; lattice differs ~33% only due to MP DB
  re-relaxation drift between snapshots). No explicit `get_conventional_standard_structure()`
  needed — that just doubles the spglib work.
- Tokenized length (`tokens_v1_mp`, 154,871 CIFs): min 187 / median 448 / mean 687 /
  p99 4,148 / max 11,745. **6,166 CIFs (3.98%) exceed block_size=2048.**
- **Truncation is architectural, not our choice:** the model uses learned absolute
  positional embeddings (`wpe = nn.Embedding(block_size, n_embd)`, 2048 rows) and
  `_model.py` asserts `past_len + t <= block_size` ("Cannot forward sequence of length
  N, block size is only 2048"). `extract_cif_embeddings.py:86` truncates to block_size to
  avoid that crash. To embed >2048-token CIFs without dropping tokens needs chunk-and-pool;
  a single forward pass beyond 2048 is impossible with this checkpoint.
- **Decision (2026-07-18): 4% tail is marginal, deferred.** Keep the current truncation
  for now; revisit chunk-and-pool or a length filter if MP-embedding quality matters.



### Steering vector definition + class band-gap stats (why negative/high alpha barely move the gap)
Vector (`compute_steering_vector.py`, layer 14, from
`metadata.parquet:dos_electronic.band_gap`):
`steering_vector = normalize(mean(insulators) - mean(metals))`, points metal→insulator.
`+alpha` = toward high gap, `-alpha` = toward metallic. Stored **unit-normalized**
(‖v‖=1), so `alpha` IS the norm of the perturbation added at layer 14; the raw
class-mean-diff norm is 17.49 (the natural metal↔insulator separation), so
alpha≈16 ≈ 1× that separation, alpha 40/60 ≈ 2.3–3.4×.

Class stats (col `dos_electronic.band_gap`, 582,596 labelled):
- **metals** (gap ≤ 0.05): n=**575,593**, mean **0.0034 eV**, median 0.0014
- **insulators** (gap ≥ 1.0): n=**1,905**, mean **2.4142 eV**, median 2.044
- middle (excluded): n=5,098
- overall mean gap 0.0138 eV, median 0.0014; separation ≈ 2.41 eV

Two structural facts that shape every steering result:
1. **302:1 class imbalance** (575k metals vs 1.9k insulators). Direction is fine
   but the insulator pole is estimated from 0.3% of the data — a fragile anchor.
2. **Dataset is ~99% near-zero-gap.** This is the floor effect: a toward-metal
   push (alpha −16) has almost nowhere to go, so alpha −16 ≈ alpha 0 in predicted
   gap is expected, NOT a bug. Also explains why MEGNet `%>0` piles up at ~0 — real
   materials genuinely cluster at zero gap. Steering IS applied correctly (verified:
   alpha 0 → zero vector = true control; hook always registered on `h[14]`,
   KV-cache-aware; validity degrades 95.5%→88.8% across alpha, proving the hook
   fires). The weak band-gap effect is the data prior, not the mechanism.


