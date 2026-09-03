# Current Tasks

Active experiments. See `README.md` for everything completed so far and the
overall pipeline.
## consider probing at the layers to see if they are actually encoding information

## Experiment settings live in `experiments/*.conf`, run via `./run.sh` (2026-08-28)
`./run.sh <name> [--local|--dry-run] [overrides]`. See CLAUDE.md for the contract.

Replaces the env-var-per-slurm convention for anything new. Three things it fixes:
squeue now shows `centroid_density_l14-l10-w3-s500` instead of eight identical
`centroid_pca_plots.slurm`; `experiments/runs.tsv` records every submission with its git
SHA and resolved command including overrides; and the boilerplate (cd / PATH / bashrc /
mkdir logs / venv activate) lives once in `slurms/_job.slurm` instead of being pasted
into 17 files.

**Migrated so far** (chosen by usage -- these five slurms account for 720 of ~860 jobs
ever run): centroid_density_l14, centroid_bandgap_mp_l14, validate_steered,
predict_bandgap, predict_density, relax_steered, steer_pca_density, steer_linear_bandgap.
The old slurm files still work and are left in place until each is verified.

**Not migrated on purpose.** `compute_pca_basis.slurm` runs two scripts with a
skip-if-exists guard and a TARGET-gated second call -- that is control flow, not
configuration. `plot_tsne_pca.slurm` and `plot_categorical.slurm` loop over layers with
several calls each. `slurm_cocluster.sh` passes positionals to a script that also reads
os.environ directly. A loop is not a config.

**Found while surveying, not fixed here:**
- `plot_categorical.slurm`, `plot_tsne_pca.slurm`, `plot_wyckoff.slurm` never pass
  `--partition`, which `utils.py:add_partition_args` makes required=True. Those jobs die
  at argparse. They have 2 log files each, so they have barely run.
- `slurm_cocluster.sh` uses `N_CLUSTERS` and `RUN_NAME` with no default and no `:?`
  guard; unset silently expands to nothing and the script crashes on `sys.argv[1]`.
- `#SBATCH -V` appears in 9 files and is a no-op -- in SLURM `-V` is `--version`; the
  export-the-environment meaning is PBS/Torque's `qsub -V`. `_job.slurm` omits it.
- Eight variables carry conflicting defaults across files (ALPHA 40 vs 1.0, N_SAMPLES
  10000 vs 3, TARGET 30 vs required vs branch-gating). Under the new layout the default
  lives in the python argparse and each config states what it wants.
- `PARTITION` means the data split here and the queue in SLURM. The config format calls
  the queue `QUEUE` to keep them apart.

**Watch out:** `--local` runs write to the same output paths as a real run, so a quick
test with reduced settings will overwrite a committed figure. It happened while building
this. `runs.tsv` records the clobbering command, and `git checkout` restores the file.

## Linear probe across all 16 layers, four properties (2026-09-01)
`scripts/analysis/property_probe.py` -- the regression twin of `symmetry_probe.py`. Ridge
on mean-pooled embeddings, scored on the `by_formula` split so no test formula appeared in
training. Four baselines: mean, lookup, composition matrix, embedding.

    property              composition   L0      peak         gain over L0
    density_atomic           0.464     0.901   0.947 (L7)       +0.046
    efermi                   0.316     0.764   0.808 (L9)       +0.044
    energy_above_hull        0.041     0.619   0.715 (L10)      +0.096
    dos_electronic.band_gap  0.150     0.202   0.227 (L3)       +0.025

**Read the L0 column first.** Layer L here is the OUTPUT of transformer block L --
`extract_cif_embeddings.py` hooks `model.transformer.h[l]` -- so layer 0 already has one
attention + MLP behind it. (An earlier version of this note said layer 0 was token
embeddings plus position with no block run. That was wrong; there is no pre-block
extraction in this pipeline.)

Even so, block 0 alone gets R^2 = 0.901 on density, 0.764 on efermi and 0.619 on
energy_above_hull, and the remaining fifteen blocks add only +0.046, +0.044 and +0.096.
Almost everything these probes read is available after a single block.

**For density, the answer is printed in the input.** `density_atomic` is defined as
`cell_volume / n_atoms` (exact to 0.0 in `density_atomic_v1.parquet`), and both terms are
literally in the CIF. Parsing them out -- no model, no regression, no fitting -- on 29,832
corpus CIFs:

    100% of corpus CIFs carry a literal _cell_volume tag
    _cell_volume / (sum of _atom_site_symmetry_multiplicity)   R^2 = 1.000000, 100% exact
    _cell_volume / (sum of counts in _chemical_formula_sum)     R^2 = 1.000000, 100% exact

**What that does and does not mean.** It removes the probe as evidence: 0.947 cannot argue
the model learned a density representation when a trivial parse gets 1.000. Any probe
score at or below the parse ceiling is consistent with the embedding merely retaining two
printed numbers.

It does NOT show the model has no internal density representation, and it does not say
density is unsteerable. A model can perfectly well build an internal representation of a
quantity that is also written down, and steering could still change which digits it emits
for `_cell_volume` and the multiplicity column. Both remain open; the probe just cannot
settle them either way.

**This is specific to density.** efermi, band gap and energy_above_hull are DFT labels
that appear nowhere in a CIF, so their probes have no parse ceiling. That is also why
efermi reaches 0.764 at block 0 while the composition baseline is only 0.316 -- the pooled
embedding carries cell lengths, angles, space group and site positions, not just
stoichiometry.

Band gap is the exception and the honest case: L0 only reaches 0.202, and the whole model
only gets to 0.227. It is the one property here that is neither surface-readable nor
learned well.

All four peak in the middle (L3-L10) and decline toward L15, consistent with the last
layers specialising for next-token prediction rather than holding property information.

CSVs: `analysis/v1_all/full/val/property_probe_{density_atomic,dos_electronic_band_gap}.csv`
and `analysis/v1_mp/full/val/property_probe_{efermi,energy_above_hull}.csv`.

## Linear compresses, the manifold shifts (2026-09-03)
`scripts/analysis/stratified_effect.py`. The pooled Cohen's d asks whether the whole
distribution moved. It cannot say whether a prompt whose true structure sits at 14
A^3/atom responds like one already at 32, and steering density UPWARD has obviously
different headroom in those two cases. This runs the same paired comparison inside
buckets of the prompt's true density, width 1.

**Buckets are on GROUND TRUTH, not on the alpha=0 generation.** Control and steered are
both noisy draws from the same prompt. Bucketing on the realised control value enriches
low buckets for prompts whose control happened to sample low, while the steered draw does
not share that noise -- the difference would then trend downward across buckets from
regression to the mean alone, with no steering effect required. Ground truth is fixed and
cannot do that. The first version of this analysis was going to bucket on the control;
that was caught before it ran.

Best arm of each method with the sample size to stratify: `manifold d2 s9` (n=936,
d +0.254 pooled) and `linear a32` (n=861, d +0.158 pooled). Both layer 7, nosg.

The prompt set thins to a handful per unit below 12 and above 28, so those are POOLED
into one bucket at each end rather than dropped -- otherwise the plot stops at 26, which
is exactly where linear gets interesting.

    true density bucket:  <12     12     13    ...    21     22     25     26    >28
    linear a32      d:   +1.64  +1.07  +0.80   ...  +0.02  -0.18  -0.15  -0.14  -0.63
    manifold d2 s9  d:   +0.09  +0.63  +0.34   ...  +0.06  +0.27  +0.40  +0.54  +0.27

The two pooled ends, in A^3/atom:

    bucket   run              n    truth   control  steered   diff
    <12      linear a32      30    10.38    10.42    11.42   +1.00
    <12      manifold d2 s9  40    10.52    10.54    10.74   +0.20
    >28      linear a32     100    36.06    33.73    32.54   -1.19
    >28      manifold d2 s9 105    36.98    34.81    35.33   +0.52

**Linear does not steer density up -- it compresses the distribution toward ~21.** The
effect decays monotonically from d +1.64 in the pooled low tail to zero around bucket 21,
then turns NEGATIVE, reaching d -0.63 and -1.19 A^3/atom in the pooled high tail. It drags
low structures up by a full A^3/atom and high structures down by more than that. The
pooled +0.158 is a strong low-density effect averaged against a strong high-density
anti-effect, and the two nearly cancel.

**The manifold shifts without that reversal.** Positive in 15 of 16 buckets, no trend with
starting density.

**But linear is better where it works.** It beats the manifold in every bucket up to 15 --
d 1.07 vs 0.63 at bucket 12. The manifold wins pooled because it does not collapse at the
top, not because it is uniformly stronger. For pushing low-density structures higher,
linear alpha 32 is the better tool.

**A consistent mechanism, not a demonstrated one.** linear adds the same fixed vector to
every token regardless of where the state sits; the manifold's step is computed from the
token's own position on the curve (encode -> step -> decode). The injection-magnitude
plot shows this directly -- linear's inter-quartile whisker is near zero because
|injection| is constant, while the manifold's spans a wide range. An adaptive step should
be less prone to helping one end of the range and hurting the other. Not tested causally.

**Limits.** Tail buckets rest on 21-28 prompts, so bumpiness above bucket 22 is partly
noise; buckets below 12 and above 27 fell under the 20-prompt floor and are absent, so
this says nothing about the extremes.

Plot and per-bucket numbers:
`analysis/v1_all/test/plots/density_stratified_effect.{png,csv}`.

## Layer 7, no space group: the first dose response (2026-09-02)
Density steering at layer 7 -- where the property probe peaks -- with the space group
withheld from the prompt. 11 runs, 1,000 prompts x 3, jobs 453318-453328. These are a
`nosg` family of their own; every earlier density arm used `--with-spacegroup`, so an
alpha=0 nosg control had to be generated alongside them.

    label             valid%   n_paired   control   median   cohens_d    p_holm
    linear a0          95.7%      --       19.25    19.25       --         --
    linear a16         93.3%     974       19.25    19.31    +0.104    1.2e-03
    linear a40         33.6%     580       19.25    19.71    +0.181    3.0e-05
    linear a80          0.0%      --       19.25      --        --         --
    manifold d2  s6    93.7%     976       19.25    19.40    +0.192    2.7e-09
    manifold d15 s2    87.0%     938       19.25    19.46    +0.150    2.1e-05
    manifold d5  s6    39.9%     551       19.25    20.78    +0.112    8.8e-03
    manifold d15 s4    29.5%     422       19.25    21.30    +0.155    4.8e-03
    manifold d15 s6     9.5%     132       19.25    26.08    +0.115    0.38
    manifold d10 s6     8.3%     132       19.25    25.05    +0.064    0.46
    manifold d15 s8     5.7%      93       19.25    27.59    +0.088    0.40

**`manifold d2 s6` is the largest effect in the density table**: d +0.192 at 93.7% valid,
p_holm 2.7e-09, against a control at 95.7%. The linear arm at comparable validity (a16,
93.3%) reaches +0.104. Linear only gets to +0.181 by falling to 33.6% validity.

**Both sweeps are monotone in the median, which layer 14 never showed:**

    delta at s=6:   19.40 -> 20.78 -> 25.05            (d = 2, 5, 10)
    scale at d=15:  19.46 -> 21.30 -> 26.08 -> 27.59   (s = 2, 4, 6, 8)

The property moves the way it was asked to and keeps moving as the push increases. First
ordered response in this project.

**Best-of-3 does not overturn it.** Collapsing each prompt's three samples by max
instead of mean shrinks every effect (the control gains most: 19.25 -> 19.85, since it has
the most draws to pick from), but the three runs whose surviving-draw count matches the
control keep significant positive effects in the same order:

    arm                smp/prompt   d mean    d max     p max
    manifold d2 s6        2.87      +0.192   +0.142     1e-05
    manifold d15 s2       2.77      +0.150   +0.112    0.0025
    linear a16            2.86      +0.104   +0.096    0.0055
    (control a0)          2.92

Runs below ~2.2 draws per prompt are NOT interpretable under max -- `linear a40` at 1.73
draws is being asked to win best-of-1.7 against the control's best-of-2.9, and its effect
duly vanishes (+0.181 -> +0.020, p 0.63). Read the mean column for those.

**Survivorship caps how far the large shifts read.** Every large median shift is on a run that
destroyed 90%+ of its output: d15 s8 reaching 27.59 is 93 surviving prompts out of 1,000,
and those rows have small d with p ~ 0.4 because the variance is enormous. The monotone
ladder may be selection -- harder steering surviving only where it did least, or most --
rather than steering. These numbers cannot separate the two. The two rows that carry
weight are `d2 s6` and `d15 s2`: near-full validity, tight p-values, modest shifts.

**Not comparable to layer 14.** These ran nosg; every layer-14 density arm ran sg.
Different prompts, different task, different control, so "+0.192 at layer 7 beats +0.173
at layer 14" conflates layer with prompt format. Settling it needs layer 14 rerun nosg or
layer 7 rerun sg.

**Layer 7 is much more fragile to linear steering.** alpha=80 produced zero valid output
-- median CIF 111 characters against the control's 843, and the job finished in 11 minutes
against ~1h50 for everything else -- where layer 14 held 88.4%. alpha=40 fell to 33.6%
against layer 14's 94.9%.

**The curvature argument for picking layer 7 did not survive.** An earlier reading of the
layer-7 centroid plot gave a count-weighted turning angle of 58.5 degrees against ~34 at
layers 9/10/14. That was an artifact: the rerun applied `centroid_density_l14.conf`'s
`--max-per-bucket 5000`, which the committed layer-7 file never used, flattening every
large bucket to weight 5,000. Uncapped it is 34.7 degrees. All four layers are the same
curve -- tortuosity 12.2-13.0x, turning ~34 degrees, pc2 at rho -0.95.

## Manifold steering at layer 14: strong under mean, second under max (2026-08-28, revised 2026-09-02)
`--method manifold` slides along a curve fitted through the density bucket centroids,
carrying the off-curve offset through unchanged. Four deltas, 1,000 prompts x 3, layer 14,
paired against the same alpha=0 control as every other density arm.

> **CORRECTION (2026-09-01).** The "matched displacement" claim below is wrong and the
> validity conclusion does not survive it. The 7.50 for d=15 is the curve step; the ~7.8
> for pca_centroid t=1.0 is the centroid OFFSET, not the injection norm. Measured on real
> per-token states at layer 14 where |h| = 130.80, the actual injections are:
>
>     manifold d=15        6.80    5.2% of |h|
>     pca_centroid t=0.5  55.60   29.3%
>     linear alpha=40     40.00   30.6%
>     linear alpha=80     80.00   61.2%
>
> So the manifold arm ran at a sixth of the magnitude of anything that ever had an effect
> here. Validity was not "fixed" -- it was never stressed, and the null below is
> uninformative about curvature. See the scale sweep at the end of this section for what
> the same method does once the magnitude is matched.

**Validity, at the magnitudes actually run** (all far below the linear arms):

    run                 displacement   valid%
    alpha0 control            0         96.3
    manifold d=5           3.88         96.4
    manifold d=10          6.44         96.5
    manifold d=15          7.50         96.7
    pca_centroid t=0.5     ~3.9         95.0
    pca_centroid t=1.0     ~7.8         12.5   <- same displacement as d=15
    pca_centroid t=2.0      --           0.0

So the CIFs were breaking because the intervention left the data distribution, not
because of how far it moved. That hypothesis is confirmed.

**The property still does not move, and moves less than before.**

    run           target   median   mean_diff   %of move      d        p
    control          --    19.26
    manifold d=2   21.5    19.24    -0.00107      -2.2%    -0.043    0.84
    manifold d=5   25.8    19.18    -0.00062      -0.5%    -0.024    0.018
    manifold d=10  34.4    19.31    +0.00049      +0.2%    +0.057    6.4e-07
    manifold d=15  39.5    19.29    -0.00049      -0.2%    -0.020    0.00023
    linear a=80      --    19.51    +0.00306      +1.6%    +0.165
    pca_centroid t=0.5 --  19.36    +0.00181      +0.9%    +0.119

|d| <= 0.057, signs inconsistent, no dose response. d=15 asks for 39.5 A^3/atom and
returns 19.29 against a control of 19.26.

**The reading (superseded -- see the correction above).** The damage and the effect came from the same thing. pca_centroid's
perturbation moved the written digits -- the causal probe measured 2.6x selectivity on
the volume tokens -- by knocking the model off-distribution. Carrying the off-curve
residual is exactly what keeps it in-distribution, and that is also what stops the output
changing. An intervention gentle enough not to break the CIF is too gentle to matter.

That reading required the displacements to be matched. They were not, so it does not
stand. What survives is narrower: at 5% of |h| the manifold does nothing, and at 62% it
damages the CIF more than a straight line does.

### Matched-magnitude scale sweep (2026-09-01)

`--variant residual --scale S` multiplies the curve step, which is otherwise capped by the
curve's extent. Two runs at delta 15, layer 14, same 1,000 prompts x 3, same alpha=0
control. Predictions are from the raw generated CIF; none of these were relaxed.

    method          strength   injection   valid%   cohens_d    p_holm
    manifold s=6       6          ~41       94.2%    +0.173    3.0e-07
    linear            80          80        88.4%    +0.165    9.1e-07
    pca_centroid       0.5        55.6      95.1%    +0.139    6.2e-05
    linear            40          40        94.9%    +0.128    6.9e-05
    manifold s=1       1           6.8      96.7%    -0.020    0.79
    manifold s=12     12          ~82       50.1%    +0.033    0.79

**Under the mean, `manifold s=6` is the largest effect in the table.** Against `linear
alpha=40` -- the arm it is magnitude-matched to -- it is +0.173 vs +0.128 at 94.2% vs
94.9% validity. Larger effect, same validity. It also edges `linear alpha=80` (+0.165),
which needs twice the injection and drops to 88.4% valid.

> **That ranking does not survive best-of-3 (2026-09-02).** Collapsing each prompt's three
> samples by their max instead of their mean reverses it:
>
>     arm                 smp/prompt   d mean    d max     p max
>     linear alpha=80        2.80      +0.165   +0.217   8.9e-11
>     linear alpha=40        2.89      +0.128   +0.141   1.1e-05
>     manifold d15 s6        2.91      +0.173   +0.125   3.3e-04
>     pca_local t0.5         2.92      +0.130   +0.108    0.0022
>
> `linear alpha=80` is the only arm that STRENGTHENS under max, and it does so with fewer
> surviving draws per prompt than the manifold arm (2.80 vs 2.91), so the draw-count
> advantage runs against it. The manifold arm weakens to second. So "manifold beats linear
> at layer 14" is true of the mean and false of the max, and should not be stated without
> naming the aggregation. Note this is the opposite of layer 7, where `manifold d2 s6`
> stays on top under both -- see the layer-7 section.

**What that supports.** Following the fitted curve does better than a straight line at
equal injection, on this property at this layer. That is the first result here where the
geometry earns its cost.

**What it does not support.** The effect is small in absolute terms: median 19.258 ->
19.292 A^3/atom, +0.17%. And it is a narrow window, not a trend -- s=12 doubles the
injection and the effect vanishes (d +0.033, p 0.79) while validity halves to 50.1%. Two
points do not establish a dose response; s=2, s=4, s=8 would.

`frac_of_target_move` is NaN for every manifold row on purpose. `target` holds an ARC STEP
for these runs, not a property value, so scoring "how far toward the target" would read
delta 15 as 15 A^3/atom -- below the 19.26 control -- and report a spurious negative.

**The two projection variants produce nothing.** `project` and `project_nomu` are 0/600
valid, so they contribute no rows. Both were run at 200 prompts rather than 1,000. Their
injection is dominated by DELETING h_perp (and mu), not by steering, so 0% valid is the
expected outcome rather than an informative one.

**Discovery.** `steered_manifold_*` stems used to fall through to `kind="linear"`, whose
strength regex does not match them, so every manifold arm was silently dropped from
`steering_runs.csv`. The stem grammar now lives once in
`plot_steering_distribution_shift.py` (`kind_of` / `sweep_target` / `sweep_strength`) and
`steering_ttest.py` delegates to it. The manifold variants are separate methods because
five runs share delta 15 and would otherwise collide on one (target, strength) key.

**Two findings that bear on every null in this section (2026-09-01):**

*The stack is fitted pooled and injected per-token.* Every vector, centroid and curve here
is fitted on mean-pooled whole-CIF embeddings, while the hook runs on individual token
states. A ridge probe trained on pooled scores R^2 = 0.759 on pooled and **-22.193** on
per-token; per-token spread is 140.91 against pooled's 13.04. This is the leading
unrefuted explanation for the uniform nulls. Deferred deliberately -- there is literature
precedent for using pooled contrast vectors per-token -- but it is not ruled out.

*`encode` works on the bulk, fails on ~10%.* On 40,000 held-out structures at layer 14:
spearman(true density, encoded density) = +0.660, MAE 3.44 A^3/atom, 57% within +-2, and
every property band peaks on its own true centre. The defect is narrower than a broken
projection: about 10% of structures collapse to arc ~6 regardless of true density, and the
high tail clamps at the curve end. Plot and per-band numbers in
`analysis/v1_all/full/val/plots/manifold_encoding_density_atomic_layer14.png`, from
`scripts/plots/plot_manifold_encoding.py`. (An earlier note in conversation called `encode`
globally broken on the strength of `bank[:8000]` -- an unshuffled leading block with a
76.5% failure rate against a 10.1% base rate. The slice was real; the generalisation was
not.)

**Note for the results table:** `steering_ttest.py --all` does not pick these up.
`discover_runs` keys on `alpha<N>` or `_t<N>_`, and `steered_manifold_test_d10_k64_layer14`
matches neither, so the manifold arms are absent from `steering_runs.csv`. The numbers
above came from a direct paired computation. Extending discovery is a small change nobody
has made because the result is null.

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

### The property IS a curved ridge in activation space (2026-08-28)
`scripts/plots/centroid_pca_plots.py` -- bucket a property into [x, x+width), average the
embeddings in each bucket, project the centroids into the PCA basis, plot in property
order. Streams the layer, so it costs one pass. `--property`/`--labels` make it work on
any scalar column; `--basis centroids` refits PCA on the centroids instead of using the
corpus one.

density_atomic, width 1, layer 14 and 9, buckets under 30 structures dropped:

    layer  part   buckets  tortuosity(all)  tortuosity(core 10-40)  turn_wtd  best pc
      14   train      82        11.80x              2.01x             27.8    pc2 -0.95
       9   train      82        12.60x              1.80x             27.3    pc2 -0.95
      14   val        53         5.54x              2.10x             35.6    pc2 -0.91
       9   val        53         5.40x              1.89x             35.7    pc3 -0.93

**The path is not straight and it replicates out of sample.** Tortuosity is path length
over end-to-end distance. Restricted to 10-40 A^3/atom, which holds 98% of structures,
it is ~2x at both layers in both partitions.

**Most of the wild wiggling is sampling noise, not geometry.** 40-90 A^3/atom is 45
buckets holding 1.7% of structures, so those centroids average a few dozen points each.
Weighting the turning angle by bucket count drops it from a median of ~78 degrees to
~28. Read the core column, not the all column.

**One principal direction tracks density almost monotonically**: pc2 at Spearman -0.95
(train, both layers), -0.91 (val), and pc3 -0.93 for layer 9 val.

**Why this coexists with nine null steering results.** The centroid path spans 8.35 units
end to end across 12 dimensions. Individual structures sit a median 13.43 from their own
class centroid. *The conditional-mean trajectory is smaller than the scatter around it.*
Density is genuinely encoded -- curved, near-monotone along pc2, reproducible on held-out
data -- and simultaneously buried under within-bucket variance larger than the signal.
Additive steering moves a hidden state along the ridge by less than the cloud's own
width, which is what d ~ 0.15 looks like.

### Bucket width is a confound, and the DIRECTION of its effect is the real diagnostic
The turning angle was being read across properties binned at different widths, so bucket
count and structures-per-centroid were confounded with the geometry. Swept the width at
layer 14:

    property            width  buckets  turn_wtd
    density_atomic        1       82      27.8    <- train
    density_atomic        2       47      50.3
    density_atomic        5       24      41.2
    band_gap (MP)         0.1     74     106.7
    band_gap (MP)         0.25    31      96.4
    band_gap (MP)         0.5     16      48.6
    energy_above_hull     0.02   107      81.3
    energy_above_hull     0.1     46      93.1
    energy_above_hull     0.25    21      59.2
    efermi                0.25    68      56.2
    efermi                0.5     36      48.4
    efermi                1       22      48.6

**The two properties respond in opposite directions.** Density gets WORSE as buckets
widen (27.8 -> 50.3), band gap gets much BETTER (106.7 -> 48.6). Coarsening does two
things at once: it averages estimation noise away, and it under-samples genuine fine
structure. So the direction of change separates them -- density's fine structure is real
and coarse bins destroy it; band gap's jitter is noise and coarse bins remove it. That is
a better diagnostic than the raw angle.

**Correction to the earlier claim.** At comparable bucket counts the properties converge:
density 41.2 at 24 buckets, band gap 48.6 at 16, efermi 48.6 at 22. The dramatic
27.8-vs-106.7 contrast holds only at fine resolution (82 vs 74 buckets, so it is a fair
comparison there) -- but "band gap has no path at all" was too strong. At coarse
resolution it has one about as good as density's.

Watch out when summarising these: a glob over `*layer14*` picks up train AND val, which
are different populations with different bucket counts. The val density row at width 1
has 53 buckets and 35.6 degrees, and is not comparable to train's 82 and 27.8.

### Path smoothness is a spectrum, and structural share does not predict it (2026-08-28)
Extended the centroid-PCA measurement to `energy_above_hull` and `efermi` on MP, to test
what makes a property's representation readable. Count-weighted turning angle, layer 14:

    property             structural share  turn_wtd   best pc   corr w/ writable
    density_atomic             57.1%         27.8     pc2 -0.95      1.00
    efermi                       --          56.2     pc5 -0.91      0.668
    energy_above_hull          55.4%         81.3     pc3 +0.91      0.283
    band_gap                   10.4%        106.7     pc2 -0.90      0.445

**The prediction failed.** energy_above_hull was chosen because its structural share
(55.4%, i.e. the fraction of variance surviving a fixed composition) nearly matches
density's 57.1%, so it should have traced a comparable path. It does not: 81.3 degrees
against 27.8. **How much of a property survives fixing the composition does not predict
whether its representation has a trajectory.** That was the whole reason for preferring
it over a density replication.

**efermi is the first non-geometric property to land in between**, 56.2 degrees at layer
14 and 57.4 at layer 9. Not density's smooth ridge, not band gap's oscillation.

The better predictor now looks like **correlation with what the model literally writes
into the CIF** (max over volume / density_atomic / density / nsites) -- monotone for
three of the four, with band gap and energy_above_hull swapping. Four points and a broken
ordering is not a law; treat it as a lead.

Caveat on efermi: it correlates 0.668 with density, so "readable because it is partly
density" is live. Against that, it loads on pc5 while density loads on pc2 -- different
directions. Worth settling before calling efermi an independent result.

`--min-value 0.001` on energy_above_hull changed almost nothing (81.3 -> 82.1) because
the 5,000/bucket cap had already cut the on-hull spike down. Unlike band gap, where the
restriction mattered.

### Band gap has a direction but no path; density has both (2026-08-28)
Same centroid-PCA measurement across layers, properties and corpora. Turning angle is
weighted by bucket count, so a vertex resting on 50 structures does not count as much as
one resting on 50,000. **A weighted turn near 90 degrees is a random walk** -- consecutive
segments perpendicular on average -- and above 90 the path oscillates around itself,
which is what estimation noise looks like.

    property            corpus   buckets  tortuosity  turn_wtd   best pc
    density L7  train   NOMAD        82     1.70x       26.6     pc2 -0.95
    density L9  train   NOMAD        82     1.80x       27.3     pc2 -0.95
    density L10 train   NOMAD        82     1.86x       28.4     pc2 -0.96
    density L14 train   NOMAD        82     2.01x       27.8     pc2 -0.95
    band gap L14 train  NOMAD        41     6.15x       94.7     pc0 +0.94
    band gap L14 gapped NOMAD        40     6.64x      102.6     pc0 +0.94
    band gap L14 all    MP           74    12.74x      106.7     pc2 -0.90
    band gap L14 gapped MP           74     5.34x      112.5     pc2 -0.94

**Density curves smoothly and the curvature grows with depth** (1.70 -> 2.01 from layer 7
to 14), with pc2 carrying it at rho ~ -0.95 at every layer, replicated on val. Note layer
7 has the straightest path AND the largest effect in the causal probe (-0.037 vs layer
14's -0.0000): straighter representation, more steerable.

**Band gap never traces a path, and it is not a sample-size problem.** NOMAD has only
6,296 gapped structures so noise was the obvious suspect; MP has 78,668 across 74
buckets, twelve times more, and the turning angle went UP (94.7 -> 106.7). Both corpora
give a strong monotone direction (|rho| 0.90-0.94) with no smooth trajectory between
buckets. That is a real difference from density, on two independent datasets, and it
matches the steering results: density moved slightly at every strength, band gap not at
all.

Restricting NOMAD to gapped materials made it *worse* (6.15 -> 6.64x, 94.7 -> 102.6 deg),
against expectation: dropping the metals removed the one well-estimated centroid and left
40 noisy ones. The metals were anchoring the path, not drowning it.

**"Gapped" means >= 0.05 eV, never != 0.** 65% of NOMAD sits in (0, 0.05) -- nonzero but
physically metallic -- so a != 0 cutoff would call 386,208 structures gapped instead of
7,003, and 98% of them would be metals. MP's own `is_metal` flag agrees with the 0.05
threshold exactly: of its 72,640 metals, zero have band_gap >= 0.05.

**pc2 carrying both density and band gap is not the two properties being correlated**:
MP band_gap vs density_atomic is Spearman -0.208.

**v1_mp has no split.** 96,229 of its 154,879 ids appear nowhere in splits_v1, so the MP
runs use `--partition all` and mix structures the model trained on with ones it did not.
Whether those 96,229 are genuinely outside the training corpus is unverified.

### Steering moves TOWARD the data, not away from it (2026-08-27)
`scripts/analysis/manifold_distance.py`. The hypothesis was that steering pushes
activations off the data manifold and that is why the CIFs stop being valid. Measured
against 100,000 real training activations (the class bank), in subspace coordinates:

    t      d_nn    d_knn   mahal   valid%
    0.0   87.54    91.71   45.06   96.3
    0.5   40.30    42.43   22.53   95.0
    1.0    8.99     9.57    0.00   12.5
    2.0   83.77    87.04   45.06    0.0

Distance to real activations FALLS monotonically with t. At t=1.0 the steered state is
10x closer to real activations than the unsteered one, and that is where validity
collapses. **The decisive pair is t=0 vs t=2.0**: both ~87-92 away, both mahal 45.06,
validity 96.3% vs 0.0%. Manifold distance does not determine validity; displacement
magnitude does.

Why "closer" still breaks things: at t=1.0 mahal is exactly 0 -- we land on the centroid,
and in 64 dimensions the mean is not a typical sample. The data lives on a shell at
radius ~sqrt(64)=8; the centre is the emptiest place in it.

Caveat: the bank is mean-pooled whole-CIF embeddings while the probe measures per-token
prompt activations, which is why even the clean row sits at mahal 45 rather than ~8. The
t=0 vs t=2.0 comparison is unaffected (both measured identically), but "distance to this
bank" is not a clean measure of the model's own per-token activation manifold.

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


