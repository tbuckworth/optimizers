# Saved-checkpoint modular-addition representation measurements

9 September 2026 · Numerical protocol fixed before first acquisition and pinned
with its implementation in the acquisition manifest.

## Question and scope

When does modular-sum information become linearly decodable from the trained
representation, relative to observed generalization? Does its temporal ordering
suggest earlier formation or later cleanup in the legacy/stable filter arms?
This is an observational analysis of already-diverged trajectories, not causal
mediation. It follows the approved paper plan and the completed five-seed
confirmation. No training run is restarted or silently extended.

## Fixed corpus and stages

Use all five seeds 100–104, arms AdamW/legacy/stable, and the ten recorded
checkpoints 0,100,500,1000,1500,2000,2500,3000,4000,6000: 150 states total.
Source metrics are the byte-hash-verified files in
`output/2026-09-08-spectral-paper-planning/grokking-confirmation-results/raw/`.
Each contains its checkpoint paths, hashes, configuration, split and model
identity. The runner verifies those identities and safely loads each file.

The first stage contains only seed100 at steps0 and6000 for all three arms
(six states). This checks the ruler on initial versus successfully generalized
models. The remaining stage contains the other144 states, with no duplicate
acquisition of the six completed ones. If measurement implementation must be
changed after calibration, preserve the calibration outputs, label the change,
and use separate evaluation states rather than silently overwrite or reselect.

## Numerical readout

Use the actual unchanged two-token/GELU/LayerNorm model, not an assumed Nanda
architecture. Extract logits and final pre-unembedding hidden vectors using
non-mutating hooks. A pre-attention residual control contains both token
positions, concatenated; it can decode individual tokens but has no learned
cross-token interaction at that layer. Different feature dimensions are
explicit, not described as identical-capacity probes.

Split never-trained pairs once per seed, stratified by true modular sum, into
probe-fit and probe-evaluation halves using a fixed generator. No training pair
enters either probe set. Across checkpoints/arms, preserve the same indices.
Input rows are sorted lexicographically by `(a,b)`. For each true sum in
ascending order, permute its sorted row indices with one CPU Torch generator
seeded `20260909 + training_seed`; assign the first floor(n_sum/2) to fit and
the remainder to evaluation. Thus the fit half is smaller for odd strata.
Archive the exact indices and their hashes. The pre-attention hook is the
input to `ln1`, flattened over both token positions; the final hidden hook is
the output of `ln_final`, exactly the input to `unembed`.
For each frequency k=1,…,56, regress the paired targets
`[cos(2πk(a+b)/113), sin(2πk(a+b)/113)]` on hidden features.

Fit a mean-loss ridge model with intercept in fp64. Center features/targets on
the fit set only; scale features by one fit-set RMS scalar, not evaluation data.
Ridge coefficient is0.001, fixed rather than optimized. Select five frequencies
by fit-set paired R² with deterministic tie breaking. Report all56 fit and
evaluation scores plus the selected-five evaluation mean. R² uses the constant
fit-target mean as its reference prediction on each evaluated set. It may be
negative and is never clipped. Do not choose frequencies on evaluation outcomes.
In detail, `s=sqrt(mean((X_fit-mean_fit(X))²))`, `Z=(X-mean_fit(X))/s`;
solve `(Z_fitᵀ Z_fit/n + .001 I) B = Z_fitᵀ(Y_fit-mean_fit(Y))/n` and restore
the target mean at prediction. Centering makes the intercept unpenalized.
Pair-R² pools squared error across the cosine/sine pair and compares with
the constant fit-mean target prediction. Ties rank smaller frequencies first.
The zero-RMS case uses unit scale and consequently an intercept-only predictor.

Repeat the identical fit/select/evaluate procedure for twenty fixed
**row permutations of sums** on never-trained pairs. Do not merely permute the
sum-to-class codebook: that would preserve a function of sum. Preserve full
per-frequency null scores, permutation hashes and selected indices. Their
maximum provides a descriptive finite null envelope, not a multiplicity-
corrected significance threshold over all checkpoints and arms.
Generate the twenty permutations sequentially using one CPU Torch generator
seeded `20261909 + training_seed`. The same row permutations apply to the same
fixed split for both representations and every checkpoint/arm of that seed.
If the generated permutation is exactly the identity, roll it by one row;
this guard is deterministic. Archive the full permutation indices with the
activations, not only seeds and hashes. Targets are globally row-shuffled before
indexing the fixed halves, so half-specific target marginals may change; each
null's own fit-target mean is used in its predictions and R² reference.

## Construct validation and interpretation

Before scientific acquisition, fixtures must show exact forward/hook parity,
no mutation, disjoint/full-coverage probe splits, fit-only preprocessing and
selection, recovery of synthetic sum Fourier features, and failure on
independent labels beyond a finite-sample null. Include scale/offset invariance
within the fixed regularized recipe and zero/degenerate-feature handling.

On initial/final calibration states, each arm must separately satisfy all four:
final-hidden selected-five evaluation R² at6000 is at least0.50, and it exceeds
(a) its maximum final-hidden null, (b) its initial final-hidden score and
(c) its final pre-attention score by at least0.20 each. These large descriptive
margins are fixed before reading the six states, not significance cutoffs.
The remaining stage requires the complete hash-bound six-state receipts and
recomputes this gate; no machine success is inferred merely from a file name.
If the ruler fails, diagnose the measurement;
do not label the optimizer hypothesis disproved by an invalid probe.

Readable sum information is not proof that the model uses that information,
that it follows a Fourier circuit, or that formation has finished. The probe
provides a temporal representation marker. Symmetry/excess-defect diagnostics
and later common-state interventions can strengthen interpretation but cannot
be replaced by reading test accuracy off the same curve.

For sensitivity to frequency switching, define a second fixed-frequency panel:
rank the mean fit R² across the three seed100 final-hidden step6000 states,
select the top five (smaller k breaks ties), and apply these same frequencies to
every saved per-frequency evaluation curve. This uses calibration fit data,
not seeds101–104 outcomes. The per-state-selected and fixed-panel scores answer
different questions; neither demonstrates a five-frequency model circuit.

The independent [construct review](construct-review.md) proposed hash-ranked
splits and separately permuted halves. The operational protocol instead uses
the already specified version-pinned Torch generators and global row nulls,
with exact indices archived. Both destroy arbitrary row association; this
choice is fixed before acquisition, not selected after null outcomes. Its
four calibration margins, both-token control, fixed-panel sensitivity and
decodability-versus-circuit cautions are adopted.

## Subsequent saved-logit diagnostics

After acquisition, use these same archived logits without rerunning inference
to measure row-centered output equivariance for input shifts1,2,4,8,16,32
on never-trained pairs, plus exchange symmetry. A wrong output shift of
`delta+37 mod113` supplies an equal-edge-count comparison. Match train-to-heldout
and heldout-to-heldout edges by axis, shift and source sum using fixed hash
ordering, then compare normalized squared equivariance defects. Retain counts,
numerators, denominators and logit RMS. Zero-energy outputs are undefined, not
perfect symmetry. Exact diagnostic code and fixture validation precede its
first analysis; the [construct review](construct-review.md) gives the equations.
This training-membership excess is not a memorization-circuit ablation; changes
in logit temperature can change it despite unchanged classification.

## Resources and evidence

Read-only model inference and small ridge solves on the desktop; one process,
one CPU math thread, optional RTX3090 inference. No optimizer step, backward
pass, paid judge or cloud launch. A verified systemd transient unit enforces
16GiB MemoryMax,0 MemorySwapMax,30min RuntimeMaxSec,Restart=no and whole-cgroup
termination. The runner has a1740s cooperative check between states. Its
10GiB output budget is checked before each array write with overhead reserve;
it refuses paths outside a new `/tmp/spectral-experiment-artifacts` descendant and requires
11GiB initial free space. Check real resource use on the six-state stage
before starting the remaining fixed roster. Scientific files/recipe are hashed
before first acquisition, outputs exclusive-create, raw activations and scalar
results retained for independent arithmetic review. No result is counted as an
independent seed merely because it is a different checkpoint.

Sources: the approved [measurement proposal](../2026-09-08-spectral-paper-planning/grokking-mechanism-design.md),
[completed confirmation](../../research/grokking_stable_confirmation_2026-09-08.md),
[Nanda et al.](https://arxiv.org/html/2301.05217v2), and
[PyTorch solve documentation](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.solve.html).
The implementation uses a linear solve, not an explicit matrix inverse.
Specifically, it reuses a checked positive-definite Cholesky factor and
`cholesky_solve` for multiple target columns. Positive ridge makes the centered
Gram system nonsingular in exact arithmetic; fixture agreement with an
independent fp64 normal-equation solve checks the implemented convention.
