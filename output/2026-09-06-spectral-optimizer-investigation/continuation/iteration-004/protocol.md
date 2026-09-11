# Iteration 004 prospective protocol: stopping metric and scalar norm control

Prepared before any iteration-004 pilot or outcome. This implements
[design-intent.md](design-intent.md); it is not launch approval. No frozen
iteration-003 or production source is edited. Pilot and confirmation require
separate parent GO, and confirmation requires the reviewed source set committed.

## Questions and estimands

Primary outcomes are paired differences in **test accuracy of the checkpoint
selected by maximum validation accuracy**: hard estimate32/project32 minus
AdamW, and hard estimate32/project32 minus scalar32_norm. Both primary contrasts
are reported together; neither is selected after outcomes. Width128 contrasts,
test cross-entropy, the minimum-validation-CE selector, endpoint and common
warmup-stop outcomes are mandatory secondary comparisons. Three seeds provide
three paired observations, not step-level or test-example-level replication.
Report every signed seed difference, mean, range and descriptive sample SD;
no p-values, equivalence claims or multiple-comparison-adjusted discovery claim.

Positive, mixed, null and adverse outcomes remain visible. A consistent positive
primary contrast would support that contrast in this small recipe, not identify
its causal mediator. A negative/mixed comparison limits the corresponding
superiority claim and does not trigger tuning or a larger sweep.

## Frozen recipe and randomness

Use fresh seed bundles **3, 4, 5**, nominal fixed replacement probability **.9**,
and the iteration-003 recipe: cached MNIST training IDX files only, first 5,000
examples of a seeded 60,000 permutation for training, next 5,000 for clean
validation; images flattened float32 divided by 255; MLP 784–64–ReLU–10 with
biases, 50,890 parameters; batch 64 sampled with replacement; 2,000 AdamW steps,
lr .001, weight decay .01, betas (.9, .999), epsilon 1e-8, foreach/fused disabled.
No schedule, clipping, augmentation, dropout, batch normalization or tuning.

Import the validated iteration-003 helper by absolute resolved file path.
Use its exact `make_plan` RNG namespace `[20260906,3,seed,stream]`, changing only
the unused seed values to 3, 4, 5; streams remain split 0, replacement decision 1,
replacement digit 2, initialization 3, training order 4. Generated auxiliary/probe
plans and tensors are unused; iteration 004 does not add gradient probes.
The same plan, initialized parameters and fixed noisy labels are paired across
all four methods. Replacement includes the original digit, so expected actual
incorrect fraction is .81; retain realized replacement and incorrect fractions.

These are fresh initialization/split/corruption/batch bundles on **reused MNIST
and the same official test set**, not an independent dataset-level replication.
No earlier outcomes are used for iteration-004 checkpoint selection.

## Four policies, common warmup and exact scalar definition

Arm identifiers: `adamw`, `estimate32_project32`, `estimate128_project32`,
`scalar32_norm`. Filtered/scalar estimators use the canonical stable update:
decay .99, relative eigenvalue floor 1e-8, absolute floor 0, scheduled repair 100,
canonical initial overweight, no diagonal normalization or adaptive rank.
Each observes its **own raw current gradient** before transformation. All
parameters form one global covariance block; projection rank is capped at 32.

Steps 1..100 pass raw gradients unchanged in every arm; filtering/scaling begins
strictly at 101. During warmup or without a basis, the candidate/applied operator
is identity and scalar alpha=1. Compare every warmup step's parameter and
raw/applied-gradient hashes across all four methods. At 100 also require exact
equality of model/gradient/AdamW moments and step counters, training flags and
RNG states. Estimation states need not match across widths; hard32 and scalar32
must have matching observer hashes throughout warmup and exactly equal complete
observer state at 100. Save the common warmup100 checkpoint prospectively.

For an active scalar32 observer, compute candidate `h=P32 g` using its current
self-inclusive basis, then pass `a=alpha*g` with `alpha=||h||/||g||`. Compute
norms with float64 reductions; multiply the float32 gradient by this scalar.
If `||g||=0`, set alpha=0 and pass a zero tensor; if `||g||>0` but `||h||=0`,
also use alpha=0. Never use `None` gradients or skip AdamW. Zero applied gradient
can still produce a nonzero update through moments and decay.

Require finite inputs/scalar/results and `0<=alpha<=1.005`; allow slight values
above 1 within that gate to accommodate approximate orthogonality, without
clamping. Require norm discrepancy `abs(||a||-||h||)` at most
`1e-6*max(||g||,||h||)+1e-12`. Scalar collinearity discrepancy is
`||a-alpha*g||/max(||g||,||a||,1e-30)`, gated at 1e-6. If both norms are positive,
also require `abs(cos(g,a)-1)<=1e-6`; otherwise cosine is null.

The observer affects training through alpha and is not measurement-only. This
matches a projected **raw-gradient norm on the scalar arm's own trajectory**,
not another arm's norm or realized AdamW step norm. Hard-versus-scalar contrasts
compare complete adaptive policies, not pure directional mediation. Alpha
varies and scaling begins with nonzero warmup moment history; constant-scaling
cancellation arguments do not automatically apply.

## Measurements and selectors

Every step records raw, candidate and applied squared norms; norm-matching
error; alpha for scalar/identity policies (null for active hard projection);
raw/applied cosine; scalar collinearity error when applicable; estimation rank
and operator type. Measure realized total displacement `d=theta_after-theta`
and nominal decay-subtracted `d_data=d+lr*wd*theta_before`. For both, retain
squared norm, raw/applied gradient dot displacement and cosine. Directional
sign tolerance remains `1e-6*||x||||d||+1e-14`; zero-norm cosine is null.
No orthoprojector leakage or in/out identity is applied to `alpha*I`; this
harness deliberately does not compute those metrics for any arm.

Evaluation grid is **0, 100, 200, ..., 2000** on all 5,000 clean validation examples.
`min_val_ce` selects the earliest strict minimum of finite validation CE;
`max_val_accuracy` selects the earliest strict maximum of validation accuracy.
Exact ties preserve the earlier step, with no secondary metric tie-break and
no epsilon-based tie grouping. Step 0 and step 100 are eligible for both.
`final` is step 2000; `warmup100` is step 100, independent of any selector. A shared
warmup candidate guarantees no worse selected validation criterion, not no
worse selected test metric. Save all four checkpoints even when identical.

All **12 training/validation-selection runs must finish before the harness
opens official test IDX files**. Then evaluate every saved checkpoint on all
10,000 test examples, reporting accuracy and CE. No test metric chooses a
checkpoint, arm, seed or stopping point. Also retain final training clean/noisy
and validation results. Confirmatory result JSON has `steps_raw`,
`validation_trajectory`, `checkpoint_steps` and `test` with the four fixed names.
The parent owns the separate frozen summarizer and its analysis specification.

## Development pilot, tests and launch gates

Plan only until parent approval: development seed **9877**, replacement .9,
**220 steps for each of four arms**, instrumentation off/on: eight traces.
Do not evaluate validation/test performance or compute accuracy. The pilot may
load training/validation tensors but never test files. Output timing, memory,
rank, hash and invariant evidence only; do not retain or inspect pilot loss,
alpha, gradient/update outcomes for choices. Timing window 129..220 includes
full-width repair at 200, also logged separately.

Instrumentation-on/off must match all 220 parameter hashes and final full
training states exactly for each arm. State snapshots around measurement calls
at steps 1, 50, 100, 101, 150, 200 must be bitwise unchanged. All warmup cross-arm and
hard32/scalar observer checks above must pass in both pilot and confirmation.
Check finite training gradients/parameters, basis orthogonality <=5e-3 at those
scheduled steps, and width128 reaching 128 by 160. Preserve every failed gate
and active seed/arm/step/phase; confirmation failures retain partial rows,
pilot failures retain no learning/gradient outcome arrays. Stop for review
before any amendment or retry.

CPU unit tests cover scalar norm/direction/zero cases, nonzero AdamW movement
after a zero applied gradient, warmup policy/optimizer equality, independent
strict selectors, schema/no-projector-identity handling, measurement-state
invariance and partial-failure preservation. They use synthetic tensors only,
not development or confirmatory MNIST training.

Before full launch, bind/hash this protocol, design intent, harness, tests,
parent analysis specification/summarizer/tests, imported iteration-003 helper,
canonical spectral source and training IDX files. Require source bytes match
committed Git and a passing development pilot, and data hashes match that pilot.
Require an explicit parent GO and `--full --confirmatory-go`. Refuse overwrite.
Use the local RTX 3090 only after occupancy review, deterministic algorithms,
TF32 off, CPU threads 1 and the validated CUBLAS configuration. No cloud or API.

## Interpretation limits

The study isolates policy and checkpoint-selection choices within a small
fixed recipe. It does not norm-match actual updates, eliminate endogenous
trajectory differences, demonstrate a unique denoising mechanism, or supply
independent test-data replication. Preserve endpoint and both selectors even
when their rankings conflict. Leave all original records intact; large plans
and checkpoints remain local with tracked hashes/sizes.
