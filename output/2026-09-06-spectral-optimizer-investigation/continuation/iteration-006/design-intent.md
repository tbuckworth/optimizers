# Iteration 006 design intent: does self-inclusive delivery help or hurt learning?

6 September 2026. Design for independent challenge, **not launch approval**.
No iteration-006 pilot, training, validation or test result exists at this stage.
The ongoing user-authorized hypothesis loop resumes from commit e88aa73.
Earlier experiments, production sources and the delivered report are immutable.

## Evidence, question and success criteria

Iteration 005 found current-gradient energy retention near .84 after observation
versus .41 under the previous basis on three reused AdamW streams. This includes
the entire observer transition, including centering, truncation and repair, and
does not isolate an outer-product effect. Iteration 004 found strong endpoint
anti-memorization but no general accuracy-selected superiority. Its own-trajectory
scalar norm control did not reproduce the endpoint gains. Relevant evidence:

- [Audited common-stream diagnostic](../iteration-005/results.md).
- [Stopping and scalar controls](../iteration-004/results.md).
- Original project synthesis (artifact not distributed in this public snapshot).
- Research state and user authority (artifact not distributed in this public snapshot).

Question: does delivering through the previous observation's basis, rather than
the basis just updated with the current raw gradient, change learning in this
fixed recipe? Does that contrast persist when the lagged direction is given the
current operator's gradient norm at its own state? Self-inclusion could admit
idiosyncratic corruption, but it could also preserve novel useful directions;
neither learning outcome is determined algebraically. This is a mechanistic
study of the existing optimizer, not a novelty, SOTA or semantic-denoising claim.

Positive, null, mixed and adverse outcomes are equally informative. The study
is successful if its frozen comparisons are faithfully executed and audited,
not only if lagging wins. No outcome-triggered tuning or added seeds. Report
every paired seed difference; no p-values, equivalence or population claims.

## Proposed fixed recipe and six policies

Use fresh seed bundles 6,7,8 in the existing iteration-003 RNG namespace and
nominal fixed uniform replacement probabilities 0 and .9. Same 5,000 MNIST
training / 5,000 clean validation / disjoint auxiliary split, 784-64-10 ReLU
MLP, 50,890 parameters, batch 64, 2,000 AdamW updates, lr .001, decay .01,
betas (.9,.999), epsilon 1e-8, no schedule/clipping/augmentation. Pair plans,
initialization, labels and batches across arms. This is 36 runs, not a sweep.
Seeds pair clean/noisy conditions too; never count those as six independent
seed replications. Data and official test set are reused, not independent
dataset-level confirmation. Preserve the exact prior data preparation code.

Every observing policy stores width 32. The three projected policies deliver
through at most rank 32; nonzero scalar controls use a full-rank scaled identity,
not a rank-32 delivery operator. All observers use the
canonical stable raw-gradient update: beta .99, relative floor 1e-8, absolute
floor 0, repair every 100 observations, existing initialization convention.
All parameters form one global block. Observe every raw gradient once, including
the common 100-step raw-gradient warmup. No filtered-gradient feedback into the
covariance. For active steps t >= 101, retain a clone of B_(t-1), update the
observer with raw g_t, and obtain B_t. Evaluate both candidates at the same
pre-Adam parameters: c=B_t B_t^T g_t; l=B_(t-1) B_(t-1)^T g_t. Actual native
float32 action is used; do not replace it by a QR projector. An absent basis is
identity. Basis cloning must prevent aliasing through the observer update.

| Arm | Gradient delivered after warmup |
|---|---|
| adamw | g |
| current32 | c |
| lagged32 | l |
| lagged32_current_norm | l times ||c||/||l|| |
| scalar_current32 | g times ||c||/||g|| |
| scalar_lagged32 | g times ||l||/||g|| |

All observers are on their **own policy trajectory**. The norm-restored lagged
arm uses its own current candidate, not a different arm's values. It changes
direction while following the same current-candidate norm prescription; it
does not numerically match cross-arm gradient or actual AdamW update histories.
Scalar arms control the corresponding own-state candidate norm, not adaptive
step norm or a uniquely identified mediator. Compare all these policies without
claiming a pure causal decomposition of their endpoint differences.

Norms use float64 reductions; delivered tensors remain float32. Scalar zero
input gives a zero applied gradient; never skip AdamW or use None gradients.
For norm-restored lagging: zero current norm gives zero; both zero gives zero;
positive current norm with zero lagged norm is an undefined direction and a
fatal preserved design-gate failure, not fallback, clipping or arm exclusion.
Finite rescaling and delivered norm/direction gates are mandatory. Do not cap
nonzero scale or quietly change the intervention after outcomes.

## Outcomes and interpretation

Primary comparisons, reported separately for clean and noisy conditions:

1. lagged32 minus current32 test accuracy at max-validation-accuracy selection;
2. lagged32_current_norm minus current32 at that same selection rule. This is
   **lagged direction with current-observer magnitude**, not a fully non-self-
   inclusive control: the current gradient still affects its delivered norm.

These are four prespecified primary condition/contrast groups, each with three
paired seed values, not four chances to select a favorable headline. A benefit
in all three noisy seeds supports that particular small-recipe comparison;
mixed/negative signs limit it. Clean changes establish costs or benefits in the
same recipe; do not silently trade them against noisy gains. No superiority
claim follows from consistency alone, and no equivalence follows from small
differences. Preserve secondary endpoint and CE-selected findings regardless.

Mandatory secondary contrasts: every unordered arm pair with a single fixed
orientation; report current/lagged and both scalar controls against AdamW,
norm-restored versus unscaled lagging, plus their controls. Retain test accuracy
and CE for final, earliest strict min-validation-CE, earliest strict max-validation-
accuracy and common warmup100 checkpoints. Validation grid 0,100,...,2000; no
test-set selection, tie-breaking or outcome-dependent windows. All 36 training
and validation-selection runs must finish before opening official test data.

Record every step's current/lagged candidate and applied norms, rescaling,
raw/applied cosine and total/nominal-decay-subtracted displacement norms and
gradient dots. Record current-minus-lagged retention on each observing policy's
stream, but do not equate this with independently useful signal. Prespecified
windows: 101-2000,101-500,1501-2000; seed-first aggregation with exact null masks.
No assumption that AdamW updates stay inside either candidate span. No new
semantic probes, SVD references or hyperparameter search is needed for the
primary policy question. If probes are proposed in review, their value and
measurement isolation must be justified before freezing the design.

## Fail-fast stages, resource scope and safeguards

These are calibrated planning estimates, not empirical success probabilities.

| Stage | Probability of passing | Hours to check | -ln(P)/hours | Gate |
|---|---:|---:|---:|---|
| Correctly distinguish delivery order and scaling | .7 | .15 | 2.38 | Synthetic policy/aliasing/zero tests |
| Measurement isolation and practical resources | .85 | .15 | 1.08 | Development-only on/off pilot |
| Useful lagged learning comparison | .5 | .25 | 2.77 | Frozen policy results; do not optimize for a win |

Dependencies take priority over that rough information-rate ordering: no neural
run before design review, synthetic tests, source freeze and explicit parent GO.
Planning/checking time is distinct from GPU runtime. A null scientific outcome
ends this fixed comparison and is written up, not retried as an engineering
failure. Engineering defects require an explicit amendment and preserved record.

Proposed one local RTX 3090 development pilot: seed 9880, noisy condition .9,
220 steps per arm, instrumentation off/on (12 traces), no accuracy/CE selection
or test data access. Retain only timing, memory, finite/rank/hash/gate evidence,
not pilot learning or gradient outcomes for tuning. Check complete final states
and every parameter hash between on/off; verify identical raw-gradient warmup
and full model/optimizer/RNG state across six arms. All width32 observer states
must match during warmup; AdamW has no observer. Include scheduled repair and
the first active step in explicit synthetic and measurement-invariance tests.

Use the verified large-volume /tmp/spectral-experiment-artifacts mount for exclusively created
plans, checkpoints and large raw JSON, with tracked hashes/sizes and manifest;
do not fill the workspace's remaining 2.4 GB. Proposed full wall-time cap 900 s,
pilot cap 180 s, no paid resources or API. Inspect GPU occupancy before launch;
no other research job may run concurrently. Reuse deterministic settings from
validated earlier harnesses (TF32 off, CPU threads 1, CUDA/BLAS settings).
Bind all scientific sources, helpers and data hashes to the committed passing
pilot. Preserve partial failure context and refuse overwrites/retries. Parent
owns source commits and separate pilot/full launch decisions.

## Design-time limitations and alternatives

Fix now: clean boundary, two stopping rules, same warmup, current/lagged scalar
controls, own-state lagged norm restoration, aliasing/state tests and strict
source/data binding. These are cheap within the local recipe.

Deferred: larger models/datasets, fresh held-out dataset, independent hyperparameter
tuning, actual adaptive-update norm matching and a causal mediation analysis.
Those ask different questions and require a separately designed experiment.
Do not promote a clean/noisy MNIST policy comparison into an application claim.

Independent challenge must decide whether any remaining construct-validity
flaw requires redesign before implementation. This file is the shared base;
reviewers must not read each other's assessments.

## Prospective challenge resolutions

The initial rank-delivery blanket statement was wrong for the scalar arms;
it is corrected above before implementation. Their observer rank remains 32.
No reciprocal current-direction/lagged-magnitude filtered arm is present, so
the six policies are not a complete direction-by-magnitude factorial.

All rescaling multiplies a float64 copy of the direction by the float64 norm
ratio, then casts once to float32. This avoids converting a large finite scale
to float32 before multiplication. Check all raw/candidate inputs for finiteness
before any zero shortcut. A positive target must produce a positive delivered
norm, and relative norm error must be <=1e-6. Check normalized direction error
<=1e-6 when both vectors are nonzero. Zero target gives exact zero delivery;
positive target with zero direction, overflow, full underflow or excessive
subnormal quantization is a preserved domain/numerical failure. No absolute
floor may turn that failure into a pass. Scalar ratios alone retain <=1.005;
the restored-lag ratio is not capped. Tests distinguish representable extreme
cases from genuinely unrepresentable ones before any neural pilot.

Log the same-state current/lagged candidate cosine with explicit zero-vector
nulls, and retain final clean/noisy training accuracy and CE from the same
fixed-state logits. These measurements never select checkpoints. Add fixed
repair-versus-other-step retention/angle summaries as specified in the analysis
plan; this does not isolate the causal effect of repair itself.

The primary estimands include each policy's validation-selection rule. Unequal
selected training exposure is part of that estimand; endpoint contrasts supply
the separate fixed-exposure comparison. Engineering validity and scientific
sign are separate: the planning table's .5 is uncertainty about a useful lagging
effect, not a pass threshold or permission to retry an adverse result.
