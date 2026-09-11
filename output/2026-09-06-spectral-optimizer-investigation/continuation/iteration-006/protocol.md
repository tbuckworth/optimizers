# Iteration 006 prospective protocol: current versus lagged delivery

6 September 2026. Implementation follows the amended [design](design-intent.md),
[analysis plan](analysis-plan.md), [implementation decision](challenge/decision.md)
and best-practices check (artifact not distributed in this public snapshot). **Implementation and synthetic
CPU tests are authorized; no pilot, dataset access, GPU execution or confirmation
is authorized by this document.** Parent-owned commits and separate launch
decisions remain mandatory. Earlier sources/results remain unchanged.

## Fixed scientific question and cells

Compare six policies on seeds 6,7,8, each with nominal fixed replacement 0 and
.9: exactly 36 cells `(seed, noise, arm)`. Arm order is AdamW, current32,
lagged32, lagged32_current_norm, scalar_current32, scalar_lagged32. All later-
listed-minus-earlier-listed pairs are retained, separately by condition.

The four primary groups are lagged32-minus-current32 and
lagged32_current_norm-minus-current32 test accuracy under maximum-validation-
accuracy selection, separately for clean/noisy training. Three paired bundles
are the replication unit, not six independent clean/noisy seeds or 72,000 steps.
Do not select a favorable primary group. Null/mixed/adverse results are valid
findings; no outcome-driven seeds, arms, windows or retries are permitted.

Use the unchanged iteration003 data/model/optimizer helpers: cached training
MNIST IDX, float32 images divided by 255, 5,000 training, 5,000 clean validation
and disjoint 5,000 auxiliary examples; 784–64–ReLU–10 model with 50,890 parameters;
batch 64 with replacement; 2,000 AdamW updates, lr .001, weight decay .01,
betas (.9,.999), epsilon 1e-8, foreach/fused false. No schedule, clipping,
augmentation, new probes, reference solves or hyperparameter tuning.

Preserve the RNG namespace `SeedSequence([20260906,3,seed,stream])` and all plan
arrays, including unused probe plans. Each seed's initialization, split and
batches are paired across arms/conditions; labels differ by the fixed condition.
Replacement can equal the clean label. Preserve the helper's realized-fraction
metadata as float32 means; nearest count/5000 is checked within `3e-8`, not
mistaken for exact count division. Accuracy metrics themselves use correct/count.
There is no fresh dataset or independent official-test confirmation here.

## Observation, delivery and arithmetic

All five observing arms use canonical stable width32, beta .99, relative
eigenvalue floor 1e-8, absolute floor zero and scheduled repair every 100
observations. Observe each raw gradient once, including warmup; do not feed the
filtered gradient into covariance. AdamW has no observer.

Steps 1..100 deliver raw gradients identically for every arm. Candidate metrics
are explicitly null during warmup even though observing states exist. Before
each active step's observation, independently clone the native previous basis
`B_previous`; then update with the raw gradient once and obtain `B_current`.
At unchanged pre-Adam parameters compute native float32 `c=B_current(B_current.T g)`
and `l=B_previous(B_previous.T g)`. An absent basis has identity action.
Never use a warmup-null returned projection as evidence that the stored basis
is absent, and never substitute a QR projector.

After warmup the delivered tensors are:

| Arm | Delivery |
|---|---|
| adamw | g |
| current32 | c |
| lagged32 | l |
| lagged32_current_norm | l times norm(c)/norm(l) |
| scalar_current32 | g times norm(c)/norm(g) |
| scalar_lagged32 | g times norm(l)/norm(g) |

All candidates/targets belong to that policy's own state. Norms use float64
reductions. Every rescaling multiplies a float64 copy of the direction by its
float64 ratio and casts once to float32. Scalar ratios alone must be <=1.005;
the restored-lag ratio is not capped. Nonzero scalar identity delivery is not
rank32, despite its width32 observer.

Check every input for finiteness before zero shortcuts. Zero target requires
exact zero delivery; positive target with zero direction is fatal. Positive
target requires positive delivered norm and relative norm error <=1e-6, with
no absolute-floor exemption. Normalized direction error must be <=1e-6 for
nonzero vectors. Overflow, total underflow or excessive quantization fails;
there is no fallback, scale clipping or exclusion of a failed arm.

Read back every parameter's actual `.grad` after assignment and apply delivery
gates there, including with optional instrumentation off. Always call AdamW,
also for zero tensors; None gradients are prohibited. Warmup hash equality
checks raw/applied gradient values and parameters across all six arms within
each seed/condition, plus full model/optimizer/RNG state. All five observing
arms also match complete observer states/hashes throughout warmup. Clean/noisy
warmups are not required to match each other.

## Measurement and checkpoint contract

Retain every step's scalar norms/energies, current/lagged/raw ratios, candidate
retention difference, mutual cosine, raw/applied cosine, policy scale and
norm/direction errors. Candidate values are null for AdamW, not a fictional
raw-gradient measurement. Zero denominators/cosines have explicit null reasons.
Retain every post-step parameter hash, warmup raw/applied/observer hashes,
actual estimator ranks and actual `stabilization_count` counters.

Compute total parameter displacement from before/after vectors and nominal
decay-subtracted displacement by adding `.001*.01*theta_before`. Log norms,
energies, raw/applied gradient dots, cosines and signs. Sign tolerance is
`1e-6*gradient_norm*update_norm+1e-14`; positive-dot denominators retain all
scheduled steps. These are directional measurements, not finite-loss changes.
No projected-span leakage decomposition or exact-orthogonality identity is used.

Full state witnesses include parameters, gradients, optimizer, complete observer,
module modes, Python RNG, NumPy-global RNG, Torch CPU RNG and all CUDA RNG states.
Local plan generators are precomputed and discarded, not advanced by training.
Check unchanged state around optional measurements at steps 1,100,101 and every
100th step, plus the final step. This is 22 checks per full cell, or five in an
instrumented 220-step pilot. The unchanged model has no dropout or batch norm.
Mandatory delivery validation runs in both instrumented and uninstrumented
paths; the optional measurement witness protects the additional update-metric
calculations. Policy/validation statelessness is separately checked by synthetic
tests and complete on/off histories, not inferred from the optional witness alone.

Validation grid is exactly 0,100,...,2000. Independently retain earliest strict
minimum-CE and maximum-accuracy checkpoints, with no cross-metric tie-break;
step0 and warmup100 are eligible. Save independently CPU-cloned tensor-only
states for `final`, `min_val_ce`, `max_val_accuracy`, `warmup100` even when
steps coincide. Final clean/noisy training CE/accuracy use identical fixed-state
logits and never select checkpoints. All training/validation counts are 5,000.

The official test loader is behind exact 36-cell identity, completed step/grid,
four-checkpoint name, artifact hash and tensor-state availability checks.
The gate also rederives earliest strict selectors from finite validation rows
and requires identical tensor-state hashes whenever checkpoint steps coincide.
Only after all training/selection completes may official test files be opened.
Evaluate every saved checkpoint on 10,000 examples: exactly 144 evaluations.
No test-based tie-break, checkpoint selection or pilot outcome is allowed.

The complete [raw schema](result-schema.md) and parent analysis plan fix all
keys, nulls, counts and aggregation. Main geometry windows are 101–2000,
101–500 and 1501–2000. Additional candidate-retention/difference/cosine windows
are named `scheduled_repair` (active step divisible by100) and
`other_steps`. Canonical code can also trigger drift repairs; these
groups do not imply that every other step is repair-free or isolate a causal
repair effect. Use actual repair counters, never assume floor(step/100).

## Pilot, provenance, resources and failure

The proposed development pilot requires separate parent GO: seed9880/noise.9,
220 updates per arm, instrumentation off/on, exactly12 traces. No training
accuracy, validation/CE selection or official-test evaluation is performed.
Do not retain learning/gradient diagnostic outcomes for tuning. Retain timings,
memory, finite/rank/counter/gate records, all220 parameter hashes and warmup
hashes. Every on/off history and complete final state must match; compare
within-condition core and five-observer warmup states. The pilot exercises
step101 and scheduled repair200; synthetic tests separately distinguish the
current/lagged and undefined/extreme-scaling cases.

Before pilot/full, require a complete committed 15-file scientific-source map
from the raw schema, including policy/tests, harness/storage/tests, old helper,
canonical optimizer, protocol/schema/design, analysis/summarizer/tests,
best-practices and implementation decision. Imported code bytes must match.
Full execution additionally requires the passing pilot's exact source/data/
environment bytes and hash-bound pilot plan/invariant artifacts. Parent launch
decisions and unrelated theory files are not extra scientific-source members.
No old-source globals are monkeypatched. Recheck sources during and after runs.

Use one local RTX3090, deterministic algorithms, one CPU thread, TF32 off,
cuDNN benchmark false and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Inspect occupancy
without treating this harness's own PID as a foreign job. Existing desktop
processes are allowed, unrelated compute jobs are not. Require >=8 GiB free GPU
and >=16 GiB available host RAM. Numerical environment must match the pilot.

Exclusively create an attempt root below `/tmp/spectral-experiment-artifacts` and verify its
actual mount is `/private-artifacts/storage` on `/dev/RECONFIGURE_FOR_LOCAL_STORAGE` or the documented UUID. All plans,
checkpoint bundles and scalar/hash-only raw histories go there; no full
gradient/basis histories are authorized. At most1 GiB total artifact bytes,
8 GiB peak allocated GPU memory and12 GiB RSS; require2 GiB initial disk
headroom and >=1 GiB plus pending payload before writes. Tracked small JSON
metadata is limited to20 MiB. Do not use the nearly full workspace for bulk.

Cooperative wall caps are180 seconds pilot and900 seconds full, including
loop/evaluation/hash/I/O work. Check time/memory each step and around writes/
evaluations. Operations are not force-killed mid-write; a detected overrun
preserves failure and stops without automatic retry or relaxed limits.

Preserve completed-cell checkpoint and pre-test JSON artifacts as each cell
finishes. Final tested JSON uses a separate exclusive filename. Failure retains
partial scalar rows/validation and available partial checkpoint tensors, plus
a compact finite error record and all completed bindings. Preserve serialization
or storage failures explicitly if bulk saving also fails. Never overwrite a
previous attempt, delete old evidence, or convert a partial study into complete.

CLI requires exactly `--pilot --development-go` or `--full --confirmatory-go`,
as appropriate, and those flags must reflect an actual separate parent approval.
CPU tests must use `CUDA_VISIBLE_DEVICES=''`; they use only synthetic tensors,
plans and temporary artifacts. No paid compute/API or production edit is allowed.

## Interpretation boundary

Selection is part of the primary estimand; endpoint results give a separate
fixed-exposure view. Restored lag has lagged direction but self-inclusive
current magnitude. Own-state scalar controls do not match numerical cross-arm
gradient histories or AdamW updates, and the policies are not a complete
direction-by-magnitude factorial. Report all four primary groups, seeds and
mandatory secondary contrasts without equivalence, population or uniquely
identified semantic-denoising claims. A negative result requires reporting,
not changing the intervention to obtain a favorable answer.
