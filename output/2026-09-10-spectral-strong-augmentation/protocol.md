# Stable rank-200 augmentation bridge — prospective protocol

Codex — Spectral Optimizer Investigation · 10 September 2026

Status: **implementation preparation; no acquisition yet**. This protocol
implements the [selected design](../2026-09-10-spectral-multiview-clean/design-decision.md).
Freeze it with all acquisition/audit sources and fixtures before the once-only
attempt. It is a new experiment, not a restart of any completed study.

## Question and scope

Does the previously successful stable global rank-200 configuration add
useful late noisy-label robustness beyond ordinary translation augmentation?
The primary comparison is **native200+translation versus raw AdamW+translation**.
Unaugmented arms test whether the historical protection pattern transfers to
this internal holdout discipline. A negative small rank-32 result does not
answer this question about the larger configuration.

Twelve trajectories, exactly three seeds `202609171`, `202609172`, `202609173`
and four arms: raw/none, native200/none, raw/translate, native200/translate.
The four-arm list rotates left by seed index 0, 1, 2. No clean-training
factorial, matrix/legacy/multiview arm, rank/LR search, new augmentation family,
paid acquisition, harmful-behavior work or production default change.

## Source, split and fixed random streams

Read only these already-present official MNIST **training** files under
`data/MNIST/raw`; no download or official-test read:

| File | SHA256 |
| --- | --- |
| train-images-idx3-ubyte | ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db |
| train-labels-idx1-ubyte | 65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5 |

For each seed, create four independent `Generator(PCG64(SeedSequence([s,seed])))`
streams, s=0,1,2,3. Pin NumPy 1.26.4 and the precise draw order below.

1. Stream0: `permutation(60000)`, first 50000 IDs training, next 5000 clean
   validation, final 5000 clean reporting. Save IDs int64 and true labels int64.
   Random splits, not exactly balanced; save class counts. Within-seed roles
   are disjoint; different seeds can overlap. Do not force balance afterward.
2. Stream1: `random(50000) < .9` boolean replacement mask, then
   `integers(0,10,size=50000,dtype=int64)` for **all** positions, including
   unselected ones. Assigned labels are replacement where masked, true
   otherwise. Save replacement labels, selected mask, assigned labels and
   `assigned != true` actual-wrong mask. Replacement may equal true: expected
   actual wrong fraction .81, not .90. Report realized counts; no redraw.
3. Stream2: 72 consecutive `permutation(50000)` draws, cast int32, stacked
   `(72,50000)`. Each row contains local training positions once. Retain a
   shared int64 boundary vector `[0,64,...,49984,50000]` of length783. The
   final batch has16 examples; none are dropped or sampled with replacement.
4. Stream3: one `integers(-2,3,size=(72,50000,2),dtype=int8)` draw, in occurrence
   order. Last axis is **(dy,dx)**. Both translated arms use the identical
   shifts; unaugmented arms use the identical occurrences without shifts.

Use fixed assigned labels across all passes/views. Validation/reporting labels
never train the model or enter the covariance estimate. The data-role change
from historical all-60000 training/official-test reporting is deliberate and
must remain explicit; this is **not exact historical reproduction**.

## Model and update

`torch.manual_seed(seed)` precedes CPU construction of the ordered FP32
Sequential model: Linear(784,256), ReLU, Linear(256,128), ReLU,
Linear(128,10), with ordinary Linear biases and initialization. Move to the
local GPU afterward. Exactly235146 trainable parameters. No dropout, batchnorm,
schedule, clipping, compiled kernel or mixed precision. All four arms within
a seed have exactly the same initial parameter digest.

Raw pixels are uint8 converted to FP32 then divided by FP32(255). Translation
is existing zero-filled integer translation in raw pixel space. Standardize
**after** translation as `(x-FP32(.1307))/FP32(.3081)`; padding is standardized
black, not zero in standardized coordinates. All evaluation inputs are original.

AdamW lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=.01,
amsgrad=False, foreach=False, fused=False, maximize=False,
capturable=False, differentiable=False. Each batch uses mean cross-entropy,
one backward and one Adam step. Moments/counters carry throughout.

Native uses the unchanged `spectral_filter.py` SHA256
`9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`:
rank=200, decay=.99, warmup=100, filter_strength=1,
energy_threshold=None, adaptive='none', normalize='none', weighting='hard',
alpha=1, soft_residual=True, stable_update=True,
relative_eig_tol=1e-8, absolute_eig_floor=0, stabilize_every=100.
The two soft-only options are inert under hard weighting, not a soft arm.
Observe the current incoming batch gradient once, then project that gradient
only on updates >100, then Adam once. No restored mean, gain, lag, history
reset, or change to canonical numerical repair. At update100, raw/native
model+Adam digests and predictions must match within each augmentation mode.

Pin Python/framework provenance; torch=2.11.0+cu128, NumPy=1.26.4. Single CPU
threads (OMP/OpenBLAS/MKL/NumExpr and torch intra/inter-op=1), deterministic
algorithms, no cuDNN benchmark or TF32, CUBLAS_WORKSPACE_CONFIG=:4096:8.
Reproducibility claims are within this pinned environment, not cross-platform.

## Exposure and saved outputs

72 full passes ×50000 =3600000 training-example exposures per arm;
782 updates/pass =56304 updates/arm; 675648 batch-gradient evaluations and
43200000 training-example exposures across12 arms. Historical60×60000 has
the same example exposure but56280 updates and fewer repeats per example.
Save per-step loss, raw/delivered/actual-update norms and exact update/observer/
basis-rank counters; these are diagnostics, not standalone utility estimates.

Readout steps are `(0,100)` followed by `782*epoch` for epochs1…72, exactly74.
At each readout save **one bounded NPZ** containing FP32 logits for the
50000 original training,5000 original validation and5000 original reporting
examples, plus the step. No giant history NPZ or all-history GPU retention.
Store initial model state per seed and complete model/Adam/filter/RNG/gradient
state at warmup and final per arm, with a new truthful two-hidden-layer
snapshot schema. Do not label these as an I9 one-hidden-layer snapshot.

## Validation-only checkpoint selection, then reporting

During acquisition calculate train and validation metrics, but **do not
calculate, print or inspect reporting metrics yet**. Reporting logits may be
saved at each fixed readout; no gradient or selection uses them.

After all12 trajectories finish, choose one readout per arm by minimum
float64 mean validation cross-entropy. Exactly equal minima choose the
earliest step. Fix the selector's numerical reduction: convert logits to
float64, subtract each row maximum, compute NumPy `log(exp(z).sum(axis=1))`
minus the selected shifted logit, then ordered Python `math.fsum` divided
by example count. Equivalent formulas can round exact ties differently;
the independent auditor reconstructs this specific selector and separately
checks metrics using logaddexp. Persist all12 choices and validation references
to exclusive `selection.json`. Verify its readback against independently
recomputed validation metrics from saved validation arrays before calculating
reporting metrics. The independent auditor repeats this validation-first
choice verification before its reporting arithmetic. Audit timestamps/source
control flow and frozen selection receipts; stored reporting data are not a
cryptographic blind, which remains a limitation of an autonomous local study.

Report fixed final clean reporting accuracy and CE as co-primary outcomes.
For each metric list all3 paired native+translation-minus-raw+translation
contrasts and their arithmetic mean; orient CE benefits as raw minus native.
Show all4 absolute endpoints, native-minus-raw without augmentation, each
optimizer's augmentation effect and their difference. No test-based checkpoint,
hyperparameter or seed selection; no significance/equivalence/optimum claim.

Predeclared secondary views, never replacements for an adverse primary:

- Each arm's reporting accuracy/CE at its **one validation-CE-selected** step,
  with the same contrasts and selected exposure. Clean validation availability
  is an explicit resource, not a universally available deployment assumption.
- Useful learning from update100 to final, allseed signs, not just low noisy fit.
- Mean reporting accuracy/CE over epoch endpoints61–72, a fixed late interval.
- All curves; original-train assigned-label and true-label accuracy/CE;
  actually-wrong-subset assigned-fit and true-fit from the same logits.
- Gradient/example counts, synchronized training, augmentation, evaluation,
  serialization/selection and total wall time; this is not a speed benchmark.

A fixed-horizon gain without selected-checkpoint gain supports robustness to
continued training, not necessarily better attainable performance. Useful
native learning without beating raw is reported as such. Neither a win nor
a loss identifies semantic selection or establishes an alignment intervention.
If the strong unaugmented reference fails here, preserve that result without
changing settings. No automatic experiment follows.

## Execution, resource and failure rules

Local RTX3090 only; paid spent/reserved=$0 against the user's $100 allowance.
New unit `spectral-strong-augmentation-001.service`, Type=exec, Restart=no,
KillMode=control-group, CPUQuota=100%, CPUQuotaPeriodSec=100ms,
MemoryMax=16GiB, MemorySwapMax=0, RuntimeMaxSec=12600 (3.5h).
Runner cooperative deadline10800s (3h), GPU allocation≤8GiB,
archive≤8GiB including1MiB reserved failure allowance. Expected1.5–3h is
provisional from historical timings, not guaranteed. Launch only after a
source-level byte inventory, synthetic fixtures and independent source review.

Require≥16GiB free disk on verified `/private-artifacts/storage` mounted from `/dev/RECONFIGURE_FOR_LOCAL_STORAGE`
and≥8GiB free GPU. Accepted existing desktop compute clients are PID2101 up
to512MiB and PID8861 up to128MiB; reject unknown compute clients rather than
displacing them. A fresh mktemp parent under `/tmp/spectral-experiment-artifacts` contains
exactly one `acquisition-001`; refuse reused parents/attempts/units.
Bind all sources/protocol/tests and old dependencies to committed SHA256s.
Never mutate frozen sources after the attempt; no unbounded processes.

Record exclusive attempt, actual unit/invocation/PID, resource receipts and
all artifact hashes. A failed/interrupted run stays failed/partial; no silent
resume, auto-retry, changed tolerance, smaller arm count or discarded adverse
seed. Verify actual handles before any later action.

One independent saved-output NumPy audit is prepared before acquisition:
no torch/checkpoint unpickling/model replay, original training labels only,
stream-hash opaque checkpoints, reconstruct plans/metrics/selection/contrasts
and artifact/source integrity. New audit unit1CPU/2GiB/noSwap,
900s cooperative/1200s hard,64MiB report outside immutable archive.
This audit is not a training replication. A specific artifact defect can be
reported; a null outcome cannot trigger a search disguised as repair.
