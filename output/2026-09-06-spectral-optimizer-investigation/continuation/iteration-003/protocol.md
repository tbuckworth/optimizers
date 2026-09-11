# Iteration 003 prospective protocol: neural gradient retention and realized updates

Prepared before the development-seed pilot. Confirmatory execution requires a
separate parent GO after this protocol and implementation are committed. No
production optimizer code is changed; no paid compute, API, or new dataset is
used. This follows the implementation check (artifact not distributed in this public snapshot).

## Question and scope

With projection rank fixed at 32, does covariance-estimation width 128 versus
32 alter retention of clean-task gradients and fixed training-label corruption
residuals, actual AdamW update geometry, and learning? This is a small neural
measurement experiment, not a generalization benchmark or a causal proof about
semantic usefulness. AdamW momentum alone can ascend the current minibatch.
The unfiltered baseline is therefore measured using the same instrumentation.

The primary probe is an independently drawn batch **from the training examples
with their fixed training labels**. Independence concerns the draw, not unseen
examples: it can overlap the current batch by chance. An auxiliary disjoint
clean-data probe tests retention of a held-out task gradient. Unseen random
labels are not substituted for the corruption that training could memorize.

## Frozen data, model and optimization

- Cached MNIST IDX files only: `data/MNIST/raw`.
  No download fallback. Images are flattened float32 pixels divided by 255;
  no augmentation, centering, or further normalization.
- For each seed 0, 1, 2, a named split RNG permutes the 60,000 official training
  examples. First 5,000 are training; next 5,000 clean validation; next 5,000
  auxiliary clean probes. Remaining examples are unused. These subsets and all
  random plans are identical across arms and corruption levels within a seed.
- Nominal replacement probabilities 0 and .9. A fixed per-example uniform
  replacement decision and independent uniform digit in 0..9 are generated
  once. Replacement may preserve the original digit, so .9 implies expected
  incorrect-label fraction .81. Record realized replacement and incorrect
  fractions. Corruption is never redrawn during optimization or probe calls.
- Deterministic `Linear(784,64) -> ReLU -> Linear(64,10)`, including biases:
  50,890 parameters. Standard PyTorch linear initialization with a separately
  named initialization seed; no dropout, batch normalization, or preprocessing
  fit on validation/test examples.
- Three arms: AdamW; stable covariance width 32 / projection rank 32; stable
  covariance width 128 / projection rank 32. All use AdamW learning rate .001,
  weight decay .01, betas (.9,.999), epsilon 1e-8, `foreach=False`, `fused=False`.
  No schedule, clipping, norm restoration, adaptive rank, soft residual or
  diagonal normalization. Every trainable parameter belongs to one global block.
- 2,000 steps, training batch size 64 sampled with replacement by a named RNG.
  Identical batch indices, initial parameters, labels and probe draws across
  arms. No early termination based on performance.
- Canonical stable covariance class, decay .99, relative eigentolerance 1e-8,
  absolute floor 0, scheduled repair every 100 steps, canonical initial
  overweight. Update its running mean and eigensystem from the **raw current
  training gradient**, then apply `V[:, :min(32,width_current)] V^T` only at
  steps strictly greater than 100. The self-inclusive basis is frozen for all
  measurements of that step. During warmup or without a basis, the applied
  operator is the identity. Width 128 never implies projection rank 128.

## RNG, precision and execution

All stochastic plans use `numpy.random.default_rng(SeedSequence([20260906,3,
seed,stream]))`, with streams: split=0, corruption decision=1, replacement
digit=2, initialization=3, training order=4, primary probes=5, auxiliary probes=6.
The initialization stream produces one uint32 seed for Python, Torch CPU and
Torch CUDA. Separate NumPy generator objects are used; probes use precomputed
indices and consume no training RNG state. Plans and source hashes are saved.

Use the local RTX 3090 only after checking occupancy. Set CPU BLAS/Torch threads
to one, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms on, cuDNN
benchmarking off, and TF32 off. Model/basis/gradients are float32; canonical small
eigensolves remain CPU float64. Scalar diagnostic reductions are float64.
Reproducibility is scoped to recorded software/hardware, not every platform.

## Measurements and mathematical definitions

Record lightweight realized-update metrics every step. At steps 1, 50, 100,
101, and every multiple of 50 thereafter, additionally evaluate independent
probe gradients at the identical **pre-update** parameters:

- `c = grad CE(primary images, original training labels)`;
- `n = grad CE(same primary images, fixed noisy training labels)`;
- `r_corrupt = n-c`;
- `c_aux = grad CE(disjoint auxiliary images, original labels)`.

Primary and auxiliary probe batches contain 256 sampled examples each.
Autograd probes use separate forward passes with `autograd.grad`; do not write
`.grad`, update the covariance, alter the optimizer, or consume random plans.
Record squared norm retention `||P z||²/||z||²` for c, r_corrupt and c_aux,
their original/projected squared norms, and clean-minus-corruption retention.
Zero-denominator ratios are null, not zero: at clean labels r_corrupt is zero
by construction. These are finite-batch loss gradients, not oracle usefulness
or all stochastic gradient noise.

Also retain `c dot r_corrupt`, `(P c) dot (P r_corrupt)`, and unprojected/projected
noisy-gradient energies, with closures of `||n||²=||c||²+||r||²+2c dot r` and
the projected version. Clean and corruption gradients can cancel; separate
retention fractions alone do not determine the retained noisy gradient.

Let g be the raw training gradient, h=P g the actual gradient passed to AdamW,
theta the pre-step parameter vector, delta=theta_after-theta, and
`delta_data = delta + lr*weight_decay*theta`. The latter subtracts nominal
decoupled decay, with floating-point rounding explicitly acknowledged. Record
both displacements, norms, g-dot-displacement, normalized cosine, and signs.
For each displacement d, record d_in=P d, d_out=d-d_in, squared-norm leakage
`||d_out||²/||d||²`, h-dot-d_in and (g-h)-dot-d_out. In exact orthogonal geometry:

`g dot d = h dot d_in + (g-h) dot d_out`.

The float32 basis is only approximately orthonormal: retain the signed closure
residual and check its absolute value divided by `||g||||d||` is at most 1e-4.
This distinguishes harmful leakage (positive residual contribution) from mere
leakage, and allows the in-subspace contribution itself to be positive.
Warmup and AdamW use P=I, so leakage is zero by definition there.

A `closure_robust_leakage_reversal` requires positive total, negative inside,
and positive outside contributions, each exceeding its own sign tolerance
**plus the observed absolute closure residual**. A positive-total/negative-inside
case that fails these stronger margins is labeled unresolved by closure rather
than a supported reversal. Plain derivative signs remain separately recorded.

Classify a directional derivative as positive/negative only beyond
`1e-6*||g||||d|| + 1e-14`; otherwise classify it as numerically near zero.
Use the corresponding two-vector norm product for signed component terms.
Record raw unrounded values as well. At probe steps record same-training-batch
loss before and after the actual update, separated from its linear prediction;
classify changes beyond `1e-6*max(1,abs(loss_before))`, else near zero. Also record
probe c/n/c_aux dot total and decay-subtracted updates. No counterfactual update
is applied to the model.

## Learning outcomes and selection

Evaluate clean validation cross-entropy/accuracy at step 0 and every 100 steps.
Select the checkpoint with smallest validation cross-entropy, ties to the
earliest step. Retain both selected and final model states. All 18 training
runs must finish before loading official test files for evaluation. Then
evaluate both checkpoints on all 10,000 official test examples, and final
training clean/noisy-label and validation outcomes. No test-based choice of
checkpoint, hyperparameter, seed, or method is permitted.

Preserve each seed and report paired width128-minus-width32 and filtered-minus-
AdamW differences. Primary mechanistic summaries average diagnostic steps
101..2000: clean retention, corruption-residual retention (corrupted condition
only), their difference, auxiliary clean retention, actual update leakage,
and signed in/out contributions. Separately summarize early steps101..500 and
late1501..2000; no bin selection after outcomes. Learning summaries include
final and validation-selected test results, not just the more favorable one.
For each window report both the arithmetic mean of finite per-probe retention
ratios and ratio of summed projected energies to summed original energies.
Null denominators remain null and their counts are reported; no undefined
corruption ratio enters the clean-condition mean. Sign frequencies use **all**
steps in the specified window, including near-zero classifications, as their
denominator. Thus the full postwarmup window has 1,900 scalar observations and
39 probe observations; early and late probe counts are 9 and 10. Each seed is
first reduced within its window, then paired seed differences are reported;
steps/probes are not treated as independent replicates. Baseline P=I leakage
is zero by definition and is not evidence that baseline updates obey a useful
learned subspace. No multiple-comparison significance claim is planned.
Three seeds provide a limited paired diagnostic, not broad replication or a
reliable tail-risk/significance estimate. Endogenous trajectories differ across
arms, so this is not a fixed-gradient causal comparison of their subspaces.

## Development pilot and invariant gates

Before any confirmatory seed, run development seed 9876, replacement .9,
220 steps per arm, both instrumentation-on and instrumentation-off: six runs.
This is a prospective resource-only amendment from the initially suggested 180
steps: it includes the first scheduled repair at full estimation width, step200.
No test loading, validation evaluation or accuracy computation during the pilot;
passive preparation of validation tensors is allowed. Do not
display or use pilot loss/gradient results for design choices. Output only
timing, memory, ranks and invariant results/hashes. Record total synchronized
elapsed time, per-step timings, and mean/median time for steps129..220 to
capture width128 steady state; record repair-step200 time separately and peak
allocated/reserved GPU memory.

Check each step's parameter trajectory hash agrees across instrumentation
settings, plus exact final equality of parameters, AdamW state, covariance
state and training gradients. At every instrumented probe, snapshot parameters,
existing `.grad`, optimizer state, all mutable estimator state, model training
flags, Torch CPU/CUDA RNG states, Python RNG state; assert bitwise unchanged
after probe calls. All training and probe parameter gradients must exist and
be finite. Check projected rank <=32 and estimation rank <=configured width;
width128 must reach128 by step160 for representative timing. Check projection
closure tolerance above, finite parameters, and basis orthogonality <=5e-3.
Any gate failure stops launch; preserve failure report, diagnose, and document
any code/protocol correction before a fresh development pilot. Timing/resource
changes require explicit amendment and cannot use observed accuracy.

Unit tests cover warmup and width/projection separation, primary corruption
decomposition and zero-corruption null ratios, probe state invariance, the
in/out identity and a harmful-leakage example, decay subtraction, and validation
selection tie behavior. They also verify window observation counts and joint
gradient-energy closures. The full run requires matching committed source hashes
and an explicit GO argument; no full experiment is part of the current task.

## Artifacts and limits

Keep protocol/source/tests, development report, full execution provenance,
per-step/per-probe JSON, all validation trajectories and seed-level results.
Large RNG-plan archives/checkpoints stay local with tracked hashes/sizes. Failed
gates and incomplete runs remain visible. A positive retention difference alone
does not establish better generalization; a null learning effect does not erase
measured geometry. Norm-matched causal controls, fixed-gradient replays and
larger-task transfer remain subsequent experiments, not implied results here.

### Provenance-only hardening before confirmatory freeze

After the first passing development pilot, but before any confirmatory run,
the parent approved binding the training IDX file hashes and the separate
`analysis-plan.md`, `summarize_results.py`, and `test_summary.py` hashes into the
pilot/full-run gate. The full launch checks all source files are committed and
unchanged from the passing pilot, and that training bytes match the pilot.
In-arm exceptions now preserve active seed/arm/step/phase and every completed
raw step/probe/validation record in a partial-failure artifact. A synthetic
failure-preservation unit test was added. These changes do not alter training,
probe definitions, thresholds or selection. The first pilot/report are retained
as attempt001, and the same development pilot is repeated on the hardened
source before full launch. No performance-based tuning or pilot accuracy
inspection is involved.
