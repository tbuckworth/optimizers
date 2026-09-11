# Iteration 005 prospective protocol: common-gradient covariance replay

Design only, prepared after iteration 004 commit
`c52d2ceb56fab7fcf88497738c6aae28e7d26e4c` and before any iteration 005
MNIST execution. This implements [design-intent.md](design-intent.md), follows
best-practices-check.md (artifact not distributed in this public snapshot), and uses the independently
checked reference identities (artifact not distributed in this public snapshot). It does not authorize
implementation, a pilot, or confirmation. Each requires the parent's approval;
confirmation additionally requires the reviewed sources and analysis committed.
No previous frozen source or production optimizer is changed.

## Question, comparison and units

On one identical AdamW gradient stream per seed, compare canonical stable
estimation widths **32 and 128**, evaluating the leading **32 columns** of each.
Both observers are measurement-only: neither changes gradients, parameters or
AdamW state. These are not two training policies.

The single primary diagnostic is reference covariance energy captured by the
rank-32 **orthonormalized span** at **step 2000**, following
[analysis-intent.md](analysis-intent.md). Report width128 minus width32; a positive
difference favors width128 on this matched-rank fidelity measure. Boundary ties
do not invalidate its optimal-energy denominator. Rank-32 normalized squared
projector distance, where its positive-rank/gap gates pass, is secondary; negative
differences favor width128 on that measure. Full-width
covariance reconstruction error is mandatory secondary evidence with explicitly
**unequal representational capacity**. Native-operator probe retention and
clean-minus-corruption retention are separate secondary quantities. Improved
covariance fidelity need not improve useful retention or a training policy.

Use seed bundles **3, 4, 5**, each **2,000 steps**, nominal fixed replacement
**.9**. They are the same three AdamW trajectories used in iteration 004, not
fresh learning replications or an independent dataset. Snapshot steps are
**200, 500, 1000, 2000** with no outcome-selected additions. These four overlapping
history prefixes are dependent repeated measurements, not twelve independent
replicates. Preserve all seed/snapshot results, including null or adverse ones.

## Fixed trajectory and historical anchors

Reuse the unchanged iteration-003 model/data/optimizer helper recipe imported
by iteration 004: cached MNIST training IDX images and labels; float32 flattened
images divided by 255; MLP 784–64–ReLU–10 with biases and 50,890 parameters;
5,000 training examples; batch 64 with replacement; AdamW lr .001, weight decay
.01, betas (.9, .999), epsilon 1e-8, foreach/fused false; no schedule, clipping,
augmentation, dropout or batch normalization. No filter is ever applied.

Regenerate and compare every array of each saved iteration-004 plan, using its
exact NumPy namespace `SeedSequence([20260906,3,seed,stream])`: stream 0 split,
1 replacement decisions, 2 replacement digits, 3 initialization, 4 training
batches, 5 primary probe batches, 6 auxiliary probe batches. A replacement may
equal the original digit; preserve realized replacement and incorrect fractions.
Training uses the same fixed noisy labels, not labels redrawn on each visit.

Use the local RTX 3090, deterministic algorithms, TF32 off, cuDNN benchmark
false, one CPU thread and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Bind Python,
NumPy, PyTorch/CUDA versions and device identity to the accepted pilot and
historical execution manifest. A changed historical numerical environment
requires review before claiming exact reproduction.

Before replay, verify source/data/plan/result/checkpoint hashes against the
audited iteration-004 manifests. Load its three AdamW checkpoint bundles with
`weights_only=True` on CPU. Match every retained parameter tensor bitwise at
each historical named checkpoint's recorded step: `warmup100`, `final`,
`min_val_ce`, `max_val_accuracy`, including step 0 if present and duplicate
selectors without dropping their checks. These steps are historical verification
anchors, not new selectors. Match all 100 saved warmup parameter/raw/applied
gradient hashes and the complete step-100 core-state hash (parameters, gradients,
AdamW moments/step counters, modes and RNG, excluding observers). Every replay
step must also exactly reproduce the stored AdamW raw squared norm and total/
decay-subtracted displacement squared norms and raw-gradient dot products.

Historical full-run parameter-vector and gradient-vector hashes were **not**
saved beyond warmup. Therefore this verifies all available retained anchors and
scalar diagnostics, not independent historical full-vector equality at every
step. Save new raw-gradient, innovation and per-step parameter hashes to support
a fuller future audit. Failed exact comparisons stop the replay; do not silently
relax them to an approximate match.

Do not open official test IDX files or compute any new accuracy, validation
criterion or selector. Historical validation forwards are omitted: the model
has no mutable forward buffers or stochastic layers, and a synthetic state/RNG
invariance test of the unchanged evaluation helper must establish this omission
does not change persistent training state. Existing historical checkpoints and
scalar diagnostics remain the confirmatory reproduction gates.

## Observation, state timing and exact reference weights

Let `theta_(t-1)` be parameters before AdamW step t and `g_t` its raw float32
training gradient. At each step, compute `g_t` once and give the exact same tensor
values to both canonical observers, incrementing each step counter before its
stable update. Never call the wrapper's gradient-setting or optimizer-step path.
Use decay beta=.99, relative eigenvalue floor 1e-8, absolute floor 0, repair period
100, canonical startup overweight, hard weighting, no normalization or adaptive
rank. Observer warmup is 100 but does not suppress observation. At scheduled
states both are beyond warmup; evaluate only their leading 32 columns regardless
of estimation width. Inspect model `.grad` immediately before AdamW: it must be
bitwise equal to the original raw gradient.

The canonical float32 mean arithmetic is:

    m_1 = clone(g_1)
    m_t = fl32_add(fl32_mul(m_(t-1), beta), g_t, alpha=1-beta)
    c_t = fl32(g_t - m_t).

Here `fl32_add` specifies the actual sequential in-place PyTorch
`mul_(beta).add_(g_t, alpha=1-beta)`, not an algebraically rearranged expression
or float64 mean. Compare both observer means and an independently maintained
mean using this operation sequence bitwise on every step. Save the same rounded
`c_t` used by this check. `c_1=0`; do not assume the first accepted innovation is
step 2. Log `s`, the first step at which initialization accepts
`c_t.norm()>0` in canonical float32 and creates a basis; require the same s for
both widths. Require prior innovations to be exactly zero: a nonzero vector
whose norm underflows before acceptance is a reference-convention failure.

Before s set the reference C_t=0. At/after s define the **algebraic uncapped
finite-history target on rounded innovations**:

    C_s = c_s c_s^T
    C_t = beta C_(t-1) + (1-beta)c_t c_t^T
        = beta^(t-s)c_s c_s^T
          + sum_(j=s+1)^t (1-beta) beta^(t-j)c_j c_j^T.

Thus the first accepted innovation has weight beta^(t-s), not
(1-beta)beta^(t-s). Construct float64 weighted columns X_t from saved float32
innovations: the first column is beta^((t-s)/2)c_s and later columns are
sqrt(1-beta)beta^((t-j)/2)c_j. All weighting and reference linear algebra use
float64; no history tail is dropped, no bias correction is applied, and no
covariance-sized p-by-p array is formed. Save weights as well as their defining
indices and beta.

The target is not a population covariance, an infinite-history covariance, or
bitwise equivalence to an uncapped execution of production code. Production
also has residual rejection, eigenvalue pruning, float32 basis normalization,
rotation and repair; even its startup V/S outer product need not exactly equal
the float64 outer product of the rounded c_s. Discrepancies may contain all of
these effects, not truncation alone. Log basis creation, ranks and repair
counters. Any transition from an established basis to no basis, or subsequent
reinitialization, stops the comparison for review rather than silently assigning
a fresh reference startup weight.

At each scheduled t, copy the previous leading basis B_(t-1) **before** observation.
Then observe g_t, copy both updated states, and evaluate probes at the unchanged
`theta_(t-1)` **before** AdamW. Consequently C_t/B_t include g_t. Save phase,
step, parameter hash, means, complete observer state, previous leading bases,
and current g_t. After the measurements, deliver unchanged g_t to AdamW.

## Independent probes and native action

At each scheduled t, use row t-1 of the existing stream-5 plan to sample 256
training examples independently of the training batch draw. On that identical
batch/state, obtain `v_clean` and `v_noisy` by `autograd.grad` with clean and
**fixed training noisy** labels; define the float32 residual
`v_residual=v_noisy-v_clean`. Use row t-1 of stream 6 for a 256-example clean
auxiliary gradient on the disjoint 5,000-example auxiliary pool. Sampling
independence does not imply distinct examples or gradient independence conditional
on the learned state. Auxiliary gradients are held-out task probes; residuals
are the actual fixed training corruption, not newly sampled unseen noise.

Compute this probe bundle once, save the four full vectors, and use exactly the
same vectors for both observers and the reference. Never feed probes into either
observer. Require complete model/gradient/optimizer/mode/Python/NumPy/Torch/CUDA
RNG and both observer states to be bitwise unchanged around probe calls and
other measurement-only operations. No dropout or batch normalization is present.

For an observer with saved float32 leading columns B, the native action is
`native_P(v)=B @ (B.T @ v)` in float32. Reductions are float64. Report

    R_B(v) = ||native_P(v)||^2 / ||v||^2,
    S_B = R_B(v_clean) - R_B(v_residual).

If an observer has no basis, its canonical native action is **identity**, not
the zero map: nonzero vectors have retention 1, zero vectors remain null,
Chat=0, and matched-rank span diagnostics are null. Apply the same identity rule
to an absent previous basis in the self-inclusion diagnostic. At the scheduled
post-warmup snapshots an available basis always uses the explicit rank-32 cap.

Zero input norm makes that retention and any difference requiring it null; do
not insert zero. Preserve input/output energies. Report all four probe types,
clean/residual cosine, their signed cross term before and after projection,
noisy-gradient retention, and raw/projected energy-closure residuals for
`||n||^2 = ||c||^2 + ||r||^2 + 2 c^T r`. Here c in this identity denotes the
clean probe, not the temporal innovation. The identity is approximate for the
saved rounded residual and native float32 action; gate its absolute residual
divided by the sum of all absolute contributing terms at 1e-5, with 1e-30
denominator floor. Do not explain noisy-gradient behavior from separate
retentions while ignoring cancellation or fixed-label conditioning.

Report the secondary self-inclusion increment
`R_(B_t)(g_t)-R_(B_(t-1))(g_t)` for each observer at the four snapshots.
These are current-gradient measurements, not independent-probe evidence, and
do not require another reference solve. Reference probe retention uses the
float64 U_32 operator below only when its rank/gap gates pass. Also compare each
native action with the float64 action of its saved B; require relative action
error `||native_P(v)-B64(B64^T v64)||/max(||v||,1e-30)<=1e-5`.
Do not silently QR-repair the native operator or clamp its retentions to [0,1].

## Dual reference and metric equations

After all three trajectory replays pass, process each saved seed/snapshot in
fixed seed/step order. Form K=X^T X in float64 on the local GPU; check symmetry,
then explicitly symmetrize `(K+K.T)/2`. Solve on CPU, one thread, with
`torch.linalg.eigh`, reverse to descending eigenpairs, and save the full spectrum,
the Gram matrix and leading 33 dual vectors. For positive lambda_i construct
`u_i=X q_i/sqrt(lambda_i)` and save the leading reference vectors needed below.
No backward pass through an eigensystem is permitted.

Let reference eigenvalues be lambda_1>=lambda_2>=..., and
`E32=sum_(i=1)^min(32,n) max(lambda_i,0)`; this optimal orthoprojector energy
remains meaningful at a boundary tie. It does not use a numerically pruned
reconstruction of C. For each stored full observer basis V with variances
`d_i=S_i^2`, define Chat=V diag(d) V^T. Cast saved V to float64 without repairing
it. With G=V^T V and A=V^T X:

    F_ref^2 = ||K||_F^2
    F_est^2 = sum_ij d_i d_j G_ij^2
    inner = sum_i d_i ||A_i,:||^2
    relative_covariance_error = sqrt(F_ref^2+F_est^2-2*inner)/F_ref.

Use all stored columns for this explicitly unequal-capacity secondary metric;
also save its terms and the estimator trace `sum_i d_i G_ii`. Do not replace
F_est^2 with sum d_i^2 unless exact orthogonality has actually been established.

For matched-rank diagnostics take B=V[:,:32], obtain float64 reduced QR `B=QR`,
and verify Q is orthonormal and B has full column rank. QR is an **analysis-only
span construction**, not an observer change. Define:

    span_energy_fraction = ||Q^T X||_F^2 / E32
    span_projector_distance = 1 - ||Q^T U32||_F^2 / 32.

The primary span fraction requires 32 observer columns, at least 32 numerically
positive reference eigenvalues and E32>0, but **not** the boundary-gap gate.
The distance additionally requires orthonormal Q/U32 and the reference
boundary-gap gate. Lower is closer. Save unnormalized energies, overlaps and
rank diagnostics.
Also report the actual stored-operator quantities, with H=B^T B and Z=B^T X:

    trace_P_C = ||Z||_F^2
    represented_operator_output_energy = sum(Z * (H @ Z))
    represented_operator_energy_fraction = represented_operator_output_energy / E32.

These float64 algebraic diagnostics describe the operator represented by saved
B, apart from additional native float32 application roundoff. `trace(P C)` and
`trace(P C P)` are not interchangeable. Only the orthonormal-span fraction has
the exact upper bound 1; the represented/native operator can slightly exceed
that bound. Better reference capture is not proof of better useful retention.

## Numerical, null and failure rules

All input tensors, weights, operator states, Gram matrices, eigenvalues and
reported non-null scalars must be finite. The following tolerances are frozen
before any MNIST reference calculation; diagnostic nulls are not runtime failures.

| Check | Rule |
|---|---|
| Gram symmetry before explicit symmetrization | `||K-K.T||F/max(||K||F,1e-300) <= 1e-12` |
| Reference negative eigenvalues | No eigenvalue below `-1e-10*lambda_1`; retain raw values, use zero only for tiny negative values in spectral-energy sums |
| Numerical positive rank | Count eigenvalues strictly above `1e-10*lambda_1`; report threshold and count |
| Unique rank-32 reference | At least 32 numerically positive values and `(lambda_32-max(lambda_33,0))/lambda_1 > 1e-6`; missing lambda_33 is zero |
| Reference eigensolve | Every saved dual eigenvector residual `||Kq-lambda*q||/lambda_1 <= 1e-8`; full dual orthogonality `||Qdual.T Qdual-I||F <= 1e-8` |
| Reference spectral checks | Trace/spectrum-sum and Frobenius/spectrum-norm relative discrepancies <=1e-9 |
| Mapped leading covariance vectors | For each mapped vector, `||X(X.T u)-lambda*u||/lambda_1 <=1e-8`; `||U.T U-I||F <=1e-6` |
| Native observer basis | Saved full `||V.T V-I||F <=5e-3`; no analysis-side repair of V |
| QR span validity | 32 columns; smallest/largest singular-value ratio of B >1e-8; `||Q.T Q-I||F <=1e-10` |
| Covariance-error cancellation | Save raw squared error; clamp a negative value only if its magnitude <=`1e-12*(F_ref^2+F_est^2+2*abs(inner))`; otherwise fail |
| Bounded span metrics | Permit only [-1e-6,1+1e-6] for distance/energy fraction; retain raw values rather than clamp |

For an exactly zero X, C and E32 are zero, positive rank is zero, and normalized
reference metrics are null. All-zero Gram eigenvalues with nonzero X, or a
nonpositive leading eigenvalue for nonzero X, fail. Tiny positive eigenvalues
are not discarded from C or its Frobenius norm; the positive threshold only
controls inverse-square-root mapping and rank claims. Construct mapped vectors
only for leading modes that pass this threshold; do not amplify numerical nulls.

A failed positive-rank or boundary-gap criterion makes the unique reference
projector, reference-probe retentions and projector distances null with a reason.
Reference optimal energy, covariance errors and actual-observer retentions
remain reportable; a gap failure alone also leaves the primary span fraction
reportable. Insufficient reference positive rank makes that matched-rank fraction
null while retaining its unnormalized energy. An observer with fewer than 32
columns has null matched-rank span comparisons; its available-rank native
retention and covariance error are
still reported with actual rank. Never synthesize missing directions or replace
a null with a favorable/zero score. Numerical residual, finite, reset, state or
trajectory failures stop execution; preserve the failure before any review.

## Aggregation fixed before outcomes

Report all twelve scheduled seed/state rows and both observer values. For each
metric, form width128-minus-width32 within the same seed/state. The primary is
the **step-2000 span-energy-fraction contrast**, with all three seed differences
and their descriptive mean, median, range and sample SD. The pooled primary
requires all three valid final-state pairs; otherwise report it null with
counts/reasons while retaining individual pairs. Do not obtain a new primary
by dropping a difficult spectrum or replacing step 2000 with another state.

For every metric additionally show fixed-step paired means across seeds with
valid counts. Equal-weight four-snapshot means within each seed are secondary,
defined only if all four values exist; their across-seed summary requires all
three complete seed means and reports mean, median, range and sample SD. Any
available-case summary is explicitly secondary and states its mask. Probe
summaries use means of per-probe ratios, not ratios of summed energies; retain
summed input/output energies as separately labeled secondary quantities. No
step-level inference, p-values, equivalence claim, best-snapshot selection or
post-outcome tuning.

## Prospective tests, pilot and resource guards

Before any MNIST pilot, require synthetic CPU tests of sequential rounded mean
updates and first-nonzero acceptance (including delayed and absent initialization),
startup weight versus regular-weight counterexamples, weighted recurrence versus
dense/dual covariance for beta .9/.99/.999, orientation and descending ordering,
rank deficiency, exact/near boundary ties, covariance cancellation, deliberately
nonorthogonal V, native-versus-represented action, all zero/null cases, reset
detection, checkpoint cloning, omitted-evaluation state invariance, unchanged
observer/model/optimizer/RNG state around probes, and finite failure preservation.
Reuse the mathematical checks in `check_reference_identities.py` without edits;
add implementation-specific tests separately after implementation approval.

Proposed pilot, requiring separate parent GO: development seed **9878**, .9
replacement, **220 steps**, with observers/probes off and on. Require all 220
post-step parameter hashes and final complete core training states to match
bitwise. The instrumented trace checks both widths together; scheduled pilot
probes/state copies are at 200 and 220, and steady-state timing covers 129..220,
including repair step 200. No accuracy, loss, retention or fidelity outcomes
are retained or inspected for choices; retain timing, ranks, hashes and numerical/
state gates only. This short trace does not bound a full reference solve.

Therefore also require one **worst-size synthetic reference resource check**:
p=50,890, n=2,000 float32 innovation columns, fixed dedicated seed **9879**,
first column exactly zero, remaining standard-normal columns, s=2, beta=.99.
Run the same weighted-Gram/eigh/top-33 mapping and residual path plus the full
raw/innovation artifact write/hash path. This is a numerical/resource check,
not an MNIST outcome. Save only resource/gate summaries and provenance, not
synthetic performance evidence. Pilot wall-time cap is 10 minutes. Report
measured training, Gram, solve, mapping, hashing/I/O times separately, peak RSS,
and allocated/reserved GPU memory. Parent reviews extrapolation before GO;
confirmation wall-time cap is 15 minutes including replay and references.
Exceeding a cap preserves partial artifacts and stops without automatic retry.

Before pilot and confirmation, inspect GPU occupancy; do not compete with another
compute job. Require at least 8 GiB free GPU memory, 16 GiB available host RAM,
and 10 GiB free on the verified large-volume filesystem. Pilot peaks must stay
below 8 GiB allocated GPU memory and 12 GiB RSS. These are resource gates, not
instructions to alter ranks, dtype, snapshot schedule or precision to fit.

The workspace has only approximately 2.2 GB free. **All bulky tensors, binary
plans/checkpoints, partial streams and synthetic scratch files must be under a
new uniquely scoped directory inside `/tmp/spectral-experiment-artifacts/`.** Verify its resolved
path and mount are `/private-artifacts/storage` on `/dev/RECONFIGURE_FOR_LOCAL_STORAGE` (or the documented volume UUID),
not an unmounted lookalike or the workspace filesystem; create exclusively and
reject overwrite. Raw and innovation streams alone occupy
`2*3*2000*50890*4 = 2,442,720,000` bytes. Store each float32 stream once, not every
weighted prefix; build X on demand and release it after each reference. Budget
at most 6 GiB persistent artifacts for confirmation. Small code/JSON/Markdown
metadata may live in iteration005, with a 20 MiB metadata budget. Check disk
headroom before stream allocation and each reference write; no deletion of
previous evidence or automatic cleanup to recover capacity.

## Provenance, recoverability and launch boundary

Bind SHA256, sizes, shapes, dtypes, axes, seed/step/phase and exact paths for raw
streams, innovations, means, previous/current bases, S, probe vectors, reference
weights, Gram matrices, spectra/dual/mapped vectors and checkpoint comparison
artifacts. Keep original raw arrays unrounded; hashes may describe bulk files
outside Git without copying them into the workspace. Record row completion
counts so preallocated unwritten tails cannot appear complete. Flush completed
chunks before advancing durable progress. Never overwrite another attempt.

Before confirmation, require committed-byte equality and passing pilot hashes
for protocol, design intent, best-practices/reference identity files, harness,
tests, analysis/summarizer/tests, all imported helpers and canonical source.
Bind both training IDX hashes, saved iteration-004 plan/checkpoint/raw-result
hashes, original execution commit and audited-results commit. Require the same
source/data bytes as the accepted pilot plus explicit parent GO. A subsequent
source/protocol change preserves the old attempt and invalidates its launch gate.

Manifest status transitions distinguish replay, reference and completed stages;
record UTC timestamps, active seed/step/snapshot/phase, completed rows, resource
observations and every gate. Confirmation failure artifacts preserve partial
streams/states/metrics with finite JSON serialization and a separate error record;
pilot failures preserve resource/invariant context without learning/retention
outcome arrays. No amendment or rerun follows failure without parent review.
No training policy comparison, new test evaluation, paid compute, API call,
source optimizer edit, or automatic enlargement is authorized by this protocol.
