# Prospective neural anchor-clustering pilot

Status: fixed design, no MNIST acquisition run by the implementation agent.
The user explicitly requested a neural clustering experiment after the synthetic
pilot; main reviews/freezes sources and launches the one exclusive batch.

## Question and interpretation

Can parameter-gradient clustering support useful neural learning, and does it
change late memorization of fixed random labels? Report clean held-out accuracy
**and** cross-entropy together, plus training fit. An action that merely prevents
learning is not evidence of useful anti-memorization. This is a three-seed
exploratory policy comparison, not an optimizer-superiority, AI-safety,
semantic-cluster, or best-checkpoint claim. Preserve null/adverse outcomes.

The synthetic pilot and its results remain unchanged. Its graph/action helpers
are reused byte-identically. The neural adaptation deliberately replaces its
uncentered streaming second-moment estimator with the repository's accepted
stable **centered** covariance observer, also used by the hard-spectral control.
Thus this does not reproduce the synthetic pilot's estimator, and the comparison
does not isolate clustering from every difference in projection timing/geometry.

## Frozen roster and recipe

- Seeds: `202609091`, `202609092`, `202609093`; independent new RNG namespace
  `SeedSequence([20260909,31,seed,stream])`, streams 0 split, 1 initialization,
  2 replacement mask, 3 replacement digit, 4 batches.
- Per seed: 5,000 training and 5,000 disjoint clean held-out images sampled from
  the original 60,000 MNIST **training** examples. No official test file is read.
  Image pixels divided by 255; no augmentation/normalization search.
- Conditions: `clean`, `fixed_uniform_0p9`. In the latter, each selected example
  receives one fixed uniform digit in 0–9, which can equal its original label.
  A 0.9 replacement probability implies approximately 81% actually incorrect
  labels in expectation, not 90%. Report both realized counts.
- Architecture: Linear(784,64), ReLU, Linear(64,10), 50,890 trainable parameters,
  including all biases. Existing I9 model initialization and Adam helper reused.
- AdamW: learning rate 0.001, weight decay 0.01, betas (0.9,0.999), epsilon 1e-8,
  foreach=False, fused=False, no schedule/clipping/rescaling.
- Each seed/condition performs one common 100-step raw-gradient warmup while
  observing covariance; fork its exact model, Adam, observer and RNG state into
  all four arms. All arms use the same initialization, split, fixed targets and
  2,000 precomputed batch-with-replacement indices of size64. Warmup is included
  in the 2,000 steps. No restart, selector, sweep, adaptive stopping or retry.
  First-action raw-gradient hashes must match across all4 forks; first
  post-observe full tracker hashes must match across the3 filtered forks.
- Arms: `adamw`, `hard32`, `cluster32`, `mixed32`. Total24 trajectories;
  6×100 shared warmup updates plus24×1,900 branch updates =46,200 actual updates.
  A priori arm order rotates by (seed index + condition index) mod4 to reduce
  systematic order effects in timings. Timings remain descriptive, not a
  controlled performance benchmark.

## Exact observer, graph, and actions

Let raw flattened gradient be g_t. The stable repository observer increments its
step counter and updates its running mean mu_t = .99 mu_(t-1) + .01 g_t, with
first mean initialized to g_1. The centered observation is g_t-mu_t. Its stable
rank32 update and repair schedule100 estimate the exponentially weighted
centered second moment in V_t,S_t. `S` stores **singular values**, so the factor
is F_t = V_t diag(S_t), with F_t F_t^T = V_t diag(S_t^2) V_t^T.
The first centered observation is zero. Existing eigenvalue tolerance1e-8,
absolute floor0, weighting hard, normalize none, adaptive none are unchanged.

On each filtered branch, observe the **raw** current gradient before action.
The baseline AdamW branch needs no observer after the common warmup. Hard32
applies h_t = V_t V_t^T g_t (or the repository identity fallback if no basis).

For clustering, refresh membership on steps101,201,...,1901, **after** observing
that step's raw gradient and immediately before applying that step's action.
Membership is held fixed between refreshes, while its covariance observer still
updates every step. This differs from hard32's continually updated projection.
There are19 refreshes per clustering trajectory,228 in the full batch.

At a refresh:

1. Normalize each nonzero row F_i to x_i=F_i/||F_i||. Norm≤1e-12 rows are
   isolates. Reuse the frozen farthest-point initializer with seed equal to
   the trajectory seed, selecting up to64 current row IDs; select fresh anchors
   at each refresh (no stale basis-coordinate anchors).
2. W_ia = exp(-||x_i-x_a||²/(2 sigma²)), sigma=.35. For numerical safety subtract
   each row's minimum squared distance before exponentiating; subsequent row
   normalization cancels this common factor exactly in real arithmetic.
   Z_ia=W_ia/sum_b W_ib; Lambda_aa=sum_i Z_ia; R=Z Lambda^(-1/2).
   Implicit affinity A=R R^T is nonnegative, symmetric, row-stochastic on active
   rows. Never construct a parameter-by-parameter matrix.
3. Eigendecompose only R^T R, take up to32 eigenvalues>1e-10 and their lifted
   eigenvectors. Include the constant mode, row-normalize this embedding, then
   frozen farthest-initialized k-means with32 requested clusters and at most30
   iterations. Empty clusters keep their center. Actual nonempty count can
   differ from32 and is reported.
4. If group c has indicator b_c and size n_c, define
   C = sum_c b_c b_c^T/n_c + sum_(i isolate) e_i e_i^T.
   Thus C=C^T=C²; `(Cg)_i` is the mean gradient of its assigned cluster and
   isolates pass through unchanged. Cluster32: h=Cg. Mixed32: h=.5g+.5Cg.
   Applied values are accumulated in NumPy float64, then cast to gradient dtype
   (float32) before AdamW. Neither action is norm-matched. Adam preconditioning
   means its final parameter update need not stay in the cluster subspace.

Rank32 matches the known small stable-hard recipe.64 anchors permit a32-mode
graph embedding without forcing it to retain the entire anchor space. The fixed
100-step cadence bounds CPU graph work. These are pragmatic prospective choices,
not tuned optima. Pure clustering can tie incompatible tensors/scales or cancel
gradients. The fixed half-identity mixture preserves half the complementary
gradient to steelman this bottleneck, but is not a mechanistic norm control.
Cluster means enforce equal coordinate updates, not amplitude-weighted
proportional co-movement; normalized-row affinity alone does not remove this
distinction. These are parameter-coordinate groups, not groups of examples,
facts or demonstrated semantic features.

## Frozen measurements and reporting

Evaluate at steps0,100,200,...,2000. The primary endpoint is **step2000 only**:
clean held-out CE (nats/example; lower favorable) and accuracy (fraction; higher
favorable), jointly. Curves show learning dynamics without selecting a checkpoint.
Report all3 seed values and paired differences, not only a best arm/seed.
Suggested descriptive contrasts are each nonbaseline arm minus AdamW and each
clustering arm minus hard32, separately by condition. Mean, sample SD and sample
SE may summarize these3 paired differences; no p-value/composite/rank selection.

Secondary: train clean accuracy/CE; train fixed-target accuracy/CE;
`train_corrupted_accuracy` means accuracy **against fixed wrong targets on the
actually changed subset**, i.e. fixed_target != clean_target. It is null with
denominator0 in the clean condition, never silently assigned0. These distinguish
fitting wrong labels from retaining the clean signal.

Every evaluation saves raw float32 train and clean-heldout logits. Scalar CE sums
are recomputed in float64; counts and sums accompany derived means. Initial and
warmup logits are shared identically across forks; no official-test selector.
Graph diagnostics: actual anchors, factor rank, eigenvalues, k-means iterations,
nonempty cluster sizes, isolate count/fraction, effective projector rank
(nonempty clusters + isolates), retained arrays, refresh seconds, and ARI between
successive memberships on commonly assigned coordinates; report that subset's
size and isolate-status change separately. Isolates are not all treated as one
cluster for this ARI. C's self-adjoint/idempotent properties are tested in tiny
fixtures, not inferred from a low action norm. Each postwarmup step records raw
and applied squared norms and applied/raw norm ratio (null for zero raw norm).

Timing: shared warmup measured once, branch time includes training, graph
refreshes and evaluation but excludes checkpoint/logit serialization. Report
warmup+branch per trajectory; do not multiply shared warmup into actual batch
wall time or call this a speed benchmark. Adam's postwarmup path has no dummy
observer overhead. Batch wall time includes all work and writes.

## Artifacts, provenance, and resource admission

Runner: `experiments/anchor_graph_mnist.py --output-dir NEWPARENT/acquisition-001
--device cuda`. Main supplies exclusive parent on verified `/private-artifacts/storage`
mount `/dev/RECONFIGURE_FOR_LOCAL_STORAGE`, source-hash-pinned launch guard and bounded service. CPU is an
explicit CLI alternative but not an automatic fallback/retry. Existing system
Python3, torch2.11.0+cu128, NumPy; cached training IDX files read-only.

`manifest.json` pins all imported scientific source bytes, protocol and fixtures,
git commit, environment, full roster, dataset hashes, process and invocation ID.
Accepted original-training IDX hashes (also checked before acquisition):
images `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db`;
labels `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5`.
These match the archived I9 manifest, not just whichever files are present.
Data/source pins are reread before completion. Completion requires all24 rows,
finite losses/logits/actions, shared-fork identity checks and unchanged pins,
not favorable results. A failed run retains partial artifacts and `failed.json`
and is not silently relaunched or considered complete.

Files per seed: `plan-sSEED.npz` contains train_indices[5000],heldout_indices[5000],
scalar initialization_seed,replacement_mask[5000] bool,replacement_digits[5000],
training_batches[2000,64],train_clean_labels[5000],heldout_clean_labels[5000].
Each condition binding records array hashes (dtype+shape+bytes), fixed-target and
actually-changed-mask hashes, selected/changed counts, initial model digest,
common full-state digest and exclusive warmup checkpoint receipt.

Each arm saves `curve-ID.json` with21 `{step,sufficient_statistics,metrics}` rows,
`logits-ID.npz` with `steps[21]`, `train_logits[21,5000,10]`,
`heldout_logits[21,5000,10]`; `actions-ID.json` with1,900 action rows;
`clusters-ID.json` with19 refresh rows for clustering arms, empty for controls;
final full model/Adam/observer/RNG snapshot; all19 clustering label vectors in
`labels-ID.npz` (`steps[19]`, `labels[19,50890]` int64) if present, enabling
independent membership/size/isolate/churn checks without training replay.
`results.json` merely lists all24 endpoint records and their curve receipts; it
does not select or recompute a winning comparison. `complete.json` binds every
artifact receipt `{path,size_bytes,sha256}` with relative direct-child paths.

Hard service limits:1 CPU,16GiB RAM,no swap,1,800 seconds,no restart.
Cooperative total limit1,740 seconds checked every training step, before and after
each graph refresh, and before each new cell/write. GPU allocation capped8GiB,
output2GiB with a1MiB metadata reserve, filesystem free reserve1GiB. Expected
outputs below0.6GiB (about0.2GiB logits,0.09GiB labels plus snapshots/JSON); graph working
arrays scale as O(P*64), not O(P²). Runtime is unmeasured prospectively; do not
promise completion or increase limits if it exceeds them. No paid/cloud use.