# Independent design and source review

Date: 2026-09-09
Scope: prospective neural anchor-clustering pilot only
Verdict: **PASS — no material design or source blocker found for the fixed, bounded acquisition.**

I reviewed the current runner, nine synthetic fixtures, protocol, implementation
handoff, frozen anchor-graph helper, canonical spectral observer behavior, I9
MNIST model/optimizer/snapshot helpers, launch guard, and the relevant
noise-memorization knowledge record. I did not read MNIST arrays, run inference
or training, launch an acquisition, or inspect outcomes.

Reviewed source receipts:

- runner `a2e47651cada793619e5b6e7ef1646e6c553c93ba1c8898f45a9681c3e7dcde4`
- fixtures `41c75156b53b7f65a6d0c5dcc935799b71f29f3c4f3dfa1408b7e5cf40f30b38`
- protocol `0dc04e3efa96666501975d48b33f98f4248cdfd60aa4ff8789e7e05e54e56839`
- launch guard `231a7ef025a47213d1b409c891f0eebe46431fe86f232cd824b47d3693e578e3`
- frozen canonical spectral source `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`
- frozen graph helper `2581010abe529d1b3da2af0a71106710dcc59b81ec03e58b93b56f2f4790f89e`

The implementation now resolves the pre-freeze issues raised during review:
accepted I9 IDX hashes are static; the covariance factor is correctly
`F = V diag(S)` because `S` contains singular values; first post-warmup raw
gradients are hash-bound across all four forks and post-observe tracker states
across the three filtered forks; all 19 membership vectors per clustering
trajectory are retained; projector identities are checked at runtime and in
fixtures; and capped NPZ writing has a passing round-trip fixture after the
NumPy 1.26 file-like compatibility fix. Main reports all nine synthetic fixtures
passing. These are resolved findings, not remaining qualifications.

## Mathematical assessment

For a fixed membership, the implemented operator is

`C = sum_c b_c b_c^T / n_c + sum_{i isolate} e_i e_i^T`.

It is the Euclidean orthogonal projector onto vectors that are constant within
each nonempty cluster, with isolate coordinates passed through independently.
Consequently `C = C^T = C^2`, `g^T Cg = ||Cg||^2`, and `||Cg|| <= ||g||`.
Its actual rank is the number of nonempty clusters plus the number of isolates,
not necessarily 32; the runner records that rank. The mixed action
`T = .5(I + C)` has eigenvalue 1 on `range(C)` and .5 on its complement, so it
retains half of every rejected component and is full-rank. This is a sensible
fixed steelman of a potentially over-restrictive cluster-only action, not a
norm-matched control or an outcome-selected hyperparameter.

Both `Cg` and `.5(g + Cg)` have nonnegative inner product with the raw gradient
in the Euclidean first-order SGD geometry. That fact does not carry through as
an equivalent statement about the actual AdamW parameter displacement: carried
moments, coordinatewise preconditioning, epsilon, and decoupled weight decay can
move the update outside the cluster subspace. Endpoint differences therefore
test complete optimizer policies, not a pure causal effect of an isolated
projector.

The graph clusters normalized rows of a rank-32 approximation to centered,
exponentially weighted gradient covariance. Similar profiles indicate
approximately proportional historical co-movement in that retained geometry;
they do **not** imply equal current raw-gradient amplitudes. The unweighted
cluster mean retains indicator directions and enforces equal applied values
within a cluster, rather than retaining arbitrary proportional co-movement.
This is the principal construct limitation. It can be especially consequential
because one flattened graph crosses weights, biases, layers, and heterogeneous
coordinate scales and is not invariant to reparameterization. The fixed mixture
reduces, but does not resolve, amplitude weighting or cross-layer scale mismatch.
No extra arm or sweep is warranted in this first test; these are conditional
follow-up mechanisms if the fixed result motivates them.

## Construct validity and reporting

The roster is a fair bounded first usefulness test: three prospectively fixed
seeds, clean and fixed-uniform-0.9-replacement conditions, common 100-step
model/Adam/observer/RNG state, fixed batches, and four fixed arms. Clean training
and clean held-out CE/accuracy expose failure to learn; wrong-target fit on the
actually changed subset helps distinguish suppression of noisy-label fitting.
They do not prove that a parameter cluster is semantic, that retained directions
are generally useful, or that rejected directions are memorization-specific.

Hard32 updates its basis every step, whereas cluster membership is rebuilt at
101, 201, ..., 1901 after observing the current raw gradient and held for the
following block. Anchors are selected afresh at each rebuild. Cluster and mixed
branches then develop endogenous observers and memberships. Thus hard-versus-
cluster and cluster-versus-mixed comparisons are policy comparisons rather than
exact graph-only or action-only ablations. The current-step inclusion at refresh
is explicit and acceptable, but it must not later be described as a predictor
built solely from prior gradients.

The primary endpoint is fixed at step 2000, with held-out clean CE and accuracy
reported jointly. All three seed values and predeclared paired contrasts should
be shown, including adverse and conflicting metrics. The report must not select
the best clustering arm, seed, checkpoint, condition, or metric after inspection.
The data are reused MNIST training examples with a disjoint internal held-out
split, not the official test set or fresh validation. Three seeds support a
limited multi-seed diagnostic, not a broad optimizer-superiority claim.

## Source, provenance, and resource assessment

The source implements exactly 24 trajectories and 46,200 actual updates, with
one shared warmup per seed/condition and exact full-state restoration before
each arm. There is no resume, search, adaptive stop, fallback, or retry path.
Raw logits, sufficient statistics, action norms, memberships, refresh
diagnostics, final states, and exclusive-write hash receipts are adequate for an
independent saved-data audit. Static accepted data hashes and source pins are
checked before work and again before completion.

The guard verifies committed worktree sources against on-disk bytes, the
committed main guard, an unused large-volume output parent, the intended Python
environment, prior-job terminal state, recognized GPU occupants, deterministic
environment variables, and effective service limits. The runner additionally
enforces a 1,740-second cooperative deadline, 8 GiB GPU allocation cap, 2 GiB
output cap with metadata reserve, and disk reserve. The service imposes one CPU,
16 GiB RAM, no swap, 1,800 seconds, no restart, and cgroup-wide termination.
Runtime is intentionally unmeasured prospectively. Expiry would be a recorded
null acquisition, not grounds to relax limits or retry.

No further pre-launch scientific change is required. The main agent should
freeze the reviewed receipts and preserve the protocol's one-shot and reporting
constraints. This review is source/design acceptance only; it does not report or
prejudge an experimental result.
