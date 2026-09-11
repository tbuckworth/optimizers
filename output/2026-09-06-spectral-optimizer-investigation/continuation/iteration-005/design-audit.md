# Prospective design audit — iteration 005

**Conceptual verdict: pass for the stated common-stream approximation question.**
This is not implementation, pilot, numerical-validation or launch approval.
The review covers [design intent](design-intent.md),
best-practices check (artifact not distributed in this public snapshot),
reference identities (artifact not distributed in this public snapshot),
[analysis intent](analysis-intent.md) and the complete
[prospective protocol](protocol.md), including its final-primary and no-basis
amendments. No training, reference experiment, test suite, source change outside
this audit, commit or subagent was used by the auditor.

Reviewed SHA-256 identifiers:

- Protocol: `eb8f49f4a5247e9c33a23a0d15a572f2c5cec08355f2b4d05ada50f435ce0035`
- Design intent: `b72f1989b27b46a92b738b8b08fdab4e8571093f4ea86b6ab6834b6aa02bea37`
- Best-practices check: `b7c466ef12d9e4412b72bad311f925cbe414ce33d56cfb03cf00eb43c0408eed`
- Reference identities: `8afb8291018f58e1f91319fe69481442b00c2ae4c36cc9aa2f76b949307bfe43`
- Analysis intent: `2ab9e1ebd5df584e284b8686f81d4433bf85ef7bfd2e12e1f9a1ffbee804baca`

## Reference target and initialization

The protocol correctly freezes the **post-mean-update rounded innovation**,
not the gradient centered on the previous mean or a float64 recomputation.
Both observers and a separate sequential mean calculation must agree bitwise.
For first accepted innovation s, its reference weight at t is beta^(t-s);
later innovations receive (1-beta) beta^(t-j). This reproduces the canonical
startup overweight in the algebraic target rather than silently applying the
ordinary observation weight to the first innovation.

The target is an uncapped finite-history second-moment matrix of these rounded
innovations. It is neither a population covariance nor exact execution of a
full-rank version of the canonical implementation. Residual rejection,
eigenvalue pruning, rounded normalization/rotations and repair can all contribute
to the difference. Even the initial represented V/S covariance need not equal
the float64 innovation outer product exactly. Reinitialization after basis loss
would invalidate a shared single-startup convention; the protocol detects and
stops on it rather than changing the reference. Delayed/all-zero initialization
and float32 norm-underflow behavior are also specified.

Consequently, a width effect identifies a difference between the two complete
canonical estimator recurrences on a fixed stream, not truncation alone with
every other numerical effect removed.

## Common state, self-inclusion and replay coverage

Time alignment is explicit: raw g_t and all probe gradients are evaluated at
theta_(t-1), while the inspected observer state has already incorporated g_t.
Both observers receive exactly the same g_t, innovation history and probe
vectors. Neither changes the gradient delivered to AdamW. The independent
training-batch probes are not fed back into covariance estimation; auxiliary
clean probes use a disjoint example pool. Full state/RNG checks around probes
guard against measurement changing either the model or observers.

This removes the between-policy trajectory confound present in iterations 003
and 004 for the estimator/probe comparison. It does not make the baseline state
or learned covariance exogenous, nor establish what would happen if either
observer controlled training. The current-gradient before/after-observation
retention increment is a separate dependent update diagnostic, not independent
probe evidence or an isolated semantic-noise effect. Independent batch draws
can overlap in examples and remain statistically related through the model.

Historical verification is accurately bounded. Inspection of iteration 004's
`train_arm` implementation confirms that every-step parameter hashes were
saved only for its pilot; confirmation retains warmup hashes, named checkpoint
states and per-step scalar geometry. Iteration 005 must match all those available
anchors, including optimizer/RNG core state at step 100 and every historical
scalar comparison, but must not claim independent historical full-vector
equality at every later step. New full raw/innovation streams and parameter
hashes improve future auditability. Omitting validation forwards is conditioned
on a state/RNG-invariance check. No new validation selection, accuracy or test
evaluation is part of this experiment.

## Independent algebra check: stored operator versus repaired span

The reference identities are correct even when the stored basis is not perfectly
orthogonal. For C=XX^T, Chat=V diag(d) V^T, G=V^T V and A=V^T X, cyclic trace
identities give

    ||Chat||F² = sum_ij d_i d_j G_ij²
    trace(C Chat) = sum_i d_i ||A_i,:||².

For B's native represented operator P=BB^T, H=B^T B and Z=B^T X,

    trace(P C P) = trace(H Z Z^T) = sum(Z * (H Z)),
    trace(P C) = ||Z||F².

These differ under orthogonality drift. The native float32 application is a
third numerical object, explicitly compared with the float64 represented
operator. A QR basis Q defines an analysis-only orthoprojector onto the same
span; it does not repair the observer that produced the saved data.

The sole primary, final-step `||Q^T X||F²/E32`, compares **matched rank-32 spans**.
Its theoretical upper bound is one. Native/represented output-energy ratios
need not satisfy that exact bound under nonorthogonality. Full-width covariance
reconstruction error remains secondary because width 128 has four times the
representational capacity; its improvement alone does not show better delivered
rank-32 filtering.

## Degeneracy, inference and storage boundaries

- The sum of the leading 32 reference eigenvalues is well-defined at a boundary
  tie. The protocol correctly keeps the primary energy ratio when only the
  eigengap gate fails. Unique-reference projector distances and reference-probe
  retentions become null with reasons; actual-observer retention remains
  reportable. The protocol additionally requires 32 numerically positive
  reference values and a full-rank observer span for its primary. Numerical
  null gates must not be relaxed after outcomes.
- Dual/leading-vector residuals, orthogonality, spectrum/trace/Frobenius checks,
  conditioning/rank and gap thresholds are prospective requirements, not
  guarantees established by this design review. Successful eigensolver return
  alone does not certify them. Gaps are used for uniqueness, not to censor an
  adverse but otherwise valid energy result.
- Missing observer basis means canonical **identity** action, not an empty-B
  zero map. Nonzero probe retention is one, zero-input retention is null,
  Chat is zero and matched-rank comparisons are null. The same rule applies to
  an absent previous basis in the self-inclusion diagnostic.
- The three reused seed streams are three paired descriptive observations,
  not fresh learning replications. Four overlapping history prefixes do not
  increase that to twelve independent replicates. Final-step span capture is
  the single primary; complete-pair means, null masks and mandatory secondary
  outputs prevent selecting another snapshot or an easier spectrum after
  observing results. There is no significance, equivalence or broad
  generalization claim from this design.
- Clean versus fixed-corruption residual retention is conditional on the same
  baseline model. Its signed cross terms and noisy-energy closures are retained;
  the residual is not pure independent noise. Better reference fidelity without
  better selectivity is a valid result, not a failed experiment requiring a new
  metric. Neither quantity proves semantic denoising or better learning.
- Raw and innovation arrays alone require 2,442,720,000 bytes, exceeding the
  stated workspace headroom before any states are saved. The uniquely scoped,
  mount-verified large-volume directory, exclusive creation, manifests and row
  completion markers are necessary. Weighted prefixes are reconstructed on
  demand rather than duplicated. Partial writes must not appear complete;
  previous evidence must not be deleted to meet a budget.
- A short training pilot cannot bound the largest dual-Gram solve. The protocol
  therefore includes a separate worst-size synthetic reference/I/O resource
  check, explicit memory/storage/time ceilings and parent review. It does not
  authorize automatic rank/precision changes or retries if a gate fails.

Recommendations on initialization, exact state timing, historical verification
coverage, matched-rank versus unequal-capacity metrics, absent-basis behavior,
degeneracy and bulk storage were incorporated before this verdict. The
researcher-review guidance informed the separation of direct planned evidence
from broader claims. Implementation checks, a passing bounded pilot, final
analysis/source binding and explicit parent GO remain outstanding launch gates.

## Pre-implementation handoff addendum

The parent subsequently added version-matched NumPy 1.26 array-persistence
guidance to the best-practices check. Re-read that file completely; its current
SHA-256 is `cfe6c28b82298fc4468cf75329ab7d3a982ec1c9cf02dcb9d31e9deaaff2fd7a`.
The earlier hash above records the first review rather than being erased.
Protocol SHA-256 remains
`eb8f49f4a5247e9c33a23a0d15a572f2c5cec08355f2b4d05ada50f435ce0035`.

The added read-only audit maps, explicit dtype/shape, flush-before-progress and
separate completed-row counts align with the reviewed recovery design. They
do not imply that allocated file shape certifies completion or that flushing
alone guarantees crash-proof persistence. There is **no remaining conceptual
implementation blocker** from this audit. Approval to implement is separate
from approval to execute a pilot or full replay; no run is authorized here.
