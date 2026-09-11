# Proposed primary estimand for the shared-stream replay

Before iteration005 pilot or MNIST reference outcomes; subject to final protocol
alignment and source freeze. This is not execution approval.

Use the final scheduled state (step 2000) as the primary time point. For each
observer take an explicitly orthonormal QR basis Q32 for the span of its first
32 stored columns and compute

    rho = trace(Q32^T C Q32) / sum_(i=1)^32 lambda_i(C).

This is matched-rank span capture relative to the optimal reference rank-32
energy, not the covariance error of an estimator allowed four times as many
columns, and not the actual approximately orthogonal float32 delivery operator.
The primary contrast is width128-minus-width32 in rho, with all three seed
differences and descriptive mean/median/range/sample SD. No independent-step
inference, p-values, discovery or equivalence claim. Other scheduled states
and full-width covariance error are mandatory secondary quantities.

The optimal-energy denominator is well-defined even with a tie at the rank-32
boundary. Do not drop this primary merely because a unique leading projector
is ambiguous. Positive rank, nonzero denominator and numerical validity remain
required. The boundary-eigengap gate belongs to unique-reference-subspace
distances and reference-projector probe retention. Keep well-defined observer
probe retentions and energy quantities when those reference diagnostics are
null, with explicit reasons and per-seed availability.

Common clean, fixed-corruption-residual, noisy and disjoint-clean probe
retentions measure the native observer operator at the same pre-update model
state. Their clean-minus-corruption gap is a secondary property, not semantic
purity or a population-noise estimate. A covariance-fidelity improvement with
no selectivity improvement is a valid outcome, not a reason to change metrics.

Mean later-window or all-snapshot contrasts may be reported descriptively if
frozen in the final analysis, but must not replace the primary final-state
comparison. Repeated prefixes and reused seed/data bundles remain dependent.
