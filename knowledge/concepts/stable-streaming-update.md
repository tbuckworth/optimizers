---
title: Stable streaming covariance update
type: concept
status: current
updated: 2026-08-12
sources:
  - spectral_filter.py
  - research/stable_filter_evaluation.md
  - results/stable_filter/benchmark_summary.json
---

# Stable streaming covariance update

The original streaming rank-one update gradually lost orthogonality in fp32.
That makes `V V^T` cease to be a true projection and can change the effective
filter strength. The current update fixes this without forming the conceptual
`p×p` covariance.

## Current algorithm

For each centered gradient, the implementation builds an orthonormal augmented
basis from the existing `V` plus its residual, diagonalizes the covariance in
that small coordinate system in fp64, removes numerically negligible modes,
rotates back to model dtype/device, and periodically repairs orthogonality.
Two-pass orthogonalization reduces cancellation when the new direction nearly
lies in the existing subspace.

`stable_update=True` is the default in both the global and per-matrix wrappers.
`stable_update=False` and the experiment flag `--legacy_update` exist only for
historical comparison.

## Evidence

In a 600-gradient fp32 diagnostic, stable mode reduced spectral-norm
orthogonality error from `3.846×10^-5` to `7.714×10^-7`, about 50-fold. It also
slightly improved two-moons and five-epoch MNIST. On sparse parity it reached
the grok threshold earlier in all three seeds, but one seed later regressed,
so the benchmark evidence is mixed rather than uniformly positive. Direct
metrics are in the [stable benchmark summary](../../results/stable_filter/benchmark_summary.json)
and interpretation in the [evaluation report](../../research/stable_filter_evaluation.md).

The default decision rests primarily on restoring the intended projection
geometry and secondarily on the small benchmark improvements—not on a claim
that every downstream training curve improves.

## Related pages

- [Temporal gradient-covariance filtering](temporal-gradient-covariance.md)
- [Current recommendations](../decisions/current-recommendations.md)

