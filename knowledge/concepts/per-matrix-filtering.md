---
title: Per-weight-matrix filtering
type: concept
status: current
updated: 2026-08-12
sources:
  - matrix_spectral_filter.py
  - research/per_matrix_spectral_evaluation.md
  - research/noisy_mnist_hard_curves.md
  - results/per_matrix_spectral/benchmark_summary.json
  - results/noisy_mnist_hard_curves/summary.json
---

# Per-weight-matrix filtering

[matrix_spectral_filter.py](../../matrix_spectral_filter.py) assigns an
independent [stable temporal covariance filter](stable-streaming-update.md) to
each trainable matrix or higher-dimensional tensor. There is no globally
flattened gradient basis. For blocks with sizes `p_j` and ranks `k_j`, basis
storage is `Σ_j p_j k_j` and the work decomposes across blocks.

## LoRA and parameter selection

Only parameters owned by the base optimizer and marked trainable are included.
Frozen base-model weights therefore do not consume covariance state. LoRA `A`
and `B` matrices become separate blocks, matching their distinct parameter
geometry rather than concatenating all adapters into one global basis.

## Bias decision

The default `bias_mode="joint"` joins a matrix with its matching output bias,
analogous to treating the affine layer in homogeneous coordinates. This is
inspired by layerwise/blockwise curvature methods such as EKFAC, but it is a
design analogy rather than an implementation of EKFAC. Unmatched vectors such
as LayerNorm parameters pass through unfiltered. Alternatives are:

- `separate`: filter each trainable vector independently;
- `exclude`: leave all one-dimensional parameters unfiltered.

## Empirical picture

On clean LoRA MNIST, soft rank 4 with `alpha=2` matched plain LoRA (`0.9676`
versus `0.9662` mean final accuracy over three seeds) while using `0.716 MiB`
of basis storage, 12.5 times less than global rank 50. Hard global and hard
per-matrix LoRA both underfit. At 90% label noise, the selected soft LoRA
configuration was worse than plain LoRA.

On the full MNIST MLP with 90% random relabeling, a single-seed 60-epoch hard
comparison found per-matrix rank 64 retained `0.8174` final clean-test accuracy,
global rank 200 retained `0.7877`, and AdamW fell to `0.3893`. The per-matrix
basis used `57.41 MiB` versus `179.40 MiB` globally. Because rank 64 was selected
on a scout using the same seed, this is supported descriptive evidence, not an
unbiased multi-seed estimate. See [Noise memorization](../findings/noise-memorization.md).

## Interpretation

Per-matrix filtering is a tractability mechanism, not a different covariance
algorithm and not a guaranteed accuracy improvement. Smaller ranks become
possible because each block has a smaller ambient dimension, but the useful
rank still depends strongly on workload: rank 4 was adequate for soft LoRA,
whereas the full noisy MLP favored hard rank 64 per matrix.

## Related pages

- [Current recommendations](../decisions/current-recommendations.md)
- [Boundary conditions](../findings/boundary-conditions.md)

