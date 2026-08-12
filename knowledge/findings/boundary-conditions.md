---
title: Boundary conditions and negative results
type: finding
status: current
updated: 2026-08-12
sources:
  - research/weight_covariance_v2_summary.md
  - research/targeted_ablation_findings.md
  - research/per_matrix_spectral_evaluation.md
  - research/grokking_v2_findings.md
  - results/mnist_normalize
---

# Boundary conditions and negative results

The filter keeps recurring gradient structure. Several experiments show why
that is not equivalent to keeping useful or generalizing structure.

## Known failures and costs

- **Clean-data underfitting:** an overly small rank suppresses directions needed
  to fit the task. Raising effective rank can recover clean accuracy but also
  reintroduces noise memorization.
- **Sparse parity:** fixed-rank filtering delayed grokking, and very small ranks
  destabilized it. See [Grokking](grokking.md).
- **Adaptive spectral rank on CIFAR-10:** energy, effective-rank, and eigengap
  rules chose far fewer directions than the function required. Spectral
  concentration measures how gradient variance is distributed, not the task's
  intrinsic functional dimension.
- **Covariance normalization:** correlation and degree-normalized bases both
  underperformed raw covariance on 50%-noise MNIST. Per-weight variance carried
  useful information in that regime.
- **Coherent unwanted signals:** a backdoor can be a persistent consensus
  direction and therefore survive or be amplified. Supervised direction
  ablation removed a backdoor in a linear model but failed completely in an
  MLP, where the behavior was distributed across many paths. See the
  [targeted-ablation report](../../research/targeted_ablation_findings.md).
- **LoRA:** hard global and hard per-matrix filters underfit the clean LoRA
  benchmark. Soft per-matrix matched clean LoRA, but the selected soft setting
  did not improve 90%-noise LoRA.
- **Financial time series:** the external corrected `p×p` experiment that
  motivated the stable update did not improve the financial metric. Numerical
  correctness transferred; task benefit did not. See the
  [stable-update evaluation](../../research/stable_filter_evaluation.md).

## Claims this evidence rules out

The repository does not support any of the following:

- top temporal covariance directions are always generalizing directions;
- anti-memorization automatically improves out-of-distribution extrapolation;
- the filter replaces weight decay;
- low spectral concentration implies a low-dimensional function;
- hard projection is universally better than soft weighting;
- per-matrix filtering is always more accurate than a global basis;
- a few gradient directions can generally remove a distributed unwanted
  behavior in a nonlinear model.

These limits should remain visible when updating the
[Project synthesis](../overview.md) or [Current recommendations](../decisions/current-recommendations.md).

