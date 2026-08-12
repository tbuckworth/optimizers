---
title: Project synthesis
type: overview
status: current
updated: 2026-08-12
sources:
  - research/weight_covariance_v2_summary.md
  - research/stable_filter_evaluation.md
  - research/per_matrix_spectral_evaluation.md
  - research/noisy_mnist_hard_curves.md
---

# Project synthesis

The repository studies a temporal gradient-covariance filter: estimate the
dominant directions of batch-mean gradients across optimization steps, project
the current gradient into that learned subspace, then let a conventional base
optimizer apply the update. The central interpretation is a **coherence
amplifier**: recurring directions survive and transient directions are
suppressed. See [Temporal gradient-covariance filtering](concepts/temporal-gradient-covariance.md).

## Current conclusions

1. **Noise robustness is the strongest result.** On MNIST and CIFAR-10 with
   corrupted training labels, ordinary optimizers learn useful structure early
   and then increasingly memorize corruption. The spectral filter preserves
   substantially more clean-test accuracy late in training. This is supported
   across seeds and datasets. See [Noise memorization](findings/noise-memorization.md).
2. **The learned, rotating subspace matters.** A fixed random subspace at the
   same nominal dimension performs much worse, so the effect is not explained
   by generic parameter-count restriction. See the
   [v2 master summary](../research/weight_covariance_v2_summary.md).
3. **The method amplifies persistence, not goodness.** It accelerates grokking
   on modular addition but delays sparse parity; it does not replace weight
   decay. Coherent unwanted features may be retained or amplified. See
   [Grokking](findings/grokking.md) and
   [Boundary conditions](findings/boundary-conditions.md).
4. **Numerical stability is now part of the method.** The default streaming
   covariance update uses a small fp64 eigensystem and regular
   re-orthogonalization. The older update remains only for reproducing old
   runs. See [Stable streaming update](concepts/stable-streaming-update.md).
5. **Per-matrix filtering is viable but not universally superior.** It reduces
   basis storage for LoRA-scale parameter subsets and, with hard rank 64,
   matched the global filter's anti-memorization behavior on a full MNIST MLP.
   Clean LoRA instead favored soft rank 4, and noisy LoRA did not improve. See
   [Per-weight-matrix filtering](concepts/per-matrix-filtering.md).

## Practical status

The stable global filter is the mature default. The per-matrix wrapper is an
optional scaling mechanism whose rank and hard/soft weighting must be selected
for the workload. Neither should be described as an optimizer that generally
improves test accuracy: the reliable claim is narrower—under suitable rank and
decay, temporal subspace restriction can prevent late memorization of
incoherent label noise.

Use [Current recommendations](decisions/current-recommendations.md) before
running a new experiment, and consult [Open questions](questions/open-questions.md)
before extending the research agenda.

