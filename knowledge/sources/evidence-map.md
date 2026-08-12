---
title: Evidence map
type: source-map
status: current
updated: 2026-08-12
sources:
  - README.md
  - research/README.md
  - results
  - experiments
---

# Evidence map

## Implementations

- [spectral_filter.py](../../spectral_filter.py) — canonical global temporal
  covariance filter and stable/legacy update paths.
- [matrix_spectral_filter.py](../../matrix_spectral_filter.py) — per-matrix
  wrapper, block construction, bias policies, and memory diagnostics.
- [run_single_weight_cov_v2.py](../../experiments/run_single_weight_cov_v2.py)
  — main MNIST/noisy-label harness, including AdamW and global/per-matrix modes.
- [Tests](../../tests/) — numerical invariants, dtype behavior, block isolation,
  LoRA parameter subsets, checkpointing, and memory estimates.

## Canonical reports

- [v2 master summary](../../research/weight_covariance_v2_summary.md) — broad
  synthesis of label noise, controls, grokking, CIFAR, rank rules, normalization,
  and targeted ablation.
- [Stable-update evaluation](../../research/stable_filter_evaluation.md) —
  external implementation audit, numerical diagnostic, and legacy comparison.
- [Per-matrix evaluation](../../research/per_matrix_spectral_evaluation.md) —
  block design, LoRA rank scouts, memory, runtime, and initial verdict.
- [Noisy-MNIST hard curves](../../research/noisy_mnist_hard_curves.md) — fresh
  AdamW/global/per-matrix stable-hard comparison with an epoch-level graph.
- [Grokking report](../../research/grokking_v2_findings.md) — modular addition,
  sparse parity, rank sweeps, and adaptive filtering.
- [Targeted ablation](../../research/targeted_ablation_findings.md) — coherent
  backdoor directions in linear and nonlinear models.

## Raw results

- [results/weight_covariance_v2](../../results/weight_covariance_v2/) — MNIST
  label-noise sweeps, long 90%-noise runs, LoRA, and random-subspace controls.
- [results/noisy_mnist_hard_curves](../../results/noisy_mnist_hard_curves/) —
  matched stable-hard curves and per-matrix rank scout.
- [results/per_matrix_spectral](../../results/per_matrix_spectral/) — full-model
  and LoRA global/per-matrix comparisons.
- [results/stable_filter](../../results/stable_filter/) — stable-versus-legacy
  benchmark summary.
- [results/cifar_noise](../../results/cifar_noise/) — CIFAR label-noise, fixed
  rank, adaptive rank, and normalization-related evidence.
- [results/grokking_v2_seeds](../../results/grokking_v2_seeds/) and
  [results/sparse_parity_grok](../../results/sparse_parity_grok/) — contrasting
  algorithmic-task evidence.

## Historical material

[research/findings.md](../../research/findings.md) and `experiments/legacy/`
describe the earlier per-sample `B×B` consensus line. They remain useful for
provenance but are not the current `p×p` temporal method. When historical and
current documents disagree, follow the source-priority rules in
[the schema](../schema.md).

