---
title: Noise memorization
type: finding
status: current
updated: 2026-09-11
sources:
  - output/2026-09-10-spectral-strong-augmentation/results.md
  - research/weight_covariance_v2_summary.md
  - research/noisy_mnist_hard_curves.md
  - results/weight_covariance_v2/noise90_long
  - results/cifar_noise
  - results/noisy_mnist_hard_curves/summary.json
---

# Noise memorization

The repository's strongest conclusion is that temporal gradient-covariance
filtering can prevent late memorization of corrupted training labels.

## Strongest recent controlled comparison (three paired seeds)

With approximately 81% actually wrong fixed training labels, final clean held-out
accuracy was 32.0% (AdamW), 79.8% (spectral + AdamW), 85.0% (AdamW + translations),
and 66.8% (both interventions). The filter tracked rank 200 across all 235,146
model parameters. It improved from 36.7% at activation to 79.8%, demonstrating
continued useful learning. It fit only 2.5% of actually wrong assigned labels,
versus 39.2% for unaugmented AdamW.

The important contrary evidence: augmentation outperformed the combination in
every seed on accuracy and loss at fixed, validation-selected and late readouts.
Filtering beat unaugmented early stopping on selected accuracy but lost on
selected cross-entropy. Data roles were disjoint subsets of the official
training split; these are not official-test-set results or independent training
replications. [All seeds and audits](../../output/2026-09-10-spectral-strong-augmentation/results.md).

## Earlier evidence

On MNIST with 20% and 40% label corruption, the global filter retained higher
final clean-test accuracy than Adam after Adam's early peak decayed. At 90%
noise over 60 epochs and three seeds, global rank 200 peaked late and finished
around 80%, while Adam, LoRA, and fixed-random-subspace controls collapsed or
finished much lower. The learned basis outperformed a random basis of the same
rank, showing that capacity restriction alone does not explain the result. See
the [v2 master summary](../../research/weight_covariance_v2_summary.md) and raw
[90%-noise runs](../../results/weight_covariance_v2/noise90_long/).

CIFAR-10 reproduced the qualitative effect on a convolutional model: Adam
peaked early and collapsed as label noise rose, while appropriately constrained
fixed-rank filters retained substantially higher final accuracy. The clean-data
cost could be reduced by increasing effective rank, but doing so also admitted
more noise. This establishes rank and decay as a capacity/noise tradeoff rather
than a free improvement.

## Stable hard global versus per-matrix

Fresh matched full-model runs used AdamW, 90% random relabeling, stable
covariance, hard projection, 60 epochs, and seed 42:

| Method | Peak test | Mean test | Last-10 test | Final test | Final noisy-train |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.8235 | 0.5395 | 0.4069 | 0.3893 | 0.3651 |
| Global hard, rank 200 | **0.8548** | **0.7728** | 0.8026 | 0.7877 | 0.1726 |
| Per-matrix hard, rank 64 | 0.8435 | 0.7500 | **0.8146** | **0.8174** | 0.1781 |

Both filters kept accuracy against the corrupted labels near 0.17–0.18 while
preserving about 0.8 clean-test accuracy; AdamW increasingly fit corruption as
clean accuracy fell. The [curve report](../../research/noisy_mnist_hard_curves.md)
and [epoch-level JSON](../../results/noisy_mnist_hard_curves/) contain the
direct evidence. This comparison is single-seed and rank-selected on the same
seed, so relative global-versus-matrix ordering remains provisional.

## Mechanism and limits

The evidence supports conditional restriction of learning, not a universal
division into useful recurrent and harmful transient gradients. Centering,
sampling, optimizer state and directional alignment all matter. It does not follow that
the filter detects incorrect labels, that top directions are intrinsically
good, or that it improves clean-data training. See
[Boundary conditions](boundary-conditions.md).
