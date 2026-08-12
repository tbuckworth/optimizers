# Per-weight-matrix spectral filtering

## Design

The global filter represents a rank-k approximation to the temporal gradient
covariance over all optimized parameters. Its dominant state is `p × k`, which
is too large for full fine-tuning and still wasteful for large LoRA adapters.

`PerMatrixSpectralGradientFilter` uses a block-diagonal approximation: each
trainable matrix or higher-order tensor has an independent streaming covariance
basis. LoRA A and B parameters become separate blocks. Frozen parameters and
parameters absent from the base optimizer are ignored.

Biases use a homogeneous-coordinate analogue by default: a matching affine
bias is concatenated with its module's weight block. Unmatched 1D parameters
(for example LayerNorm scale and bias) pass through unchanged. The behavior is
configurable with `bias_mode={joint,separate,exclude}`.

This borrows only the layerwise/block-structured intuition from KFAC/EKFAC; it
is not a Fisher, Kronecker-factored, or natural-gradient implementation. EKFAC
motivates scalable layer-local curvature representations and correcting
directional scale in a factored eigenbasis:
https://arxiv.org/abs/1806.03884.

For LoRA evaluation, the adapter remains `W + (alpha/r) BA`, A and B are both
filtered, and learning rate is swept independently. This follows the practical
warnings in *LoRA Without Regret*: LoRA's product parameterization changes its
optimization dynamics, all weight matrices matter, and optimal learning rate
must not be assumed to match full fine-tuning:
https://thinkingmachines.ai/blog/lora/.

## Complexity

For block sizes `p_j` and local ranks `k_j`, basis storage is
`sum_j p_j k_j`; no global gradient vector or global `p × k` basis is formed.
The update cost is the corresponding sum of block-local matrix-vector products
plus one small eigensystem per active block.

On this repository's rank-32 LoRA MLP (46,948 trainable adapter parameters):

| Filter state | Basis storage |
|---|---:|
| Global rank 50 | 8.95 MiB |
| Global rank 200 (historical setting) | 35.82 MiB |
| Per-matrix rank 16 | 2.87 MiB |
| Per-matrix rank 4 | 0.72 MiB |
| Per-matrix rank 2 | 0.36 MiB |

These figures exclude Adam state and the one-gradient-vector running mean,
which both variants require. Per-matrix rank 4 cuts basis memory 12.5× versus
global rank 50 and 50× versus global rank 200.

## Rank scouting

Rank selection used seed 42; seeds 43–44 were confirmation seeds. Concurrent
scout wall times are intentionally excluded from runtime comparisons.

Full-model hard per-matrix final clean-test accuracy after five epochs rose with
rank: rank 1 0.8948, 2 0.8893, 4 0.8981, 8 0.9141, 16 0.9191, 32 0.9383,
and 64 0.9510. Rank 64 was selected.

LoRA hard per-matrix filtering underfit at every tested local rank: rank 1
0.8599, 2 0.8639, 4 0.8683, 8 0.8694, and 16 0.8810. Preserving the residual
gradient and softly reweighting only the retained directions fixed the clean
optimization problem. At rank 4, alpha values 0.5, 1, and 2 produced 0.9631,
0.9640, and 0.9682 after three epochs; rank 4 / alpha 2 was selected.

For 90%-label-noise LoRA, a separate seed-42 scout selected the milder soft
rank 2 / alpha 0.5 setting. Ranks 2, 4, and 8 and alpha values 0.25, 0.5, and 1
were evaluated.

## Final results

All final reruns used PyTorch 2.11.0+cu128 on an RTX 3090. Means are over seeds
42–44. The repository contains every raw run under
`results/per_matrix_spectral/` and a machine-readable summary in
`results/per_matrix_spectral/benchmark_summary.json`.

### Full MLP, clean MNIST, five epochs

| Optimizer | Rank | Final test accuracy | Isolated time |
|---|---:|---:|---:|
| Adam | — | 0.9760 ± 0.0045 | 10.4 s |
| Global hard filter | 50 | 0.9376 ± 0.0044 | 18.7 s |
| Per-matrix hard filter | 64 | 0.9503 ± 0.0025 | 31.6 s |

The per-matrix filter is 1.27 percentage points better than the global filter
but remains 2.57 points below Adam. This is a partial benefit, not a win over
the baseline optimizer.

### Rank-32 LoRA MLP, clean MNIST, five epochs

| Optimizer | Rank | Final test accuracy | Isolated time |
|---|---:|---:|---:|
| Plain LoRA + Adam | — | 0.9662 ± 0.0036 | 13.5 s |
| Global hard filter | 50 | 0.8940 ± 0.0039 | 20.5 s |
| Per-matrix hard filter | 16 | 0.8942 ± 0.0044 | 42.6 s |
| Per-matrix soft filter, alpha 2 | 4 | 0.9676 ± 0.0055 | 49.1 s |

Soft per-matrix rank 4 is competitive with plain LoRA (+0.14 percentage
points in this three-seed run) and avoids global basis storage. It is 3.6×
slower than plain LoRA in this small implementation because every block solves
a CPU eigensystem each step; the result supports memory tractability, not a
speed claim. Hard projection fails for both global and per-matrix layouts.

### Rank-32 LoRA MLP, 90% label noise, 20 epochs

| Optimizer | Rank | Final test accuracy | Peak test accuracy |
|---|---:|---:|---:|
| Plain LoRA + Adam (existing matched runs) | — | 0.5809 ± 0.0113 | 0.8156 |
| Per-matrix soft filter, alpha 0.5 | 2 | 0.5636 ± 0.0109 | 0.8062 |

The per-matrix method is worse by 1.73 final-accuracy points and 0.94 peak
points. It does not reproduce the global full-model filter's strong long-run
label-noise robustness on LoRA adapters.

## Verdict

The implementation makes spectral filtering tractable for large sets of small
trainable matrices and works directly with frozen-base LoRA models. Its useful
operating point here is soft, residual-preserving, and very low rank. It matches
plain LoRA on clean MNIST with much less spectral state than a global filter,
but adds runtime overhead and provides no noisy-label benefit. It should remain
an explicit optimizer choice rather than replacing plain Adam/AdamW.
