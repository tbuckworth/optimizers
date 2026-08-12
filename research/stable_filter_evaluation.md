# Stable streaming covariance update

## What changed

The corrected update from the financial-timeseries research repository was
ported into `SpectralGradientFilter`. The historical implementation recovered
basis vectors by dividing through small singular values. In fp32 this gradually
made `V` non-orthogonal, so `V V^T` was no longer the projection the optimizer
claimed to apply.

The stable implementation instead diagonalizes the covariance directly in the
orthonormal augmented basis `[V, q]`, solves only the small eigensystem in fp64,
uses a scale-relative eigenvalue cutoff, performs two-pass orthogonalization,
and periodically repairs the factorization without changing its represented
covariance. `stable_update=True` is now the default. Historical behavior remains
available with `stable_update=False` and `--legacy_update` in the principal
benchmark CLIs.

## Local evaluation

All paired reruns used PyTorch 2.11.0+cu128 on an RTX 3090. Full machine-readable
results and configurations are in
`results/stable_filter/benchmark_summary.json`.

| Evaluation | Legacy | Stable | Change |
|---|---:|---:|---:|
| fp32 basis orthogonality error (lower is better) | 3.85e-5 | 7.71e-7 | 49.9x lower |
| Two-moons final clean accuracy | 0.880 | 0.886 | +0.006 |
| MNIST 5-epoch test accuracy | 0.9409 | 0.9427 | +0.0018 |
| Sparse parity mean grok epoch, 3 seeds | 2816.7 | 2616.7 | 7.1% earlier |
| Sparse parity mean final accuracy | 0.9912 | 0.9677 | -0.0235 |

The parity result is mixed: every stable run crossed the primary 0.9 threshold
earlier, but seed 2 later fell from its grokked state and ended at 0.907 versus
0.988 for the legacy update. This is not evidence of a universal optimization
gain. The change is defaulted because it restores the optimizer's stated
mathematical operation and improves the other benchmark outcomes; the legacy
path is kept for reproducibility and ablation.

## External financial result

The corrected global p×p filter did not rescue the Numerai MLP. With rank 16
selected on validation, held-out mean per-era correlation was +0.000609 versus
+0.016863 for AdamW (paired difference -0.016254, 95% block-bootstrap CI
[-0.022137, -0.010983]). That negative result is about the filter's suitability
for the financial task, not the numerical correction: the exact-covariance,
scale-invariance, resume, fp32 stress, and real-gradient orthogonality gates all
passed first.
