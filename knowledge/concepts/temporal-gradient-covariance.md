---
title: Temporal gradient-covariance filtering
type: concept
status: current
updated: 2026-09-11
sources:
  - research/spectral_final_core_report_2026-09-11.md
  - spectral_filter.py
  - research/weight_covariance_v2_summary.md
  - research/weight_covariance_v2_findings.md
---

# Temporal gradient-covariance filtering

Let the flattened batch-mean gradient at step `t` be `g_t ∈ R^p`. The global
filter tracks a rank-limited eigenspace `V_t` of an exponentially weighted,
mean-centered covariance over the sequence of gradients. After warmup, hard
filtering replaces the gradient with

```text
g_filtered = V_t (V_t^T g_t).
```

The base optimizer—usually Adam or AdamW—then consumes that filtered gradient.
The implementation is [spectral_filter.py](../../spectral_filter.py).

## What `p×p` means here

The conceptual covariance has parameter-space shape `p×p`, but the code never
materializes that dense matrix. It stores a `p×k` basis and `k` singular-value
state, updating them through a small augmented eigensystem. The dominant storage
term is `O(pk)` and the small solve is `O(k^3)`; gradient projection is `O(pk)`.

## Hard and soft use the same covariance

- **Hard:** project into the tracked top-rank subspace.
- **Soft:** reweight tracked directions according to their covariance strength;
  optionally preserve the out-of-basis residual.

Hard is the default and has the clearest anti-memorization result on the full
MNIST model. Soft was better for the small clean LoRA benchmark. These are
workload-dependent projection policies, not competing covariance estimators.

## Mechanistic interpretation

The observer tracks centered gradient variation across steps. At fixed weights,
if `g = s + noise` with constant shared component `s`, then `Cov(g) = Cov(noise)`:
the shared component itself disappears. Coordinated variation can nevertheless
span useful directions, conditionally. A changing wrong association can also
dominate. The current gradient updates the observer before projection.

Moreover, Adam's inherited momentum, coordinatewise scaling and weight decay
mean that projecting its input does **not** confine the actual parameter step
to that subspace. A zero supplied gradient is not necessarily a frozen model.
The legacy nonorthogonal basis can change gains within the retained span;
the stable policy repairs its basis. These versions must be reported separately.
[Code-grounded account](../../research/spectral_final_core_report_2026-09-11.md).

Persistence can also describe memorization or a coherent unwanted feature.
That is why the method's scope is defined by [Boundary conditions and negative
results](../findings/boundary-conditions.md), not by a blanket claim that top
eigendirections are beneficial.

## Related pages

- [Stable streaming covariance update](stable-streaming-update.md)
- [Per-weight-matrix filtering](per-matrix-filtering.md)
- [Noise memorization](../findings/noise-memorization.md)
