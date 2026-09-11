---
title: Current recommendations
type: decision
status: current
updated: 2026-09-11
sources:
  - research/spectral_final_core_report_2026-09-11.md
  - README.md
  - spectral_filter.py
  - matrix_spectral_filter.py
  - research/stable_filter_evaluation.md
  - research/per_matrix_spectral_evaluation.md
  - research/noisy_mnist_hard_curves.md
---

# Current recommendations

The core investigation is complete. Its defensible output is an empirical and
mathematical characterization, not a general optimizer or safety-defense claim.
No archived experiment should be relaunched automatically. Public distribution
does not authorize new compute or make archived machine-specific launchers
portable; see [publication scope](../../PUBLICATION.md).

For future evaluations, include a strong ordinary training recipe, separate
clean validation selection from final reporting, and report accuracy **and**
loss. The strongest recent comparison favors augmentation over filtering and
finds their combination adverse. Check actual parameter-step geometry when
reasoning about Adam: projected input is not a projected Adam step.
[Final evidence](../../research/spectral_final_core_report_2026-09-11.md).

## Defaults

- Use `stable_update=True`. The legacy update is only for reproducing results
  from before the numerical fix.
- Use raw covariance (`normalize="none"`) unless a workload-specific experiment
  shows otherwise.
- Start with hard weighting for full-model anti-memorization experiments. Treat
  rank and decay together as a capacity knob.
- Use Adam or AdamW as the base optimizer; momentum/adaptive scaling has been
  complementary to filtering in the tested noisy-label regime.

## Global versus per-matrix

Choose the global [temporal filter](../concepts/temporal-gradient-covariance.md)
when the `p×k` basis is tractable and the goal is the best-established behavior.
Choose [per-matrix filtering](../concepts/per-matrix-filtering.md) when a global
basis is too large, especially for frozen-base LoRA/PEFT models. Do not infer a
per-block rank mechanically from the old global rank; scout ranks on the actual
workload.

For biases, keep the per-matrix default `bias_mode="joint"` for affine layers.
Leave LayerNorm-like unmatched vectors unfiltered unless there is direct
evidence for separate filtering.

## Hard versus soft

- **Hard:** strongest current full-model noisy-label evidence; easy to interpret
  as subspace restriction.
- **Soft:** retained because it matched plain LoRA where hard LoRA underfit.

Soft has not been abandoned. Weighting choice is task-dependent and must be
reported alongside rank, decay, warmup, and whether residual directions are
retained.

## Evaluation protocol

1. Always include the unfiltered base optimizer with the same learning rate and
   weight decay.
2. Report train and test curves, not only endpoints; memorization is temporal.
3. Record fixed and late outcomes; use validation rather than test outcomes for
   checkpoint selection, and report loss alongside accuracy.
4. Separate hyperparameter-selection seeds from confirmation seeds.
5. Use multiple seeds before claiming one spectral layout beats another.
6. Record basis memory and wall time for global/per-matrix comparisons.
7. Preserve negative configurations and selection protocol in committed JSON.

See [Noise memorization](../findings/noise-memorization.md) for the model
comparison that motivates these rules and [Boundary conditions](../findings/boundary-conditions.md)
for cases where filtering should not be expected to help.
