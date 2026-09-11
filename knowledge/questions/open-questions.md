---
title: Open questions
type: question
status: provisional
updated: 2026-09-11
sources:
  - research/spectral_final_core_report_2026-09-11.md
  - research/weight_covariance_directions.md
  - research/per_matrix_spectral_evaluation.md
  - research/noisy_mnist_hard_curves.md
  - research/weight_covariance_v2_summary.md
---

# Open questions

These are scientific unknowns, **not an active experiment queue**. The core
investigation is complete. Its most important unresolved bridge is predicting
when dominant centered variation selects useful learning rather than a shared
error or excludes a legitimate rare case. Unique accumulated mediation, broad
safety efficacy and practical superiority remain unestablished.

The augmentation question below is partly answered: on the three-seed strong
MNIST comparison, ordinary augmentation wins and adding filtering harms it.
This settles that recipe, not every task or augmentation.
[Final report](../../research/spectral_final_core_report_2026-09-11.md).

## Historical candidate questions (not authorized follow-up work)

1. **Does per-matrix hard filtering replicate across seeds?** Repeat the
   60-epoch 90%-noise MNIST comparison with rank selection separated from
   confirmation. This would determine whether its better late-window result
   than global rank 200 is real or seed/selection noise.
2. **How should block rank scale?** Compare fixed rank, rank proportional to
   matrix dimension, and a capped square-root rule while matching total basis
   memory. Current ranks were selected empirically and do not establish a
   scaling law.
3. **Does per-matrix filtering help real language-model LoRA?** Test frozen-base
   adapters on clean validation loss plus controlled label/instruction
   corruption. The current LoRA proxy is a small MNIST MLP.
4. **Can per-block updates avoid the CPU eigensystem bottleneck?** The LoRA
   per-matrix run used much less basis memory but was substantially slower than
   plain LoRA because many small stable updates were solved separately.

## Mechanism

5. Which measurable property predicts whether useful features occupy dominant
   temporal covariance directions, distinguishing modular addition from sparse
   parity before a full run?
6. Can a rank rule estimate functional dimension rather than spectral
   concentration? Energy, entropy effective rank, and eigengap all starved the
   CIFAR model.
7. How much of noisy-label robustness comes from mean centering, rotating basis
   adaptation, and projection separately under the stable implementation?

## Scope and safety relevance

8. Can coherent unwanted behavior be isolated as a larger subspace or in
   activation space when single-gradient-direction ablation fails in nonlinear
   networks?
9. Does the anti-memorization effect survive modern augmentation, schedules,
   convolutional/transformer scale, and realistic data noise rather than
   uniform synthetic relabeling?

Resolved questions should move into a finding or decision page, and the
resolution should be recorded in the [knowledge log](../log.md).
