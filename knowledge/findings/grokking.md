---
title: Grokking is task-dependent
type: finding
status: current
updated: 2026-08-12
sources:
  - research/grokking_v2_findings.md
  - results/grokking_v2_seeds
  - results/sparse_parity_grok
  - results/sparse_parity_ranksweep
---

# Grokking is task-dependent

The filter changes grokking dynamics, but the direction of the change depends
on the task.

## Modular addition

Across five seeds, global filtering plus AdamW reached 90% test accuracy at
`2550 ± 79` epochs versus `3720 ± 84` for AdamW, about 31% earlier. Switching
the filter on only after memorization produced essentially the same timing as
filtering from the start. This localizes the gain to the post-memorization
transition rather than initial fitting. Direct analysis is in
[grokking_v2_findings.md](../../research/grokking_v2_findings.md).

The filter did not cause grokking without weight decay. Weight decay supplies
the pressure that favors the generalizing solution; filtering changes how the
optimizer follows persistent directions after that pressure exists.

## Sparse parity

On sparse parity, the filter delayed grokking by roughly threefold. Very low
ranks could prevent or destabilize grokking; moderate ranks eventually worked,
but no tested fixed rank beat AdamW. An adaptive effective-rank rule became the
first adaptive filter variant to grok parity, yet still did not establish a
general speed advantage.

## Synthesis

"The filter accelerates grokking" is false as a general claim. A narrower
statement fits both tasks: projection amplifies the dominant temporal gradient
subspace. On modular addition that subspace becomes useful during the
weight-decay-driven transition; on sparse parity the weak generalizing feature
can sit outside the dominant memorization directions, so filtering delays it.

This supports the broader [temporal coherence](../concepts/temporal-gradient-covariance.md)
mechanism and places grokking among the method's important
[boundary conditions](boundary-conditions.md).

