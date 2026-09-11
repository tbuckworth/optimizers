# Implementation check before coding

9 September 2026. Best-practices validation against the **installed/frozen
PyTorch2.11** documentation, not an instruction to upgrade to the current stable
release. This continues the reviewed one-policy design, not a fresh scope.

## Aligned with documented behavior

- Restore model/optimizer state dictionaries, filter state and RNG using the
  existing receipt-bound helpers. Their checkpoint loader already uses
  `weights_only=True`; no unsafe-pickle fallback is needed.
  [Serialization](https://docs.pytorch.org/docs/2.11/notes/serialization.html).
- Reuse the unchanged reduced QR/small-R SVD calculation, with explicit
  positive numerical-rank threshold and no dense P-by-P matrix. Singular-vector
  signs are not unique, so compare projectors/actions rather than signed basis
  entries. No autograd through this decomposition is required.
  [SVD](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.svd.html).

## Gaps addressed by the new runner and fixtures

- Zero delivered tensors still receive an ordinary AdamW step. `grad=None`
  would skip that parameter and change the inherited-moment/decay intervention.
  Test this explicitly with nonzero carried moments and decay.
  [AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).
- Preserve the original parameter ordering and complete optimizer state;
  match incoming norm after casting and separately measure actual movement.
  This latter measurement is our experimental requirement, not a PyTorch
  guarantee that equal incoming norms imply equal parameter updates.
- Hash every trajectory JSON when acquiring it. This remedies an old receipt
  omission prospectively without silently strengthening the old evidence.

## Concerns retained, not “fixed” by changing the experiment

Full-state/RNG restoration does not guarantee bitwise CUDA future trajectories
across environments. Keep the original library/backend settings, record the
actual environment, and retain the archived-reference limitation rather than
toggle deterministic modes or repeat old branches.
[Reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

Recommendations: implement a new policy module and new fixed-roster runner;
reuse existing trusted primitives without modifying their frozen bytes; use
synthetic fixtures and independent source review before one bounded acquisition.
No fresh seed, extra arm, paid compute, dependency installation or old-run
replay is required by this check.

## Operational service documentation addendum