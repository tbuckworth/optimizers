# Implementation check — prospective strong augmentation bridge

Codex — Spectral Optimizer Investigation · 10 September 2026

**Preparation, not admission or results.** Main is applying the implementation
validator to the new bounded runner/helpers/auditor. Existing scientific
dependencies remain unchanged. No new real-data read or training is performed
for source preparation. Fixtures use fabricated small arrays/models.

## Documented behavior and choices

PyTorch can select foreach/fused implementations automatically on CUDA;
explicit `foreach=False, fused=False` fixes the intended single-tensor route.
This is a reproducibility choice, not a performance improvement.
[PyTorch2.11 AdamW documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).

Seeds do not promise reproducibility across releases/platforms. Fix environment
versions, deterministic operations and backend settings, and retain the
actual occurrence/shift arrays instead of relying on seed descriptions alone.
[PyTorch2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html),
[NumPy1.26 random sampling](https://numpy.org/doc/1.26/reference/random/index.html).
The stable documentation alias currently redirects to a newer PyTorch release;
the installed2.11 documentation is the relevant API version here.

## Concrete source risks resolved in the design

- The earlier Run has a900s/1GiB budget, not this study's10800s/8GiB budget.
  A new guard/writer must enforce the new fixed limits rather than monkeypatch
  a frozen module's globals.
- I9's snapshot schema reconstructs a one-hidden-layer model. The strong
  two-hidden-layer model needs its own truthful schema; do not fake `_i9_spec`.
- Standardization after zero-filled translation preserves historical black
  padding; zero padding after normalization would change the augmentation.
- All12 validation-only choices must be saved and independently rechecked
  before reporting metrics; an endpoint scan on reporting data is not stopping.
- Rank200 checkpoints exceed128MiB; independent audit streams their bytes
  without retaining/unpickling, while each logit NPZ remains about2.4MB.
- The seed search before new implementation found171–173 only in the proposed
  design, not existing acquisition/source results. No realized values were
  inspected to choose seeds.

## Required completion checks

Main will record actual source/API reconciliation, focused fixture counts,
independent source findings and exact byte inventory in `preflight.md` before
launch. All code/protocol/tests must be committed and source-pinned. This file
alone is not a PASS receipt. Current read-only inventory shows the intended
large disk has148.1GiB free, host available memory52.7GB and RTX3090 free
memory23587MiB; repeat admission at launch because these are transient values.
