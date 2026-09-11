# Implementation validation before scientific acquisition

Codex · 10 September 2026

Version-matched primary documentation checked before implementing new helpers:

- [PyTorch2.11 AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html):
  decoupled decay is separate from moments. Use actual single-tensor AdamW
  from the accepted code and a distinct rounded decay-only parameter vector.
  A post-Adam norm match is not equivalent to rescaling its input gradient.
- [autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html):
  returns requested derivatives rather than accumulating in input .grad.
  No retained/create-graph or batched prototype API is required for this tiny
  fixed model. Verify model/gradient/optimizer/observer parent immutability.
- [torch.load](https://docs.pytorch.org/docs/2.11/generated/torch.load.html):
  load only own hash-bound snapshots with explicit weights_only=True and CPU
  mapping; validate tensor/schema shapes before device transfer. A restricted
  loader is not a reason to trust arbitrary external serialized objects.

All numerical computation retains the installed2.11.0+cu128/NumPy1.26.4 platform
and deterministic settings from the accepted ordinary run. No new dependency,
API backend, default or model training is needed. Remaining scientific limits
are explicit in protocol.md: empirical four-view decomposition, changed warmup
parents, carried-Adam state and the difference between local and long-run utility.

The independent design review found no construct blocker and requested the
cheap fixes now in the protocol: actual rounded decay reference, both norm-
matching directions, shared scaled decay atα0.1, materialized norm checks and
within-parent attribution. Source review then checked the guarded runner, helper
and independent auditor. It found and resolved prospective dependency pins,
rank-zero canonical identity fallback, explicit FP32 displacement conventions,
actual Adam counter assertions and no-symlink parent receipt admission.
All29 focused synthetic tests and128 augmentation-family tests pass before
acquisition. The end-to-end synthetic fixture compares actual private Adam
proposals/readouts against the independent NumPy audit on fabricated CPU inputs.
Initial fixture-only failures (unittest import invocation and dummy basis
rank mismatch) were corrected before admission; no scientific state was read
or numerical tolerance relaxed. No scientific outcome selected these definitions.
