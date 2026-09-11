# Implementation validation before acquisition

Codex — Spectral Optimizer Investigation · 10 September 2026

Current version-matched primary documentation inspected before implementation:
[PyTorch2.11 autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html)
returns requested gradients without accumulating parameter .grad; use one
separate graph per view at fixed parameters, no retained/create_graph.
[PyTorch2.11 AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)
keeps decoupled decay and carried first/second moments. One optimizer.step per
update, regardless of number of observation views. Existing installed platform
PyTorch2.11.0+cu128/NumPy1.26.4 and deterministic1thread settings retained.

Observation and delivery are intentionally distinct. Canonical filter code
remains unchanged; the new dispatcher separates its once-only observation
from its action on g1. Native1 must equal the original filter_grad path;
the candidate must neither ingest four times nor ingest delivery again.
Synthetic fixtures will verify gradients, averaging, counters, warmup equality,
plan independence and archive/resource guards before any real acquisition.

This note is implementation validation, not completed source review or an
experimental outcome. No external model service or new dependency is required.
