# Main decision: prepare saved-vector alignment accounting

Codex — Spectral Optimizer Investigation · 10 September 2026.

Main has read the complete design (artifact not distributed in this public snapshot), SHA256
`a3c6222868de4e9833fce9e48878db08bb1b02128c858fbafd5f5ca747e2dc72`, and
[independent review](next-design-review.md), SHA256
`2ef6e4c77ea8369438b11a6690ac0f8ff7778803260b5a9569629a8d90feb2b4`.
Select this bounded saved-data question as the next continuation. It adds
missing cross terms that retention and actual-step utilities cannot identify.
It is within the approved understanding-first scope, not a new user-approval
gate or permission for another training experiment.

Before scientific array access, prepare and review the small analysis source,
fabricated fixtures and receipt inventory with these mandatory clarifications:

- Encode raw/zero alignment identities explicitly: `F_raw=B`, `F_zero=0`,
  `K_raw=0`, `K_zero=−B`. Retain all parents, label cells, probes and controls.
- Convert vectors to FP64 before differences. Report signed dot products,
  norms and cosines separately; zero-norm cosines are undefined.
- Use an independent compensated summation of FP64 products as a reference.
  The proposed `128 eps64 × sum(abs(products)) + 1e−12` is an operational
  consistency threshold, not a worst-case theorem for arbitrary long sums.
  Freeze a cancellation-aware comparison of direct difference products to
  paired products, using their absolute constituent sums, in source/tests
  before outcomes. Do not adapt tolerances to scientific results or treat a
  small unresolved sign as evidence for a hypothesis.
- Bind all 21 regular-file inputs to existing size/hash receipts. Metadata
  totals 189,384,513 bytes; no stream replay, model construction, inference,
  observer/Adam update or additional tensor members are in scope.

Keep the proposed one CPU/thread, CUDA-hidden, 4 GiB/no-swap, 180-second hard /
150-second cooperative, 100-MiB output plus 1-GiB reserve envelope, no retry.
Joining accepted actual-step and finite-loss scalars is not a new independent
measurement of them. Direction access, present signal and persistent learning
remain distinct; this cannot identify an endpoint mediator or safety guarantee.

No new analysis source, tensor calculation, output parent, job or reservation
has been created. Continue from this selected design only when ordinary
resource and usage controls permit; no completed study should be repeated.
