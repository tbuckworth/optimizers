# Fixed-model observer pathway: execution protocol

Codex — Spectral Optimizer Investigation, 10 September 2026.
Prospective source preparation; no new scientific measurement yet.

The complete scientific contract is the unchanged, main-reviewed
matched-parent observer-only discriminator (artifact not distributed in this public snapshot),
SHA256 `2fbb538f4da88680322d64ec6506e16545fbd8a4bd7e068d9cb0599af6ed21fd`.
Its [design review](../2026-09-10-spectral-batch-composition/next-design-review.md)
remains applicable. This page adds operational identities, not new arms,
scientific gates, tolerances or interpretive claims.

The implementation uses CPU NumPy 1.26.4 FP64 means of the saved FP32 streams
and a NumPy FP32 cast of their symmetric mean as the canonical numerical
construction of `g_star`. This fixes reduction order before measurement;
the original mathematical formula and `128 * eps32 * B + 1e-10` bound are
unchanged. Both native arms and raw receive these common bytes; the explicit
zero reference, by definition, receives zero instead.

Use three accepted step-100 parents, seeds 202609121/122/123, both Clean and
Diffuse targets, and both first-block histories. Measure the fixed-model
50-batch streams, compare their FP64 means at the fixed tolerance, form the
common FP32 symmetric mean, and deliver the canonical self-included actions
from observer 151 to a fresh copy of Adam 100. Adam ends at 101, not 151.
The 600 stream gradients, six oracle gradients, 12 native + six raw + three
shared-zero physical steps and 21 new train/held-out logit pairs are fixed.
All parents and earlier completed trajectories remain immutable. This is
outcome-informed local mechanism evidence, not a fresh endpoint confirmation.

## Operational identities and limits

New producer: `experiments/spectral_observer_pathway.py`.
New independent checker: `scripts/audit_spectral_observer_pathway.py`.
Their fabricated tests and implementation notes must be reviewed and frozen
before measurement, along with all imported scientific helper sources.

- Acquisition unit: `spectral-observer-pathway-001.service`; one new exclusive
  large-volume parent and `acquisition-001` child, named in the launch record.
- Local RTX 3090 only; one CPU math thread and CPU quota, 16 GiB host memory,
  no swap, 8 GiB allocated GPU cap, 600-second hard and 480-second cooperative
  deadline. Output cap 1 GiB, plus 1 GiB free-disk reserve. The implementation
  must enumerate the full conservative artifact inventory before launch.
- Audit unit: `spectral-observer-pathway-audit-001.service`, sibling `audit-001`;
  CPU only, one thread/quota, 4 GiB host/no swap, 300-second hard and 250-second
  cooperative deadline, 100 MiB output. Recompute saved-array arithmetic only;
  no model, neural inference, gradients, observer-stream replay or prior audit.
- Both units use Type=exec, Restart=no and KillMode=control-group. No retry,
  source/tolerance/cap changes after launch, paid spend or new cloud reservation.

Main owns admission and the once-only launch, immediately saves handles, and
monitors existing handles on continuation. Results retain every case and seed,
absolute rare/common CE improvements and their paired contrasts; there is no
effect-size acceptance gate, selective extension or new approval workflow.

## Implementation contract validation

The best-practices validation guidance directs attention to state copying and
control semantics. PyTorch documents that `autograd.grad` returns derivatives
without accumulating parameter `.grad`, whereas optimizer gradients of zero
and `None` can have different step behavior. The zero action therefore assigns
actual FP32 zeros; it does not skip inherited momentum and weight decay.
Loading optimizer state does not use parameter names to establish semantic
correspondence, so parameter order and full moments/flags are explicitly bound.
See the official [gradient contract](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html),
[zero-gradient contract](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html),
and [state-loading contract](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.load_state_dict.html).

The installed experiment environment remains Torch 2.11.0+cu128 and NumPy
1.26.4, not the current documentation version 2.14. These general contracts
guide checks; no upgrade or cross-version bitwise-equivalence claim is made.
Pinned local source plus fabricated tests and saved-state invariance checks
establish the actual implementation behavior. Research guidance supplies
explicit counterfactuals and evidence discipline, not a fresh workflow or
permission gate.
