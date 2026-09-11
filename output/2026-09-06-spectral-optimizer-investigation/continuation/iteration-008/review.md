# I8 independent review and evidence boundaries

Main and `/root/current_report`, 7 September 2026. Reviewer performed no training,
restarts, file edits or commits. `/root/math_covariance` independently developed
the finite-design proposal and Adam geometry alongside main's implementation.

## Before launch

- Five focused tests passed for main and reviewer separately, not ten distinct
  tests. Finite-design autograd gradients, Hessian I, balanced/pairable batches,
  exact warmup, frozen basis and oracle-SGD nuisance freeze were checked.
- Reviewer clarified primary live-vs-raw identification, seed-first comparison,
  raw's larger stopping window and state-only step 0. Main resolved these before
  source/protocol commit `b02e06d` and the once-only launch.

## Completed evidence audit

- All 512 unique expected IDs and 64 cells, with 2001 finite clean-risk entries
  per trajectory; completion hashes and source bindings verified independently.
- Every stored minimum, endpoint, selected step and sparse snapshot checked
  against raw curves. Risk reconstruction maximum discrepancy about 2.22e-16.
- Batch hashes independently regenerated, pairings and exact steps 0–100 warmups
  checked; frozen alignments constant; no-variation repetitions deterministic.
- Primary useful-SGD wins 8/8 at each angle; nuisance 0/8, both 3/8; SGD-none has
  no meaningful effect. Adam's coordinate-dependent deterministic success is
  retained, not removed as a presumed implementation error.
- Out-of-subspace energy ratios recompute from the stored energy totals, and
  sparse gradients/steps verify their intended geometric meaning. The full
  per-step displacement series was **not** retained, so those totals cannot be
  independently reconstructed from sparse snapshots alone. This is an audit
  limitation, not a claim of complete-state neural measurement.
- Adam recurrence expansion, scalar bias factors, sign-normalization regime,
  geometric line floor and fixed-P warmup formula independently checked. All
  stated endpoint/minimum, epsilon and raw-memory qualifications are required.
- Full-batch H=I identity `centered g = centered theta` verified, including
  update-before-centering initialization. Reinterpreting the same deterministic
  final state with clean target v gives risk .9416107422092398, versus
  .0010141536448106257 for u. This is post-hoc arithmetic, not new training.
- Final report tables and knowledge-base additions passed a separate narrative
  read: post-warmup versus unrestricted minima, primary versus control status,
  paired rotations, deterministic repeats, full-scale figure and limitations
  are all represented correctly. No material issues remained.

## Execution record

Unit `spectral-selective-learning-i8-001.service`, invocation
`c2d702ab6f014308a3ad32f8598a65d7`, PID 1685807; launched once 7 September 13:03:01 UTC.
Completion timestamp in raw JSON is 13:08:03.868 UTC, exit 0, inactive/MainPID 0.
Kernel limits were 900s wall, 2 GiB charged memory, one CPU quota and no visible
CUDA device. Journal reports 303.101 CPU seconds and 341.8 MiB peak charged memory;
that is not a userspace maximum-RSS measurement. The source manifest and raw
completion are authoritative; the manifest's initial `running` status is a
historical launch record superseded by `completion.json`.

No cloud compute, credential-bearing file, budget reservation, canonical optimizer
edit, I7 retry or new neural result occurred. Reminder remains enabled/active.
