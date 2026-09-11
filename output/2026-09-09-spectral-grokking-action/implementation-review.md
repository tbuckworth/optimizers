# Independent prelaunch implementation review

9 September 2026

**Verdict: PASS for source freeze; no remaining scientific implementation
blocker found.** This is an implementation review, not an audit of real
checkpoint execution or results. I used only source inspection and synthetic
fixtures.

The initial review found two material defects, both fixed before this verdict:

- Norm matching had used the pre-cast float64 orthogonal norm rather than the
  norm of the delivered gradient tensor, and did not fail/label the frozen zero,
  sub-floor and post-cast mismatch cases. The helper now uses the delivered
  tensor, applies the `1e-30` rule, fails zero-orthogonal/positive-native cases,
  labels clamped/degenerate cases, and enforces
  `max(1e-12, 10*eps(dtype))`.
- Continuation initially computed every policy on every update. That allowed an
  unused norm-control validation to abort the orthogonal branch. The wrapper now
  requests only its selected policy; the common first-action calculation still
  requests all three.

## Contract checks

- The runner restores the exact audited legacy step-1500 checkpoint, regenerates
  and validates the fixed split, computes one full-batch raw gradient, increments
  the counter once, and invokes the inherited legacy estimator update once.
  Native action equality is checked bitwise against the original
  `SpectralGradientFilter` implementation.
- Each first action starts from a separately constructed model, AdamW and filter.
  Restoration checks model, optimizer, filter/configuration, CPU/all-CUDA RNG,
  split, parameter layout and device identity. The shared updated filter is
  independently cloned into each fork, the common post-forward RNG is restored,
  and each supplied action receives exactly one AdamW step.
- Native is captured only at step 1501. Orthogonal and norm-matched continue
  directly with updates 1502 through 2500 and capture 1501/2000/2500. The roster
  is therefore the prescribed 35 new states, with no native continuation.
- The numerical projector uses device float64 reduced QR followed by CPU reduced
  SVD of `R`, retains `s > max(P,k)*eps64*s_max`, and reconstructs
  `Q = Q0 U_keep`. It saves the full first-step basis/actions/coefficients and
  geometry diagnostics; continuation records `P`, `k`, singular values,
  tolerance, rank, `s_max`, truncation residual and action timing each update.
  Stored legacy `V` is never replaced by `Q`.
- The Adam diagnostic correctly uses carried post-step `m` and `v`, their
  per-parameter step counters and bias corrections, and separates mathematical
  adaptive movement, decoupled weight-decay movement and actual displacement.
  An additional synthetic float32, 1501-step check produced a maximum
  decomposition residual of `6.21e-9`, consistent with float32 update rounding;
  the full movement components and residual norm/maximum are saved, so the
  residual tensor is reconstructable rather than asserted equal to zero.
- Step-1501 checkpoints are written before evaluation. Later original-grid
  evaluations assert unchanged CPU and all-CUDA RNG. Source hashes bind the
  helper, runner, batch wrapper, both fixtures, inherited implementation and
  protocol, and are rechecked before completion.
- The batch wrapper has the fixed seed order 100--104, uses seed 100 only as a
  completion/resource admission, has no scientific-outcome gate or retry, and
  passes the same seed-100 completion to all remaining seeds. Output creation is
  exclusive under a fresh large-volume parent. Prospective per-write accounting
  enforces the cumulative 20 GiB envelope and 1 GiB free reserve.

## Validation and launch conditions

All 37 `test_grokking*.py` tests passed under one-thread math settings in 1.668
seconds; the three new source modules and two new fixture modules also compiled.
The focused action suite contributed 10 passing synthetic tests, including
native exactness, rank loss, norm edge cases, estimator recurrence, fork restore
and carried-moment Adam decomposition.

The pass is conditional only on operational requirements that are outside these
Python sources: commit the reviewed frozen sources before acquisition; launch
the fresh exclusive parent under the declared 16 GiB RAM, zero-swap, 400% CPU,
three-hour-per-seed and 12-hour-overall limits; verify the effective cgroup
values before observation; and do not restart or alter the fixed roster after
seed-100 admission. The code's seed subprocess timeout supplies the per-seed
backstop, while RAM/swap/CPU and overall runtime require the external service
unit described in the protocol.

The stronger review suggestion to retain projector orthogonality/reconstruction
residuals at every continuation update is not implemented: these are saved for
the common first action, while every update saves the decomposition and rank
quantities required by the protocol. This is a transparent diagnostic
limitation, not a change to an action or estimand. Likewise, historical CUDA
backend flags were not archived; the runner records current flags and correctly
does not claim historical bitwise replay.

Method: change-focused review of all new files and their inherited restore,
filter and optimizer paths, following the repository review workflow; no other
reviewer's verdict was imported.
