# Parent decision: bounded development pilot GO

6 September 2026, before any iteration006 dataset execution. The parent has
re-read the active user objective, workflow state, protocol and code review.
The previous goal turn made progress by committing the reviewed implementation
and exact theory; this decision advances the empirical question without
changing its outcomes, policies or analysis.

Approve **one development pilot only** under the frozen protocol: seed9880,
nominal replacement .9, six arms, 220 updates each with instrumentation off/on
(12 traces). No validation/test access, accuracy computation, learning-outcome
selection, extra seeds, automatic retries or full study is approved here.

## Evidence at GO

- Scientific-source commit: `1221d70f1d95c8c8b60e701a08c9cedd3c3b5772`.
  All fifteen files match the committed bytes and implementation-checks.json;
  the gate was independently rerun with CUDA disabled. Worktree was clean.
- Three design challenges and parent code review are complete. Sixty synthetic
  CPU tests, seventeen repository tests and 1,320 actual tiny-CPU producer-row
  checks pass. The pre-run pilot-binding bug is fixed and regression-tested.
- No iteration006 pilot, results directory or pilot log exists at this decision.
- Fresh inspection shows the single RTX3090 with 23,475 MiB free, only the
  permitted remote-desktop and Stremio compute entries, and 48 GiB available
  host RAM. The harness must repeat its own occupancy/resource gates at launch.
- Bulk storage resolves to /private-artifacts/storage on /dev/RECONFIGURE_FOR_LOCAL_STORAGE, with the documented
  UUID and about751 GiB free. The workspace has only2.3 GiB available; keep
  bulk artifacts exclusively below verified /tmp/spectral-experiment-artifacts.

## Execution and stopping

Commit this authority record before launch. Run the unchanged harness once
with `--pilot --development-go`, RESEARCHER_BACKEND=codex, retaining complete
stdout/stderr in pilot-run.log. The launch itself records environment, source,
data and artifact bindings. Preserve any partial failure and stop; do not
overwrite or relabel a failed attempt.

After completion, independently inspect the pilot's raw invariant/timing
artifacts and hash/source bindings. Only a separate parent full-run decision
may approve the 36-cell learning comparison. Pilot success is not learning
evidence, and no full-run outcome exists at this GO.
