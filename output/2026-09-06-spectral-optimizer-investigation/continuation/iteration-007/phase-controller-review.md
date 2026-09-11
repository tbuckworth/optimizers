# Single-root CPU phase checkpoint

Codex, Spectral Optimizer Investigation, 6 September 2026.
Evidence status: **deterministic synthetic engineering**, not an MNIST finding.

## Result and implications

The actual tiny plan -> eight-update source -> six-branch -> independent numerical
audit pipeline now works in one capped artifact root, including close/reopen at
ready, primary-complete and final-complete boundaries. This removes the earlier
separate-plan-store fixture limitation. The native study remains disabled.

The source is the registered 26-parameter MLP fixture, primary bundle0, anchor5,
eight total updates. Its source completion core checks actual optimizer/observer
counters at eight before any branch starts. The full smoke path retains the real
anchor/witness, one complete endpoint, actual branch results and independent raw
audit. It verifies 24 Adam parameter checks / 72 tensor screens and all 390
independent contrast checks. It has no real data, scientific plan or CUDA context.

Nine immutable event files and their receipts bind the same root device/inode,
header hash, full verified plan/receipt reference, context digests, previous
event, exact preceding file inventory, operation outputs and runtime snapshots.
The finished smoke has 15 payload receipts and 32 regular files in total. Observed
tiny-profile bytes/timing/RSS are in phase-fixture-measurements.json (artifact not distributed in this public snapshot). These are
not full-width sizing, an upper bound, or a complete native-phase certificate.

## Implementation and review

- The held plan adapter shares the existing exact byte/receipt/schema verifier
  and keeps the writer's exclusive lock. Pinned reopen validates the same root
  and complete nonterminal inventory, restores its original byte cap/reserve,
  and never creates a replacement budget root.
- The fixture controller uses a closed event sequence, explicit fixture-only
  decisions, exact new payload/receipt membership and an operation-start event
  sealed before calling the body. Ready/primary-complete/complete are the only
  writable recovery boundaries. Source-complete is deliberately not a boundary:
  one primary guard continues across source and branches without resetting.
- The fixed runners finish the real source, reload its saved inputs, execute
  both branch orders through the existing producer, then use the same-root raw
  numerical auditor. Default CLI calls and importing either entrypoint are inert.
- Producer guards now cover source/capture/file boundaries, candidate creation,
  every canonical/reverse branch and existing measurement loss chunks. Completed
  raw files are retained on guard/operation failure; the store becomes terminal.

Independent design review supplied exact fixture membership, update8 completion,
full plan receipt binding, boundary inventory closure and crash/callback tests.
Two separate leaf implementations covered store I/O and producer guards. Parent
reviewed their diffs and wrote the phase controller, runners and actual smoke.
A fresh read-only reviewer checked the complete integration and found no remaining
blocking defect within its documented fixture scope after these corrections:

1. Completion timestamps originally reused the pre-body timestamp. They now
   obtain a fresh UTC value after the body, before sealing completion.
2. The first wording overstated what callbacks prove. A reviewer demonstrated
   that deliberately fabricated common-envelope callback outputs could reach
   fixture `complete` without a real branch/audit, while the scientific flag
   remained false. The contract now explicitly distinguishes trusted callback
   attestations/common-schema checks from actual execution in the fixed smoke
   entrypoint. The manifest is **not** a full semantic validator or independent
   execution-history certificate. This trust boundary is not suitable as a
   native scientific launch gate by itself; native construction remains rejected.
3. Parent replaced loose inventory value comparison with direct typed equality.
   A regression rehashes the event and receipt after substituting boolean false
   for the lock's integer zero byte count, then requires rejection.

The initial phase test harness had two errors, not failing scientific results:
one strict JSON read omitted its required byte cap, and the abrupt-exit test
forked an already guarded process. An isolated reproduction showed the child
CPU clock had reset (`cpu_seconds=-1.232859847`, `clock_regression`), so the guard
correctly stopped before the intended crash point. The test now starts a fresh
process owning its clocks and exits abruptly after the started marker, optionally
after a retained raw file. No production clock check was relaxed. Both cases
refuse the stale boundary pin and leave the files present until fixture cleanup.

## Verification and limits

The final 14 phase tests cover the actual source/branch/audit path and three
reopens, ordering, forged success, GO reuse, strict profile separation, source
and plan changes, stale/extra-file boundaries, abrupt exits, wrong completed
counts7/9, typed inventory corruption, shared byte cap/failure reserve and inert
entrypoints. Eight store-I/O tests and five producer-guard tests add targeted
lock/pin/replacement/cap and pre/postwrite/branch/chunk coverage. The complete
regression passes 283 I7 tests (171.873 seconds), 17 repository tests
(1.282 seconds) and six reminder tests (2.156 seconds): 306 tests total. The
separate final phase suite passes 14 tests in 33.294 seconds. Knowledge lint
(14 pages / 11 indexed content pages) and whitespace checks pass. Exact
commands, results and current source hashes are in
phase-controller-checks.json (artifact not distributed in this public snapshot); older checkpoint hash records remain historical.

The research-workflow skill kept this checkpoint on the existing step9
continuation with independent review. The best-practices check used primary
Linux flock/Python os documentation before implementation. Hypothesis-first
debugging isolated the test's child-clock assumption before changing the harness.

## Next actual work