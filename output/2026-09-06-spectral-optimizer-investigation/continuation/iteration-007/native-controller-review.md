# Development controller integration review

Codex / Spectral Optimizer Investigation, 7 September 2026.
Base: `6b51871796eca3c0ea47c81b728ed18b9b01d250`.

## Outcome and evidence boundary

The [implementation decision](native-controller-decision.md) preserves the study
and its limits. A passing pilot does not authorize primary training. Native
primary preparation is explicitly disabled until its own permission/environment
path is integrated. Only the synthetic contract test exercises that schedule
position, using primitive payloads rather than seeded plans.

## Implemented

- `RuntimeGuard` accepts paired launcher-entry wall/CPU observations and their
  entry-captured PID, measures cumulatively from those origins and rejects a
  different process before probing. Default clock-call behavior and scientific
  probe/limit restrictions remain unchanged. A defensive origin property is
  available to the controller; absolute origins/PID are not added to scientific
  resource records. This is cooperative enforcement, not hard preemption.
- `native_branches` loads the verified saved plan and registered saved anchor/
  witness, delegates the fixed numerical branch producer and binds its return.
  It restores/verifies caller RNG and terminalizes even a post-seal failure.
  A redundant preliminary path-based restricted decode remains; the subsequent
  authoritative held-descriptor reread/exact comparison still governs execution.
- `native_controller` validates the initial bootstrap's actual saved metadata,
  journal, root and entry clocks. It records started before each fixed producer,
  authenticates actual output/receipt bytes and exact membership, validates saved
  source transactions and returned values, then records sealed. Completion plan
  and nested anchor/witness refs are bound to the actual files. A valid unequal
  pilot retains all seven source-pair outputs and their sealed operation before
  terminalizing; neither branch nor the boundary can follow. Post-boundary guard
  failure also terminalizes, so the syntactic boundary cannot authorize reuse.

## Independent review and corrections

Reviewer `i7_diagnostics_audit` read the fixed producer APIs and all six final
implementation/test snapshots; the parent also reviewed the code and ran the
controller/full regressions. No remaining material static finding in the bounded
components. Exact reviewed hashes are in checks (artifact not distributed in this public snapshot).

Review found and resolved missing PID binding in the origin API, missing PID in
controller construction, and a wrong three-field plan-artifact argument where
the producer requires ordered `{sha256,size_bytes}`. It also found that common
envelope validation alone did not bind saved payload semantics/producer returns,
and that nested completion refs were not yet tied to actual files. These checks
were added before any native execution. They were real wiring/binding defects,
not scientific failures. The first controller fixture run passed eight tests;
the expanded ten-test run additionally covers nested refs and exact native API
wiring with fail-before-work doubles. Those doubles do not generate study data.

## Verification

The parent full I7 suite passed **412 tests in 278.539s**, optimizer regressions
passed 17 in 1.339s, and reminder tests passed six in 2.385s: **435 distinct
tests**, 20 more than the base. Commands, terminal handles and source hashes are
recorded in checks (artifact not distributed in this public snapshot). Focused runs are subsets,
not additional independent tests. The controller uses
actual bounded temporary stores with synthetic-contract metadata/payloads; it
does not validate 220-update native execution. Branch tests exercise the real
registered eight-update CPU MLP source-to-branch path with synthetic IDX. All
new writer tests retain the verified-big-volume, hidden-CUDA, single-thread,
2-MiB store/1-MiB reserve, 1-GiB free-space and cooperative 120-second/2-GiB-RSS
test discipline. Existing regressions keep their established scopes.

## Remaining finite launch path

1. Connect the pre-import bootstrap guard and a boundary-only close/read-only
   inspection handoff, including exit cost and structured partial-root failure
   identity. Current development return still holds its writer lock; it is not
   itself an independently inspected boundary or reopen permission.
2. Integrate separately scoped primary/sensitivity permissions and fresh phase
   environments, then their fixed source-before-branch loops. Instantiate one
   primary guard before long-plan preparation and retain it through primary GO;
   do not reuse a prior process's CPU timestamp or silently reuse development
   metadata after reopening.
3. Connect all 16 groups to the separate CUDA-uninitialized CPU audit process,
   with its own complete resource accounting and GO.
4. Freeze the 12-primary/four-sensitivity collector before outcomes. Finish
   actual shared storage accounting, including external permission/journal/
   failure copies, metadata, receipts and transient copies. The earlier
   958,225,106-byte conditional projection is not a native fit certificate.
5. Bind the complete assembled scientific source membership, commit it clean,
   independently review the resource gate, and only then consider the one
   predeclared development pilot. No such GO is issued by this checkpoint.

The scientific collector still has its prior 45-file membership; new adapters
are not yet a complete bound source set. Production `spectral_filter.py` and
inspection001 evidence are unchanged. No durable scientific conclusion changed,
so no knowledge-base update is warranted. The best-practices check informed the
separate wall/CPU and same-process accounting design, as cited in the decision.
