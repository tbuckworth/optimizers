# Closed root accounting: engineering checkpoint

Codex / Spectral Optimizer Investigation, 7 September 2026.
Started from clean b22cedc8ea0fe78472e2b8381715018fad0bf781.

## Finding and implication

The defined payload/receipt/control-record baseline is now explicitly accounted
for and exercised through the actual storage writer using tiny synthetic
bodies. This closes a storage-plumbing uncertainty, not the native feasibility
gate. No optimizer result changes and no experiment was restarted.

The investigation still lacks a native failure-record adapter: after a store
write failure, the normal API refuses both subsequent payload writes and the
proposed chained phase-failure record. Reopening that terminal store for writing
is also refused. A valid in-memory failed transcript is not a retained one.
The test preserves this negative finding; it does not change store behavior.

## Conditional accounting

The machine-readable projection (artifact not distributed in this public snapshot) uses the
existing zero/ones component sizes in
the final measurement (artifact not distributed in this public snapshot), not a new
serialization run. Both patterns have the same eleven component sizes.

| Charge | Logical bytes |
| --- | ---: |
| 82 projected component payloads | 945,341,202 |
| Their actual-encoder receipts | 20,896 |
| 90 phase bodies at their 128-KiB ceiling | 11,796,480 |
| Phase receipts under the chosen fixture names | 17,730 |
| Scientific default header and empty lock | 222 |
| Conditional retained total | 957,176,530 |
| Shared failure reserve, capacity rather than a file | 1,048,576 |
| Total including reserve | 958,225,106 |
| Unallocated remainder below the 1-GiB cap | 115,516,718 |

This approximately 110.17-MiB remainder is **not a native-fit certificate**.
The component bodies are a historical storage-only projection, not native size
upper bounds. Native copied source/environment metadata, CUDA RNG layout and
any additional adapter files remain unresolved. Outer admission/write caps can
reject oversized records; rejection does not prove every required record fits.

The closed baseline has 346 regular files: 82 scheduled payloads, 90 synthetic
phase records, 172 receipts, one header and one empty lock. The fixed policy has
41 work operations, four decisions and four boundaries, yielding 90 successful
records. The helper verifies this actual schedule, rather than relying only on
a hardcoded record count. The `storage-phase-NNN.json` convention belongs to
this fixture, not an implemented native adapter.

## Tiny actual-writer checks

The tests use primitive-only, explicitly storage-only/nonsemantic bodies under
all 82 scheduled names. The store header uses the existing MLP fixture profile;
scientific membership strings in phase records are labelled
`synthetic_contract_fixture` and `execution_enabled=false`. No model, training
data or scientific plan is generated. The Torch RNG snapshot is observation
for an unchanged-state assertion, not a source training state.

Checks cover raw payload lengths/hashes, strict JSON or restricted CPU Torch
roundtrips, actual receipt encodings, all 90 persisted/reloaded records, the
declared hash chain, exact inventory and pinned close/reopen. Import and default
CLI remain Torch-free and inert; explicit accounting APIs lazily import the
actual CPU policy/store.

Each disposable store has a 2-MiB logical cap, an internal 1-MiB failure reserve,
and a 1-GiB free-space floor on the verified large volume. CUDA is hidden;
numerical thread settings are one. The 120-second body and 2-GiB RSS checks are
cooperative, not hard preemption. Scoped contexts close stores and remove only
their temporary fixture directories. No bulk root is retained.

Resource fields are pre-record-write snapshots. The last record therefore
excludes its own body and receipt; a separate final inspection reports the
actual total. Allocation is the sum of regular-file `st_blocks * 512`, excluding
directory metadata. Clocks cover metadata plumbing and reopen/hash work is
cache-warm. Persisting `development_go` after creating the fixture store does
not test the required online decision-before-root lifecycle.

Parent full-suite observations: the success fixture had 346 regular files,
172 receipts, 287,911 logical bytes and 1,429,504 allocated bytes. Its final
record/receipt added 1,208 logical and 8,192 allocated bytes after the last
pre-write snapshot. The separate terminal fixture had five regular files, one
failure, 1,071 logical bytes and 16,384 allocated bytes; all three subsequent
write/reopen refusals passed. Both temporary roots were removed and no matching
test directory remained in the final read-only check. These are tiny fixture
observations, not native or full-width measurements; timing/resource/path fields
can change their serialized lengths between test invocations.

Verification evidence (artifact not distributed in this public snapshot) records 356 I7 tests passing
in 206.363 seconds, 17 optimizer tests in 1.283 seconds and six reminder tests in
2.007 seconds: 379 distinct passing tests. The implementer's focused eight-test
run is additional verification, not eight extra distinct tests. All parent
handles are terminal; the full-suite PID1094762 and old native PIDs989661/989705
were absent at the final check. The timer remained active on its original
two-hour cadence, next 03:21:37 BST when checked at 01:21 UTC.

## Review and correction history

The research-continuation workflow and independent bounded reviews selected
closed accounting followed by tiny writer tests. A repeated roughly 945-MB
specimen was deferred because it would not resolve the native metadata gaps.
Official serialization/API validation is recorded in
[the scope decision](root-accounting-decision.md).

- The inventory review found a real fail-closed gap: artifact membership alone
  did not validate the operation kinds that determine phase-record count. The
  helper now checks the actual 49-operation schedule and 90-event derivation;
  a regression changes an operation kind while preserving artifact names.
- Parent/protocol review added actual metadata clocks, exception-safe store
  closure, payload roundtrips, single-thread checks and exact benign failure
  fields. Final independent accounting and fixture reviews found no remaining
  material defect within this engineering scope; the fixture review was static.
- The implementer's first arithmetic diagnostic read the superseded
  full-storage-measurement.json. The final file's two anchor component sizes
  are 64 bytes larger each, repeated 18 times: a 1,152-byte difference. No
  padding was added, and neither historical measurement was overwritten. This
  identifies the changed components, not the original cause of their size change.
- An intermediate test edit introduced an IndentationError and failed before
  running tests. It was corrected before the implementer's final 8-test pass
  (16.326 seconds). Earlier 7-test passes predated the added environment test;
  they are not additional distinct tests.

## Remaining gates

Native attempt001 remains failed/consumed, with its original marker and failure
bytes unchanged. Its exact failure cause and CUDA-initialization outcome remain
unknown. No new inspection, I7 data/plan/pilot/source update, native retry or
full-width measurement is approved by this checkpoint.

Next safe work is a bounded design/review of terminal-safe failure recording and
online GO/root lifecycle, alongside the explicit native metadata requirements.
Any further native inspection needs a separate scope/GO decision after checking
the live goal, handles and committed state. The study cap and scientific source
membership remain unchanged; this engineering checkpoint does not achieve the
research goal or authorize a scientific run.
