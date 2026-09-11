# Native adapter component review

Codex / Spectral Optimizer Investigation, 7 September 2026.
Base commit: `809467f91c09eb5a3308fb59c0652cf7907331fe`.
Scope: [implementation decision](native-adapter-decision.md), CPU fixtures only.

## Outcome and evidence boundary

The finite native path now has concrete input, streaming-source and bootstrap/
inspection components. These are engineering results, not evidence about an
optimizer effect. No real study plan, MNIST read, GPU source update, pilot or
native inspection was executed. The consumed native-layout inspection001 and
all historical results remain unchanged. Existing production/scientific modules
were not edited; the scientific collector still has its prior 45-file membership.
The new modules cannot be treated as a complete bound native source set yet.

The parent owns inputs and integration decisions; leaf i7_current_storage_count
owns streaming/source integration; i7_native_inspection_helper owns control;
i7_diagnostics_audit is the independent read-only reviewer. No leaf launched an
experiment or wrote shared modules. Imports/default entrypoints remain inert.

## Implemented components

- `native_inputs.py`: exact historical six-stream plan generation, strict owned
  plan tensors, guard-stage calls, and bounded pinned loading of only the two
  training IDX files. Tests actually draw only bundle 0 at registered MLP fixture
  dimensions. All five scientific identities are checked using doubles that
  raise before each RNG draw; no scientific arrays are produced. The fixture
  integration generates, seals, verifies/loads and materializes a plan.
- `source_streaming.py`: one-pass completion/pilot-comparison assembly into
  existing wire formats, exact update/ref placement, direct typed-byte equality,
  final-core-only retention, and terminal in-memory handling of invalid calls.
  Reference construction validates writer receipts but does not independently
  reread them; the existing explicit external-verification limitation remains.
- `native_source.py`: fixed live factories, exact scheduled source updates and
  live anchor/witness transaction, bounded stream consumption and create-only
  completion/comparison writes. Explicit per-arm RNG save/restore isolates the
  interleaved pilot while rejecting RNG changes inside an update. Valid unequal
  rows are retained, with `invariant_pass=false`; the final controller must stop
  before branches or another GO. Exceptions terminalize a writable store without
  appending after terminality. There is no branch runner or scientific CLI here.
- `native_control.py`: exact external permission/journal and singleton scope,
  attestation formation before store creation, fixed metadata copies, read-only
  prefix/receipt inspection, nonblocking held-lock validation, positive final
  consistency checks and terminal/incomplete/pre-root classifications. Policy
  records still grant no execution authority. Cumulative clock arithmetic is
  available; full resource enforcement from launcher entry is not implemented.

The 90-record control fixture uses primitive payloads and synthetic native
metadata. Its positive inspection checks actual bytes, receipts and policy
consistency, not the 82 scientific envelopes or native hardware behavior. The
source-loop fixtures use the registered eight-update CPU MLP and real tiny
writer transactions. Injected unequal rows are adverse-fixture evidence, not
an observed native capture effect.

## Review corrections and false starts

The input adapter initially used a zero-argument checkpoint. Review caught its
incompatibility with `RuntimeGuard.check(stage)` even though the initial 11 tests
passed. The signature and closed stage sequence were corrected; a regression
now uses the actual existing guard API. Final focused input verification passed
12 tests in 0.262s. Input review found no remaining material issue.

Control review tightened exact attempt/path scope, native-versus-fixture store
profiles, nonblocking external journal reads, held-lock identity/final snapshots,
known interrupted-operation outputs, terminal precedence even after boundaries,
factual combined failure statuses and symmetric external-failure/root bindings.
No-root failures are inspectable without inventing a root. Caught interrupts
close a returned store; hard-death or pre-return partial-root identity can still
be unavailable. An initial control test expected two filenames in the wrong
sorted order; that test-only error was corrected. Final focused control tests
passed 12/12 in 26.424s; independent static review passed its frozen files.

The first streaming test mislabeled its second actual optimizer update as 3;
the existing state validator correctly rejected it. The test was corrected to
submit a valid duplicate core. A reviewer subsequently alleged that anchor
updates were skipped, then explicitly retracted the finding after reading the
full `capture_anchor_then_live_witness` implementation: that helper performs
the actual update. Adding another normal step would have doubled those updates.

Later source review found a real binding defect: a structurally valid modified
caller plan could retain the original file reference. The accepted correction
loads the actual sealed plan through `load_verified_plan_from_store`, compares
the exact reference and typed plan bytes, and runs the loaded owning plan. A
valid in-range batch mutation is rejected before initialization, materialization
or updates, retaining one store failure. Final focused source verification
passed 6/6 in 8.022s; the adjacent source/stream/history/capture run passed 32 tests
in 17.338s. These overlap the full suite rather than adding independent test counts.
Independent static review passed all eight frozen implementation/test files.

## Remaining launch-critical integration

1. Assemble the fixed phase controller and actual branch membership loops around
   these components. A false pilot invariant is a stopping gate, not a success
   return that may be ignored. No incomplete operation may be resumed.
2. Enforce cumulative wall/CPU/RSS/GPU accounting from entry, including imports,
   initialization, metadata/root work, hashing, reads/writes and exit. Do not
   reset primary preparation time at primary GO. Connect later phase environment
   observations and the separate CUDA-uninitialized CPU audit process.
3. Finish structured partial-root identity reporting when store construction
   fails before returning; do not guess directories from timestamps or retry.
4. Close the actual adapter inventory under the unchanged shared cap, including
   permission/journal copies, metadata, receipts, failures and transient copies.
   The prior 958,225,106-byte conditional projection (artifact not distributed in this public snapshot)
   is not a native fit proof.
5. Freeze the complete 12-primary/four-sensitivity collector, including domain
   masks, audit resolution and no pooling of sensitivity. Update actual scientific
   source membership only with the assembled reviewed source set.
6. Commit/review that complete path and issue a separate scoped development GO
   only if its prospective resource gate passes. No such GO is issued here.

No new research conclusion warrants a knowledge-base update in this checkpoint.
The best-practices check informed the RNG API and descriptor/lock choices; its
official-source references are retained in the implementation decision.

## Parent verification and handoff

The final full I7 suite passed **392 tests in 247.615s** with CUDA explicitly
hidden, all numerical thread variables set to 1, bytecode writes disabled and
TMPDIR on the verified big volume. The 17 optimizer tests passed in 1.305s and
the 6 reminder tests in 2.249s: **415 distinct tests**, 36 more than the base
checkpoint. Source-adjacent and focused runs are subsets, not additional tests.
Exact commands, hashes, handles, scope and remaining work are in
native-adapter-checks.json (artifact not distributed in this public snapshot).

Parent suite handle77355/PID1158231 is terminal; optimizer handle61043, reminder
handle7401 and input handle6313 are also terminal. The old native controller/
worker989661/989705 and prior test PID1094762 were freshly absent. No matching
disposable native-input/source/control fixture root remains. Only unrelated
desktop GPU processes2101 and8861 were observed. The prospective development
attempt directory was not created. Original inspection001 marker/failure and
production optimizer hashes are unchanged.

The independent reviewer passed the eight final source/test hashes, including
the last sealed-plan regression. All three leaves are complete. This checkpoint
made concrete implementation progress; it does not finish or block the research
goal. The original two-hour reminder remains enabled/active and unchanged.
