# Native-layout engineering decision

Codex / Spectral Optimizer Investigation, 7 September 2026.

Parent read the current objective, complete saved state, resource contract,
source/environment policy and actual collector code. Independent read-only
review found the proposed zero-update environment inspection distinguishable
from the development attempt, with the caveats adopted in
the proposal (artifact not distributed in this public snapshot). Official CUDA API guidance
confirms that initialization is a real side effect, even for an RNG-state read.

## Implementation decision

Approve only the new inert inspection helper and CPU-only tests at this stage.
Keep production/scientific modules, source membership, numerical tolerances,
registered profiles and all launch gates unchanged. A future positive check is
an engineering observation, not execution-history or full-fit certification.

Before native observation: parent must read the helper, pass the CPU tests,
commit the proposal/decision/code, verify a clean source checkout and fresh
GPU/mount state, then explicitly record the exact command/commit/attempt root
as development-independent inspection GO. Until then CUDA initialization is
not approved. Once started, there is no automatic retry; inspect the exact
handle and retained attempt marker on continuation.

Only this named observation may supersede the older checkpoint's no-CUDA-
initialization boundary. It does not permit a native pilot, data read, source
update, model/optimizer construction, branch or prior-experiment restart.
After it terminates, use its actual source/environment metadata in a CPU-only
storage comparison; retain the earlier component observations unchanged.

Independent review amendments adopted before implementation completion:
singleton inspection directory instead of random retry roots; separately bound
helper without source-schema changes; wrapper UUID pin and observation-only
driver; fully specified controllable settings; own-process-group timeout/reaping
and controller-owned failure publication. The native collector stores layout
only and does not exercise continuation/restore paths. Memory caps remain
cooperative observations, not virtual-address-space or CUDA-context hard limits.

## Parent inspection-only scope approval

Parent completed full code/contract review and adopts the independent review's
conditional PASS in native-layout-inspection-review.md. The frozen helper's
SHA256 is9e60b45e9e5d85a9e5b8bdb74b29f60ce3470813697a8fe8ec559561861e9c4f;
its tests are1869b9c93a5618f900224874c78e5032a1d840e5a6156d6ce27e86e1f83fa68f.
Parent full regression passes338 I7 tests (189.741s),17 repository tests
(1.349s) and six reminder tests (2.738s). Final comparator label clarifications
have an additional four-test focused pass (1.058s). The helper was not changed
during the full suite; comparator-only label edits were verified afterward.

Approve exactly one engineering native-layout observation at the fixed root
`/tmp/spectral-experiment-artifacts/spectral-i7-native-layout-inspection-001`, on GPU UUID
`GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c`, after this decision and exact sources
are committed and a fresh clean status/resource check passes. Record the actual
full commit and command in the tool transcript before executing. The command
uses the committed native_layout_inspection.py with --inspect-native-layout,
--expected-commit and --expected-gpu-uuid; no implicit/default execution.

This approval adopts the explicit45-second monitored work deadline,5-second
TERM grace, separate1-second live-descendant disappearance check and failure-
path5-second post-exit GPU query. Local filesystem publication is not hard-
preempted. Memory limits are cooperative per-controller/worker observations,
not aggregate process-tree or total CUDA-context hard limits. Initialization
is a real side effect. Missing NVIDIA own-process footprint remains null.

After the single attempt terminates, preserve its records and use a completed
passing result only in native_metadata_storage.py's CPU-hidden paired anchor
comparison. Existing CPU synthetic fixture draws/restoration are allowed for
that offline step, not for the native inspector. No repeated native attempt,
new directory, cap change, native pilot or scientific dataset/plan is approved.

## Terminal outcome

The one approved observation ran at e806973c88b3e51e472b0a42a15ac2798be44031 and
failed (terminal handle56037). The singleton is consumed; no retry is approved.
The failure and marker remain intact, with byte-identical committed copies.
No passing native metadata exists, so the paired comparator has not run against
an inspection result. See native-layout-inspection-results.md for the lost
worker-error detail, CPU-only diagnostics and unresolved native measurements.
