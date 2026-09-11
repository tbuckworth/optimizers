# Process supervision and complete accounting decision

Codex / Spectral Optimizer Investigation, 7 September 2026.
Base: `05921ac955ac47e978bf47b916986d304191aaf6`.

The preceding turn was verified engineering progress: 480 distinct passing
tests, committed phase/controller/audit integration and a clean 55-member source
collection. Fresh recovery found no live experiment or prior test process. The
goal is active; its original scientific question, membership and resource limits
remain unchanged. No native GO is issued by this decision.

## Work toward the assembled launcher

Parent owns a new `phase_worker.py` and its tests: an inert-by-default fresh
interpreter entrypoint connecting an explicit authenticated phase request to the
existing development or scientific controller, actual fixed metadata setup,
closed boundary handoff and bounded structured output. Native paths never accept
fixture callbacks, replacement seeds, alternative data or a CPU-training fallback.
The CPU audit starts in a genuinely new interpreter with CUDA hidden and remains
CUDA-uninitialized. Worker CPU origins belong to the worker; supervisor wall
accounting must include startup rather than resetting the phase after imports.

The supervisor's import/default CLI are Torch-free. Its explicit source
authentication may import CPU Torch transitively, but must check CUDA remains
uninitialized before/after collection and charge that work to its wall deadline.
This does not reuse a numerical context in the fresh exec worker.

Control leaf owns `process_supervision.py` and tests. First propose the exact
request/output/persistence API for agreement with parent and independent reviewer.
Launch one fresh interpreter per explicitly permitted phase, with explicit args,
working directory and sanitized environment; no shell or fork-inherited CUDA.
Capture bounded output, observe the actual child exit and verify the closed
handoff. A failure, timeout or incomplete record is nonresumable, never an
automatic retry. Any termination must target only the positively identified
owned child/process group, never an unrelated GPU or agent process.

Supervision evidence must survive a failure before development bootstrap creates
its attempt journal. The proposed solution is one fixed, create-only metadata
namespace beside the existing attempt namespace, never another scientific data
root. Its exact names, record ceilings and no-retry semantics need agreement
before implementation. Every byte in both namespaces counts inside the existing
shared 1 GiB budget/reserve. No per-phase unaccounted root or log allowance.
Existing consumed inspection001 is preserved and never invoked or retried.

Protocol agreement: fixed sibling `i7-native-development-attempt-001-supervision`
contains only four phase requests (8 KiB each) and four exits (16 KiB each),
96 KiB maximum charged within the shared budget. A partial/missing exit means
`launched_or_unknown_consumed_incomplete`, not recoverable ownership or retry
authority. The supervisor keeps the leader unreaped using Linux pidfd/waitid
WNOWAIT while observing/cleaning its inherited process group. Synchronous
Git/NVIDIA/mount helpers are descendants; leader exit alone does not prove cleanup.

The prospective frozen source worktree is
`/private-artifacts/repositories/optimizers-i7-native-frozen-001` (not created here).
It is not a second scientific output root. Inputs remain the previously pinned
raw MNIST training files; no alternate dataset or test split is admitted.

Review found and corrected a fresh-process setup gap: development bootstrap and
later acquisition now require an explicit native initializer after their durable
journal/transition marker, before native metadata or any root/payload write.
The initializer receives a copy of the reviewed permission and returns no payload.
Same-process wall/RSS/PID and CUDA-uninitialized checks precede it; device/UUID
and retained-origin RuntimeGuard checks immediately follow. Audit and fixture
paths forbid this callback. This API change grants no native execution authority.

Storage leaf owns `final_storage_accounting.py`, tests and a final ledger draft.
Reconcile actual native filenames, scheduled payloads, 90 phase records, metadata
and receipts, all supervisor/permission/exit files, bounded failure capacity,
temporary-write behavior and source-membership growth. Reuse existing complete
component measurements with their explicit conditional status. Separate proven
format bounds, measured CPU specimen bytes, runtime-only conditions and unresolved
native deltas. Never relabel the old conditional projection as measured native
fit, fabricate a successful native inspection or silently enlarge a limit. First
identify any remaining load-bearing measurement needed before a pilot decision.

Parent adds execution-bearing files to the exact source manifest only when the
module membership is settled. A frozen clean source worktree must remain at one
revision across all phases. Checkpoint notes and permission decisions must not
dirty or commit into that source worktree while it is the study's pinned input.
An eventual separate frozen worktree is a launch preparation step, not a reason
to rewrite source claims or restart old work.

The independent reviewer is read-only and checks design first, then frozen
interfaces/implementation and storage claims. Workers are leaf-only: no spawning,
user dialogue, commits, native invocation, actual IDX reads or scientific plans.
Verification uses registered tiny CPU/primitive process fixtures, hidden CUDA,
single numerical threads and verified `/tmp/spectral-experiment-artifacts`. A bounded new
storage-only measurement requires a separately stated specimen/resource scope;
do not repeat completed measurements as though they were interrupted.

## Platform check

Python documents explicit child environments and observed return codes, and warns
that process creation itself is not universally interruptible. Use the advanced
Popen interface only where bounded incremental output and ownership-aware cleanup
require it. A timeout result is not proof that every descendant disappeared.
Retain that distinction in reports; in-process resource guards remain cooperative.

- [Python subprocess](https://docs.python.org/3.12/library/subprocess.html)
- [Python resource](https://docs.python.org/3.12/library/resource.html)
- [Python OS process interfaces](https://docs.python.org/3.12/library/os.html)

This implements the remaining real launch path; it does not substitute another
scientific question, issue a native GO, or authorize a new inspection attempt.
