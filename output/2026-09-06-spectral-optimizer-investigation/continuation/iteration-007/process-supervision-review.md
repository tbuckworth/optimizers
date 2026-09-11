# Process supervision checkpoint review

Codex / Spectral Optimizer Investigation, 7 September 2026.

## Static verdict

PASS for the component checkpoint, **not** native launch readiness. Frozen
implementation hashes and regression results are recorded in
`process-supervision-checks.json`.

- Native initialization occurs after the durable inner journal/transition marker,
  with copied permission, own-process origins, resource checks and GPU UUID checks.
  Audit and fixtures forbid that initializer.
- Fresh workers dispatch exactly one permitted phase, normalize the closed
  handoff, use private bounded failure classes, and never grant authority through
  a success report. Audit uses a genuinely new hidden-CUDA interpreter.
- The create-once supervisor authenticates source/permission/admission pins and
  the exact predecessor request/exit/handoff. Parent-directory fsync precedes
  launching after namespace creation. Partial evidence is nonresumable.
- Supervision retains the leader unreaped while observing/cleaning its owned
  session/group. It never signals a reused PID/group after reaping. Exceptions
  retain observed child identity and report uncertainty; leader exit is not
  substituted for descendant cleanup. Output and persistent records are bounded.
- Source authentication may import CPU Torch in the supervisor, with explicit
  CUDA-uninitialized checks and cumulative wall accounting; import/default CLI
  remain inert and Torch-free.
- All 364 study files, eight supervision files, external control evidence,
  consumed inspection and two 64-KiB storage-evidence prerequisites are counted.
  Caller-supplied integers/digests cannot turn conditional arithmetic into a fit.

## Corrections made during review

The initial worker handoff shape did not match the supervisor's six-field
protocol; it now normalizes the controller handoff. Resource-limit reports can
retain over-cap observations and unavailable probes instead of becoming malformed.
Source binding was split so worker-side request authentication cannot import
Torch before development bootstrap. Final record timing cannot reject the very
overrun it must report. Exceptional post-spawn handling preserves child identity;
post-reap errors cannot trigger group signaling. Process-directory read errors
are not silently interpreted as proof of absence. Admission uses a storage-only
decision label; all evidence authority/certification flags remain false.

Storage review rejected fake-digest fit claims and declaration-only source
readiness. Runtime admission currently always raises the explicit unresolved
upper-bound error. The next scope document proposes, but does not execute, an
analytic encoded-size bound checked with CPU specimens. Source membership grows
from 55 to 59, including the old inspection module solely because the new worker
reuses its pure configuration/parser/process-identity helpers—not its runner.

## Limits and next gate

The proposed frozen worktree, native development namespace, supervision namespace
and storage-evidence files have not been created. No native GO is issued. The
consumed layout inspection is preserved and may not be retried. There are no new
scientific findings here, and no durable knowledge-base conclusion changes.

`native-storage-bound-next-scope.md` received scope-only review: it is the next
bounded engineering task, with actual analytic inequalities, admission decoder,
frozen evidence and runtime checks still to be implemented/reviewed. A storage
pass would still require separate scoped native permission.
