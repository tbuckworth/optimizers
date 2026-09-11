# Saved-input independent numerical audit checkpoint

Codex — Spectral Optimizer Investigation — 6 September 2026.

The synthetic pipeline now seals an independent numerical-audit record from
the exact saved plan, anchor, live witness and six-branch file. This makes both
successful and adverse arithmetic results inspectable without relying on the
producer's success flags. It is dataset-free CPU engineering evidence, not a
new optimizer finding, native pilot or completed scientific execution path.

## What is implemented

[audit_envelope.py](audit_envelope.py) preserves the existing
`i7_anchor_numerical_audit` schema and eight ordered payload fields. The file
adapter verifies indexed pinned files/receipts using restricted CPU loading,
verifies the saved plan separately, compares its exact decoded values/binding,
and recomputes IDX/plan/probe/data bindings. Actual source bytes are rechecked;
the auditor has its own observed CPU runtime, not a fabricated copy of the
native producer environment. The source manifest adds both new audit modules
without removing mandatory future phase/runner files.

The numerical checks cover the two represented basis operators, all six
delivery/domain rows, nc/nl and every leverage field, four parameter-level
AdamW updates per defined branch (including native endpoints and both moments),
and the existing independent NumPy loss/gradient/displacement/contrast audit.
No source trajectory or branch optimizer is rerun during audit. Existing
independent numerical modules, production optimizer and TL/Tq/delivery gates
are unchanged. A separately declared CPU64 metadata roundoff screen checks
saved norm/leverage scalar consistency; it does not replace those gates.

The normal fixture retains six branches, 24 Adam parameter checks/72 tensor
screens and 390 independent contrast checks. The engineered zero-lagged fixture
retains five branches, 20 Adam parameter checks/60 tensor screens and 260
unaffected independent contrast checks. Its restored branch is explicitly
domain_undefined, never a numerical failure disguised as a null. This constructed
edge state is not claimed as a continuation of the warm-up trajectory.

[audit_diagnostics.py](audit_diagnostics.py), implemented by a separate worker,
encodes every failing flat coordinate as a fixed-width lossless bitset with
exact dimension/count/padding/hex validation. The all-failure 50,890-coordinate
case round-trips all indices; there is no first-N truncation. Primitive node,
depth, string-byte and JSON-byte budgets reject oversized records without
raising the attempt cap or removing their sealed raw inputs. These limits are
not a measured all-phase storage certificate.

Numerical work stops at the first fatal stage. Completed preceding diagnostics
are retained; later stages remain not_run. A resource guard preserves that
completed prefix, marks the interrupted stage incomplete, and attempts bounded
publication before terminal status. If the store refuses publication on resource
grounds, its bounded failure record and previously sealed inputs remain.
The current numerical module does not expose internal partial state after an
exception; that interrupted stage is not falsely marked complete or rerun.

The adapter seals, restricted-reloads and directly compares the audit record,
then rechecks source/runtime/RNG. Failures after sealing retain the file/receipt
and restore the caller RNG. Terminal stores reject retry before numerical work.

## Independent review and corrections

One independent reviewer challenged the contract and raw structure/completion
path. Parent adopted the original frozen schema, separate pinned plan reference,
explicit undefined Adam rows, norm/leverage checks, exact counts and internal
measurement resource hooks. The reviewer initially misread the CPU environment
role and corrected that finding after inspecting the current collector: existing
scientific `cpu_audit` support was sufficient, so no parallel environment schema
was invented.

The first six-test integration run had four errors and one guard-test failure.
Hypothesis-first inspection isolated a witness-schema typo in the new adapter:
`i7_source_witness` instead of the actual `i7_source_step_witness`. Correcting the
name made all six initial cases pass; no tolerance or fixture was weakened.

Fresh read-only code review found that guard exceptions originally skipped
audit publication and that completed numerical failures continued into later
stages. Both were corrected and have regression assertions. Follow-up review
identified transient-guard revalidation as non-reproducible. Validation now
recomputes completed preceding checks and validates the declared incomplete
suffix without requiring the original resource event to recur. The abort event
itself remains an external declaration and the artifact remains fatal/incomplete.

Parent review separately fixed partial measurement completion: a returned
partial numerical report retains its detail, but completed follows its actual
audit_complete flag. It also prevented a tiny norm/ratio from borrowing a much
larger unrelated norm's absolute comparison ceiling. A new 1e-20-scale regression
rejects doubled tiny values while accepting their original native values.
Norm/leverage corruption tests consistently rehash producer attestations, then
require saved-structure success and an independent delivery-stage failure; they
do not merely detect stale hashes. A separate mutation of an actual saved Adam
moment retains the exact failing coordinate, and stops before loss auditing.

The diagnostic key-type regression rejects string-subclass keys that compare
equal to canonical names. All original fixed response/projection/Adam/loss
numerical tests remain part of the combined verification.

A subsequent 256-test run exposed an older filesystem-test assumption: moving
the same leaf inode advanced ctime by approximately one millisecond, while
device/inode/mode/link count/size/mtime and bytes were unchanged. The old test
incorrectly demanded identical ctime. It now checks the unchanged fields and
independently requires different directory identities even when ctime is ignored
for that assertion. Production `_signature` still includes and checks ctime;
no source-safety check was relaxed. This is consistent with the distinction
between content modification and inode status timestamps in the
[Linux inode documentation](https://man7.org/linux/man-pages/man7/inode.7.html).

## Verification and remaining work

The focused final-feature audit suite passes ten tests (38.959 seconds, before
the final explicit formula-ID labels); the separate diagnostic suite passes ten.
Final whole-suite verification passes 256 I7 tests (126.670 seconds), 17 repository
tests (1.341 seconds) and six reminder tests (4.239 seconds): 279 total. The source-
binding regression separately passes 13 tests (8.100 seconds). Knowledge lint
passes for 14 pages and 11 indexed pages; whitespace checks pass. Exact commands,
source hashes and scope are recorded in [audit-envelope-checks.json](audit-envelope-checks.json). Earlier
checkpoint hash files remain historical and unchanged.

The NumPy numerical references are independent of producer arithmetic; the
adapter deliberately reuses trusted producer schema/loader imports. This is not
module/process isolation. A public same-code raw-input recomputation verifies
record integrity, not a second independent numerical implementation. Pure value
APIs do not authenticate receipt declarations; only the file adapter establishes
their pinned-file association. Saved proof consistency is not independent
evidence of historical capture, source replay or observer transitions.

The research workflow kept the existing experiment and scope intact. The
best-practices skill informed strict hex and restricted-loading decisions using
the primary documentation linked in the contract. The debugging skill isolated
the schema-name failure before correction. No durable scientific conclusion
changed, so the knowledge synthesis and previously delivered report are unchanged.
