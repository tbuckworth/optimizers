# Sealed actual-source capture checkpoint

Codex — Spectral Optimizer Investigation — 6 September 2026.

The synthetic producer now seals its complete anchor before the actual source
step, retains that step's real loss/gradients/endpoint, and seals the witness
before any replay. A downstream fixture reloads those files and reproduces the
current branch exactly before independently checking six-branch measurements.
This is stronger engineering evidence for the planned mechanism experiment,
not a new scientific finding about the optimizer. The goal remains active.

## What is implemented

[source_environment.py](source_environment.py) defines exact primitive-only
source/runtime records and separate pure validators, actual collectors and saved
core/runtime cross-checks. The source collector compares required working bytes
to Git blobs, not just clean status, with bounded reads, no-follow paths, root
and intermediate-directory identity, exact leaf signatures and repeated checks.
Required scientific membership includes the future branch/phase/audit entrypoints;
it cannot silently fall back to the tiny two-file fixture repository.

The runtime collector observes only explicit numerical settings and a six-name
environment whitelist, versions, OS/device details and RNG layouts. It does not
change settings or RNG. CPU fixture/audit paths inspect only CUDA initialization
state, not availability, count, properties or CUDA RNG. The installed cuDNN
extension supplies its version without the public helper's availability probe.
This is a frozen-version implementation detail, not a portable API guarantee.

[anchor_envelope.py](anchor_envelope.py) implements the full strict anchor root
and all five binding groups. It recomputes plan/data/probe bindings, checks
expected source/runtime snapshots and their common repository root, cross-binds
native devices/RNG layout, counts actual payload tensor bytes and rejects
unsupported/aliased tensors before cloning. Pure envelope checks establish
consistency with expected inputs, not that those inputs were independently
acquired or executed.

[source_capture.py](source_capture.py) adds an actual live transaction: verify
current source/runtime against expectations; capture/build/validate/seal anchor;
verify sealed bytes without decode; recheck source/runtime and directly recapture
the same live objects; perform exactly one actual forward/backward/filter/AdamW
step; retain native loss, independent raw/current gradient records and complete
t endpoint; recheck RNG/runtime; validate/seal witness. There is no new-object
restore before the witness, mismatch repair, generic step callback or retry.
Failure terminals the existing store and preserves completed files/failure
metadata. The witness's event flags support review, not proof of history by
themselves. Its pure validator does not reproduce the live loss or certify event
order from hashes alone.

## Direct integration evidence

[test_sealed_source_pipeline.py](test_sealed_source_pipeline.py) uses actual
temporary clean-Git collection, actual CPU runtime observation, a sealed/verified
fixture plan, synthetic IDX bytes and four planned warm-up updates. It then
executes the new transaction and reloads the anchor/witness by exact file hashes.
Before constructing any replay clone it directly captures the original live
objects at t=5 and compares their actual parameters, optimizer, observer and RNG
to the loaded witness/anchor. Only after these checks and both receipts exist
does replay begin.

The saved native loss, raw/current gradients and post-ingest observer match replay
directly. All six branches run from independent anchor restorations in canonical
and reverse order; parameters/moments are exactly order-invariant and current
matches the saved live endpoint. The original anchor remains unchanged and each
branch preserves RNG and avoids reobserving. Recovered branch values feed the
measurement producer and independent NumPy audit, requiring 24 actual AdamW
parameter checks, all 390 specified independent contrast checks, and exact
4/24/6/15/15/15/10/60/150 completion counts. Nothing is inferred from test count
alone. This is one deterministic synthetic fixture, not a multi-seed study.

The plan store is separate. Anchor, witness, branch raw, measurements and audit
share another tiny store, but branch/audit files are not yet complete scientific
envelopes and no all-phase controller is exercised. Temporary fixture data is
removed on context exit; no prior research artifact was deleted or changed.

## Review findings and corrections

Independent review found that the earlier fixture sealed neither anchor nor
witness before replay and discarded actual live loss/gradient records. The new
transaction fixes this instead of relabelling replay as source evidence. The
earlier proof proposal's circular pre-capture hash and unavailable same-object
restore were replaced by explicit seal/recapture/direct-equality events.

Integrated testing reproduced a false source-collector failure when unrelated
temporary directories changed ancestor mtime/link counts. Directory identity is
now dev/inode/mode; file signatures remain strict. A regression preserves a leaf
inode while replacing nested parents to check directory identity separately.
FIFO replacements are opened nonblocking then rejected, and double-leading-slash
roots/string subclasses fail the exact input schema.

Independent anchor tests exposed input laundering: the constructor cloned and
detached a requires-grad core before strict tree validation. It now rejects that
input, oversized-storage views and other unsupported representations before any
clone. The live transaction also re-collects sources/runtime so stale supplied
snapshots cannot attest changed current code or settings. Tests deliberately
change a live setting after anchor seal and require failure before forward,
retaining the sealed anchor and terminal record.

Final independent pipeline review caught a missing direct original-live-endpoint
assertion: agreement with a fresh replay alone could not establish that the
original objects reached the witness. The test now directly binds those original
objects before any replay. This corrects the coverage claim, not a demonstrated
manufactured-witness defect in the transaction implementation.
The independent reviewer then confirmed that the original-live linkage finding
is closed. Review remains scoped to these engineering components and fixtures,
not the future scientific experiment's independent results audit.

The research skill maintained the prospective experiment boundary; the
best-practices check used primary Git/PyTorch documentation to separate observed
source/runtime state from reproducibility guarantees; hypothesis-first debugging
isolated the ancestor-metadata failure before changing its signature rule.

## Verification

The final CPU-only suite passes 211 I7 tests (38.103 seconds), 17 repository
tests (1.517 seconds) and six reminder tests (2.913 seconds): 234 total. The
strengthened sealed-source integration also passes alone (one test, 2.632
seconds). The I7 total adds 34 tests: 13 source/environment, 15 anchor envelope,
five live source-capture and one sealed-source integration test. Knowledge lint
passes for 14 pages and 11 indexed content pages; the diff check is clean.
Exact commands, test environment, source hashes and scope are recorded in
source-capture-checks.json (artifact not distributed in this public snapshot). Earlier checkpoint hash
records remain historical and have not been rewritten.

## Remaining scientific gates

Implement the complete branch-results/proof and independent-audit envelopes,
including bounded adverse diagnostics, then the immutable single-root phase
controller/manifest and actual runner entrypoints. Measure complete full-width
artifacts, all-phase storage and runtime costs before a separate native-pilot
review. The earlier incomplete 954.324516-MiB projection is unchanged and is not
a complete-fit certificate. Source verification is local/cooperative and does
not attest imported interpreter objects or prevent hostile same-user rewriting.