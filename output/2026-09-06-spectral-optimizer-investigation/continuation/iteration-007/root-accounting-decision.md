# Root accounting and tiny metadata fixture

Codex / Spectral Optimizer Investigation, 7 September 2026.
Starting checkpoint: b22cedc8ea0fe78472e2b8381715018fad0bf781.

## Decision

Continue the existing research workflow with CPU-only accounting and unit
fixtures. Do not create another queued run, GitHub issue or scheduler. The
continuation skill's read-only orientation is complete; implementation below is
within the user's existing autonomous investigation, not a new research scope.
The failed native inspection remains consumed and no native retry is approved.

Independent recount identifies 82 scheduled payloads and 90 successful phase
records. If each is an ArtifactStore payload, the defined baseline is 346
regular files: 172 payloads, 172 receipts, one header and one empty lock.
This is conditional on an adapter choosing to retain each phase record; the
native policy does not currently implement that writer or name those files.

The two reviews differ on priority. The inventory reviewer recommends pure
closed membership/accounting before more physical storage work; the protocol
reviewer recommends a tiny actual-writer fixture before any full-width root.
Adopt both in that order. Defer the roughly 945-MB bulk specimen: it repeats
known component storage while native metadata and adapter omissions remain.

## Authorized implementation and tests

Add an engineering-only root_storage_accounting.py and its tests. Change no
existing production/scientific module, profile, membership, threshold or cap.
Import/default CLI must not import Torch or create files. Native execution and
full-width specimen generation are absent from this module.

1. Enumerate the exact 82 payload names and their existing component-size keys
   from the fixed policy. Choose one explicit *fixture-only* phase filename
   convention, storage-phase-NNN.json, for its 90 records. Derive every receipt
   filename with the real writer convention; reject missing, duplicate and
   unclassified files. This is not an adopted native adapter or fit certificate.
2. Given exact positive integer component sizes, produce a conditional logical
   ledger using actual receipt/header encoders, phase-record body ceilings and
   the unchanged shared failure reserve. Keep the unresolved native payload
   delta and unscheduled adapter files explicit. Reject bool-as-int, unknown
   components and internally inconsistent membership. Parent will feed the
   already committed component measurement; do not rerun that measurement.
3. Test plumbing with tiny inert bodies under the 82 names, using Torch encoding
   for .pt and strict JSON for .json. Every body explicitly says storage_only,
   nonsemantic and execution_enabled=false; no tensor/model/data/seeded plan is
   generated. Store header uses the existing MLP fixture profile, never the
   scientific profile. Policy records retain synthetic_contract_fixture and
   execution_enabled=false; their scientific strings denote membership only.
4. Build references from actual receipt bytes; cross-check exact names, sizes,
   hashes and encodings. Persist/reload the 90 synthetic records, validate their
   declared chain, and inspect/reopen with pinned root/header identity. Require
   exactly the 346 baseline files and reject extras. Records' resource totals
   are pre-record-write observations; final actual totals are a separate
   inspection, not self-inclusive values inside the last record.
5. In a separate tiny disposable fixture, provoke a fixed benign duplicate
   write. Preserve and inspect the resulting store failure, demonstrate normal
   post-terminal writes and writable reopen are refused, and confirm a chained
   phase-failure record cannot be appended through the normal API. Do not alter
   that behavior or call it a complete native failure adapter.

Tiny test roots may use scoped TemporaryDirectory contexts below the verified
/tmp/spectral-experiment-artifacts mount and be cleaned by those contexts. No retained bulk
root or standalone measurement launch is approved here. Each fixture has a
2-MiB logical cap, with a 1-MiB internal failure reserve and at least 1 GiB free
filesystem headroom. These are smaller engineering-fixture limits; the study's
1-GiB cap is unchanged. Use hidden CUDA and single-thread numerical environment.
Report regular-file logical sizes and st_blocks allocation separately; directory
metadata is excluded from the latter. Use bounded fixed bodies, cooperative
120-second body checks and a 2-GiB process RSS check; no hard-preemption claim.

## Interpretation and unresolved gates

The root is a post-hoc storage/transcript specimen. Persisting development_go
after creating the fixture store does not reproduce its required online
decision-before-root lifecycle. Resource clocks cover metadata plumbing only,
and reopen/hash timing is cache-warm, not native all-phase timing. A valid
synthetic transcript remains incapable of executing or certifying research.

Native RNG layout, real copied source/environment metadata, final adapter
membership, adverse partial-file arrangements and native all-phase time/RSS
remain unresolved. The controlled terminal-write finding is an adapter design
gap, not an optimizer result. No new durable scientific conclusion follows.

## Official API validation

The current writer's owned CPU primitive trees, uncompressed Torch ZIP format,
explicit restricted reload and create-only fsync path match the relevant APIs.
Restricted unpickling is not a memory/denial-of-service guarantee. BytesIO views
must be released before closing/resizing. Keep existing behavior rather than
adding compression, sparse placeholders or unsafe deserialization fallbacks.

- [PyTorch 2.11 serialization](https://docs.pytorch.org/docs/2.11/notes/serialization.html)
- [Python 3.12 BytesIO](https://docs.python.org/3.12/library/io.html#io.BytesIO)
- [Python 3.12 fsync](https://docs.python.org/3.12/library/os.html#os.fsync)

Parent review, independent bounded review and CPU tests precede integration.
Passing this fixture does not approve another native inspection or pilot.
