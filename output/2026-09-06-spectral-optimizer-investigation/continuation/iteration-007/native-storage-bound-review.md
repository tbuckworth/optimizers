# Serializer component review

Codex / Spectral Optimizer Investigation, 7 September 2026.
Parent record of independent read-only leaf review (`i7_current_storage_count`)
and parent integration checks. This is engineering review, not the future
independent scientific audit of experimental outcomes.

## Verdict and scope

The four component helpers pass static review within their stated domains:

- Pickle: the pinned C-Pickler primitive opcodes, corrected list/dict batches,
  memo-width boundaries, aliases/cycles, v2 fixed subtotal152, uint32-v3
  subtotal193, key-digit growth, traversal admissions and raw-storage handoff.
  The final hashes are recorded in `native-storage-bound-checks.json`.
- ZIP: exact record membership, prefix, local-only alignment fields, conservative
  descriptors, central names, forced ZIP64 tail, decimal-index crossings and
  the below64-MiB domain. Runtime configuration checks now permit an explicitly
  already-initialized native caller while requiring initialization state to
  remain unchanged. Tests use CPU or mocks only.
- Tensor descriptors: exact current paths/dtypes at maximum width, both
  fixture-absent CUDA tensors per retained core, and corrected multiplicities.
  Parent added a direct fixture-declaration path/dtype regression without
  executing the fixture. Raw byte arithmetic does not prove serialized size.
- Pilot comparison: fixed actual schema/trajectory, counters, SHA widths,
  independent boolean maxima, summary count limits, compact ASCII syntax and
  one newline yield308,940 bytes. No partial comparison payload is emitted by
  the actual source constructor on an incomplete stream.

Review caught an additional runtime gap: `copyreg` can override Tensor reduction
through ordinary registration before `__reduce_ex__`. The final pickle gate
rejects such overrides for Tensor/OrderedDict/dtype and tests that case.
Earlier underbounds, test mistakes and corrections are retained in the check
record and component derivations, not silently replaced by a clean narrative.

Parent selected regression passed80/80 in13.999s (40 new and40 related existing
checks), with CUDA uninitialized and peakRSS684,728,320 bytes. The previous full
512-test checkpoint is historical; this does not claim it was rerun. Knowledge
lint passed, and the production spectral optimizer hash is unchanged.

## Still outside this pass

The81 Torch payloads still lack complete prospective primitive/container maxima,
especially adverse audit shapes. The corrected theoretical tensor and JSON
pieces leave an unresolved137,706,019-byte ceiling target for all pickle/ZIP
overhead, after already-bounded non-payload charges. This is not native headroom.

The actual ArtifactStore save call is unchanged. Before any GO, its actual
serializer arguments, mutable configuration, tree-domain checks and buffer
ceilings must be bound. A root-only cap does not independently enforce the
shared cap including287,990 bytes of external normal namespaces. A rejection
limit can protect safety without proving the full successful run will fit.

Both canonical evidence files and their decoder remain absent. The actual
native admission still always rejects. No native/frozen-source attempt was
created and no study specimen, dataset read, plan or experiment was run.
Installed-source/version/header identity is not independent proof of compiled
Torch provenance or the absence of hostile in-process mutation.

Current source membership is63. Every later execution-bearing topology/admission
module or writer wrapper must join that exact set before the clean launch
manifest and evidence digest are frozen. The active research goal and existing
two-hour reminder remain unchanged.
