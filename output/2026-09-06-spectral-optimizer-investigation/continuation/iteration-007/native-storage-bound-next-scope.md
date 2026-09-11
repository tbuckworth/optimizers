# Next bounded work: prove native storage admission

Codex / Spectral Optimizer Investigation, 7 September 2026.

This is a proposed next engineering scope, not a native GO, an executed
measurement, or evidence that the study fits. The production admission function
currently raises `StorageAdmissionError("reviewed native upper bound unavailable")`.
Keep that gate closed until the proof, implementation and evidence are reviewed.

## What remains load-bearing

The complete accounting ledger charges the actual 364-file study membership,
external control and supervision maxima, the consumed inspection, two bounded
storage-evidence prerequisites, and one shared failure reserve. Its conditional
total is 963,790,723 bytes **before** the unresolved native payload delta. The
109,951,101-byte difference to 1 GiB is not demonstrated native headroom.
See `final-storage-ledger.md` and `final_storage_accounting.py` for direct
arithmetic and the deliberately non-authorizing result.

The source-level topology inventory identifies full source/native environment
bindings in 18 anchors, auditor environment bindings in 16 audits, and full RNG
cores in 18 anchors plus six source-final states. The next bound must independently
verify these multiplicities and every other data-dependent primitive field.

## Proposed admission conditions and proof

Candidate runtime conditions are: CPU RNG exactly 5,056 bytes, exactly one CUDA
uint8 RNG state of 16 bytes, canonical source JSON at most 32 KiB, and each native
or auditor environment JSON at most 8 KiB. These are proposed admission gates,
not observed native measurements. A different runtime must fail before any
plan/data/optimizer work or study-root creation, not silently change this layout.

Prefer an analytic protocol-2 primitive/pickle and pinned Torch ZIP upper bound,
checked against bounded CPU specimens. It must account for memoization/alias
topology, unique strings, maximum admitted integer widths (including Python RNG
integers), tensor entries, ZIP headers/alignment, receipts, all 82 payloads and
the complete external/reserve equation. A serialized sample alone is not proof
that another native value cannot be larger.

A candidate small measurement would serialize existing-baseline and maximum
admitted representatives for each of the 11 component kinds, sequentially in
memory. Estimated encoder traffic is about 164 MiB; largest existing body is
about 33.2 MiB. Proposed limits: hidden CUDA, single numerical threads, no IDX,
no plan generation, no native initialization, 120 seconds and 2 GiB process RSS,
at most 64 MiB per serialization buffer and only one retained canonical result
of at most 64 KiB. No such measurement was run at this checkpoint. Verify the
scope/estimates and obtain independent design review before executing it.

The canonical result must bind exact frozen commit/source digest and Torch
serializer revision, admission schema, per-component analytic ceilings and
counts, aggregate 82-body ceiling, delta and complete final accounting. Its
decoder must recompute those totals. The storage-admission wrapper pins that
result but conveys no execution authority. Worker validation must authenticate
both actual files and recheck actual source/environment layout before returning.

If the analytic inequalities cannot be established reviewably, propose the
larger full-membership specimen separately; do not replace proof with a presumed
maximum or rerun the already consumed native inspection. Neither this document
nor a future storage pass authorizes native development or later phases.

## Read-only serializer orientation during checkpoint verification

Installed `torch/version.py` identifies Torch 2.11.0+cu128, source revision
`70d99e998b4955e0049d13a98d77ae1b14db1f45`. Installed `serialization.py` uses
protocol2 by default and reads mutable serialization configuration; the installed
`utils/serialization/config.py` declares alignment64, CRC enabled and pinned D2H
copy disabled as defaults. The future gate must verify the relevant actual
configuration and absence of skip-data context, not assume defaults are immutable.
No runtime setting was changed during this read-only check.

The matching upstream writer constructs the buffer archive prefix as `archive`,
initializes ZIP64, accounts for an extra padding-field header, and finalizes
version/serialization-ID records and the central directory. Thus the proof must
include those entries and distinguish padding from total extra-field bytes;
alignment alone is not a complete ZIP overhead bound. This is source orientation,
not the completed inequality or a measurement.
[Pinned upstream writer](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/caffe2/serialize/inline_container.cc).
