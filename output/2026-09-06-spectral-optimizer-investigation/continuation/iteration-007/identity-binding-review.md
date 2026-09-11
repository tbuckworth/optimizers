# Identity and plan-binding checkpoint

Codex — Spectral Optimizer Investigation, 6 September 2026. Base commit
`135d11ca01eabe3777e3ee66a52ad58c326bfd20`. Dataset-free CPU engineering only.

## Outcome and confidence

Canonical identity and frozen-plan consistency checks are now executable, with
strict field order/types, profile separation, byte-hash definitions and exact
next-batch values. Tiny fixtures pass an actual immutable-store write, restricted
CPU reload, binding write and second reload. This is progress toward reliable
replay, not an optimizer result or complete scientific provenance validation.

The plan validator checks every permutation row, exact partition order, local
training batch/probe coordinates, global auxiliary coordinates, replacement
values and initialization seed. Duplicates in batches/probes are preserved.
It rejects stale references even when changes occur outside the next batch.
The identity codec distinguishes primitive types, list/tuple and mapping order,
preserves signed-zero tensor bytes, and rejects malformed or duplicate-key JSON.

Exact prospective layouts and documentation checks are in
identity-binding-contract.md (artifact not distributed in this public snapshot). Numerical sources,
tolerances, scientific membership, optimizer and prior studies are unchanged.

## Independent review and corrections

Independent review checked both the identity codec and parent plan-binding
consumer. Before commit it found and resolved:

- A genuine encoding collision: a non-BMP Unicode scalar and a manually
  constructed surrogate pair are unequal Python strings but share the same
  ensure_ascii JSON encoding. Strings and keys now reject surrogate code points,
  while genuine Unicode scalars remain valid. Tests also check encode/decode
  round-trip, valid escaped pairs and rejection of lone surrogates.
- Detached lazy-negative tensors could pass compact-storage checks but fail
  NumPy conversion. Both codecs now reject lazy representation flags before
  hashing/acceptance, without silently resolving their values.
  A final nested-tensor check also normalizes unsupported backend errors into
  bounded codec/binding failures instead of emitting large dispatcher messages.
- Parent review rejected string-subclass identity keys; strict key types are
  now tested. Deep recursion/huge-integer conversion failures become CodecError
  without changing Python's runtime safety limit.
- Plan binding validation now compares exact typed structures directly, rather
  than comparing whole-tree hashes. Review caught an initially vacuous mocked-
  digest regression; the corrected test first accepts the valid binding under
  the same mock, then rejects the changed row. Hash references remain hashes of
  their declared raw values; they are not substituted for schema validation.

## Deliberately unclosed boundaries

The pure builder accepts caller-supplied plan values and a file receipt. It
cannot prove that those values were decoded from that receipt, or detect a caller
mutating the decoded plan before constructing a new binding. The future file
adapter must associate verified store name, completion status, encoding, receipt
and unmodified decoded values in one operation. No seeded plan regeneration,
dataset correctness or source-history claim follows from these tests.

Plan hash references do not replace the direct training-probe/auxiliary value
copies required by `bindings.probes`. Full probe/data/source/environment fields,
anchor/witness/branch proof envelopes, independent audit membership and bounded
adverse diagnostics remain to be implemented. Single-root phase accounting and
complete full-width artifact sizing still precede any native pilot review.
The earlier 954.324516-MiB storage-only projection remains incomplete; it was not
reinterpreted as a full-attempt certificate.

## Verification and continuation

The final check counts, exact commands, timing and source hashes are in
identity-binding-checks.json (artifact not distributed in this public snapshot). Tests run with
CUDA hidden, one CPU thread and no bytecode writes. They include existing
numerical fixtures, the repository tests and reminder tests. Knowledge lint and
git diff checks pass; no durable scientific conclusion changed.

The research workflow supplied independent design/code challenge; the
best-practices check informed duplicate-safe JSON parsing and explicit tensor
ownership/serialization boundaries. This use changed engineering validation,
not research conclusions or launch authority.