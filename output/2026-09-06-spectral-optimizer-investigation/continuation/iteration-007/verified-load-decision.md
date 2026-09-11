# Verified plan loading: parent implementation decision

2026-09-06; dataset-free engineering GO only. The parent read the complete
verified-load-proposal.md and adopts its descriptor-held inspection, quiescent
shared lock, strict JSON metadata, byte-before-decode verification, eight-storage
ZIP preflight and fixed size limits. Two qualifications are mandatory:

- Public API is `load_verified_plan(root, name, *, identity, profile,
  expected_sha256)`. The required external SHA-256 comes from the predeclared
  immutable manifest, not the file being loaded. It is compared to both the
  on-disk receipt and actual bytes. Reading a self-consistent replacement receipt
  alone is insufficient. There is still no caller-supplied decoded plan, decoder,
  receipt, resource override or unsafe fallback.
- The returned plan is a mutable Python/Torch value. The operation proves its
  byte/receipt/binding association **at return**, not perpetual immutability or
  cryptographic authenticity of a producer. Caller code must use those verified
  values without modifying them. A later verified-load operation rechecks disk;
  any reuse adapter must explicitly check retained content identity.

Return keys are exactly `{store,artifact,plan,binding}` as in the proposal, plus
`plan_content_sha256` last, computed with the typed-tree codec. It can detect
accidental mutation on later reuse; it does not authenticate hostile caller code.
No new input/parameter copies are persisted by loading.

Extract `_inspect_dirfd` from ArtifactStore.inspect without changing public
inspection behavior. All adapter reads use the one held descriptor. Inspect
every metadata JSON with the strict codec before the existing inventory validator;
bound each metadata file to its existing cap. Reject failure records before
decode. Check target size before the full store inspection. Existing inspection
reads all indexed payload bytes, which costs I/O/RAM and remains in the later
phase budget. Do not call these size limits a hostile-pickle memory bound.

Use runtime_guard.verify_big_volume for scientific mount identity, then retain
an absolute component-by-component O_NOFOLLOW directory descriptor and compare
the reported device number to fstat. Scientific roots must be strict descendants
of `/tmp/spectral-experiment-artifacts`; reject mount/root paths themselves. Keep the same
device below the big mount and revalidate the path/mount on completion. Fixture
profiles skip the fixed-volume requirement, not any other file-integrity gate.
Tests use only temporary tiny-profile stores; no real plan/data reads.

The supported threat model is the reviewed local writer with accidental
corruption/replacement detection, not an adversarial same-user process that can
rewrite both the frozen manifest and files or modify interpreter globals.
Advisory flock coordinates cooperative writers; it is not an access-control
boundary. Restricted Torch decoding is not a denial-of-service sandbox.

Positive-path testing corrected a parent assumption: Torch 2.11's import already
registers `_MeshLayout`, `NestedTensor`, and `_rebuild_njt`. Combined-suite testing
then exposed nine additional built-in registrations. A fresh-process reproduction
with only Linear/AdamW construction isolated the cause to ordinary lazy imports,
not leaked custom test globals. Those nine are `_DimRange`, `DeviceMesh`, `DTensor`,
`DTensorSpec`, `ShardOrderEntry`, `TensorMeta`, `Partial`, `Replicate`, and `Shard`.
The adapter allows only these twelve exact objects from their already loaded
Torch modules, or a subset, and rejects additional registrations. It neither clears
nor adds safe globals. Installed source confirms registration in
torch/nested/__init__.py, torch/distributed/device_mesh.py,
torch/_dynamo/__init__.py, and torch/distributed/tensor/__init__.py.
These objects are not accepted plan values; exact decoded plan/type checks still
apply. A regression constructs AdamW before verified loading and checks that the
loader leaves registrations unchanged. Late-race tests assert their injection
callback ran, so an earlier rejection cannot masquerade as race coverage.

Best-practices check: Python 3.12's
[os documentation](https://docs.python.org/3.12/library/os.html) supports
descriptor-relative operations and platform-specific no-follow flags.
PyTorch 2.11's
[serialization documentation](https://docs.pytorch.org/docs/2.11/notes/serialization.html)
explicitly limits weights_only security, including denial-of-service exposure.
The Linux descriptor and preflight design is our implementation choice, not a
claim that those APIs authenticate scientific provenance.

This decision leaves source/environment/proof envelopes, phase continuation,
aggregate failure accounting and full artifact sizing open. No native pilot,
scientific execution, new scientific plan or prior-experiment restart is allowed.

## Subsequent single-root fixture integration

`load_verified_plan_from_store(store, name, *, identity, profile, expected_sha256)`
now uses the same private verifier and exact return layout through the held
exclusive writer descriptor. It never opens a second shared lock or converts
the existing lock. The external pinned hash, strict metadata/inventory, eight
storage preflight, CPU restricted decode, plan/schema/binding and final path
checks are unchanged. The original closed-store public API remains supported.

`ArtifactStore.reopen` requires profile, header SHA-256 and root device/inode
pins, verifies the same canonical path and complete nonterminal inventory under
a newly acquired exclusive nonblocking lock, reconstructs actual file metadata,
and preserves the header's aggregate cap/free-space floor/failure reserve.
It does not validate phase authority itself. The fixture phase controller adds
the separate exact sealed-boundary/event/inventory check before any continuation.
See phase-controller-contract.md (artifact not distributed in this public snapshot). All new tests
are dataset-free; neither adapter authorizes a native pilot or live-source retry.
