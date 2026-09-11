# Parent decision: exact source and runtime bindings

6 September 2026. Parent read source-environment-proposal.md completely and
approves implementation of its five APIs plus state/environment cross-check,
with synthetic temporary-Git and CPU-only runtime tests. No scientific source
collection, plan, dataset, CUDA initialization or pilot is approved here.

Adopt the proposal's exact ordered source/environment layouts, runtime-role
separation, actual Git-blob/worktree-byte equality, strict source membership,
safe six-variable numerical environment whitelist and read-only RNG-preserving
runtime capture. Add `I7/source_capture.py` with role `producer` and
`I7/anchor-envelope-contract.md` and this decision with role `contract` to the
scientific membership. The proposal itself is now adopted subject to this
decision; its planned filenames remain mandatory until explicitly amended.
Future phase/entrypoint files remain absent and scientific collection must
fail; do not substitute fixture membership or drop the missing paths.

Subsequent branch-envelope checkpoint: add `I7/branch_execution.py` with role
`producer` and `I7/branch-envelope-contract.md` with role `contract`. The existing
`I7/artifact_envelopes.py` member now implements branch envelopes; mandatory
future phase/audit files are unchanged. No scientific sources have been collected
and this prospective membership amendment is not launch approval.

Subsequent numerical-audit checkpoint: add `I7/audit_diagnostics.py` and
`I7/audit_envelope.py` with role `independent_auditor`. The previously mandatory
`I7/audit-envelope-contract.md` now exists. Future phase/runner/complete-envelope
paths remain mandatory; no scientific collection or launch is implied.

Subsequent single-root fixture checkpoint: the previously reserved phase controller,
scientific/audit runner filenames and envelope/phase appendices now exist.
Membership is not weakened or replaced with fixture paths. The entrypoints are
inert by default and explicitly reject scientific execution; the live phase path
is the registered MLP fixture only. Presence of source files is not implementation
completeness or a native GO. Native multi-source/pilot-history/resource closure
remains required, as enumerated in envelope-contract.md.

Five-second bounded Git reads remain appropriate. Metadata command output is
capped at 1 MiB; an individual `git cat-file blob` read may use the explicitly
declared 2-MiB source-file cap. Total required source bytes remain capped at
16 MiB. Read actual blob bytes without text conversions/external diff helpers;
clean Git status alone is insufficient, including assume-unchanged cases.
Check root and intermediate-directory identity again on completion, not just
the final file descriptor; shared-directory rename races must fail.

Both scientific roles preserve the established versions and explicit numerical
configuration; other recorded values require an independently reviewed expected
snapshot before use. CPU fixtures observe their real settings rather than fake
the scientific settings. CPU collection must not call CUDA availability/count,
properties or RNG functions; initialized-state inspection is allowed. No broad
environment, credential, subprocess stderr or executable content is persisted.

The anchor builder additionally requires source/environment repository roots to
match and cross-binds core native devices and exact RNG layout to the retained
runtime snapshot. That is saved-value consistency, not imported-module attestation
or proof of foreign-platform replay. Live runtime comparison and phase-level
manifest binding remain required at actual execution.

The parent checked current primary Git and PyTorch documentation as linked in
anchor-envelope-contract.md. Stable status parsing and release/platform-limited
reproducibility support these checks; the exact schema, caps and membership are
our implementation decisions rather than guarantees from those libraries.

## CPU-only UUID compatibility remediation,7 September2026

The failed engineering inspection exposed a mismatch between the pinned native
Torch UUID type and the collector's accepted types. Adopt the shared pure
cuda_identity.py converter described in native-inspection-remediation.md and
add it as binding_codec immediately after state_core.py in mandatory scientific
membership. Both native state-core and environment identities use its strict
lowercase GPU-prefixed representation, matching the NVIDIA driver-query format.
Keep validators' role/device/RNG equality requirements and all numeric settings
unchanged. No native I7 record was successfully retained before this correction;
the failed attempt's original bytes remain immutable and its root consumed.
The runtime guard's native identity also uses the same converter and rejects
missing/malformed identity. Pure saved-record validators require already
canonical UUID strings; the NVIDIA driver-query boundary normalizes observed
rows before the unique-match check. This approval is implementation/CPU tests
only, not new native execution.

Independent protocol review additionally requires the controller to validate
nested success documents before publication. Move the existing pure schema and
membership definitions to source_environment_schema.py, preserving the existing
source_environment API by re-export. The controller imports only the pure
module, not Torch/NumPy. Add it immediately after cuda_identity.py as a
source_binding member (45 total). This is a shared-validator extraction, not a
new policy, collector action, or authorization for a native retry.
