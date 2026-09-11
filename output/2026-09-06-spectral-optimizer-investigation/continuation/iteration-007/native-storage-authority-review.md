# Independent authority/pin-chain review

Codex / Spectral Optimizer Investigation, 7 September 2026. Read-only review
by the existing independent leaf; parent applied fixes and recorded this note.
No specimen, measurement, native execution or production evidence was used.

Final verdict: PASS, no remaining material finding in the exact six production
file hashes recorded in checks (artifact not distributed in this public snapshot).
The reviewer verified those hashes independently after the last code change.

Corrected before freeze:

- M creation originally checked only M's exclusive slot. It now rejects any
  existing A entry, including a zero/partial file or symlink, before M work.
- The prospective later-permission helper originally encoded through an
  unbounded dumps before applying the byte cap. It now validates fields and
  uses the bounded traversal/incremental encoder. Its namespace scan also
  rejects more than seven entries without materializing an unbounded set.
- Development request authentication originally omitted exact launcher-pin
  validation and the fixed attempt path. A malformed request could have been
  consumed before worker rejection. Both checks now precede launch; tests
  reject extra fields/wrong paths and A substitution in all four phases.
- The new authority wrapper originally caught ValueError for a ZIP validator
  whose typed exception is RuntimeError-derived. It now catches ZipBoundError
  and consistently returns its advertised authority error. It was fail-closed
  before the fix too; this corrects the typed-error contract.
- Parent added a peak-buffer/maximum-observed-body consistency condition.

One preliminary combined test failed because a synthetic writer fixture used
noncanonical evidence basenames. The existing ledger rejected them correctly;
the fixture was corrected, without relaxing production checks. All intermediate
handles are preserved in checks, with overlaps explicitly not added.

The scope review rejected temporary mutation of imported analytic calculator
globals and an opaque JSON stand-in for metadata. No runner code was written.
The next recipe scope requires explicit shape-faithful construction and a
truthful CPU diagnostic metadata role, followed by separate resource review.

Validated boundaries: structural supervisor decoding remains Torch-free;
all fixed evidence/permission writes remain within charged slots, preserve
partials and verify final identity/bytes; journal/permission/request A equality
holds; manifest includes the authority and its local dependencies. The separate
runtime candidate recomputes the prospective ledger and pinned serializer
conditions, but production remains unconditionally stub-closed.

Load-bearing non-claims: hashes and the decoded record do not attest historical
measurement execution. The externally reviewed exact-permission channel is the
trust anchor under a serial, non-hostile same-user workflow. Cooperative checks
do not replace an external process watchdog. No complete recipe, actual M/A or
native storage GO exists; this checkpoint does not complete the research goal.
