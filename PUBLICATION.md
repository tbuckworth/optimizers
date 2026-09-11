# Public scientific snapshot

This release publishes the scientific work from the September 2026 investigation
without importing its private Git history. The core optimizer implementation is
unchanged from the public base. Main-investigation, clustering and J-Lens source,
synthetic fixtures, result summaries, reports and selected figures are included.
The optional familiarity/frequency experiment remains unrun; no new experiment
was performed for publication.

## What remains private

- Correspondence and its indexes/summaries, sender and message identifiers,
  mail payloads, delivery receipts, internal orchestration/handoff records.
- Machine/account paths, host identifiers and session/job routing metadata.
- Private cross-repository audit notes, source-text corpora, raw model-reader
  conversations, bulk checkpoints and compressed raw acquisition bundles.
- The original development commits and untouched source artifacts. They remain
  in local archival worktrees; this public branch contains a clean snapshot.

Scientific results are not selected by outcome. Useful, adverse and contradictory
findings remain together. Omitted links are labeled as not distributed rather
than replaced by fabricated evidence. The public package is not a complete raw
training replication archive; missing private material is a real limitation.

## Redaction and provenance

[PUBLICATION-MANIFEST.json](PUBLICATION-MANIFEST.json) lists the exported files,
their original and public SHA-256 values, and omission categories. Numeric result
values are preserved; private metadata keys/strings are removed or generalized.
Consequently, a redacted JSON file is a **publication derivative**, not a
byte-identical original. Original audit digests refer to the private originals;
they must not be presented as hashes of redacted public bytes.

Machine-specific Python paths are generalized. Where metadata-only source
changes affect a pinned helper, the public source pins are rebound explicitly
to this publication version; the manifest retains original source hashes.
This does not rerun or re-certify the historical experiment. Original source
commits and raw-artifact guards remain historical provenance, not promises that
old launch commands can run from this branch unchanged. Acquisition entry points
remain explicitly gated; hardware/mount expectations need a fresh local review.

## Reproduction and tests

The public tests use synthetic fixtures, not downloaded datasets, training
checkpoints or external model calls. See [release validation](PUBLICATION-VALIDATION.md)
for the commands and results. The optional model-acquisition and archived
training runners are not a deployment recommendation or an automatic queue.

The main report can be rendered from Markdown using the provided public report
builder. All new public HTML/PDF is generated from the redacted sources, rather
than copying private mail previews or PDFs with hidden local links.

## Interpretation

The strongest result is conditional useful learning under corrupted labels,
with an important augmentation boundary. The release does not establish new
architectural priority, general optimizer superiority, semantic clustering or
safety efficacy. The technical manuscript is a working research document, not
a conference submission or an assertion of acceptance.
