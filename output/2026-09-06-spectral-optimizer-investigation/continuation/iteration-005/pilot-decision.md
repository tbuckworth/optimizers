# Parent development-pilot decision — GO

2026-09-06, after the independent replay/reference/storage and summary reviews
passed and before any iteration005 MNIST or worst-size execution.

Approve **only the frozen protocol's local development/resource pilot** after
committing the reviewed source bytes. No confirmatory replay is approved here.

Scope: development seed9878, two 220-step AdamW traces with measurement off/on,
plus the seed9879 synthetic 50,890-by-2,000 worst-size reference and raw/innovation
write/hash check. Require exact paired core/parameter equality, all numerical
and resource gates, and no new accuracy/validation/test selection or retained
learning/retention outcomes. Ten-minute hard cap; no automatic retry or changed
tolerance after a failure. Preserve the attempt and its partial artifacts.

All bulk files belong to a uniquely created, mount-verified directory under
/tmp/spectral-experiment-artifacts. Existing evidence is never deleted. Enforce the fresh GPU
process/available-memory gate, 8 GiB GPU allocation peak, 12 GiB RSS peak,
10 GiB large-volume headroom and the protocol's persistent-size limits. No paid
compute, network training API or production-source changes.

Evidence: 22 harness and15 summary tests pass, along with247 independent dense
assertions and65 earlier reference-identity checks. The separate real-producer
CPU integration passes for explicitly artificial p48/n64 snapshots; it is not
MNIST evidence. Both independent reviews pass. Prospective review fixes are
documented in those audits and pre-pilot-verification.md.

Reviewed core SHA256:

- replay_harness.py: 35d99dd08fd7b9831a43820c8a5e1df40ba5ff759cc78d004daf4de08ddfb862
- reference_math.py: 3256c40cfe110edb035ecf38a0f1625534a941b5266da7b8c2f99a95595e50ac
- artifact_store.py: f26f0a478c6cb7feee5d5e39e52e1cac6d2174466d0b2692395fad7539c6c053
- test_harness.py: d8c6f46d45250300d47d33e6d6f795ad9730c340af07d5f9d7dade57d95c6217
- summarize_results.py: ba3f6c3947fe19e435b839d89cea673b76f7f61eb7fb49411f8122718d3a9c77
- test_summary.py: 19e7c0714d906c6da94868d1bec1375f8d539ab10e1d46a0620aedabef187397
- protocol.md: eb8f49f4a5247e9c33a23a0d15a572f2c5cec08355f2b4d05ada50f435ce0035

The pilot manifest binds all19 required source files and historical/data hashes.
Confirmation needs a separate parent review of measured runtime/memory,
unchanged committed/pilot source bytes, fresh occupancy and an explicit GO.
