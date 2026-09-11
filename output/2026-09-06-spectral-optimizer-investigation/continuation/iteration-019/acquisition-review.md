# I19 acquisition source and resource review

Status: PASS for source freeze and first bounded acquisition, 8 September 2026.
No actual scientific stream, root, attempt or service existed at this review.
The subsequent launch record must bind the frozen commit and actual unit limits.

## Source roles and verification

Main authored the protocol/runner and reviewed the complete final core/tests
from current_report (0363d0,fea1e0,2d9d96,8623d5). Independent i17_analysis
reviewed the protocol/runner/tests/API note; its requested raw/cross-CP roster
clarification is now explicit: each CP versus every other policy. Main reviewed
the added Kalman controls, source/preflight fixtures and prospective storage
correction before source freeze. These changes used no scientific outcomes.

Main final combined test bef5e1: **17/17 PASS,0.952s**. Fixtures cover exact
canonical construction/unused first process entry, rotation, no RNG/optimizer
step, mean/native/full-moment conventions, policy initialization, scalar
endpoint identities, DEMA, both Kalman recurrences and zero-process prefix
averages, callback/schema, source-byte mismatch, exclusive output and physical
mount/device/space/empty-root checks. Core imports canonical native source by
path. Numerical API semantics and version distinctions are recorded separately.

Seven-source SHA256 closure9b489c:

| Source | SHA256 |
|---|---|
| spectral_filter.py | 9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943 |
| stochastic_tracking_core.py | 7e89cf23d3903992e0937ea5bb89019572f27924cb74716b4aaffc7287786799 |
| test_stochastic_tracking_core.py | c5f449ccaf241df42d4068ce9152fd5de43fd9a1af52e2b1a4b095505e443475 |
| run_stochastic_tracking.py | 6eec2cda3513c2065c3f2b3a72e9e6405504214a62150d61041a3cf720dc0cec |
| test_stochastic_tracking_runner.py | 1d34595acfe84e0c3922b9fd8d0070f46c73b8fb5fef5da27bc505abfe92e742 |
| protocol.md | 959fec105b29f6c7437492106cad85f11af4eb3d8965614b233aee66ab5ba28b |
| best-practices-check.md | ab5e91f7255f99fb04e311648a67e1c75a5ba95737fccfbf3c771eeccbe931c5 |

## Resource admission

Main c7b82a deterministic two-observation fixture measured1,442/1,458 bytes per
observation (zero/nonzero process variance), so1,115,648,000 raw array bytes
across the full grid. The initial proposed1GiB cap was too small and was
corrected before any scientific draw to2GiB arrays/3GiB root, retaining all
seeds/arms. Main deterministic1000-observation fixture e8954b uses
sin(.13t),cos(.17t),sin(.23t) as supplied canonical columns, Q1=.1,rotation0:
1.482518s, projecting1138.574s core for768000 observations. That leaves about
461s (29% of the1600s cooperative cap) for I/O/variation. This is feasibility
timing, not a scientific observation or performance result. First-import cost
is included in the short fixture and not deducted to obtain a favorable bound.

Read-only checks939cc3/8594ad show no I19 bulk root or unit. ab92e2 confirms
/dev/RECONFIGURE_FOR_LOCAL_STORAGE is mounted at/private-artifacts/storage;494a1d has688GiB free;e28f3b has51,164MiB
available RAM. Existing unrelated swap is untouched. Actual proposed unit:
4GiB RAM,CPU1,zero swap,CUDA hidden,three numerical threads1,Nice10,
1600s cooperative/1800s hard+5s stop grace,Restart=no. Paid spend/reservation0.

Create one exclusive mktemp root on that mounted volume only after source
freeze, verify source closure, then launch once. Record PID/invocation and
actual unit properties. Never restart a live or completed stream. Independent
NumPy audit source is being prepared separately and must be main-reviewed and
frozen before the first saved-array analysis; no outcome may be interpreted
before its accepted integrity checks.

## Concurrent draft provenance

Commit d1e3c3807be459c29ce4b0fde5032ee7ce23b221 at12:08:21UTC tracked an earlier
core draft while its importer was committed. It is co-authored Claude Opus5;
main and current_report did not issue the git write. The originating process
or hook is unverified. Preserve that draft checkpoint; it is not acquisition
admission and changed no live/completed experiment source. This review binds
the later, fully reviewed seven-source closure, not that earlier core draft.
