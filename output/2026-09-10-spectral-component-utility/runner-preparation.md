# Next work: fixed runner and saved-array audit

Codex — Spectral Optimizer Investigation · 10 September 2026

The design and four import-inert helpers are prepared. Main and independent
fabricated CPU suites each pass 48 tests. No real parent load, neural diagnostic,
archive, attempt or service exists yet. This document selects implementation
work under existing autonomous authority, not an immediate experiment launch.
All earlier scientific acquisitions/audits remain complete and must not restart.

## Fixed integration requirements

1. New runner imports the new strict restore, panels, actions and objective
   helpers. Preserve all old frozen dependencies. No tuning or extra arm.
   Default invocation is inert; actual execution requires explicit committed
   source/input manifest and exclusive attempt/unit/directory identity.
2. Before deserialization, verify producer/audit/source/protocol and original
   data receipts plus every exact parent/plan/readout binding in
   `parent-inventory.json`. Original readout has step/train/validation/reporting
   arrays; restore equality uses first500 reporting rows with the matching
   original reporting IDs and producer500-example batching/FP32 environment.
   Use explicit `cuda:0` for admitted GPU restoration, never implicit `cuda`.
3. Process one parent and one action draw at a time. Fix all panel IDs/labels/
   shifts first. All action inputs are translated, including none-trained
   parents. Apply raw-pixel translation before original standardization and
   enumerate all25 evaluation shifts in image/view/class layout. No alternate
   role, class balancing, test-set input, download or model training continuation.
4. Preserve the parent, including modes/gradients/Adam/observer. Compute each
   sampled action's assigned-CE gradient once, then feed it to both private
   branches. Record raw observer clock unchanged and native advanced once;
   do not relabel raw endpoints as canonical synchronized native snapshots.
5. Store baseline FP32 logits/parameters, all six objective gradients and
   required detached values. Drop **all** original objective/logit graph
   references before actions: the last H_T gradient does not necessarily free
   a separate earlier I graph. Do not accumulate all parents' bases or graphs.
   Compute actual full/0.1 materialized-path effects and same-fraction decay
   accounting. Enforce protocol identities/tolerances without outcome adjustment.
6. Write all fixed original/per-view metrics, wrong-subset reductions, action
   vectors/moments/counters/post-bases and registered seed-level summaries.
   Average two draws within parent; do not treat them as independent seeds.
   Preserve failures and raw tiny values. No fallback, automatic rerun, roster
   shrink, post-result step rescaling or selective warmup/endpoint reporting.
7. Independent saved-array audit uses separate scalar/gradient-dot/path/Adam/
   projection calculations, exact hashes/roster/panel/label bindings and summary
   reconstruction. It streams one basis/artifact at a time under2GiB. It does
   not replay the neural model or claim to reconstruct full observer history.

## Prospective payload upper bound, not resource admission

Current helpers save nine P-length FP32 vectors per action record, two six-entry
int64 clock arrays, and one native post-basis/S pair per draw. With P=235146,
48 raw/native action records,24 native bases and12 baseline parents, the
following conservative bound allows even redundant path records. Each of144
endpoint/path records may store two FP32 points plus three FP64 displacements;
actual writer deduplication is optional and must not be assumed for admission.
156 baseline/endpoint logit sets contain I[256,25,10], R[128,25,10] and R[128,10].

| Component | Upper bytes |
|---|---:|
| 48 action tensor records | 406,336,896 |
| 24 post native bases | 4,514,803,200 |
| 24 post native singular values | 38,400 |
| 12 baseline six gradient sets | 67,722,048 |
| 144 path records upper | 1,083,552,768 |
| 156 original fp32 logit sets | 60,702,720 |
| metadata container panel reserve | 268,435,456 |
| failure reserve | 1,048,576 |
| **Total prospective upper** | **6,402,640,064** |
| Hard archive cap | 8,589,934,592 |

This is arithmetic from the proposed/helper payload schema, **not an enforced
runner byte inventory or measured runtime**. The implemented writer must
account for actual tensor storage/container headers, auxiliary diagnostics and
failure reserve and reject output before exceeding the cap. Parent checkpoints
are referenced by receipts, not copied into the new archive. A basis can exceed
128MiB; an inherited reader's smaller artifact cap cannot silently be reused.

Enforce one-parent/live-branch object lifetimes, gradient-graph release and
bounded CPU/GPU allocations, not just a favorable output sum. The current
read-only hardware snapshot (18:53UTC) is /dev/RECONFIGURE_FOR_LOCAL_STORAGE mounted at /private-artifacts/storage,
144.1G free, RTX3090 with23587MiB free; only known clients2101/260MiB and8861/27MiB.
These transient observations do not reserve resources or admit a run. Recheck
at launch and do not displace unknown work.

## Admission after source preparation