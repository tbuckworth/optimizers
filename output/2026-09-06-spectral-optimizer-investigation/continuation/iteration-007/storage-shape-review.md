# Full-width storage-only checkpoint

Codex, Spectral Optimizer Investigation, 6 September 2026. Base 353ec26.
Dataset-free CPU engineering evidence; no scientific result or pilot approval.

## Measured result

The three full-width storage specimens serialize to **48,136,892 bytes per
anchor group** (45.906918 MiB). This includes the 48,041,056-byte core tensor
ledger, 42,560 bytes of required exact index arrays, and 53,276 bytes of this
specimen's serialization overhead. The specimen intentionally omits incomplete
scientific metadata; its size is not a full-artifact upper bound.

| Storage-only object | Owned tensors | Tensor bytes | Serialized bytes | Specimen overhead |
|---|---:|---:|---:|---:|
| Anchor with required indices | 22 | 7,370,992 | 7,378,401 | 7,409 |
| Source witness | 21 | 7,735,552 | 7,742,628 | 7,076 |
| Branch results | 132 | 32,977,072 | 33,015,863 | 38,791 |
| Total | 175 | 48,083,616 | 48,136,892 | 53,276 |

Direct measurements, all tensor paths/shapes/dtypes, serialized-byte hashes and
runtime observations are in storage-shape-measurements.json (artifact not distributed in this public snapshot).
Both zero and one tensor fills yield identical sizes with otherwise equal
metadata. Every ZIP entry is uncompressed; storage entry counts and bytes agree
with the owning-tensor inventory. Restricted CPU reload compares every tensor
byte and checks compact independent ownership again. No binaries were retained;
these were bounded in-memory specimens, not files in a scientific attempt root.

The timed measurement bodies took 0.353884/0.364111 wall seconds for zero/one
fills, with lifetime peak RSS 813,465,600/813,297,664 bytes. This excludes Python/
Torch import startup and measures no disk I/O or GPU/native runtime. The
environment was Python 3.12.3, Torch 2.11.0+cu128, one CPU thread, CUDA explicitly
hidden and uninitialized. Torch RNG was preserved.

## Corrections and independent checks

Independent schema review found the old statement that all indices were external
hash-only bindings was incorrect. Anchor-schema section 8 already requires 64
next-batch indices, 256 training-probe indices and 5,000 auxiliary indices.
The prospective owned CPU-int64 representation adds 42,560 bytes per anchor;
the existing 48,041,056-byte **core** tensor count remains correct. Exact
synthetic indices in the specimen are placeholders, not a generated data plan.

An initial zero/one test differed by 64 bytes: the metadata string `zero`
matched a branch key, changing pickle string memoization. Replacing the
specimen's pattern string with an integer fill marker makes the comparison
isolate tensor values. It does not change the scientific schema or a tolerance.
This observed failure is why complete container metadata must be measured,
not inferred from tensor size or assumed constant across layouts.

Independent read-only review agrees with all fixed fields: five P-by-32 bases,
44 P-float32 equivalents, 16 P-float64 equivalents, three float64 S vectors,
32 float32 counters and the three index arrays. It found no material defect
within this limited scope and independently passed the original four tests.
Parent added a fifth check for inert default CLI and explicit CPU boundaries.
No existing scientific/MLP fixture profile was expanded or relaxed.

## Resource implication and next action

Eighteen measured specimen groups plus the inherited 128-MiB shared allowance
sum arithmetically to **1,000,681,784 bytes (954.324516 MiB)**. The residual under
the unchanged 1-GiB cap is **73,060,040 bytes (69.675484 MiB)**. This allowance
has not been shown to cover the omitted fields. It is not spare approved
capacity, a full-attempt forecast or a reason to launch.

The independent envelope closure review (artifact not distributed in this public snapshot) identifies
the remaining exact binding/proof/RNG/manifest layouts, worst-case adverse
audit memberships, and writer/phase accounting needed before complete sizing.
The current ArtifactStore uses bounded RAM serialization then direct create-new
writes plus receipts, with no disk-temporary copy. That does not remove the
required RAM, pending-write, failure-output and all-phase disk accounting.
No switch to another publication method is authorized by this observation.

Next, freeze that ordered envelope appendix and implement its strict validator
and actual phase writer. Extend the specimen to complete artifacts and adverse
failure cases. A native pilot decision still follows separately.

## Verification and boundaries

With CUDA hidden, one OpenBLAS/OpenMP thread and bytecode disabled:

```text
python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007 -p 'test_*.py'
120 tests, OK, 7.406 seconds.
python3 -m unittest discover -s tests
17 tests, OK, 1.460 seconds.
python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation -p 'test_goal_reminder.py'
6 tests, OK, 2.067 seconds.
python3 scripts/lint_knowledge.py
Passed: 14 pages, 11 indexed content pages.
```

Total 143 tests. Reproduction of each sizing body uses
`storage_shape_fixture.py --measure --pattern zero` or `--pattern ones`
with the same CPU-only environment. Without `--measure`, the CLI only shows
help. Source hashes are recorded; this engineering fixture was not claimed to
have a pre-execution scientific source-freeze commit.

The research workflow required independent schema/accounting review and kept
this separate from scientific execution. The best-practices check informed
owned-storage, restricted-load and BytesIO handling; official sources are linked
in [storage-shape-decision.md](storage-shape-decision.md). No numerical scientific
conclusion changed, so the knowledge synthesis is preserved.