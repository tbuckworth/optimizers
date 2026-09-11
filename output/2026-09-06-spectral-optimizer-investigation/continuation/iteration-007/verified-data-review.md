# Verified plan and data/probe integration checkpoint

Codex — Spectral Optimizer Investigation — 6 September 2026.

The next experiment now has a tested path from a pinned plan file and supplied
IDX bytes to actual tiny-model optimizer steps and independently checked
measurements. This closes a concrete provenance gap in the engineering pipeline.
It supplies no new scientific evidence about optimizer effectiveness and does
not approve a pilot. The broader research goal remains active.

## Implemented and checked

- [verified_plan_load.py](verified_plan_load.py) verifies the pinned actual file
  bytes, receipt and inventory under one held directory descriptor and shared
  nonblocking lock, then performs bounded ZIP preflight and restricted CPU
  decoding. Strict plan validation and binding creation occur in that operation.
  Path, root, lock and inventory are rechecked before returning. Public store
  inspection behavior is retained through the extracted `_inspect_dirfd` helper.
- [data_probe_bindings.py](data_probe_bindings.py) verifies supplied immutable
  byte hashes/sizes before exact IDX parsing, reproduces float32 normalization
  and the frozen corruption rule, and binds exact local/global row order,
  repeated indices, all four probes and all ten auxiliary chunks. Validation
  recomputes values with exact typed/tensor-byte equality and whole-output/plan
  ownership checks; hash equality alone cannot substitute for those checks.
- [test_bound_data_pipeline.py](test_bound_data_pipeline.py) carries synthetic
  30-row IDX data and a verified plan through four planned warm-up steps, the
  actual live fifth source step before any restore, six branch endpoints,
  serialized/reloaded measurements and the independent NumPy auditor. A separate
  warm reconstruction matches model, optimizer, observer and RNG directly.
  Exact membership and passing statuses are required for 390 independent
  contrast checks, plus 24 actual per-parameter AdamW endpoint/moment checks.
  Required audit completion counts are 4/24/6/15/15/15/10/60/150. CUDA stays hidden
  and uninitialized; derived data and plan content remain unchanged.

The integration uses a quiescent plan store and a separate result store, both
temporary tiny-profile stores. It is not the scientific single-root phase path.
Synthetic tensor artifacts are temporary and removed by fixture cleanup; none
of the prior scientific evidence is modified or removed.

## Corrections and review

The initial MLP plan fixture's 2/3/4 batch/probe/auxiliary counts did not match
the established measurement fixture's 4/4/10. The explicit MLP-only plan schema
v2 now uses 30 rows in three partitions of 10, batch 4 and probe 4; old MLP v1
is rejected. Scientific v1 and the original linear fixture v1 are unchanged.
See data-probe-contract.md (artifact not distributed in this public snapshot) for the prospective decision.

Positive-path loader tests exposed an incorrect empty-safe-globals assumption.
Torch 2.11 registers three built-ins on import. A combined-suite failure then
exposed nine additional built-ins; a fresh-process Linear/AdamW reproduction and
installed registration source isolated normal lazy imports as the cause. The
loader now recognizes exactly those twelve built-in objects, without modifying
registrations or permitting arbitrary additions. The AdamW regression and custom
registration rejection both pass. Race tests assert their callback actually ran.
See [verified-load-decision.md](verified-load-decision.md) for threat-model limits.

Independent workers supplied the load design/adversarial tests and data codec;
a separate reviewer checked the materializer and integration and requested
stronger exact RNG and numerical-membership assertions, now implemented.
A final read-only review of the allowlist correction and strengthened tests
found no material defects; this is a code-review result, not a scientific audit.
The research workflow kept this stage dataset-free; the best-practices check
informed descriptor/restricted-load limits; hypothesis-first debugging isolated
the Torch initialization interaction before changing the allowlist.

Final checks: 177 I7 tests pass in 10.908 s; 17 repository tests in 1.390 s and
six reminder tests in 1.989 s: **200 total**. Knowledge lint passes (14 pages,
11 indexed content pages), as does diff whitespace validation. Source bindings
and exact commands are in verified-data-checks.json (artifact not distributed in this public snapshot).
Earlier checkpoint hash records remain historical, not rewritten to new hashes.

## Still required before a native-pilot review

The loader proves association at return, not perpetual immutability, seeded plan
regeneration, producer authenticity or a hostile-pickle memory sandbox. Full
inventory inspection and deterministic data recomputation cost I/O and RAM;
their phase resource feasibility remains unmeasured.

Freeze and implement the remaining source/environment, capture/proof, audit and
manifest layouts; the proof proposal (artifact not distributed in this public snapshot) remains a
proposal only. Integrate a single-root phase controller and complete retained
failure accounting, then measure all complete full-width files and adverse
diagnostic cases through the actual writer. The previous 954.324516-MiB
projection remains incomplete; no complete-fit certificate follows here.