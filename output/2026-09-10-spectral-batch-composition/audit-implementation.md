# Independent batch-composition audit: implementation readiness

Codex — Spectral Optimizer Investigation, 10 September 2026.
**Prepared before outcome access. No scientific audit or experiment was run by
this implementation agent.** Main owns source freeze and both once-only launches.

## Ready for main review

The checker is audit_spectral_batch_composition.py (artifact not distributed in this public snapshot).
Its 11 fabricated CPU tests (artifact not distributed in this public snapshot)
pass in **0.341 seconds** under the pinned system Python, NumPy 1.26.4 and
PyTorch 2.11.0+cu128, with all four math-thread limits set to one. The fixtures
create no neural model or optimizer object and perform no inference, gradient
estimation, covariance update or scientific-data loading. Recorded three-
coordinate Adam states are constructed with explicit arithmetic.

Reviewed source identifiers at this handoff:

| File | SHA-256 |
| --- | --- |
| New independent checker | `51f22b18ce0f97d8d6b2501a4aaeac3d6fe93347189e148192668052aa59eb3e` |
| New checker fixtures | `0a747d5666dc416e790e64a98c83a6e68461a84cf32cff4e379bc5e24d0735ca` |
| New producer | `35c4b3f59000e862dd4103bc96bb1b20beb9b21a67b7cc8dce342f0ac5cc8b1a` |
| New producer fixtures | `ab45329b2267102dc6516264bac2a8762a159f37e13e7102b1ec099f6009f2cc` |
| Protocol | `e1b0efc220be2a1f67d643862f1852ba3862bb0cf71c47207cf9ecf7d8e02dbc` |
| Inert prior independent arithmetic helper | `a992842846c767c0af44cf41a19fb3042b4aebc951089d78469e26b0f3465527` |

The helper is the prior independent checker, **not the scientific producer**.
Its source hash is checked before import. Only inert receipt, array, snapshot,
metric and arithmetic helpers are reused; its old audit, bundle and main are
never invoked. No previous scientific results are inputs to this new checker.

## Indispensable acceptance checks

The checker requires the main-supplied completion hash, manifest hash and exact
six-source acquisition pin map. It binds all **323 receipted artifacts**
(179 fixed plus 144 diagnostic JSON/tensor files), rejects missing, additional,
duplicate, escaped or symlinked files, checks sizes/hashes and full completion,
and rechecks input/source bytes after analysis. The completion itself is
separately bound by its supplied hash. The acquisition directory therefore has
324 files including completion. Resource receipts and fixed environment,
36-trajectory roster, 72-event count and output inventory are checked.

From the pinned training IDX bytes it independently reconstructs the fresh
split, fixed labels, all accepted underlying plan arrays and streams 6–9.
Every occurrence is preserved, including duplicates, with exact within-group
ordering and shared slot permutations. Both complete schedule arrays, per-batch
counts, block statistics, all permutations, and distinct count-only high/low
anchors are compared. True rare labels remain unchanged; wrong membership is
the actually changed population. Majority probes use six examples per true
non-8 class; rare probes use 32; wrong probes use the fixed random subset, with
the same inputs for assigned/corrected targets. Input hashes independently use
the declared CPU NumPy FP32 division before transfer convention.

All 36 curves must contain exactly 21 fixed states. The checker recomputes
**756 evaluations** from saved FP32 train/held-out logits: every class's true
accuracy and CE, rare, majority macro, balanced total, assigned-label fit and
actually changed-label fit. Exact shared initial/warmup logits and bound full
warmup states are verified. It independently constructs all 56 endpoint
metrics, warmup changes, six schedule contrasts, eight policy contrasts and
four schedule-policy interactions, preserving all three seed values, mean,
sample SD/SE and signs. Missing Clean wrong-label metrics remain null.

For all 72 native events it verifies full before/after snapshot hashes,
parameter order, probe timing, membership, targets and counter semantics:
before Adam is at update u−1 while the already-observed tracker is at u.
Both are at u after the recorded step. Saved delivered gradients, unchanged
observer/RNG/gradient buffers around probes, native action arithmetic, Adam
recurrences, actual parameter displacement, decay/adaptive decomposition,
numerical-span geometry, signed mean-gradient utilities and finite group CE
changes are checked. Each saved group is one FP32 P-vector, not per-example
gradients. FP32 mean differences and their geometry are reconstructed. A final
state is compared to diagnostic after-state **only if its selected anchor is
actually update 2000**; an earlier last anchor is not treated as the endpoint.

## Fixed numerical tolerances

These are inherited before new outcomes, not fitted to discrepancies:

- Hashes, array membership, labels, counts, nulls and structural identity are
  exact. No CPU/CUDA bitwise numerical-equivalence assumption is made.
- FP64 metric, paired-summary and signed-utility comparisons use
  `atol = 1e-10`, `rtol = 1e-10`.
- Native FP32 probe energy/retention uses `atol = 1e-7`, `rtol = 5e-4`.
  Raw and numerical-span energies also receive the tighter independent check.
- Native delivered-action L2 error is bounded by
  `3e-5 * ||raw_gradient|| + 1e-12`.
- Numerical-span and displacement quantities use `atol = 1e-10`,
  `rtol = 1e-8`; numerical rank uses the fixed FP64 threshold
  `eps64 * max(shape) * largest_singular_value`.
- Action-history scalar checks retain their tight FP64 tolerances and the
  prospective `10 * eps32` post-cast native-norm law.
- Coordinatewise Adam moment recurrence tolerances use `16 * eps32` times
  the sum of absolute recurrence terms, plus `1e-30`. Parameter recurrence
  uses `16 * eps32 * (abs(old_parameter) + abs(ideal_adaptive_change)) + 1e-30`.

No favorable scientific sign, minimum effect size, best checkpoint, composite,
p-value, equivalence or noninferiority condition is an acceptance gate.

## Launch and renderer contract

The guarded CLI is:

```text
/usr/bin/python3 scripts/audit_spectral_batch_composition.py
  --execute
  --acquisition-dir ABSOLUTE_ACQUISITION_DIR
  --output-dir EXCLUSIVE_PARENT/audit-001
  --expected-complete-sha256 MAIN_SUPPLIED_SHA
  --expected-manifest-sha256 MAIN_SUPPLIED_SHA
  --expected-sources-json MAIN_FROZEN_SOURCE_MAP
```

Only `spectral-batch-composition-audit-001.service` is admitted: Type=exec,
one CPU quota and one math thread, 4 GiB memory, no swap, five-minute hard
deadline, 250-second cooperative deadline, no restart, control-group kill.
The checker has no GPU API calls. Tensor loading is restricted, CPU-only and
streamed; no saved state is instantiated as a model. Exclusive output is
outside acquisition, capped at 100 MiB. Source pins for the checker, fixtures,
this note, inherited helper and main's source-map file are captured and
rechecked. No outcome-driven omissions or cap expansion are implemented.

The sole result is `result.json`, schema
`spectral_batch_composition_independent_audit_v1`, with `status`, `errors`,
`checks`, discrepancies, tolerances, input/source hashes and resource receipts.
Renderer fields are `acquisition_dir`, `expected_completion_sha256`,
`expected_manifest_sha256`, `input_receipts` (path, size_bytes, sha256), and
`independent_summary` with exactly the producer's summary schema. A full PASS
also carries `trajectories_checked = 36`, `evaluations_recomputed = 756`,
`native_events_checked = 72` and independently computed diagnostic summaries.
On failure these acceptance counts are null; incomplete work is not certified.

Limits: saved logits and mean gradients are checked as recorded evidence;
their neural forward/backward provenance relies on the bound, reviewed
producer and its state-preservation assertions. This is not an independent
inference or gradient recomputation. Numerical checks validate the recorded
policy execution, not causal mediation or semantic selectivity. Runtime is a
bounded target rather than a pre-execution promise.
