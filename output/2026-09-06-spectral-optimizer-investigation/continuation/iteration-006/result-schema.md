# Iteration 006 raw result contract

Schema version 1, published before implementation execution or outcomes.
The parent-owned analysis plan remains authoritative for aggregation. This
contract describes a standalone producer, not an extension of iteration004's
hard-coded four-arm/noisy-only orchestration.

## Identity, paths and completion

Arm order: `adamw`, `current32`, `lagged32`, `lagged32_current_norm`,
`scalar_current32`, `scalar_lagged32`. Full cell identity is `(seed, noise, arm)`
with seeds `[6,7,8]`, noise `[0.0,0.9]`. A textual `run_key` is
`seed{seed}-noise{noise:g}-{arm}`. Require all 36 unique cells exactly.

Small metadata is in `results/execution.json` (or `pilot/execution.json`).
All plans, tensor checkpoint bundles and raw run JSON are in an exclusively
created directory below verified `/tmp/spectral-experiment-artifacts/`. Every artifact binding
has absolute `path`, `size_bytes`, `sha256`; run/checkpoint bindings also have
`run_key`, `seed`, `replacement_probability`, `arm`. Plans have `seed`.

Full execution fields:

- `schema_version=1`, `mode="full"`, `status="complete"`,
  `all_gates_passed=true`, `completed_runs=36`, `test_evaluations=144`.
- `started_utc`, `all_training_completed_utc`, `test_first_opened_utc`,
  `completed_utc`, with test open strictly after all training/selection.
- `completed_cells`: 36 identity dictionaries (`run_key`, `seed`,
  `replacement_probability`, `arm`); exact set equality, not just length.
- `runs`: 36 final raw-JSON bindings; `checkpoints`: 36 bundle bindings;
  `plans`: three plan bindings; `training_data_artifacts`: two training IDX
  bindings; `test_data_artifacts`: two official-test IDX bindings.
- `source_sha256`: exact 15-member repo-relative map listed below;
  `repository_revision`, `environment`, `bulk_root`, `bulk_mount`,
  `occupancy_before`, `resources`, `passing_pilot` (execution binding).
- `official_test_opened=true`, `warmup_checks_passed=true`.

Statuses before completion are `training`, `test`, or `failed`; they are never
accepted by the summarizer. Results are finalized only after all four tests;
their final hash bindings replace no old files: pre-test JSON, if persisted,
uses a distinct filename. Failure preserves completed artifacts plus separate
partial-cell/failure metadata, without disguising an incomplete cell as complete.

## Exact scientific-source membership

Prefix the first 13 paths with
`output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006/`:

1. `delivery_order_harness.py`
2. `policy_math.py`
3. `test_policy.py`
4. `test_harness.py`
5. `artifact_store.py`
6. `protocol.md`
7. `result-schema.md`
8. `design-intent.md`
9. `analysis-plan.md`
10. `summarize_results.py`
11. `test_summary.py`
12. `best-practices-check.md`
13. `challenge/decision.md`
14. `output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-003/neural_harness.py`
15. `spectral_filter.py`

Pilot and full maps must match current bytes and the committed source at each
launch. Source, training data and numerical environment must match the accepted
passing pilot. Parent launch-decision documents are authority records, not
additional scientific-source files whose later creation changes the pilot map.

## Full raw run JSON

Each cell records `schema_version`, `run_key`, `seed`,
`replacement_probability`, `arm`, `steps=2000`, `instrumented=true`,
`all_invariant_gates_passed=true`, `warmup_checks_passed=true`,
`delivery_gate_steps=2000`, `measurement_state_checks=22`, and
`realized_replacement_fraction`, `realized_incorrect_fraction`.
Clean condition must have both realized fractions zero. All arms within a
seed/condition share those fractions and the same `plan_sha256`.
Realized fractions preserve the unchanged helper's float32-mean values; validate
against the nearest count/5000 within `3e-8`. They are not exact count-division
metadata. Accuracy metrics separately retain exact correct/count arithmetic.

Required histories:

- `steps_raw`: 2,000 records, step 1 through 2000, described below.
- `validation_trajectory`: 21 dictionaries with `step` on grid
  `[0,100,...,2000]`, `accuracy`, `cross_entropy`, `count=5000`.
- `trajectory_parameter_sha256`: 2,000 post-Adam parameter hashes.
- `warmup_trajectory_hashes`: 100 dictionaries with `step`, `parameters`,
  `raw_gradient`, `applied_gradient`; raw and applied match bitwise.
- `warmup_observer_hashes`: 100 hashes for every observing arm, empty for AdamW.
- `warmup_core_sha256`, `warmup_observer_sha256` (null for AdamW).
- `estimation_rank_by_step`: 2,000 integers (zero for AdamW);
  `repair_count_by_step`: 2,000 integers (zero for AdamW).
- `step_elapsed_seconds`: 2,000 timing floats; `elapsed_seconds`, `resources`.

Checkpoint names are exactly `final`, `min_val_ce`, `max_val_accuracy`,
`warmup100`. `checkpoint_steps` maps all four names to steps. `final=2000`,
`warmup100=100`; selectors are reconstructed independently from the exact
validation grid, using strict improvement and earliest exact ties. Step 0 is
eligible. No cross-metric tie-break. `checkpoint_sha256` maps each name to its
tensor-state hash; coincident selectors still have independently cloned storage.
`checkpoint_path` points to one tensor-only CPU bundle mapping each of the four
names to a model state dictionary. No full model/optimizer/RNG object is pickled
into these portable bundles.

`test` maps all four checkpoint names to `{accuracy,cross_entropy,count:10000}`.
`final_training_clean`, `final_training_noisy`, `final_validation` use the same
metric dictionary with count 5000. Both training labelings are evaluated from
the same fixed-state logits; they never select checkpoints. `final_validation`
equals the step-2000 validation metrics (without its `step` field).
All accuracies are fractions, never percentages; all non-null numbers are finite.

## Step record and policy interface

Each row has `step`, `policy` metadata, the following flat scalar keys and
`null_reasons` (every null scalar has its same-key string reason):

- `raw_norm`, `raw_squared_norm`, `current_norm`, `current_squared_norm`,
  `lagged_norm`, `lagged_squared_norm`, `applied_norm`, `applied_squared_norm`.
- `current_raw_norm_ratio`, `lagged_raw_norm_ratio`, `applied_raw_norm_ratio`.
- `current_energy_retention`, `lagged_energy_retention`,
  `current_minus_lagged_retention`.
- `current_lagged_cosine`, `raw_applied_cosine`, `policy_scale`.
- `norm_matching_relative_error`, `direction_relative_error`.

The policy API is
`transform_gradient(arm, tracker, raw) -> applied,current,lagged,metadata` and
`delivery_metrics(raw,current,lagged,actual_delivered,metadata) -> scalars`.
The latter always checks the actual flattened `.grad` after assignment, including
instrumentation-off runs. AdamW and all warmup steps have null current/lagged
candidates; unscaled policies have null `policy_scale`. Zero denominator/cosine
norms remain null, not zero. Norm/direction gates are mandatory even when no
step row is retained by a pilot.

`policy` keys are `arm`, `observing_step` (null for AdamW), `policy_active`,
`active_policy` (`identity_warmup`, `identity_baseline` or active arm),
`current_basis_rank`, `lagged_basis_rank`, `current_basis_missing`,
`lagged_basis_missing`, `current_candidate_operator`, `lagged_candidate_operator`,
`scale`, `scale_direction`, `target_norm`, `delivery_operator`. Rank/operator metadata describes the
candidate basis, not a false rank-32 claim about scalar-identity delivery.

Baseline step/ranks/missing/operator metadata is null; observing warmup has
stored ranks 0..32 and missing booleans, but null candidate operators. Active
candidate operators are `native_basis` or `identity_missing_basis`.
`delivery_operator` is `identity`, `native_current`, `native_lagged`,
`scaled_native_lagged` or `scalar_identity`; it names the policy family even
when that candidate's absent basis acts as identity. `scale_direction` is
`raw`, `current` or `lagged`; `target_norm` is always a finite nonnegative norm.

Each row also has `total` and `decay_subtracted` update dictionaries:

```text
norm, squared_norm
raw_gradient_dot_update: {value, tolerance, sign}
applied_gradient_dot_update: {value, tolerance, sign}
raw_gradient_update_cosine, applied_gradient_update_cosine
null_reasons
```

Signs are -1/0/+1 with tolerance `1e-6*gradient_norm*update_norm+1e-14`.
The displacement is actual after-minus-before parameters; nominal decay
subtraction adds `.001*.01*theta_before`. Sign-frequency denominators include
every scheduled step even if a cosine is null. No leakage decomposition is used.

## Pilot and failure evidence

Pilot execution: `mode="pilot"`, `status="complete_passed"`,
`completed_traces=12`, `all_gates_passed=true`, `warmup_checks_passed=true`,
`validation_or_accuracy_computed=false`, `official_test_opened=false`.
One development plan, seed9880/noise.9/220steps; six arms each off/on.
`timing-and-invariants.json` contains six reports with `arm`,
`trajectory_bitwise_identical=true`, `final_state_bitwise_identical=true`,
`warmup_checks_passed=true`, `uninstrumented`, `instrumented`.
Those inner records retain timing, resources, all 220 parameter hashes,
warmup hashes/core/observer hashes, ranks/repair counters and gate counts only.
They contain no training/validation/test outcomes or gradient metric histories.
Instrumented pilot measurement-state checks occur at 1,100,101,200,220 (five);
off has zero optional measurement checks but 220 mandatory delivery gates.
Pilot execution's `timing_and_invariants` binding points to that JSON in bulk.
Both modes' `environment` keys are exactly `python`, `numpy`, `torch`, `cuda`,
`gpu`, `cpu_threads`, `deterministic_algorithms`, `tf32`, `cudnn_benchmark`,
`cublas_workspace`, `foreach`, `fused`, with exact pilot/full equality.
Full execution also retains `training_runs`: 36 distinct immutable pre-test JSON
bindings, for recovery; `runs` binds the 36 final tested JSON files. Global
test-open ordering follows the exact complete-cell/checkpoint gate, not per-run
timestamp heuristics. Repair counters are actual monotone canonical counters;
drift-triggered repairs mean they are not forced to equal floor(step/100).

The 1 GiB total artifact, 8 GiB allocated GPU, 12 GiB RSS and pilot180/full900
second limits are fatal cooperative guards. Failure JSON records finite error
context, completed-cell identities/artifacts and partial progress; it is never
consumed as a completed study. Each attempt refuses overwrites and requires
separate explicit launch flags plus parent approval.
