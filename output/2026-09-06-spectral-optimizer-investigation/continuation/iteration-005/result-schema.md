# Iteration 005 result schema (implementation contract)

Schema version 1. This is a prospective contract, not observed results. The
parent owns analysis-plan.md, summarize_results.py and test_summary.py.

`results/execution.json` records status (`replay`, `reference`, `complete`, or
`failed`), source/data/historical artifact hashes, the unique external bulk root,
timestamps, resource/gate evidence, completed replay seeds and completed snapshot
identities. Confirmation is complete only with seeds [3,4,5] and all 12 snapshots.
`results/replay-seed{seed}.json` records historical comparisons, every new
parameter hash, observer ranks/initialization, timings and bulk artifact metadata.
Require execution `mode="full"`, `all_gates_passed=true`,
`official_test_opened=false`, `new_accuracy_or_checkpoint_selection_computed=false`.
Each replay requires `steps=2000`, `instrumented=true`,
`all_historical_gates_passed=true`, `all_state_gates_passed=true`,
`historical_scalar_comparison_steps=2000`, `snapshot_count=4`,
`probe_state_check_count=4`, four named `checkpoint_comparisons` each with
`bitwise_equal=true`, 100 `warmup_trajectory_hashes`, and 2000 entries each in
`trajectory_parameter_sha256`, `raw_gradient_sha256`, `innovation_sha256`.

Each `results/snapshot-seed{seed}-step{step}.json` has:

```text
schema_version: 1
seed: 3|4|5
step: 200|500|1000|2000
state_timing: pre_adam_after_current_gradient_observation
reference: {metrics, diagnostics, null_reasons, energies}
observers:
  width32: {estimation_width: 32, actual_rank, metrics, null_reasons,
            energies, diagnostics}
  width128: {estimation_width: 128, actual_rank, metrics, null_reasons,
             energies, diagnostics}
artifacts: [{path, size_bytes, sha256, ...shape/dtype/axes metadata}]
all_numerical_gates_passed: true
```

Every `metrics` entry is a finite number or null. Every null has a same-key
string reason in `null_reasons`. A scientific rank/gap null is not a failed
execution. NaN/Inf never serve as nulls. All raw numerical diagnostics are kept.

Observer metric keys are fixed:

- `span_energy_fraction` (single primary, step 2000 only)
- `span_projector_distance`
- `relative_covariance_error`
- `represented_operator_energy_fraction`
- `trace_P_C`
- `covariance_estimator_trace`
- `native_clean_retention`
- `native_noisy_retention`
- `native_corruption_residual_retention`
- `native_auxiliary_clean_retention`
- `native_clean_minus_corruption_retention`
- `native_clean_corruption_cosine`
- `native_projected_clean_corruption_cosine`
- `native_current_gradient_retention`
- `native_previous_current_gradient_retention`
- `self_inclusion_retention_increment`

Reference metric keys are `optimal_rank32_energy`, `covariance_trace`,
`covariance_squared_frobenius`, `clean_retention`, `noisy_retention`,
`corruption_residual_retention`, `auxiliary_clean_retention`, and
`clean_minus_corruption_retention`. Reference diagnostics include numerical
positive rank/threshold, boundary relative gap, reference-projector validity,
all residual/orthogonality/spectral errors, and first accepted innovation step.
The full raw spectrum lives in a hashed external array, not rounded prose.

Observer `energies` contains `reference_squared_frobenius`,
`estimator_squared_frobenius`, `covariance_inner_product`,
`covariance_squared_error_raw`, `span_captured_energy`,
`represented_operator_output_energy`, `probes` (each vector name maps to
`input_squared_norm` and `output_squared_norm`), and `joint` with raw/projected
clean-corruption dot products and noisy energy-closure residuals. Native action
roundoff and full-basis/QR orthogonality are separate diagnostics.

The sole paired direction is width128 minus width32. The primary is the final
state's `span_energy_fraction`; other times, covariance capacity, native probe
and self-inclusion layers remain secondary. Analysis must require all complete
seeds/snapshots, preserve null masks, and follow protocol.md without choosing
an outcome-favorable time or metric.
