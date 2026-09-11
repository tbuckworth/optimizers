# Raw-direction paired JSON analyzer — source implementation

9 September 2026. **Source and synthetic fixtures only.** No real checkpoint,
NPZ member, measurement or saved-data numerical analysis was loaded/executed by
this implementation task. Existing accepted JSON was inspected only for schema.

The new `experiments/analyze_grokking_raw_direction_results.py` imports only
the unchanged stdlib-only action analyzer. It reuses its 14 metric definitions,
five-seed sample-SD/SE arithmetic, explicit undefined-secondary treatment,
receipt helpers, exclusive output writer and resource checks. It does not
invoke the old analyzer's main or repeat the old contrasts.

## Contract

- Pin the accepted archived action summary to SHA256
  `249a8b9f5b680f72f7e9a402f56a9a4734814d99dc7d8097442eac4c965aa237`.
  Retain its unchanged 35 action-reference rows and 15 archived-native rows.
- Admit exactly the ordered 15 new states, including 1501; do not treat 1501
  as an additional primary endpoint. Bind completed measurement, raw batch,
  every seed completion, exact artifact roster, checkpoints, first-action
  tensors, histories, scalar JSON, analyzed JSON and opaque NPZ receipts.
  Hashing is streaming; neither tensors nor NPZ members are interpreted here.
- Verify current original/acquisition/measurement source maps. The measurement
  map must be the full old-recipe plus acquisition plus new collector/protocol/
  guard/test union. Tie each new parent to its archived seed and fork identity.
- Require the original recipe and per-seed split contracts unchanged. Scalar
  and analyzed schemas are intentionally different: compare shared behavioral
  fields and selected/null probe fields, not whole dictionaries. The analyzed
  state adds margins and summarizes probes. Independent array audit remains
  required for those transformations and all derived readouts.
- Output raw-minus-norm-matched (primary), raw-minus-native and
  raw-minus-orthogonal (secondary), each at fixed 2000 and primary 2500. Keep
  five pairs/differences, arithmetic mean, sample SD, SE, all signs and favorable
  count. For descriptive metrics favorable count is null; for incomplete
  secondary ratios all aggregate/sign/favorable fields remain null. Accuracy
  stays a fraction. No new selection, p-value, equivalence or composite metric.
- Copy all 15 scalar and 15 analyzed JSONs byte-identically into exclusive
  output; summary rows omit only `full_symmetry` and add receipt fields. Source
  state copies retain that field. Check receipt/source stability before writing
  completion. Failure preserves output and a failure marker where writable.

Resources are the fixed one-CPU, 2-GiB/no-swap, five-minute and 100-MiB output
limits with 1-GiB free reserve. The code has cooperative checks and bounded
exclusive output; the main agent must provide/read back the hard unit/cgroup
controls before execution. No source freeze or real analysis launch is implied
by this document.

## Synthetic checks

Command:

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 -m unittest discover -s tests -p test_grokking_raw_direction_analysis.py -v
```

**11 tests PASS, 0.454 seconds** in this implementation run. Fixtures contain
small JSON and deliberately invalid opaque checkpoint/NPZ bytes: success
therefore does not require array/model loading. Tests cover arithmetic/units,
mixed signs and zero mean without equivalence, exact roster/order, nonfinite
primaries including 1501, undefined ratios without selective aggregation,
realistic transformed schemas, changed raw bytes, changed analyzed shared
metrics despite updated receipt chains, substituted checkpoints, failure
markers, incomplete source maps and import without torch/NumPy.

The main source review identified the guard-source additions and the
scalar/analyzed-schema mismatch during development. Both were corrected before
any execution on scientific data. These were prelaunch implementation defects,
not defects or findings in the live experiment. Final source review, committed
freeze, measurement admission and later independent saved-array/first-action
audit remain main-agent responsibilities.
