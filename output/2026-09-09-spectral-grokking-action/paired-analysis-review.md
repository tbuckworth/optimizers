# Independent review: paired action analysis

9 September 2026.

## Verdict

**PASS.** I found no material defect in the pure-JSON analyzer or its seven
focused fixtures against the frozen paired-analysis protocol and the measurement
schema at `b86099c`. This is a source-and-fixture review, not an audit of the new
scientific outcomes. I did not open any live action endpoint result, run
inference or training, or load a scientific NPZ/checkpoint.

Reviewed working-source identities:

- `experiments/analyze_grokking_action_results.py`:
  `e0a57f6ac5dfc0e6fb5638c0777efe1de5fba7b455c2421880514cfe28d2c628`
- `tests/test_grokking_action_analysis.py`:
  `abdf1f63e79e8a03ccf60f8b5183f784032029ba2c2624df4d02d609c69ceb5e`
- `paired-analysis-protocol.md`:
  `8e2b685bf809a44f351826bde6222070b2411484b47a25d9d5a1b44b410346bc`
- `experiments/measure_grokking_action_states.py`:
  `09743be2cc16f34c559b9d1b0f6813429fb832e0efe52340b3cf7d68b8d7c288`

The last hash exactly matches the file at commit `b86099c`; the imported
representation analyzer also matches that commit.

## Checks

- Admission is fail-closed on a measurement failure marker and requires the
  exact ordered 35-state roster, all five seeds, the declared schemas, status,
  state count, fixed frequency panel, source maps, and manifest/prior-summary
  bindings (`verify_measurement`, lines 267–350). It hashes every new raw,
  scalar, and analyzed-state receipt and checks scalar/state/checkpoint
  provenance before admitting a metric row.
- The accepted archived summary is fixed to SHA-256
  `df4f1516f1d1db60c46e2344643f08c4d162892720b8e33e18a1b0069ec09480`.
  Metadata-only inspection confirmed its 150-row exact roster and 6+144 raw
  receipt partition. `load_prior_summary` selects exactly the 15 legacy rows for
  seeds 100–104 at steps 1500, 2000 and 2500 and binds each to its archived raw
  receipt (lines 353–388).
- The six endpoint rows are exactly two endpoints times three fixed contrasts.
  Each metric uses one difference per seed, defined as left minus right. The
  implementation reports the arithmetic mean, sample SD with denominator four,
  SE as SD divided by square-root five, and exact positive/negative/zero counts
  (lines 207–264). No checkpoint, class, frequency or null draw is treated as a
  replicate.
- Primary field mappings are correct: held-out CE from
  `behavior.test.loss`, held-out correct-class margin from
  `behavior.test.correct_class_margin_mean`, and final-hidden selected-five
  evaluation R² from `probes.final_hidden.selected_eval_mean_r2` (lines 45–54).
  Units and favorable directions are explicit; accuracy remains a fraction.
- Undefined energy-normalized secondary values remain `null`. If any seed is
  undefined, its pair remains present but the analyzer suppresses the incomplete
  five-seed mean, SD, SE and sign counts instead of silently aggregating a
  favorable subset (lines 187–189 and 221–236).
- All 35 new rows, including step 1501, and the 15 archived native reference
  rows are retained in the summary. The operand identities and archived-native
  provenance distinguish the two archived-reference contrasts from the
  norm-matched-versus-orthogonal within-acquisition contrast. The output text
  explicitly disclaims p-values, equivalence, pseudo-replication and AUC.
- Output creation is exclusive under `/tmp/spectral-experiment-artifacts`, with per-write
  cumulative-size and free-reserve checks (lines 391–431). The analyzer pins its
  source closure, rechecks the measurement source maps, all admitted new
  artifacts, and the measurement manifest/completion before writing its sole
  success marker (lines 523–538). A post-output exception instead preserves a
  failure marker.

## Fixture evidence

The converted test uses only the standard library and preserves seven cases.
Both commands passed under the pinned system interpreter and one-thread math
environment:

```text
/usr/bin/python3 -m unittest discover -s tests -p 'test_grokking_action_analysis.py' -v
Ran 7 tests in 0.104s — OK

/usr/bin/python3 -m unittest discover -s tests -p 'test_grokking*.py'
Ran 53 tests in 2.064s — OK
```

The focused cases cover exact contrast order/arithmetic/units/signs, failure on
missing or non-finite primary values, undefined energy ratios without subset
aggregation, duplicate/missing endpoints, exact archived roster and raw-receipt
binding, exact 35-state admission with raw corruption detection, and live source
hash rejection.
