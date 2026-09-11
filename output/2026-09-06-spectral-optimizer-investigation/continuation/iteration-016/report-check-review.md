# I16 report-check source admission

Status: PASS

8 September 2026. Main reviewed the independent standard-library corroborator
and its synthetic tests. Source is frozen in checkpoint `07b5589`:

- `corroborate_report.py`: `557c548b12af76f0f1a04383dccb3bb23eeca22d2f62a95e85876617a9ac6ba8`
- `test_corroborate_report.py`: `6a560d2bb780aa445a689d328590ce773d442293e93218e6b98e301857c6eba7`

Five tests pass, including exact selector ties, missing-member unavailability,
pin rejection, synthetic scalar reconstruction and exclusive output refusal.
The checker independently reconstructs all eight primary contrasts, 12 joint
choices, 32 per-k selected comparisons, 16 endpoint comparisons, 30 curves,
960 scheduled aggregates, 20 path rows and 84 component rows. It uses pinned
I14/I15/I16 lossless JSON archives, not the analyzer's calculation helpers.
It retains missing legacy measurements as unavailable rather than zero.

This is a corroboration of derived scalars, not a new acquisition or a full
semantic-audit replay. The accepted independent I16 audit separately binds
complete states, original sources, raw references and runtime inventory. A
passing result does not validate every narrative claim automatically, identify
a causal mechanism, or increase the number of experimental seeds.

Admit one CPU-only execution with6GiB host memory, no swap, CPUQuota100%,
300-second runtime ceiling and absent `analysis-001/report-audit.json` output.
No model imports, tensors, forwards, updates, retries or paid resources.

## Final test invocation caveat

The final combined unittest invocation (9e175c) passed42 tests, but its last
six were the older I15 report-checker suite, not I16's five. A read-only module
origin check (1aeba1) confirmed that imports of historical helpers change the
search path from I16 to I15 for the duplicate `test_corroborate_report` name.
The36 preceding analysis/collection/core/runner tests are I16 tests. Running
the I16 checker suite in a fresh process independently passes all five tests
(1aeba1,0.338s; also628ae0 before real admission). Do not report42 I16 tests.
Use two separate processes for these suites. No frozen source was changed;
the actual once-only corroborator was launched by its explicit I16 file path
and its accepted output is unaffected.
