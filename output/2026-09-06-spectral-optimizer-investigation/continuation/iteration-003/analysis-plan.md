# Pre-outcome aggregation specification

Prepared 2026-09-06 after a runtime/invariant-only development pilot, before any
confirmatory seed or outcome inspection. This supplements, rather than changes,
the training design in [protocol.md](protocol.md).

The primary estimator contrast is width128-minus-width32 in the .9 nominal
replacement condition. The clean condition and each filtered-minus-AdamW
contrast remain fully reported boundary checks. Primary retention uses the
arithmetic mean of finite per-probe ratios; ratio of summed energies is a
separately named secondary quantity with numerator/denominator metadata.
Undefined values remain null. Paired averages require matching null-step masks
and all three seed pairs; otherwise report unavailable comparisons explicitly.

The primary actual-update geometry uses the nominal **decay-subtracted**
displacement. Total displacement is also fully reported, not selected after
seeing which is favorable. This prioritizes the data-update question without
claiming to remove historical effects of weight decay. Report all postwarmup
steps and both predeclared early/late windows.

Current-gradient ascent, positive outside contribution, nominal leakage
reversal, closure-robust leakage reversal and closure-unresolved reversal are
distinct frequencies. Use every scheduled step in each denominator, including
near-zero classifications. The robust event requires positive total, negative
inside and positive outside terms, each farther from zero than its own sign
tolerance plus the absolute observed closure residual.

Aggregate each seed first, then report all three paired differences, mean,
median, range and descriptive sample SD. No p-values or confidence intervals
based on correlated steps, no result-dependent window choice, and no claim of
equivalence from a near-zero difference. Accuracy remains a fraction in JSON.
Final and validation-selected checkpoint outcomes are both mandatory.

[summarize_results.py](summarize_results.py) accepts only the completed 18-run
confirmatory output, verifies condition labels and complete observation
schedules, and refuses to overwrite a saved summary. It does not read pilot
outcomes. Its synthetic checks are [test_summary.py](test_summary.py).
