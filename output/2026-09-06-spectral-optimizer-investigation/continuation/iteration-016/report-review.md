# I16 final report integration review

**Verdict:** Pass; conditional findings and their limitations are preserved.

8 September 2026. This is main-agent integration review of the leaf-authored
results and mathematical interpretation, plus main's prospective design. It is
not represented as an independent empirical replication. Independent evidence
comes from the previously frozen semantic audit and separately implemented
stdlib raw-JSON corroborator, which agree with the reported quantities.

## Approved document hashes

- `results.md`: `ba09907c95e96fca248d9d4eaf2b23fea0f65e50ae94f9035c2e1c8938010fdd`
- `mathematical-interpretation.md`: `f96fad47f320a776625312279f2f089ae69d680f2069fcfb43ef61875dacdfc2`
- `next-gain-matched-design.md`: `d3681823f330b918a0ed5fadd02dae17901baaf6bbb2ed008e36afbec3b0cffd`

## Checks and corrections

- All eight primaries retain each seed and the mean, both selectors and both
  auxiliary metrics. Seven means favor scalar; fixed/min-CE-selected accuracy
  is the positive spectral exception with mixed seeds. The small mixed CE
  effect under fixed/max-accuracy selection is not called equivalence.
- All12 selector choices,30 curves/180points, fixed-k endpoints, h100 progress,
  realization fitting, confidence and displacement geometry remain traceable
  to the pinned summary and original archived JSON. The independent check
  compares22,814 numeric/discrete values with zero maximum difference:
  [report-audit.json](analysis-001/report-audit.json), SHA
  `52bada70c9e071fca20aa914d1d019f95b76bc55132256e0fc0ceb5dda3c8c18`.
- The k0 endpoint witness is favorable on both metrics, both targets, all
  three seeds, with actual progress. Its clean underfitting relative to raw,
  slightly greater realization fitting than spectral, and radically different
  update dose prevent a unique-mechanism or semantic-noise-removal claim.
- I8's constructive spatial result and I15's genuine neural progress remain.
  Reused seeds, adaptive selection of the I15 comparator and unequal within-I16
  validation candidate counts are explicit. No official-test, universal ranking,
  production-default or finished-overall-goal claim is made.
- Main corrected the ambiguous opening about which prediction failed, the
  near-equality wording and an overstrong causal implication of a decay-only
  future test. Local algebra residuals are not long-run sensitivity bounds.
- The prospective normalized-response formulas hold under their stated fixed
  action/exogenous-stream or constant-state assumptions. The white-noise
  calculation is explicitly not a neural noise model. The reproducible
  [kernel check](check_normalized_response.py) passes mass, lag, variance and
  nonnegativity identities and verifies the nonmonotone lag/variance example.
  A moving action, inherited state and finite-horizon dose remain limitations.

## Remaining work, not defects hidden by this review

The24-branch matched-gain proposal has not been implemented or admitted. It
requires a separately frozen protocol, code/state checks, resource admission
and independent analysis. Fresh-panel confirmation is a subsequent proposal,
not something I16 has already supplied. No old experiment should be rerun.