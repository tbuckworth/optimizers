# Packaging-only report builder

9 September 2026. `scripts/build_grokking_raw_direction_report.py` is prepared
for main review; **it has not opened new experimental results or run on real
inputs**. No frozen acquisition, measurement, analysis or audit source changed.

Five expected SHA256 values must be supplied explicitly for the fixed
`analysis-001/{summary,complete,manifest}.json` and
`{tensor-history-audit-001,readout-paired-audit-001}/result.json` files under the
existing raw-direction batch. Both audits must be PASS, have no errors and
cover this exact completed analysis/batch and all five seeds. No failed audit
can be bypassed by invoking the builder.

The new exclusive `results/` package contains:

- One three-panel primary raw-minus-archived-projected difference plot at
  2,000 and 2,500: every seed point, a separate stored mean ± sample-SE marker,
  zero reference, and explicit favorable direction (CE lower; margin/R² higher).
- Primary contrast table, all six contrast tables covering all 14 metrics and
  every paired value/difference, all-seed endpoint tables, and secondary accuracy.
  Accuracy percentages/percentage-point conversions are labeled explicitly.
- Byte-identical copies of the five small input JSONs and a packaging manifest
  with input, builder and artifact hashes. The complete 15-state summary remains
  unchanged, including descriptive step 1,501.

The builder formats stored values; it does not recompute differences, means,
SDs, SEs, probes or inference. Undefined values remain undefined. It adds no
p-value, equivalence, mechanism, safety or general-speed conclusion. Output is
exclusive and capped at 100 MiB with 1 GiB free reserve; combined small-JSON
inputs are capped at 80 MiB. A partial packaging failure is preserved, not
silently overwritten or retried.

Source checks: Python compilation and `--help` pass. One in-memory synthetic
exercise passes complete receipt/schema admission, FAIL-audit and wrong-analysis
rejection, all six contrast tables, 45 all-seed rows, undefined-value formatting
and PNG encoding. A final in-memory mixed-sign plot check also passes after
separating the mean marker from every seed point.
The plot was generated only into memory; no synthetic result artifact or real
result package was written. Final source/visual review and actual execution
remain with the main agent.

Future invocation, with verified hashes substituted by main:

```text
python3 scripts/build_grokking_raw_direction_report.py \
  --summary-sha256 VERIFIED_SUMMARY_SHA256 \
  --complete-sha256 VERIFIED_ANALYSIS_COMPLETION_SHA256 \
  --manifest-sha256 VERIFIED_ANALYSIS_MANIFEST_SHA256 \
  --tensor-audit-sha256 VERIFIED_TENSOR_AUDIT_SHA256 \
  --readout-audit-sha256 VERIFIED_READOUT_AUDIT_SHA256
```