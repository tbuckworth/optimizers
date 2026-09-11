# Independent saved-result audit and plot plan

Prospective preparation for the user-authorized clustering MNIST pilot;
no real results have been inspected for this plan. The producer protocol and
runner source have now been read completely. The fixed roster is three seeds
202609091–093, clean/fixed_uniform_0p9, AdamW/hard32/cluster32/mixed32,
21 evaluation states per trajectory, and step2000 endpoints only. This checker does
not train, infer, load model tensors, or restart any experiment.

Use the researcher's independent evidence/claim checks. The user's explicit
fixed-roster and no-restart scope overrides the plugin's fresh-seed/split
re-execution requirement. The resulting verdict concerns saved-data arithmetic
and provenance, not independent replication of the learning experiment.

## Checks

- Bind protocol, source freeze, manifest, completion and every curve/result
  JSON by SHA-256; reject duplicate/missing stage, seed, condition or arm.
- Verify matched initialization, warmup state, batch schedule, clean data,
  corruption labels and masks across arms within each paired seed/condition.
  Distinguish replacements selected from labels actually changed.
- Recompute each reported accuracy and CE independently in NumPy float64
  from the saved FP32 logits and bound labels, then verify integer counts,
  CE sums, denominators, finite logs and common evaluation-step roster.
  Verify labels against training IDX, disjoint 5,000/5,000 splits, and the
  documented split/init/corruption/batch pseudorandom streams. No official
  test data or model checkpoint tensor is loaded; checkpoint bytes are hashed.
- Recompute fixed-endpoint per-arm summaries and all frozen paired contrasts
  directly from those curves. Report every seed, arithmetic means, sample
  dispersion/SE where registered, and direction counts. No outcome-selected
  intervals, checkpoint selectors, extra hypothesis tests or composites.
- Validate logged action/cluster diagnostic ranges and summary arithmetic
  where sufficient scalars are saved. All19 saved membership arrays additionally
  permit independent cluster counts, isolate churn and common-assigned ARI;
  verify the saved first raw-gradient hash across four arms and the first
  post-observation tracker hash across three filtered arms. Do not infer dense geometry
  or claim that pre-Adam gradient clustering constrains the actual Adam step.
- Preserve failures and partial acquisition evidence. No automatic rerun.

## Plot and claim checks

Use clean and fixed-replacement panels on the same training-step axis, with
all frozen arms and common warmup visibly identified. Accuracy axes are
fractions or explicitly converted percentages; plot seed means and faint
individual-seed curves (three seeds, descriptive evidence). Bind plot arrays
to checked curve values. Useful noisy-label protection requires retained
clean-label learning, not merely lower corrupted-label fit. Show endpoint
trained-label fit in a compact table or additional panel if informative.

## Execution bounds and numerical comparison

Main must run at most once, after independently obtaining the successful
acquisition completion hash, under one CPU, 4 GiB RAM, no swap, and a hard
two-minute service limit. The checker imposes a 110-second cooperative deadline
at file admissions and caps new output at100 MiB. All math thread variables
are1. Exact source is the exclusive `acquisition-001` under
`/tmp/spectral-experiment-artifacts/spectral-clustering-mnist-20260909.51ggnu`; output is the
new sibling `audit-001`. Failures are retained; no retry is authorized here.

Saved float64 CE sums from FP32 logits are compared against independent NumPy
log-sum-exp with absolute and relative tolerance1e-10; counts are exact.
Action norm ratios use the same scalar tolerance; cluster/mixed contraction
and the half-identity norm floor allow2e-6 for FP32 delivery. Full curve values,
three-seed endpoint summaries, five contrasts per condition and their complete
seed values are emitted so main can plot without importing producer helpers.
Parameter states and actual later Adam steps are not independently reconstructed:
only their saved identities or
available scalar algebra are checked.
