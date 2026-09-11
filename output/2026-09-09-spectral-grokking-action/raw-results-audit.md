# Independent raw-result audit

9 September 2026, completed 13:00 UTC. Auditor: independent Codex leaf
`action_raw_audit`; acquisition, scientific code and report owned by the parent.

## Disposition

**Numerical/provenance corroboration passes, with one corrected auditor-only
schema comparison recorded below.** No scientific source, raw result, selected
seed, endpoint or analysis recipe was changed. This is not a fresh-seed
replication and not a blanket `SUPPORTED` verdict under the researcher plugin's
fresh-run rubric. The user's explicit no-restart instruction takes precedence:
no training, model inference or completed experiment was repeated. CPU-only
saved-array rederivation is distinct from replication.

All five seeds100–104 and all35 new states were included. Exact primary
endpoint contrasts and all secondary paired arithmetic are corroborated.
The results support a conditional continuation-policy observation, not semantic
clustering, identified circuit formation, pre-fork selection, safety or general
optimizer speed.

## Scope and independent calculations

Frozen anchors were [action protocol](protocol.md),
[measurement protocol](measurement-protocol.md) and
[paired analysis protocol](paired-analysis-protocol.md). I read the producer's
action/intervention, Fourier, symmetry and paired-analysis implementations;
the audit scripts do not import their numerical functions or execute a model.

The [tensor auditor](audit_action_tensors.py) used all five original fork states,
shared-first-action bundles, combined Adam bundles, first-step checkpoints,
completion records and hashes of every one of the35 new checkpoints. It checked:

- shared post-estimator identity and complete restored scientific-state hashes;
- saved Q orthogonality, V reconstruction, singular values through small QᵀV,
  rank threshold, coefficients, raw and delivered action norms/cosines;
- native V(Vᵀg) in independent CPU float64, projected action and post-cast
  norm matching, with precision-aware tolerances;
- carried first/second-moment recurrences, counter1500→1501, exact checkpoint
  moment identity, bias correction, adaptive direction, separate decay and
  adaptive movement, actual displacement and decomposition residuals;
- all saved first-step summary norms and displacement/direction cosines.

This required no dense parameter-by-parameter matrix and loaded one seed at a
time. Tensor audit: **685 checks, zero errors**,60.82seconds, peak RSS2.423GB.

The [readout auditor](audit_action_readouts.py) independently reconstructed
stratified fit/evaluation indices and all20 null permutations from the frozen
seeds, and checked structural arrays across states. For every one of the35
states it used saved logits and final/pre-attention arrays to rederive:

- train/test CE, accuracy, correct-class margin and centered-logit RMS;
- fit normalization, ridge.001 coefficients with NumPy inverse/multiplication
  rather than the producer's Torch Cholesky solve, all56 fit/evaluation
  frequency scores, selected-five frequencies and means, the fixed panel,
  and all20 row-shuffle nulls for both features;
- outcome-independent held-out/exchange edges, all matched training/held-out
  edge groups and hashes, shift/wrong-shift defects, pooled ratios and excess;
- every paired seed value/difference, mean, sample SD, sample SE and sign count
  in all six endpoint contrasts and all14 metrics, including undefined handling.

There are82,320 frequency rows (observed plus nulls), each with two independently
rederived R² values and two target means. Those rows are not independent samples.
The raw audit made26,366 grouped comparisons in89.78seconds, peak RSS805.46MB.
Maximum absolute discrepancies were3.56×10⁻¹⁵ for behavior,8.66×10⁻¹⁵ for
probe R² and5.56×10⁻¹⁷ for paired arithmetic. The maximum symmetry discrepancy
2.98×10⁻⁸ concerns large unnormalized sums; all relative comparisons passed
the1×10⁻⁹ tolerance. No substantive numerical mismatch occurred.

### Preserved auditor correction

The first raw-audit artifact is deliberately retained with `status: FAIL` and
exactly one error: `analysis rows exactly match measured states`. That audit
comparison incorrectly compared the analyzer's compact row to the full state
without accounting for the documented removal of `full_symmetry` and addition
of three receipt fields. This was my audit-harness mistake, not a defect in the
scientific analyzer. It was disclosed to the parent before completion.

The separate [JSON-only supplement](audit_action_schema_supplement.py) corrects
that comparison, checks the exact transformed row for all35 states, verifies
the15 archived-native references against their accepted source rows, and checks
all copies and additional provenance/source bindings. It passed **896 checks,
zero errors**,1.72seconds. It also verifies that the original audit's sole
failure is the documented schema error. No raw arithmetic, inference or
training was repeated, and neither original script nor artifact was overwritten.

## First-step geometry

All common post-estimator bases have full column numerical rank. These are the
post-update1501 column counts, not the preceding1500 parent's counts.

| Seed | Columns = rank | Native/orthogonal action cosine | Native/orthogonal displacement cosine | Norm-match scale |
|---|---:|---:|---:|---:|
|100|199|.855028|.980190|1.171207|
|101|180|.930917|.999555|1.137367|
|102|200|.984368|.999790|1.021325|
|103|199|.805431|.997915|1.206967|
|104|175|.918825|.999429|1.117133|

Maximum Q orthogonality error4.45×10⁻¹⁵; V reconstruction relative residual
2.51×10⁻¹⁵. CPU-float64 native reconstruction differs from the saved GPU-fp32
native action by at most1.25×10⁻⁶ relative norm, compatible with different
precision/order. All norm matches are nondegenerate and within the frozen
post-cast tolerance. Full-rank numerical evidence licenses the qualified
same-span description at this shared first action, not an exact arithmetic
identity for every cast or an unchanged span throughout later training.

Actual Adam displacement is not the supplied action: direction differences are
much smaller after carried moments and decay. Pre-Adam norm matching leaves a
smaller-than-native actual displacement norm in every seed; it is not a
post-Adam norm match. Moment recurrences agree to at most2.77×10⁻⁸ (m) and
5.12×10⁻⁸ (v) relative norm. The maximum actual-movement minus fp64
adaptive-plus-decay residual is1.66×10⁻⁶ in norm and6.77×10⁻⁸ coordinatewise.
These are explicitly retained finite-precision residuals, not silently zeroed.

## What the fixed outcomes earn

Plain orthogonal continuation is worse than archived native on all three
primaries in all five seeds at both2000 and2500. Norm-matched orthogonal
continuation is better than plain orthogonal on all three primaries in all five
seeds at both endpoints. Versus archived native, norm-matched R² is higher in
all five seeds at both endpoints; CE/margin signs are mixed at2000 but all
three primaries favor norm matching in all five seeds at2500.

At2500, norm-matched minus archived native gives CE−.365745±.073803,
margin+.503453±.170917 and selected R²+.064330±.007043 (sample SE across
five seeds). Plain orthogonal minus native gives CE+1.849550±.414968,
margin−2.205693±.514191 and R²−.221107±.060989. All fixed endpoints and
adverse early comparisons must remain in the report.

This is constructive evidence that replacing unequal within-span gains by an
orthogonal direction need not destroy the later favorable trajectory when its
input norm is matched. It disfavors an unconditional claim that those particular
unequal gains are indispensable after this fork. It does **not** identify a
unique amplitude-mediated cause: later gradients, estimators and Adam states
diverge; the scale is recomputed from each branch's own counterfactual native
action, not the archived native action. It does not prove fixed subspace
sufficiency from initialization, rank selection usefulness in isolation, or
equivalence between the original and normalized policy.

An endpoint improvement is not completed grokking. At2500, mean held-out
accuracy is33.23% native,8.10% orthogonal and40.00% norm matched; mean correct
margin remains negative in all three policies. Readability is not a demonstrated
rule circuit, and all2000 mean accuracies are near chance.

## Limitations and disposition

- **Addressed free:** all five seeds, all35 states, all fixed endpoints and full
  null/frequency arrays retained; independent numerical/provenance rederivation.
- **Addressed free:** distinguish numerical full rank, pre-Adam matching and
  measured displacement; preserve finite-precision residuals and audit failure.
- **Residual design limitation:** native2000/2500 references are archived and
  noncontemporaneous, whereas norm-matched versus orthogonal is within the new
  acquisition. Known CUDA trajectory sensitivity remains; a concurrently
  repeated native control would require a separately authorized new design,
  not quiet replay of this completed batch.
- **Residual scope:** five paired seeds, one small modular-addition architecture,
  late fork1500 and fixed horizons2000/2500. No new power, CI, p-value,
  equivalence or broad generalization claim was introduced. Fresh-task or
  initialization-to-endpoint generalization requires an explicitly different
  experiment; no such work was launched here.
- **Residual causal limitation:** common initial action geometry does not fix
  subsequent subspaces or Adam histories; no isolated causal mediation fraction.

## Exact audit artifacts and resource handles

Raw parent: `/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD`.
All three JSON audit artifacts remain there, with full source/input receipts.

| Artifact | SHA256 |
|---|---|
|batch-complete.json|`22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45`|
|measurement-001/complete.json|`fe11e1404e8ab20c792d8d441477f325f48a2bb9fdc4c868620a3ffeeafbb9ee`|
|analysis-001/summary.json|`249a8b9f5b680f72f7e9a402f56a9a4734814d99dc7d8097442eac4c965aa237`|
|tensor-audit-001.json|`c1534ef2efabea41ac7f2edf596df1e7582a957223ccb43aca328f6e9aebb6a8`|
|readout-audit-001.json (original failure preserved)|`1ae2f3d60d15a4f6f87c5818ddd1befdcee541f9fcc3d4e2d1816f106b116425`|
|schema-audit-supplement-001.json|`a919f385c1c8e3d32cb9b2279e0e4659b8d5c8d1c1ec10fc16d0802bba258945`|

Script SHA256s respectively:
`f4511e99c9ded00cd0c7d3b4c207d93effcfc1df169851d1f64b811506681dea`,
`a3921ac10c111dea3ec0d1c80231f23a0df27cb4f86bc1a3b2e8ed70014d14db`,
`2f1274e0ce35578fab7a47273072863e6f414cac72e83fb56e653e9f38bb047d`.

CPU-only `Type=exec` services, `Restart=no`, `KillMode=control-group`, no swap:

- `spectral-action-tensor-audit-001.service`: invocation
  `cfddd7e45483442bba073fc0e3fbb8f3`, PID2877423,4GiB/one CPU/10minutes;
  successful terminal13:54:35BST.
- `spectral-action-readout-audit-001.service`: invocation
  `f8adc4dbcc0745748c9d0ba272991682`, PID2880786,4GiB/one CPU/10minutes;
  terminal13:59:00BST with the sole preserved auditor comparison failure.
- `spectral-action-schema-audit-001.service`: invocation
  `699e71ccad8d4b26b28f725a31578184`, PID2882150,2GiB/one CPU/5minutes;
  successful terminal13:59:38BST. No cloud spend or GPU work.

RSS figures above come from process `getrusage`; transient systemd journal
memory-peak lines underreported this usage and are not used as measurement.
