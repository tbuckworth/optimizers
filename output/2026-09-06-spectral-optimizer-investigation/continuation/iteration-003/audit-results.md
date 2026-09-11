# Independent result audit — iteration 003

**Verdict: pass within the replay scope below.** This is a three-seed,
fixed-configuration measurement study, not a tuned optimizer benchmark or a
causal demonstration that a learned subspace selectively removes label noise.
No discrepancy was found between independent scalar reaggregation, the frozen
summary, and independently evaluated retained checkpoints.

## Evidence and actual coverage

The standalone [audit script](audit_completed_results.py) imports neither the
training harness nor its summarizer. It first requires the completed
confirmatory/18-run marker before reading outcomes or official test data. The
[machine-readable audit](audit-results.json) records every per-seed metric,
paired contrast, checkpoint replay, and source/raw-result hash. The audit ran
2026-09-06 11:04:26–11:04:32 UTC on CPU; it performed no optimizer training.

- All 18 conditions are present: three seeds × two replacement probabilities
  × three optimizer arms. Each has 2,000 ordered update rows, 42 probe rows,
  and 21 validation evaluations: 36,000 updates, 756 probes and 378 validation
  records in total. Rank caps, delivered rank, identity warmup and recorded
  runtime gates agree with the frozen design.
- Reconstructed all three saved RNG plans exactly, including disjoint splits,
  initialization seeds, corruption uniforms/digits, training order and both
  independent probe-batch streams. Within a seed, all arms share these plans.
  At nominal 0.9 replacement, seeds 0/1/2 replace respectively
  4,494/4,517/4,476 of 5,000 labels; 4,011/4,084/4,016 labels actually change
  (80.22%/81.68%/80.32%). Uniform replacement can reproduce the clean digit.
- Recomputed retention ratios from stored energies, null-denominator handling,
  clean-minus-corruption retention, joint-energy closure arithmetic, signed
  update decompositions, thresholded event classifications, closure-robust
  reversals, and finite same-batch loss changes. All-postwarmup denominators
  are 1,900 update rows and 39 probes per run, not independent replicates.
- Independently reaggregated all 122 metrics per condition: 2,196 per-seed
  values and 732 paired metric contrasts agree with the producer summary.
  Supplementary read-only checks also compared every paired seed difference,
  ordering, and minimum/maximum, not just the paired mean.
- Verified the selected step is the **earliest minimum validation cross-entropy**
  in every run. Independently evaluated all 36 retained states (final and
  selected for each condition) on the 10,000-example official test set and
  the corresponding 5,000-example clean validation set: **72 evaluations**.
  Direct tensor inference used CPU float32, one Torch thread, and
  `torch.load(..., weights_only=True)`. Every accuracy matches exactly;
  the largest absolute cross-entropy difference is `7.324218764814816e-8`.
  The prospectively declared comparison tolerances were `5e-5` CE and one
  example per evaluation set. These tolerances are not a proven rounding bound.

## Findings preserved by the audit

The primary noisy-label contrast is estimate128/project32 minus
estimate32/project32, using seed-first means of probe retention ratios.

| Seed | Clean retention difference | Corruption-residual retention difference | Selectivity difference |
|---|---:|---:|---:|
| 0 | +0.014583 | +0.015613 | −0.001031 |
| 1 | +0.016111 | +0.015481 | +0.000630 |
| 2 | −0.006522 | −0.006178 | −0.000344 |
| Mean | +0.008057 | +0.008305 | **−0.000248** |

Thus, widening estimation did not improve the prespecified clean-versus-
corruption retention difference on average. The mixed, small differences do
not establish equivalence. The residual is the difference between fixed noisy-
label and clean-label gradients at the current model, not an independent,
orthogonal noise component; stored cross terms are retained in the audit.

At 0.9 replacement, test results averaged over the three paired seeds are:

| Arm | Final accuracy | Validation-CE-selected accuracy | Validation-CE-selected test CE |
|---|---:|---:|---:|
| AdamW | 30.67% | 44.85% | **1.876904** |
| Estimate32/project32 | 49.79% | 48.80% | 1.976578 |
| Estimate128/project32 | 52.93% | 51.65% | 1.960750 |

## Provenance and limits

The full source/protocol freeze is commit
`d3e93cce20325e84a0ab98dafb9fa62c9289cf80`, timestamp 10:58:12 UTC.
Recorded training starts at 10:59:21.292583 UTC and all training ends at
11:03:04.627460; test is first loaded at 11:03:04.668406 and execution completes
at 11:03:07.104039. Source and analysis hashes match the freeze and current
files. A supplementary check confirms the canonical optimizer Git blob also
matches and the passing pilot completed at 10:53:19.711809 with the same
source/protocol/test/analysis hashes. Exactly 18 condition JSON files exist,
with no partial-failure record. Source, commit and recorded timestamps support
the sequence; this is not an independent operating-system data-access log.

SHA-256 identifiers:

- Audit script: `b2fa26f8973df2e1ae7776dbdb55011d2593ccee97d09d70f98120d5a0c64f81`
- Audit JSON: `73eb790a1c676e45fdaa434a08e3b71348a75cad3c7327cf8ae7722c676e2d6e`
- Producer summary: `e52fca3be27a7bbb19b614d84abec63c31a52ab15e1d4438eb863add635ae555`
- Execution record: `aead1a8976ee624c5cdce605fb66bf2fc72d15749f32d033af49bc8c2ae4362a`

Only saved scalar/norm records were reaggregated for gradient geometry. Full
per-step vectors, bases and intermediate checkpoints are absent, so this is
**not** an independent replay of training, probe-gradient computation, or every
projector/update identity from underlying vectors. CPU checkpoint evaluation
independently supports final/selected loss and accuracy, not those unsaved
trajectory quantities. The adaptive basis includes the current training
gradient; independent probe sampling does not make the basis exogenous. Arms
subsequently follow different parameter trajectories, so retention differences
are conditional on their own states. Three seed bundles combine split,
corruption, initialization and sampling variation; multiple reported metrics
and dependent time points provide no extra independent seeds. No universal
Adam convergence claim, neural-frequency claim beyond this configuration, or
causal explanation of the earlier studies follows.

The researcher-review guidance informed the distinction between direct
checkpoint replay and interpretation. Original evidence and frozen source
were not modified; this note and the new audit artifacts are the only outputs
of this audit task. No commit was made by the auditor.
