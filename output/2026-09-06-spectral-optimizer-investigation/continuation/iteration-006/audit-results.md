# Results Audit — iteration 006

## Overall disposition: HONEST-NEGATIVE
## audit_exit_reason: beneficial-lagging-not-supported-and-noisy-direction-seed-unstable

Artifact audit verdict: **PASS**.

This is a CPU, raw-artifact audit. It imports neither the producer nor the frozen summarizer, performs no training, and does not duplicate the parent-owned checkpoint forward replay.

Verified 15 scientific source bindings, 111 full-run bulk bindings, 36 final raw runs, 36 pretest/final identities, and 72000 ordered step indices.

Independent aggregation covers 171 scalar metrics per cell, 2052 arm summaries, 5130 paired metric contrasts, and 4 prespecified primary groups. Exact null-step masks are part of every paired geometry comparison.

## Four prespecified primary groups

| Noise | Contrast | Seed differences | Mean | Median | Min | Max | Sample SD |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | lagged32-current32 | 6:-0.003300, 7:-0.002000, 8:-0.003500 | -0.002933 | -0.003300 | -0.003500 | -0.002000 | 0.000814 |
| 0 | lagged32_current_norm-current32 | 6:-0.004300, 7:-0.002000, 8:-0.003600 | -0.003300 | -0.003600 | -0.004300 | -0.002000 | 0.001179 |
| 0.9 | lagged32-current32 | 6:-0.037800, 7:-0.069400, 8:+0.002400 | -0.034933 | -0.037800 | -0.069400 | +0.002400 | 0.035986 |
| 0.9 | lagged32_current_norm-current32 | 6:-0.058100, 7:-0.058400, 8:-0.004600 | -0.040367 | -0.058100 | -0.058400 | -0.004600 | 0.030975 |

Accuracy values and differences are fractions. These are three paired seed bundles per condition; no confidence intervals, p-values, equivalence claims, or population claims are made.

## Re-execution summary

The separately frozen bundle 60006 completed six fresh current/lagged cells and 24 test evaluations. Its source, plan, data, 21 bulk bindings, six pretest/final identities, 12,000 step rows, and stored four-contrast file were independently checked here. It is not pooled with the three primary seeds.

| Noise | Fresh contrast | Accuracy difference |
|---:|---|---:|
| 0 | lagged32-current32 | -0.021200 |
| 0 | lagged32_current_norm-current32 | -0.021800 |
| 0.9 | lagged32-current32 | +0.002100 |
| 0.9 | lagged32_current_norm-current32 | +0.024700 |

The fresh clean contrasts remain adverse. Both fresh noisy contrasts reverse the signs of their corresponding three-seed primary means. Thus the artifact arithmetic is supported, but a beneficial lagging claim is not: the noisy direction is seed-unstable and the clean evidence is adverse.

## Findings

### Finding 1 — primary experiment — artifact integrity and arithmetic

- **Verdict**: SUPPORTED
- **Severity**: Low
- **Evidence**: all 36 raw runs, 36 immutable pretest records, 111 bound bulk artifacts, 72,000 ordered step rows, 171 seed-level metrics, and 5,130 paired metric contrasts agree with independent calculations. The terminal log records `FULL_PROCESS_EXIT=0 FULL_PROCESS_ELAPSED_SECONDS=624.57`; the separate CPU verifier reproduced all 360 primary and 60 fresh accuracy evaluations exactly, with maximum CE discrepancies below 1e-7.
- **Would fixing this plausibly flip PASS/FAIL?**: No.
- **Finding to hand back**: no raw-artifact or aggregation defect found.

### Finding 2 — primary plus fresh bundle — beneficial lagging is not supported

- **Verdict**: TRUE-NULL
- **Severity**: High
- **Evidence**: primary clean means are adverse for lagged (-0.002933) and restored lagged (-0.003300); primary noisy means are adverse (-0.034933 and -0.040367), while the separately frozen fresh noisy bundle is positive (+0.002100 and +0.024700). The bound fresh log ends with `audit_reexecution_complete` and `test_evaluations: 24`.
- **Would fixing this plausibly flip PASS/FAIL?**: No identifiable implementation or analysis defect exists to fix; additional bundles could change the descriptive pattern but would answer a larger replication question.
- **Finding to hand back**: the fixed study does not establish a robust learning benefit from lagged delivery; noisy effects are seed-unstable and clean effects are adverse.

## Unresolved findings for the write-up

The evidence is limited to one small MNIST recipe, three primary paired bundles plus one separately reported fresh bundle, reused validation/test data, and validation-accuracy-selected checkpoints. The fresh noisy sign reversals must be reported alongside the primary adverse means.

## Limitation triage

| Limitation | Disposition | If fixable now: what + cost | If future-work: resources a fix would need |
|---|---|---|---|
| Only three primary paired bundles | future-work | — | Prospectively freeze additional independent bundles; each six-cell current/lagged-only bundle costs roughly two GPU-minutes on the same RTX 3090. |
| One small MNIST model/recipe | future-work | — | A separately designed study on a larger model and at least one non-MNIST dataset; hardware/time depend on that design. |
| Reused official test set and adaptive research history | future-work | — | A genuinely untouched dataset or locked external evaluation not previously used in this research line. |
| Checkpoint numerical replay is separate | fix-now-free | Parent-owned CPU verifier, already run; reference its independent replay record in the final synthesis. | — |
| No uncertainty interval by prospective rule | future-work | — | More prospectively sampled bundles sufficient for a predeclared inferential analysis; do not retrofit a CI to three bundled seeds. |

The primary artifact math is approved. Scientific `SUPPORTED` is not earned for a beneficial lagging claim; the appropriate outcome is an honest adverse/mixed result, not another tuning loop.

[Machine-readable audit](audit-results.json), SHA256 `5a8a2f95bb4a4db148aa6b2bceb141d5be04b7093703a2a1e6b2f7e2fe22a048`.
