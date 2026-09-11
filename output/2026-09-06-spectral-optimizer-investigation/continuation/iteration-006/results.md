# Iteration 006 continuation results

> **Status:** Final, [parent-reviewed](parent-review.md) continuation result record. The completed independent raw-artifact audit reports PASS.

## Headline

In the predeclared three-seed primary comparison, lagged delivery was lower than current delivery in all three clean pairs and in two of three noisy pairs. Rescaling the lagged projection to the contemporaneous current projection's norm, both computed on the restored arm's own state, did not reverse that pattern: its primary mean contrast was negative in both conditions. The independently seeded bundle agreed on the clean sign but had positive noisy contrasts, including +2.47 percentage points for restored lagging. Those fresh values are reported separately and are not pooled with the primary seeds.

Accuracy cells below are percentages; deltas are treatment minus control in percentage points. Each primary cell is `treatment − control = delta` at that arm's maximum-validation-accuracy checkpoint.

| Condition and contrast | Seed 6 | Seed 7 | Seed 8 | Primary mean | Median | Range | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean, lagged − current | 87.59 − 87.92 = **−0.33** | 87.68 − 87.88 = **−0.20** | 88.16 − 88.51 = **−0.35** | **−0.293** | −0.330 | [−0.350, −0.200] | 0.081 |
| Clean, restored-lagged − current | 87.49 − 87.92 = **−0.43** | 87.68 − 87.88 = **−0.20** | 88.15 − 88.51 = **−0.36** | **−0.330** | −0.360 | [−0.430, −0.200] | 0.118 |
| Noisy, lagged − current | 50.70 − 54.48 = **−3.78** | 54.19 − 61.13 = **−6.94** | 55.25 − 55.01 = **+0.24** | **−3.493** | −3.780 | [−6.940, +0.240] | 3.599 |
| Noisy, restored-lagged − current | 48.67 − 54.48 = **−5.81** | 55.29 − 61.13 = **−5.84** | 54.55 − 55.01 = **−0.46** | **−4.037** | −5.810 | [−5.840, −0.460] | 3.098 |

The fresh bundle used independent `SeedSequence([20260906, 60006, stream])` streams. It is one additional bundle, not a fourth primary seed:

| Fresh condition and contrast | Control checkpoint | Treatment checkpoint | Fresh delta |
|---|---:|---:|---:|
| Clean, lagged − current | current 88.42% at step 2000 | lagged 86.30% at step 800 | **−2.12 pp** |
| Clean, restored-lagged − current | current 88.42% at step 2000 | restored 86.24% at step 800 | **−2.18 pp** |
| Noisy, lagged − current | current 54.80% at step 1200 | lagged 55.01% at step 1500 | **+0.21 pp** |
| Noisy, restored-lagged − current | current 54.80% at step 1200 | restored 57.27% at step 1300 | **+2.47 pp** |

**HONEST-NEGATIVE:** no benefit from lagging was consistent across the primary and fresh bundles. This is not a statistically powered claim of zero effect. The results also do not identify which part of the whole optimizer policy caused the differences.

## Scope and evidence

The primary execution comprised 36 cells: six arms × two noise conditions × three paired seeds (6, 7, 8), with 2,000 updates and four retained checkpoints per cell. The independent execution comprised six cells at bundle 60006 for current, lagged, and norm-restored lagged arms. Both used the same 784–64–10 network and the same 5,000/5,000/5,000 split recipe, but each bundle generated its own independent split. They shared the official MNIST test set. Thus the evidence is three primary seeds plus one independent bundle on one model and one dataset; repeated checkpoint use of the same official test set limits broader inference.

The primary execution manifest is bound by SHA-256 `b44ac3901b286cfd202dcb956979facf5cbfb7316f5dec8f7d3e3182ac7f998e`; the independent execution manifest is bound by `77c63fa4741eccbc884e63472d08932beb69075462297e5213526d73334f0799`. Exact summary rows are in [summary.json](summary.json), and the fresh raw artifacts are indexed by audit-reexecution/execution.json (artifact not distributed in this public snapshot).

## Checkpoint selection and fixed endpoints

The primary estimand selects each arm independently by validation data. `Max` is the maximum-validation-accuracy selector; `Min CE` is the minimum-validation-cross-entropy selector. Entries in the step columns are `training step [post-warmup exposure]` for seeds 6/7/8. Reported accuracies are mean official-test percentages and CE values are means over seeds.

| Noise | Arm | Max steps [exposure] | Test at Max: acc / CE | Min-CE steps [exposure] | Test at Min CE: acc / CE |
|---:|---|---|---:|---|---:|
| 0 | AdamW | 1800/2000/1900 [1700/1900/1800] | 93.160 / 0.241880 | 1600/2000/2000 [1500/1900/1900] | 93.127 / 0.240977 |
| 0 | Current-32 | 1800/2000/1800 [1700/1900/1700] | 88.103 / 0.422957 | 1800/2000/1900 [1700/1900/1800] | 88.257 / 0.422047 |
| 0 | Lagged-32 | 1000/100/200 [900/0/100] | 87.810 / 0.466392 | 1000/900/1000 [900/800/900] | 87.487 / 0.446345 |
| 0 | Restored lagged-32 | 1000/100/200 [900/0/100] | 87.773 / 0.466945 | 1000/900/1000 [900/800/900] | 87.500 / 0.446647 |
| 0 | Scalar current | 1800/2000/1900 [1700/1900/1800] | 93.130 / 0.242224 | 1600/2000/2000 [1500/1900/1900] | 93.090 / 0.241078 |
| 0 | Scalar lagged | 1900/2000/2000 [1800/1900/1900] | 93.033 / 0.241016 | 1600/2000/2000 [1500/1900/1900] | 93.050 / 0.240885 |
| 0.9 | AdamW | 200/300/100 [100/200/0] | 52.047 / 1.976102 | 700/600/400 [600/500/300] | 47.987 / 1.887186 |
| 0.9 | Current-32 | 1400/1000/600 [1300/900/500] | 56.873 / 1.990360 | 800/1800/1800 [700/1700/1700] | 54.003 / 1.957581 |
| 0.9 | Lagged-32 | 1400/1000/700 [1300/900/600] | 53.380 / 2.021677 | 600/1300/1800 [500/1200/1700] | 49.903 / 2.006884 |
| 0.9 | Restored lagged-32 | 1400/1100/700 [1300/1000/600] | 52.837 / 2.038644 | 600/1800/1600 [500/1700/1500] | 47.180 / 2.009554 |
| 0.9 | Scalar current | 200/300/100 [100/200/0] | 52.683 / 1.975267 | 700/1300/400 [600/1200/300] | 44.653 / 1.880412 |
| 0.9 | Scalar lagged | 200/300/100 [100/200/0] | 53.557 / 1.980143 | 700/1100/400 [600/1000/300] | 46.100 / 1.881537 |

The fixed endpoints separate checkpoint selection from exposure. All arms shared the same step-100 warmup state within each condition and seed.

| Noise | Arm | Warmup test acc | Final test acc / CE | Final train clean/noisy acc | Final noisy-label CE |
|---:|---|---:|---:|---:|---:|
| 0 | AdamW | 87.473 | 93.203 / 0.241000 | 99.253 / 99.253 | 0.045936 |
| 0 | Current-32 | 87.473 | 88.210 / 0.425010 | 88.607 / 88.607 | 0.418880 |
| 0 | Lagged-32 | 87.473 | 86.803 / 0.456860 | 86.933 / 86.933 | 0.461283 |
| 0 | Restored lagged-32 | 87.473 | 87.070 / 0.453994 | 87.180 / 87.180 | 0.459238 |
| 0 | Scalar current | 87.473 | 93.160 / 0.240791 | 99.233 / 99.233 | 0.047710 |
| 0 | Scalar lagged | 87.473 | 93.167 / 0.240498 | 99.153 / 99.153 | 0.049130 |
| 0.9 | AdamW | 45.543 | 29.580 / 2.073533 | 29.907 / 43.733 | 1.669717 |
| 0.9 | Current-32 | 45.543 | 47.920 / 1.997856 | 48.367 / 17.700 | 2.253690 |
| 0.9 | Lagged-32 | 45.543 | 44.757 / 2.041600 | 44.960 / 16.340 | 2.270790 |
| 0.9 | Restored lagged-32 | 45.543 | 44.490 / 2.049051 | 44.673 / 16.287 | 2.272038 |
| 0.9 | Scalar current | 45.543 | 29.903 / 2.074155 | 30.213 / 43.913 | 1.677994 |
| 0.9 | Scalar lagged | 45.543 | 30.807 / 2.059701 | 30.827 / 43.053 | 1.697904 |

The independent bundle showed the same selector tradeoff and was not pooled:

| Noise | Arm | Max step [exposure], acc / CE | Min-CE step [exposure], acc / CE | Final acc / CE | Warmup acc |
|---:|---|---:|---:|---:|---:|
| 0 | Current-32 | 2000 [1900], 88.42 / 0.427426 | 2000 [1900], 88.42 / 0.427426 | 88.42 / 0.427426 | 86.82 |
| 0 | Lagged-32 | 800 [700], 86.30 / 0.480595 | 600 [500], 86.80 / 0.462089 | 86.43 / 0.467878 | 86.82 |
| 0 | Restored lagged-32 | 800 [700], 86.24 / 0.480944 | 600 [500], 86.79 / 0.462770 | 86.47 / 0.470400 | 86.82 |
| 0.9 | Current-32 | 1200 [1100], 54.80 / 2.008797 | 1500 [1400], 53.44 / 1.968927 | 48.91 / 2.004321 | 45.64 |
| 0.9 | Lagged-32 | 1500 [1400], 55.01 / 2.011126 | 1500 [1400], 55.01 / 2.011126 | 49.72 / 2.043237 | 45.64 |
| 0.9 | Restored lagged-32 | 1300 [1200], 57.27 / 2.010556 | 1500 [1400], 56.63 / 1.972707 | 52.09 / 2.036797 | 45.64 |

Cross-entropy and accuracy did not select the same exposure. For example, primary noisy Current-32 moved from 56.873% / 1.990360 at the accuracy-selected checkpoints to 54.003% / 1.957581 at the CE-selected checkpoints. Clean lagged arms likewise had lower CE but slightly lower mean test accuracy at the CE selector. These are two predeclared summaries, not interchangeable rankings.

At the noisy final checkpoint, the projected-gradient arms fitted corrupted labels much less than AdamW and the scalar controls: 16.29–17.70% noisy-label training accuracy versus 43.05–43.91%, while their clean-label training accuracy was 44.67–48.37% versus 29.91–30.83%. The fresh bundle was directionally similar (15.84–17.66% noisy-label accuracy). This is a memorization tradeoff under the defined corruption process; it does not establish a unique denoising mechanism.

Primary noisy Current-32 exceeded AdamW by 4.827 pp at the accuracy-selected checkpoint (seed deltas +3.68, +5.65, +5.15 pp) and by 18.340 pp at the final checkpoint. This seed-6–8 result conflicts with iteration 004's seed-3–5 result, where the corresponding selected hard-32/current contrast against AdamW was −2.38 pp. The cross-iteration sign change is preserved as unresolved instability rather than reconciled away.

## Current-versus-prior retention

Retention values are fractions of squared gradient norm retained by the projection; cosines compare projections of the same current gradient through the current and previous bases. Means were formed seed-first. Each observing arm contributed 19 scheduled-calendar steps and 1,881 other post-warmup steps per seed: 57 and 5,643 seed-step observations per arm-condition group. These are calendar-partition counts, not counts of repair versus no-repair events.

| Noise | Arm | Scheduled current / prior / difference | Scheduled cosine | Other current / prior / difference | Other cosine |
|---:|---|---:|---:|---:|---:|
| 0 | Current-32 | 0.9257 / 0.8648 / 0.0609 | 0.9888 | 0.9220 / 0.8527 / 0.0693 | 0.9860 |
| 0 | Lagged-32 | 0.9208 / 0.8546 / 0.0662 | 0.9876 | 0.9168 / 0.8426 / 0.0742 | 0.9848 |
| 0 | Restored lagged-32 | 0.9202 / 0.8536 / 0.0666 | 0.9874 | 0.9169 / 0.8428 / 0.0741 | 0.9848 |
| 0 | Scalar current | 0.8760 / 0.7692 / 0.1068 | 0.9752 | 0.8664 / 0.7606 / 0.1058 | 0.9765 |
| 0 | Scalar lagged | 0.8763 / 0.7680 / 0.1083 | 0.9742 | 0.8669 / 0.7608 / 0.1060 | 0.9764 |
| 0.9 | Current-32 | 0.8030 / 0.4786 / 0.3244 | 0.8783 | 0.8092 / 0.4780 / 0.3312 | 0.8728 |
| 0.9 | Lagged-32 | 0.7919 / 0.4981 / 0.2938 | 0.9047 | 0.7920 / 0.4915 / 0.3005 | 0.8990 |
| 0.9 | Restored lagged-32 | 0.7842 / 0.4891 / 0.2951 | 0.9019 | 0.7882 / 0.4881 / 0.3000 | 0.9005 |
| 0.9 | Scalar current | 0.8395 / 0.4063 / 0.4331 | 0.8027 | 0.8281 / 0.4155 / 0.4127 | 0.8201 |
| 0.9 | Scalar lagged | 0.8160 / 0.4049 / 0.4111 | 0.8228 | 0.8271 / 0.4155 / 0.4116 | 0.8215 |

Current-basis retention exceeded previous-basis retention in every listed group. Scheduled-calendar steps were not uniformly more discrepant than other steps: the difference was smaller there for all three clean hard-projection arms and all three noisy hard-projection arms. Drift-triggered repairs may occur among the 1,881 “other” steps, so this is explicitly not a repair/no-repair partition. The phase summaries are descriptive calendar diagnostics rather than an isolated intervention effect.

## Gradient and update geometry

`Grad ratio` is the norm of the delivered pre-Adam gradient divided by the raw gradient norm. For the restored arm, the rescaling target is the norm of the contemporaneous current projection evaluated on that same arm's own state—not the raw-gradient norm and not the Current-32 arm's cross-trajectory norm. It is not the norm of the AdamW parameter displacement. `Total/decay-sub step` reports actual optimizer-step norms; the two are nearly identical here because explicit decay was small. Dot-product cosines and ascent fractions are directional diagnostics, not finite loss changes.

| Noise | Arm | Grad ratio | Total / decay-sub step norm | Raw / applied cosine | Raw / applied ascent fraction |
|---:|---|---:|---:|---:|---:|
| 0 | AdamW | 1.0000 | 0.036116 / 0.036148 | −0.2250 / −0.2250 | 0.0096 / 0.0096 |
| 0 | Current-32 | 0.9599 | 0.040944 / 0.040946 | −0.1634 / −0.1637 | 0.0263 / 0.0325 |
| 0 | Lagged-32 | 0.9172 | 0.041104 / 0.041107 | −0.1341 / −0.1588 | 0.0537 / 0.0460 |
| 0 | Restored lagged-32 | 0.9573 | 0.041220 / 0.041223 | −0.1338 / −0.1587 | 0.0518 / 0.0428 |
| 0 | Scalar current | 0.9297 | 0.035822 / 0.035854 | −0.2223 / −0.2223 | 0.0123 / 0.0123 |
| 0 | Scalar lagged | 0.8708 | 0.035416 / 0.035448 | −0.2226 / −0.2226 | 0.0107 / 0.0107 |
| 0.9 | AdamW | 1.0000 | 0.055523 / 0.055552 | −0.2977 / −0.2977 | 0 / 0 |
| 0.9 | Current-32 | 0.8985 | 0.043200 / 0.043200 | −0.2552 / −0.2681 | 0.0014 / 0.0040 |
| 0.9 | Lagged-32 | 0.6979 | 0.035081 / 0.035082 | −0.1433 / −0.2280 | 0.0412 / 0.0328 |
| 0.9 | Restored lagged-32 | 0.8867 | 0.035466 / 0.035467 | −0.1415 / −0.2282 | 0.0391 / 0.0268 |
| 0.9 | Scalar current | 0.9092 | 0.055001 / 0.055030 | −0.2961 / −0.2961 | 0 / 0 |
| 0.9 | Scalar lagged | 0.6404 | 0.052910 / 0.052938 | −0.2926 / −0.2926 | 0 / 0 |

Per-step relative-error gates validated the intended own-state equality between the restored lagged projection norm and that arm's contemporaneous current projection norm. The cross-arm grad ratios in the table are descriptive and are not evidence for that invariant because the arms follow different states. Restoration did not match actual parameter steps: in noisy data, restored and Current-32 decay-subtracted step norms were 0.035467 and 0.043200. Their early-to-late step norms also differed: 0.043218→0.032532 for restored versus 0.048546→0.039125 for Current-32. In clean data the corresponding paths were 0.041148→0.042455 and 0.041076→0.041717. AdamW's nonlinear state makes validated pre-Adam projection-norm matching distinct from update matching.

Consequently, the experiment cannot isolate a causal mediator. Gradient direction, delivered magnitude, optimizer-state trajectory, selected exposure, and actual parameter displacement remain coupled features of each whole policy. The scalar controls are own-state controls and do not cross-arm match those trajectories.

## Verification and completed audit

CPU checkpoint replay passed for all 360 primary evaluations and all 60 independent-bundle evaluations, with exact accuracy reconstruction in 420/420 cases. Maximum absolute CE reconstruction error was `7.915496835764202e-08` for primary and `9.630918507141928e-08` for the independent bundle; see checkpoint-replay.json (artifact not distributed in this public snapshot) and [checkpoint-replay-audit.json](checkpoint-replay-audit.json).

Those checks verify 420 checkpoint-to-metric evaluations. The completed independent raw audit separately reports PASS: all primary sources and rows matched; 171 metrics across 36 cells, 2,052 arm-group recomputations, and 5,130 paired recomputations matched exactly. For the independent bundle, all sources and rows, all six cells, and all four contrasts matched. The audit and checkpoint replay therefore cover different failure modes.

## Limitations triage

| Limitation | Consequence | Disposition |
|---|---|---|
| Raw-artifact or selector implementation error | Could change the reported numbers | **Addressed:** independent source/row, metric, arm-group, pair, and fresh-contrast audit passed |
| Three primary seeds plus one unpooled bundle | Noisy contrasts are heterogeneous, and the fresh noisy signs disagree with the primary means | **Reported now; defer extension:** predeclare any further bundles and retain every result |
| One MNIST model and repeated use of the same official test set | Evidence does not establish transfer across architectures, datasets, or data regimes; repeated checkpoint reporting can induce test adaptivity | **Deferred to a new benchmark study** with a fresh validation/test protocol |
| Restoring pre-Adam norm does not restore actual optimizer displacement | The study cannot attribute outcomes specifically to direction versus magnitude | **Deferred to a targeted mechanism experiment** with reciprocal and displacement-aware controls |
| Primary noisy Current-32 versus AdamW reverses the iteration-004 sign | The baseline comparison is not stable across iterations | **Preserved as a contradiction;** do not use either iteration alone as a general claim |
| Scheduled versus other-step phase split is observational | Phase differences cannot establish that scheduled repair caused the learning effect | **Interpretation constrained now;** a causal phase intervention would require a new design |

## Higher-level summary

**Codex | Spectral Optimizer Investigation.** Using the filter's previous state did not yield a dependable improvement. It reduced performance on clean data; on noisy data, the result changed direction with the random seed. The existing filter still limits late memorization of incorrect labels, but its advantage over stopping ordinary training early remains uncertain. All 42 training runs were independently audited, including 420 reproduced accuracy measurements. Keep the current default and test stability under a plan fixed before seeing new results. New experiments require their own recorded launch decision.
