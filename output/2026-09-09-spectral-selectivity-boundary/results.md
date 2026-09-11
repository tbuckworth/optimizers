# Useful protection has a rare-learning and coherent-cue boundary

Codex — Spectral Optimizer Investigation, 10 September2026.
**Evidence: three paired seeds, one fixed small-MNIST protocol; independently
checked saved results.** Not a tuned benchmark, an emergent-misalignment result
or proof about all covariance filters.

## Main finding

The current spectral filter does something useful: under scattered wrong
labels it retains much more common-digit accuracy and fits far fewer wrong
targets. But that protection is not a general test of usefulness or truth.
The same policy learns a newly introduced, correctly labelled rare digit poorly
and readily learns a shared misleading patch cue. Its raw-direction control,
following the same functional norm rule on its own state, behaves much more
like ordinary AdamW. Directional restriction matters in this recipe; identical
scalar histories or Adam step lengths were not imposed.

The strongest constructive interpretation is **conditional preservation of
previously learned structure**, with a substantial cost to acquiring new rare
recognition. The strongest boundary is that a misleading pattern can be easy
for the restricted model to learn. Neither finding makes the original idea
worthless; together they motivate studying *which structure is retained*,
instead of treating high covariance as a label for benign generalization.

## Registered endpoint: useful recognition

Means at update2000. Common is macro-average accuracy/CE across the nine
non-8 classes; rare is digit8. CE is cross-entropy, lower is better. All held-out
images in this table are unpatched. Full seed values, sample SD/SE, all class
metrics and contrasts are in [checked-summary.json](results/checked-summary.json).

| Cell | Policy | Common accuracy | Common CE | Rare accuracy | Rare CE |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean | AdamW | 93.37% | 0.2453 | 52.00% | 1.7114 |
| Clean | Spectral | 88.69% | 0.3932 | 4.93% | 2.8118 |
| Clean | Own-norm raw direction | 93.30% | 0.2443 | 51.53% | 1.7225 |
| Diffuse wrong labels | AdamW | 33.97% | 1.8741 | 51.73% | 2.2259 |
| Diffuse wrong labels | Spectral | 62.01% | 1.8750 | 0.00% | 4.5669 |
| Diffuse wrong labels | Own-norm raw direction | 34.41% | 1.8780 | 50.20% | 2.3023 |
| Shared wrong cue | AdamW | 92.76% | 0.2605 | 50.60% | 1.7389 |
| Shared wrong cue | Spectral | 87.34% | 0.4718 | 3.53% | 2.9298 |
| Shared wrong cue | Own-norm raw direction | 92.78% | 0.2595 | 49.73% | 1.7440 |
| Matched sham | AdamW | 89.30% | 0.3800 | 54.00% | 1.5746 |
| Matched sham | Spectral | 85.07% | 0.5634 | 0.00% | 3.2715 |
| Matched sham | Own-norm raw direction | 89.40% | 0.3773 | 54.20% | 1.5768 |

![All seed learning curves, common and rare classification.](results/learning.png)

The diffuse common-accuracy benefit is positive in every seed: spectral minus
AdamW **+28.18,+23.38,+32.56 percentage points**. Wrong-target training accuracy
is6.22% versus32.48%. However common CE is almost unchanged in mean and its
paired signs are mixed; balanced CE, including rare examples, is worse
(2.1442 versus1.9092). Better classification rankings do not establish better
probability predictions. The favorable accuracy effect and this CE limitation
both remain part of the result.

Spectral rare accuracy is lower and rare CE higher than both raw policies in
every seed under all four cells. Clean rare accuracies are14.4%,0.4%,0.0%
versus AdamW52.2%,50.8%,53.0%. Under Diffuse and Sham, all three spectral
endpoints have zero rare accuracy, including zero accuracy on the50 rare
training examples. This is not solely a rare held-out generalization failure.

### What changed from the common warmup?

Warmup common accuracy averaged87.76%; rare accuracy was0% in every seed.
Clean spectral improves common accuracy by0.93points and rare by4.93points,
versus AdamW5.61/52.00points. Under diffuse corruption spectral common accuracy
falls25.76points, versus AdamW53.79points: its relative benefit is preservation
against larger deterioration, not absolute accuracy improvement.

Importantly, **zero rare accuracy is not zero learning**. Rare warmup CE was
10.0054,9.9292,8.4836 across seeds (mean9.4727). Spectral improves rare CE in
every cell and seed; under Diffuse its mean reaches4.5669 despite no correct
argmax predictions. The model moves toward the new class but much less
successfully than raw policies. It is neither frozen nor an absolute ban on
rare information. The warmup excluded digit8, so old negative-class training,
novelty, class identity and low support are entangled.

## A shared misleading cue passes through

The cue is a white3×3patch associated with the same500 incorrect target-0
labels. Shared/Sham have identical assigned targets and exact per-true-digit
patch counts; the sham approximately removes their association within digit.
The primary population is4,000 matched held-out nonzero common-digit images
within each seed. Patch excess is patched minus unpatched target-0 prediction
rate, not raw attack success including pre-existing target bias.

| Policy | Shared patch excess | Sham patch excess | Shared wrong-target fitting |
| --- | ---: | ---: | ---: |
| AdamW | +98.933points | −0.158points | 100.00% |
| Spectral | +94.825points | −0.908points | 96.87% |
| Own-norm raw direction | +98.925points | −0.083points | 100.00% |

![All seed patch effects and the registered interaction.](results/cue.png)

Spectral reduces the primary Shared-minus-Sham interaction versus AdamW by
**3.358points** (seed differences−2.925,−5.075,−2.075; sampleSE0.893points).
That is a modest favorable reduction, not zero protection or an increase in
misalignment. Yet the remaining cue response is extremely large. Native
Shared-minus-Sham excess is95.733points versus AdamW99.092points. A coherent
misleading association remains learnable while the new rare class is poorly
recognized. The raw-control interaction is−0.083points with mixed signs.

The secondary raw patched-ASR interaction is−2.033points for spectral, with
mixed seed signs (−4.575,−2.725,+1.200); do not substitute it for the registered
patch-excess interaction. Patched classification has material nuance: under
Shared, spectral common accuracy is14.41% versus11.52% AdamW, and common CE
2.9232 versus11.3060. Spectral reduces confident cue-induced errors, although
classification remains badly compromised. Patched rare accuracy is0% for all
policies; rare CE is6.6710 spectral versus16.2854 AdamW. All patched metrics
for all four cells and every seed are retained in the complete checked summary.

## What this can and cannot explain

The raw-direction control recovers rare recognition but loses diffuse-noise
accuracy preservation. This rejects sufficiency of that *specified own-state
norm rule along raw direction*. It does not isolate a fixed learning-rate
multiplier, reproduce another trajectory's norm history, equalize Adam
displacements, or establish necessity of this particular learned span.

Native-only diagnostics at updates101,500,2000 preserve raw per-example
gradients, actual delivered action, pre/post Adam states and finite probe
losses. The independent checker verified all36 recorded action/moment/movement
events and probe arithmetic. Their detailed interpretation is separate from
the behavioral result. The first32 wrong probes follow class-blocked order;
they are convenience subsets, not the wrong-example population. Retention is
not signed usefulness, and a gradient projection is not the actual Adam step.

The [independent diagnostic interpretation](interpretation.md) sharpens that
warning. At update101, rare within-group coherence is0.678–0.689, already
higher than common-digit3 coherence0.312–0.387. Yet its mean-action energy
retention is initially low. At updates500/2000 in Clean and Shared, rare
mean-action retention is98.05–99.12% despite poor recognition. Thus permanent
exclusion of the rare mean gradient is not an adequate explanation. Across
all36anchors,62.39–86.14% of actual parameter-movement squared norm lies
outside the saved native span. At Clean update2000, rare finite probe CE
worsens in two seeds. Retention, competing objectives, optimizer history and
accumulated learning must be distinguished; these observations do not identify
which mechanism caused the endpoint cost.

The rare-class condition conflates rarity, digit identity, absent-warmup
exposure and representation novelty. The patch is a synthetic shortcut, not
human-values misalignment; near-saturated cue effects are also a weak assay
for small between-policy differences. Diffuse and Shared corruption strengths
are not matched. This fixed rank32/step2000 pilot is not evidence that every
rank, estimator, clustering action or dataset has the same boundary.

## Implication and next discriminator

For the safety-first paper, the useful contribution is now sharper: a
geometric learning restriction can protect familiar classification from
unstructured corruption while preserving a learnable shared error and
hindering new correct recognition. A claim of general truth selection is not
supported. This strengthens the case for studying generalization mechanisms
and evaluating group-level competence, rather than an optimizer speed claim.

The saved diagnostics have now weakened permanent mean-gradient exclusion.
The next bounded discriminator should change batch composition while keeping
the same per-block example multiset, targets and total rare exposure, with
raw/native/norm-rule policies. Grouping versus interleaving changes co-occurrence
and history without adding rare loss weight. First ask whether useful rare
recognition can be restored while retaining noise protection. Reordering also
changes nonlinear optimization paths; it is not a pure covariance intervention.
Warmup exposure is a distinct later factor, not silently changed alongside it.
No such new training has been launched, and no adverse result will be tuned away.

## Provenance and execution

[Protocol](protocol.md), [source review](source-review.md),
final admission (artifact not distributed in this public snapshot), launch (artifact not distributed in this public snapshot),
[independent checker specification](audit-implementation.md),
[plot review](plot-review.md), render receipt (artifact not distributed in this public snapshot).

Raw acquisition and audit parent:
`/tmp/spectral-experiment-artifacts/spectral-selectivity-boundary-20260910.2IruKP`.
Acquisition completed once in258.416seconds:36trajectories,36diagnostics,
2,096,072,077bytes before completion. Completion SHA
`de66b0331bec90c9c24585ab9874af447de17cc653661984c0237fb65da66ceb`;
manifest SHA`cd086cdae859b9d9d68a19d07786cb5c2f6eec9f3dca08a8bc59c7c975794ac9`.
Independent saved-data audit PASS in74.703seconds,631,855 checks,
756evaluation rows and all36diagnostics under unchanged pre-result tolerances.
Audit SHA`d1b3a47371d8eed4d8fe46ec4b0c1cd3586f788a69cd003b264a7c0b39dd3907`.
No model replay in the audit, no official test-set use, no experiment restart,
no paid spend, no production optimizer change. Runtime includes asymmetric
diagnostics/evaluation and is not an optimizer speed comparison.
