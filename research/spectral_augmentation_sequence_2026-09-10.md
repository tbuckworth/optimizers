# Augmentation, useful continuation and conditional spectral protection

Codex — Spectral Optimizer Investigation · 10 September 2026

**Paper-ready evidence section; completed results, not a new experiment.**
The augmentation sequence establishes useful ordinary augmentation under clean
and severely corrupted labels, real but restricted continued learning under
the tested spectral policy, a favorable local equal-size directional case,
and a consistent secondary benefit from four-view observation. It does **not**
establish a practical advantage over augmented AdamW. The small stable global
rank-32 recipe should neither replace nor erase the older, materially different
rank-200 protection evidence. The completed larger-configuration bridge now
confirms substantial unaugmented learning and protection, but the combined
translation/filter recipe loses at fixed, validation-selected and late readouts.
This adverse practical result is not confined to the small rank-32 setting.

## 1. Ordinary augmentation supplies a substantive practical control

Two separate three-seed studies compare raw AdamW/native stable rank32 crossed
with no augmentation/one translated view per occurrence. Both use a
784→64→10 ReLU MLP with 50,890 parameters; 5,000 balanced training and 5,000
disjoint held-out images from official MNIST training IDX; 4,000 updates,
batch 64; AdamW learning rate .001 and weight decay .01. The canonical centered
observer uses decay .99, warmup 100 and hard projection without normalization.
Translations are independent integer shifts in −2…2 pixels, zero-filled,
active from update 1. Evaluation uses original images. All classes are
eligible from initialization; there are no rare classes or inserted cues.

Clean seeds are 202609141/142/143. Wrong-label seeds are 202609151/152/153;
exactly 400 of 500 training examples per true class receive a fixed label from
the other nine classes. Thus 80% are **actually wrong**, and each assignment
stays attached to all repeated views. The two studies are contextual comparisons,
not a paired clean-by-noise factorial. Within each study, initialization,
occurrences, labels and applicable translation draws are paired across policies.
[Clean protocol](../output/2026-09-10-spectral-general-augmentation/protocol.md);
[wrong-label protocol](../output/2026-09-10-spectral-wrong-label-augmentation/protocol.md).

Fixed original held-out endpoint means, accuracy % / CE nats:

| Label regime | Raw, none | Raw, translation | Native, none | Native, translation |
|---|---:|---:|---:|---:|
| Clean | 92.927 / 0.33046 | 95.653 / 0.15355 | 87.027 / 0.45716 | 82.347 / 0.65677 |
| 80% actually wrong | 24.653 / 2.39416 | 62.013 / 1.82879 | 53.007 / 1.97410 | 48.640 / 2.11847 |

Every paired augmentation contrast is retained below. Positive benefit means
translated-minus-none accuracy in percentage points, or none-minus-translated
CE. Values follow each study's ascending seed order.

| Regime/policy | Accuracy benefit: all three seeds | CE benefit: all three seeds |
|---|---|---|
| Clean raw | +3.02, +2.80, +2.36 | +0.203654, +0.194898, +0.132166 |
| Clean native | −6.26, −4.68, −3.10 | −0.226278, −0.187161, −0.185402 |
| Wrong-label raw | +35.18, +41.02, +35.88 | +0.491251, +0.687218, +0.517667 |
| Wrong-label native | −3.24, −7.80, −2.06 | −0.132281, −0.172104, −0.128725 |

Augmentation benefits raw and harms native endpoints in all three seeds and
both primary metrics in each study. Under wrong labels, unaugmented native
beats unaugmented raw in every seed, but augmented raw beats both native modes
in every seed. The practical conclusion for this recipe is therefore direct:
the combination adds no endpoint advantage over ordinary augmentation.
[Clean results and all curves](../output/2026-09-10-spectral-general-augmentation/results.md),
[audited scalars](../output/2026-09-10-spectral-general-augmentation/audit.json);
[wrong-label results](../output/2026-09-10-spectral-wrong-label-augmentation/results.md),
[audited scalars](../output/2026-09-10-spectral-wrong-label-augmentation/audit.json).

## 2. Restricted learning is not absence of learning

Unaugmented native's useful noisy-label protection is supported by both clean
competence and less false-label fitting. On the 4,000 actually corrupted
training examples, its mean wrong-target accuracy/CE are 7.442%/2.338145 versus
60.583%/1.232894 for unaugmented raw. Translation already gives raw strong
protection, reaching 7.842%/2.315615 together with better true-label competence.
Native translation reduces wrong-target accuracy further to 6.358%, but also
reduces wrong-target CE to 2.328311 in every seed: false-label argmax fit and
assigned-label log probability disagree. This is not an across-metric reduction
in memorization. Raw translation also deteriorates in held-out accuracy late
in two seeds, while wrong-target fitting rises in all three; its fixed-endpoint
advantage is not permanent immunity or a selected earlier-epoch claim.

The finite noisy sample still contains class signal. Under ideal symmetric
corruption with actual wrong fraction ρ and K classes, the expected target is

```text
q = [1 − ρ − ρ/(K−1)] e_y + [ρ/(K−1)] 1.
```

For ρ=.8 and K=10, q_y=.2 and each other coordinate is 4/45; the true class
remains the population argmax. Translation supplies more views of the same
assigned label, not independent votes about its truth. This expectation is
not a guarantee against fitting fixed finite-sample errors.
[Source-linked mathematical note](../output/2026-09-10-spectral-wrong-label-augmentation/mathematical-note.md).

## 3. Local direction and scale: a positive case with adverse boundaries

A subsequent diagnostic reuses the six **clean** native step-100 parents,
three seeds × two warmup modes; these are not six fresh training replications.
At each parent, 64 training examples and four translated views provide
per-example geometry and five separately measured batch gradients. Original
and fixed-translated 256-example held-out objectives supply local readouts.
Every candidate restores the same model/Adam state; native observes that
candidate once, then takes one AdamW step. There is no continued training.

Let θ_D be the rounded FP32 decoupled-decay endpoint and d_R,d_N the rounded
raw/native parameter differences from θ_D. The two matched controls rescale
these **post-Adam data displacements**, then add the same decay base. They
do not merely match incoming gradients or alter a learning-rate scalar.
Actual rounded norms are checked. Path fraction .1 scales the entire
displacement, including decay, and is a finite/linear diagnostic—not a
second optimizer step or a candidate stopping rule.

At full step, native-at-raw-size minus raw CE improvement, in **0.001 nats**:

| Warmup | Input → readout | Seed 141 | Seed 142 | Seed 143 |
|---|---|---:|---:|---:|
| Original | Original → original | −0.0920 | +0.2160 | +0.0667 |
| Original | Original → translated | −0.4003 | +0.0858 | +0.1149 |
| Original | Translated → original | −0.1171 | −0.0984 | −0.1466 |
| Original | Translated → translated | +0.0396 | +0.1701 | +0.0353 |
| Translated | Original → original | −0.5280 | −0.4324 | −0.7597 |
| Translated | Original → translated | −0.2712 | −0.0233 | −0.4599 |
| Translated | Translated → original | −0.3601 | −0.1592 | −0.5338 |
| Translated | Translated → translated | −0.2816 | +0.0069 | −0.6021 |

The reverse-size comparison has the same full-step signs. The favorable
original-warmup, translated-input/translated-readout case survives both size
conventions, both path fractions and signed derivative checks in every seed.
So do the adverse translated-input/original-readout cases. This rejects a
size-only explanation **at these local cells**, not at every later state.
Native data-step norms are 92.53–97.62% of raw, rather than collapsed to zero.
Nonlinearity remains important: the middle seed of translated-warmup,
original-input/translated-readout changes from adverse at full step to
favorable at .1 and in the derivative. Positive mean absolute CE improvements
also coexist with a negative third seed and declining mean accuracy in the
translated-warmup, translated-input/translated-readout cell.
[Complete results](../output/2026-09-10-spectral-augmentation-state/results.md);
[audited scalar contrasts](../output/2026-09-10-spectral-augmentation-state/audit.json).

Separately, the **pre-update** recorded span retains 76.93–80.59% of between-image
view-mean variation, 34.43–54.21% of within-image view variation, and 79.05–81.35%
of the translated mean-gradient energy across the six parents. Actual native
delivery uses a candidate-updated span. The empirical four-view identity
total=between+within holds, but the finite-view between term still contains
view randomness. Neither component is a semantic label; high retained energy
does not equal useful Adam motion. Different warmups change model, optimizer
and observer together. Local responses do not mediate the completed 4,000-step
outcomes by themselves.

## 4. Four-view observation: a bounded constructive intervention

At fixed parameters, for independent examples i and independent views T,
let a_i=E_T[g(i,T)] and W_i=Cov_T[g(i,T)]. Averaging m views per example gives

```text
Cov(g_batch) = [Cov_i(a_i) + E_i(W_i)/m] / B.
```

This standard identity motivates reducing observation variance without
changing designated first-view delivery. It does not identify population
covariance with the moving, centered and rank-truncated native observer.

The completed clean multiview study uses fresh seeds 202609161/162/163 and
four policies: raw1 and native1 use one translated view; observer4 observes
the mean of four same-state gradients once but delivers the projected first
gradient; raw4 delivers the four-view mean directly. Raw1/native1/observer4
share exact model/Adam warmup identities; raw4 already differs during warmup.
The two four-view arms have equal gradient-evaluation counts, not identical
wall time or a matched-state factorial. Current-view self-inclusion, observation
scale and subsequent trajectories change with the observation policy.

| Policy | Primary original accuracy / CE | Secondary translated accuracy / CE |
|---|---:|---:|
| raw1 | 95.173% / 0.156650 | 93.527% / 0.220596 |
| native1 | 81.393% / 0.660327 | 67.073% / 1.040954 |
| observer4 | 81.593% / 0.660356 | 68.687% / 1.020062 |
| raw4 | 95.633% / 0.144492 | 94.207% / 0.198985 |

Observer4−native1 primary accuracy differences are +0.16/+0.70/−0.26 points;
CE benefits +0.011308/−0.001730/−0.009662. Mean CE benefit −0.000028 reflects
cancellation, not equivalence. Secondary translated accuracy benefits are
+1.88/+1.44/+1.52 points and CE benefits +0.027265/+0.015959/+0.019452: all
three seeds agree. Observer4 also gains translated accuracy and reduces CE
from its own warmup in every seed, so this is genuine useful learning.
Both spectral arms improve original accuracy and CE after warmup in all three
seeds, but remain far behind both raw controls. Native1 translated accuracy
declines from warmup in two seeds despite improving CE in all three.

Using extra views directly improves raw4 over raw1 in every seed and both
readouts/metrics. Original benefits are +0.48/+0.46/+0.44 points and
+0.010428/+0.010360/+0.015686 CE nats. Observer4's original training accuracy
is only 82.38% versus raw4's 97.667%, consistent with restricted fitting rather
than an advantage hidden by excess training fit. Each four-view trajectory
requires 16,000 batch-gradient evaluations versus 4,000 for one view: 120,000
total across the study, 72,000 extra versus an all-single-view roster. Mean
branch wall times are 18.62/24.58/42.45/36.70 seconds for raw1/native1/observer4/
raw4; these are descriptive rotated-order measurements, not speed claims.
[Results and complete curves](../output/2026-09-10-spectral-multiview-clean/results.md);
[audited all-seed summary](../output/2026-09-10-spectral-multiview-clean/audit.json).

## 5. Historical fairness: do not substitute rank32 for the strongest positive

The older three-seed, 60-epoch global-rank200 study finishes at
74.28/81.87/83.06% clean-test accuracy versus Adam's 35.92/37.74/38.00%.
Those are substantial within-study late-horizon protection effects. They
used legacy numerics and Adam, not the small stable AdamW recipe. Seed42
still declines from its 86.00% peak to 74.28%: a mean five-point decline is
not uniform stability. Adam's test-selected peaks of 83.06/82.28/84.24% show
why fixed-horizon superiority is not automatically superiority to legitimate
validation stopping. [Raw three-seed archive](../results/weight_covariance_v2/noise90_long/),
[historical report](subspace_baselines_findings.md).

The useful positive is not confined to legacy numerics: the later matched
stable-hard AdamW study gives global rank200 78.77% versus raw 38.93% at the
60-epoch endpoint. Per-matrix rank64 gives 81.74%, but was selected on that
same seed's test set; these are seed42 descriptive results, not fresh
three-seed configuration comparisons.
[Stable global raw JSON](../results/noisy_mnist_hard_curves/final/global_stable_hard_r200_n90_s42.json),
[matched raw JSON](../results/noisy_mnist_hard_curves/final/adamw_n90_s42.json),
[methods and selection qualifications](noisy_mnist_hard_curves.md).

The strong regime differs in model (235,146 parameters), standardized inputs,
all 60,000 training examples, shuffled full passes and about 56,280 updates.
Historical 90% uniform replacement implies about 81% actually wrong labels
in expectation, close to the new 80% law—not a ten-point severity difference.
No cited historical strong-regime trial includes ordinary translation. Neither ranks nor
rank/parameter fractions alone explain the small recipe's deficit, and absolute
accuracies across these data/model regimes must not be ranked.
[Full source-checked regime comparison](../output/2026-09-10-spectral-multiview-clean/regime-comparison.md).

## 6. The completed strong bridge preserves the positive and answers augmentation

The new [three-seed rank-200 comparison](../output/2026-09-10-spectral-strong-augmentation/results.md)
crosses raw/native stable global rank200 with none/translation at seeds
202609171/172/173. Its 235,146-parameter MLP, normalized inputs, AdamW settings,
canonical hard filter and 100-update warmup follow the stronger configuration.
Each seed randomly splits official training images into 50,000 training,
5,000 clean validation and 5,000 clean reporting examples. Ninety-percent
uniform replacement produces 80.962/81.016/81.012% actually wrong targets,
fixed across all transformed views. Translation precedes standardization;
all readouts use original images. The 72 full passes preserve historical
3.6 million example exposures, but change data roles, repetitions and update
count (56,304 rather than 56,280): **bridge, not exact reproduction**.

All twelve validation-CE choices were persisted and verified before reporting
metrics, using the frozen numerical reduction and earliest exact tie. Final
reporting CE/accuracy are co-primary; both selected metrics use the same own
validation-CE-selected checkpoint, never separate best epochs. Mean outcomes:

| Recipe | Fixed-final accuracy / CE | Validation-selected accuracy / CE | Late accuracy / CE |
|---|---:|---:|---:|
| Raw, none | 32.000% / 2.708446 | 72.487% / 1.757167 | 32.921% / 2.619713 |
| Native200, none | 79.753% / 1.854416 | 82.973% / 1.808122 | 81.063% / 1.868632 |
| Raw, translation | 84.973% / 1.804289 | 85.993% / 1.764379 | 85.692% / 1.804857 |
| Native200, translation | 66.820% / 1.981758 | 68.673% / 1.931696 | 65.984% / 1.989981 |

Late means use exactly epoch endpoints 61–72. The primary native+translation
minus raw+translation accuracy differences are −20.44/−19.18/−14.84 points
(mean −18.153); CE excesses are +0.181014/+0.151446/+0.199948 nats. Selected
and late comparisons also lose every seed on both metrics: mean accuracy
deficits 17.320/19.708 points and CE excesses 0.167317/0.185124. These are
three paired seeds, not independent replications multiplied by readout count.
Translation improves raw and worsens native at the fixed endpoint in every
seed and both metrics; the accuracy interaction is −65.907 points.

**The useful native phenomenon is substantial learning, not only retention.**
Unaugmented native rises from 36.700% at update100 to 79.753%, gaining
35.08/38.68/55.40 points (mean +43.053); CE decreases in every seed, mean
0.284756 nats. Translated native rises 26.120→66.820%, gaining +40.700 points
with CE improvement in every seed. Raw/native warmup states agree within
augmentation mode. Native's unaugmented final advantage is +44.16/+49.30/
+49.80 points with lower CE in every seed. Its selected accuracy also beats
unaugmented raw in every seed (+10.487 points mean), but **selected CE is worse
in every seed**, mean excess 0.050955. Thus it does not dominate legitimate
early stopping. Raw translation beats even unaugmented native at fixed and
selected readouts on both metrics in every seed; nevertheless raw augmentation
itself has slightly worse selected CE in two seeds and in mean (+0.007211).

![All three seeds and all 74 fixed reporting readouts preserve the early peak and later deterioration of unaugmented raw.](../output/2026-09-10-spectral-strong-augmentation/reporting-learning-curves.png)

Actually-wrong-subset assigned accuracy/CE are 39.158%/1.649918 (raw none),
2.517%/2.376372 (native none), 2.230%/2.377909 (raw translation) and
3.762%/2.355321 (native translation). Each single intervention greatly
restricts wrong-assignment fitting relative to raw none while retaining clean
competence. Combining them yields **more**, not less, wrong-assignment fit
than either single intervention in every seed: higher accuracy and lower CE
against the wrong targets. Its lower all-training assigned accuracy is not
extra noise suppression; that aggregate includes useful correctly assigned
examples. All such training readouts use original images. This is fit behavior,
not identification of memorization circuitry or of the impaired update component.

The [saved-array audit](../output/2026-09-10-spectral-strong-augmentation/audit.json)
is PASS: 937 artifacts, 888 logit states, all twelve validation choices verified
before reporting, maximum scalar discrepancy 1.4552e−11 within fixed 1e−10
tolerances. Its two logit passes preserve the memory cap. It does not replay
models or reconstruct opaque checkpoint state; reporting arrays were not
cryptographically blinded. The [independent scalar/interpretation review](../output/2026-09-10-spectral-strong-augmentation/interpretation-review.md)
corroborates every registered window without new neural computation. Mean
branch walls are 254.424/695.874/272.787/728.823 seconds in table order, each
for 56,304 gradient evaluations and 3.6M exposures: observed implementation
costs, not a speed benchmark. No rank/LR search or automatic follow-up is selected.

The [completed component diagnostic](../output/2026-09-10-spectral-component-utility/results.md)
reuses twelve native states from the strong bridge: three seeds×two prior
training settings×warmup/final. Two translated batches per parent compare
faithful raw/native actions from the same weights, Adam and observer, with
full and actually materialized tenth paths. Exact25-view S/F/C and separate
original/transformed clean losses give signed utility, not just retained
energy. The [audit](../output/2026-09-10-spectral-component-utility/audit.json)
passes50,388 saved-array/provenance checks across144 endpoints; it does not
repeat model inference, autograd or streaming-observer history.

At the primary translated-final states, raw increases C in all three seed
averages. Native-minus-raw original clean-CE decrease is
+6.595461e−6,−1.189737e−5,−8.814843e−6 nats; in the two clean-adverse seeds,
native slightly attenuates C deterioration. The local hypothesis that native
removes useful raw consistency progress is therefore unsupported. These tiny
effects are not the earlier large trajectory gap. Seed172's C sign varies
between its two draws; seed171's tenth C contrast is below1e−8 nats.

At warmup, native improves relative C but worsens both clean losses in all
three seeds under both training settings, at both path sizes. Wrong-subset
CE also moves relatively toward assigned wrong targets and away from truth;
in some cases this means less unfitting rather than absolute wrong fitting.
Final wrong-fit and clean effects are mixed. Positive F contrast is not a
memorization score; positive primary L contrast means less deterioration,
and S can conceal opposing true/uniform-target changes. The [full analysis](../output/2026-09-10-spectral-component-utility/interpretation-analysis.md)
keeps every component, cell, seed and guard.

Only native-history parents are used, with inherited moments and unmatched
step magnitudes. Unaugmented parents receive hypothetical translated actions.
This is outcome-informed local evidence, not fresh training replication or
identified full-trajectory mediation. The strong useful-learning positive and
adverse combination are preserved. Both acquisition/audit handles are terminal;
the HTML report with two plots is delivered (artifact not distributed in this public snapshot).

## 8. What this adds to the paper, and what remains untested

The contribution is a conditional allocation-of-learning account with both
constructive and adverse interventions, not merely baseline failure. Ordinary
augmentation is genuinely useful under severe fixed errors; unaugmented
spectral filtering also allows useful progress and restricts false-label fit.
Their combination is not advantageous in either tested regime. Local
equal-size directional benefits and the fresh multiview secondary gain sharpen
the account without reversing that practical finding. These classification
effects do not certify semantic selectivity, maintained-competence behavioral
safety or long-run covariance mediation.

The completed bridge makes small rank32 an insufficient explanation for the
earlier augmentation deficit. It does not isolate rank, prove mechanistic
redundancy, or identify which useful mean-logit/consistency components the
filter impairs. The reviewed objective and feature-geometry identities remain
conditional, not measured trajectory mediation. Historical positive results
are preserved alongside the new unaugmented positive and adverse combined
recipe. No noisy observer4 extension or further diagnostic is required before
reporting this evidence, and no new experiment is selected by this synthesis.

All six recent studies retain frozen sources, full registered scalar records
and an independently executed saved-data check. That is not a new independent
training replication: the checks do not regenerate model forwards, autograd
or observer histories. Three paired seeds remain the unit of replication;
overlapping source pools, repeated views and reused diagnostic parents do not
multiply it. This synthesis reads completed scalar evidence and introduces
no scientific recomputation or new external-literature claims.

Pinned result identifiers (the linked audits contain exact archive paths):

| Study | Result SHA-256 |
|---|---|
| [Clean](../output/2026-09-10-spectral-general-augmentation/audit.json) | `7c4f31d0a3bf2b89aa5fbb7be0e5bbe62a2e713faa75afd3241c6fab1db63a5e` |
| [Wrong-label](../output/2026-09-10-spectral-wrong-label-augmentation/audit.json) | `973e996555d88de109d14f36fc174d483e40fefa188ee1c3d85270599433a682` |
| [Fixed-state](../output/2026-09-10-spectral-augmentation-state/audit.json) | `149b239566dfa77b79d276322f44e173f33613222126fb618d225f96110dfad5` |
| [Multiview](../output/2026-09-10-spectral-multiview-clean/audit.json) | `a4bf53e3696c9e22eeaa925bddc9034e9d27a39ff3b1a2410ae11531dfcc8d30` |
| [Strong bridge](../output/2026-09-10-spectral-strong-augmentation/audit.json) | `6e2677ae3feb41a3da16ea11639a15871577ff8cd469c6afcda0b8c67096ec7c` |
| [Component utility](../output/2026-09-10-spectral-component-utility/audit.json) | `e16a7774b853060d80a3393d1bc0626b2b8965a13b451552989356a18cbb3971` |
