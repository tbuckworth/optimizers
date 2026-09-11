# Independent scalar and interpretation review

Codex — Spectral Optimizer Investigation · 10 September 2026

## Evidence and arithmetic scope

- Frozen question and reporting rules: [protocol.md](protocol.md).
- Pre-acquisition source review: [review.md](review.md).
- Lifecycle/producer receipt: completion.md (artifact not distributed in this public snapshot).
- Numerical source: [audit.json](audit.json), schema
  `spectral_strong_augmentation_audit_v1`, status `PASS`, SHA-256
  `6e2677ae3feb41a3da16ea11639a15871577ff8cd469c6afcda0b8c67096ec7c`.
  Its bound producer SHA-256 is
  `e98125d3bcbd85e402c8269797e6cb48797b9ebbe9d9715897f8244203c69e8a`.

An independent short, single-threaded standard-library command parsed the
audit JSON, checked its hash, reconstructed all registered reporting summaries
from `branches[*].metrics`, and only then compared arithmetic against
`summary`. It checked all 12 exact 74-readout schedules, count/accuracy and
CE-sum/mean relations for every one of the 888 records and all six data roles,
selected index/step/exposure consistency, each arm's final/selected/late/warmup
summary, all registered pairwise contrasts and augmentation interactions.
**10,916 numerical checks agreed; maximum absolute discrepancy was
2.220446049250313e-16.** No acceptance tolerance was changed.

For selection, this corroboration uses the audit's already verified canonical
choice, not a new argmin over its alternate-formula metric values. It checks
that the selected alternate CE agrees within 1e-10 and is not above any
candidate by more than 1e-10. The producer/auditor's exact shifted-exp,
ordered-`fsum` tie convention cannot be reconstructed again from alternate
scalar CEs alone; its prior validation-array audit remains the evidence for
that exact choice. It verified all 12 choices before reporting calculations.

All seed tuples below are ordered `202609171`, `202609172`, `202609173`.
Accuracy is percent; accuracy differences are percentage points (pp).
CE is mean cross-entropy; lower is better for clean prediction, but lower
assigned-label CE means **more assigned-label fit**.

## All registered reporting windows

Means are equal-weight arithmetic means across the three seeds. The late
window is each seed's mean over exactly epochs 61–72, then the seed mean.

| Arm | Final accuracy | Final CE | Validation-selected accuracy | Selected CE | Late accuracy | Late CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw, no augmentation | 32.0000 | 2.708446 | 72.4867 | 1.757167 | 32.9211 | 2.619713 |
| Native200, no augmentation | 79.7533 | 1.854416 | 82.9733 | 1.808122 | 81.0628 | 1.868632 |
| Raw, translation | 84.9733 | 1.804289 | 85.9933 | 1.764379 | 85.6917 | 1.804857 |
| Native200, translation | 66.8200 | 1.981758 | 68.6733 | 1.931696 | 65.9839 | 1.989981 |

The predeclared primary is native200+translation versus raw+translation at
fixed final, not the largest favorable entry in this table:

| Window | Native-minus-raw accuracy, by seed (pp) | CE benefit, raw-minus-native, by seed | Mean accuracy / CE benefit |
| --- | --- | --- | --- |
| Final | −20.44, −19.18, −14.84 | −0.181014, −0.151446, −0.199948 | −18.1533 / −0.177469 |
| Validation-selected | −17.74, −18.08, −16.14 | −0.189591, −0.155279, −0.157081 | −17.3200 / −0.167317 |
| Epochs 61–72 | −19.3833, −21.2100, −18.5300 | −0.188008, −0.168452, −0.198912 | −19.7078 / −0.185124 |

Thus this combined policy loses in **all three seeds and both metrics in
every registered window**. No favorable secondary window displaces that result.

Other registered mean contrasts, oriented as accuracy and CE benefits:

| Contrast | Final accuracy / CE | Selected accuracy / CE | Late accuracy / CE |
| --- | ---: | ---: | ---: |
| Native − raw, no augmentation | +47.7533 / +0.854029 | +10.4867 / −0.050955 | +48.1417 / +0.751081 |
| Translation effect on raw | +52.9733 / +0.904157 | +13.5067 / −0.007211 | +52.7706 / +0.814856 |
| Translation effect on native | −12.9333 / −0.127341 | −14.3000 / −0.123573 | −15.0789 / −0.121349 |
| Native minus raw augmentation effect | −65.9067 / −1.031498 | −27.8067 / −0.116362 | −67.8494 / −0.936205 |

Unaugmented native's final accuracy benefits are +44.16/+49.30/+49.80 pp,
and CE benefits +0.878372/+0.875899/+0.807817. Its validation-selected
accuracy benefits remain positive in all seeds (+5.30/+11.84/+14.32 pp),
but selected CE is worse in every seed (benefits
−0.049070/−0.062350/−0.041446). Raw augmentation's selected CE effect is
mixed (+0.001667/−0.012479/−0.010823), despite all-seed accuracy gains.
All other contrast signs in the table agree across the three seeds.

Raw+translation also exceeds **unaugmented** native in accuracy and CE for
all seeds at final, own validation-selected checkpoints, and the late window.
That is a secondary cross-policy comparison, not a new primary.

## Continued learning, not just endpoint preservation

| Arm | h100→final accuracy change, by seed (pp) | CE improvement, by seed | Mean accuracy / CE improvement |
| --- | --- | --- | --- |
| Raw, none | −9.08, −10.62, +5.60 | −0.617898, −0.597420, −0.492501 | −4.7000 / −0.569273 |
| Native200, none | +35.08, +38.68, +55.40 | +0.260474, +0.278479, +0.315316 | +43.0533 / +0.284756 |
| Raw, translation | +64.04, +52.10, +60.42 | +0.440049, +0.365270, +0.419684 | +58.8533 / +0.408334 |
| Native200, translation | +43.60, +32.92, +45.58 | +0.259035, +0.213824, +0.219736 | +40.7000 / +0.230865 |

Within each augmentation mode, native/raw share exactly the h100 state.
Unaugmented h100 accuracy is 42.60/42.04/25.46%; translated h100 accuracy
is 22.28/32.14/23.94%. Both native policies subsequently improve clean
reporting accuracy and CE in every seed. Calling this strong native result
“only freezing the warmup” would be incorrect. Calling the combined policy
“unable to learn” would also be incorrect.

## Training-label fit

Actually wrong counts are 40,481/40,508/40,506 out of 50,000:
80.962/81.016/81.012%. These are distinct from the 90% replacement-mask
probability, since replacement sometimes equals truth. All readouts use
original, unaugmented training images, including for translated training arms.

| Arm / label view | Final accuracy by seed (%) | Mean accuracy | Final CE by seed | Mean CE |
| --- | --- | ---: | --- | ---: |
| Raw none / all true | 33.600, 31.338, 32.200 | 32.3793 | 2.826838, 2.847511, 2.795862 | 2.823403 |
| Raw none / all assigned | 44.762, 45.112, 44.994 | 44.9560 | 1.565983, 1.540712, 1.547725 | 1.551473 |
| Raw none / wrong-subset true | 24.7400, 22.5610, 23.5916 | 23.6308 | 3.228250, 3.250033, 3.182507 | 3.220263 |
| Raw none / wrong-subset assigned | 38.5267, 39.5626, 39.3843 | 39.1579 | 1.670909, 1.637019, 1.641826 | 1.649918 |
| Native none / all true | 77.302, 80.714, 81.362 | 79.7927 | 1.846175, 1.850518, 1.868265 | 1.854986 |
| Native none / all assigned | 17.212, 17.650, 17.628 | 17.4967 | 2.276097, 2.273098, 2.273958 | 2.274384 |
| Native none / wrong-subset true | 76.9003, 80.3792, 81.0053 | 79.4282 | 1.850079, 1.854033, 1.871607 | 1.858573 |
| Native none / wrong-subset assigned | 2.6803, 2.5378, 2.3330 | 2.5170 | 2.381095, 2.375634, 2.372388 | 2.376372 |
| Raw translation / all true | 86.088, 84.540, 84.816 | 85.1480 | 1.784288, 1.818920, 1.809696 | 1.804301 |
| Raw translation / all assigned | 18.622, 18.390, 18.296 | 18.4360 | 2.260358, 2.264510, 2.265467 | 2.263445 |
| Raw translation / wrong-subset true | 85.4080, 84.0254, 84.3480 | 84.5938 | 1.792525, 1.825241, 1.815347 | 1.811037 |
| Raw translation / wrong-subset assigned | 2.0775, 2.3748, 2.2367 | 2.2297 | 2.380541, 2.375243, 2.377943 | 2.377909 |
| Native translation / all true | 66.014, 65.982, 70.876 | 67.6240 | 1.963226, 1.970589, 2.007944 | 1.980586 |
| Native translation / all assigned | 15.834, 15.936, 16.244 | 16.0047 | 2.283627, 2.282722, 2.283049 | 2.283133 |
| Native translation / wrong-subset true | 65.9173, 65.7993, 70.7574 | 67.4914 | 1.964592, 1.971639, 2.009130 | 1.981787 |
| Native translation / wrong-subset assigned | 3.9376, 4.0264, 3.3205 | 3.7615 | 2.360335, 2.356912, 2.348716 | 2.355321 |

The combined policy has **more**, not less, wrong-assignment fit than raw
translation in all three seeds by both wrong-subset metrics: higher assigned
accuracy and lower assigned CE. Its lower all-training assigned accuracy does
not establish extra noise suppression because that aggregate also includes
correctly assigned examples, on which useful learning can be lost. The
subgroup truth metrics make the competence cost explicit. These are measured
fit proxies, not a semantic identification of memorization circuitry.

## Selection exposures and costs

Canonical indices are zero-based; index0 is initialization, index1 is h100,
and index `epoch+1` is a completed-epoch readout. Both selected metrics use the
same selected checkpoint; no accuracy-specific or reporting-set selection.

| Seed suffix / arm | Index | Epoch | Step | Training exposures |
| --- | ---: | ---: | ---: | ---: |
| 171 / raw none | 8 | 7 | 5,474 | 350,000 |
| 171 / native none | 69 | 68 | 53,176 | 3,400,000 |
| 171 / raw translation | 39 | 38 | 29,716 | 1,900,000 |
| 171 / native translation | 40 | 39 | 30,498 | 1,950,000 |
| 172 / raw none | 12 | 11 | 8,602 | 550,000 |
| 172 / native none | 27 | 26 | 20,332 | 1,300,000 |
| 172 / raw translation | 51 | 50 | 39,100 | 2,500,000 |
| 172 / native translation | 50 | 49 | 38,318 | 2,450,000 |
| 173 / raw none | 13 | 12 | 9,384 | 600,000 |
| 173 / native none | 48 | 47 | 36,754 | 2,350,000 |
| 173 / raw translation | 66 | 65 | 50,830 | 3,250,000 |
| 173 / native translation | 62 | 61 | 47,702 | 3,050,000 |

All final arms have 56,304 updates and 3.6 million examples; totals are
675,648 gradients and 43.2 million examples. Equal exposures do not make the
historical 60×60,000 setting identical to this 72×50,000 internal-split bridge.

| Arm | Training seconds by seed | Augmentation/preparation seconds by seed | Evaluation seconds by seed | Branch wall seconds by seed |
| --- | --- | --- | --- | --- |
| Raw none | 220.82, 246.09, 243.96 | 6.86, 7.81, 7.49 | 8.38, 9.25, 9.37 | 237.07, 264.27, 261.93 |
| Native none | 649.80, 689.57, 688.72 | 8.48, 8.82, 8.67 | 8.17, 9.03, 9.22 | 668.67, 709.87, 709.07 |
| Raw translation | 226.14, 241.33, 236.74 | 27.22, 29.15, 28.40 | 8.43, 8.97, 8.79 | 262.81, 280.54, 275.01 |
| Native translation | 682.80, 687.06, 687.90 | 31.19, 31.40, 31.29 | 8.89, 8.98, 9.43 | 725.35, 729.94, 731.18 |

Mean branch walls are 254.424/695.874/272.787/728.823 seconds in table order.
Native translation uses 2.922× raw translation's measured training time and
2.672× branch wall time (ratios of totals). These observations are not an
optimized speed benchmark or time-to-target result. The audit JSON carries
audited branch timings but not producer global serialization/selection
durations. This review therefore does **not independently corroborate** those
global fields or infer them by subtracting rounded branch totals. Its own
auditor wall time is 90.702 seconds, distinct from acquisition wall time.

## Interpretation and criteria discipline

**Strongest positive:** this faithful larger-config bridge confirms a useful
conditional learning phenomenon. Unaugmented stable rank200 both learns
substantially beyond shared warmup and resists destructive fixed-wrong-label
fitting, with large all-seed final gains over unaugmented raw. It also retains
an all-seed validation-selected accuracy advantage, although not a CE advantage.
The positive is stronger than a claim of endpoint preservation alone.

**Strongest adverse:** in this same preregistered setting, ordinary translation
alone outperforms the combined policy by large margins and beats unaugmented
native in the registered reporting windows. Combining them worsens both clean
prediction and wrong-assignment fit relative to raw translation. The selected
strong configuration therefore does not add demonstrated practical robustness
over this ordinary augmentation baseline; the adverse result is not confined
to the earlier small-rank regime or to a last-checkpoint accident.

These findings do not establish a universal negative for spectral filtering,
a proof of augmentation/filter mechanistic redundancy, or the specific
augmentation-gradient mediation hypothesis. They do not identify semantic
clusters, safety selectivity, deployment calibration, an optimum learning
rate/rank, or a capability/safety intervention. Three paired seeds share a
dataset and their split IDs can overlap across seeds. Generalization beyond
this architecture, fixed corruption, integer translations and fixed schedule
remains untested here. Clean validation selection is an explicit resource.

The reporting discipline is to preserve the original adverse primary and the
genuine unaugmented positive together, not promote a secondary accuracy
advantage into overall superiority or erase useful learning because an
ordinary alternative performs better. No automatic follow-up is selected.

## Main report check

The report correctly distinguishes prospectively fixed seed IDs from an
adaptively revisited research task. It retains the unaugmented selected
accuracy/CE disagreement and raw augmentation's mixed selected CE, rather than
claiming universal selected-metric superiority. It states the saved-array
audit's non-replication scope. Global acquisition timing, serialization,
selection/reporting phase durations, replacement-selected counts and peak
resource fields are not in the audit branch metrics and were **not independently
recomputed here**; the main report's direct producer/resource checks remain
their provenance. This is a scope limit of this scalar corroboration, not a
detected contradiction.

Reviewed content hashes (SHA-256):

| File | Hash |
| --- | --- |
| `results.md` | `790f1ece6513892423c3559aa36bb35d16151630dc7fc91fc39acfd6707d525e` |
| `report.html` | `57d563a79e0d0cc27b86ef6c3c91e184dd0d4aa265c6cc008f602d4bfa497cc6` |
| `deliver.py` | `e04f09bb27f857a6ddef8c1999ba83719bc01b2897c06d11f24bba5b0d8055dd` |