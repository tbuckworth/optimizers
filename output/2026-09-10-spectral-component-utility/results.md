# What does one spectral-filtered update actually change?

Codex — Spectral Optimizer Investigation · 10 September 2026

**The registered local test does not support the proposed “lost useful
consistency learning” explanation.** At the three translated final parents,
the raw update itself increases the consistency penalty. Keeping the filter
changes clean prediction loss only slightly, with mixed signs across seeds.
In the two clean-adverse seeds, it reduces the consistency deterioration
slightly rather than suppressing a consistency improvement.

## Direct evidence and design

- [Independent saved-array audit](audit.json), statusPASS,50,388checks,
  SHA256`e16a7774b853060d80a3393d1bc0626b2b8965a13b451552989356a18cbb3971`.
- [Protocol](protocol.md), source/input manifest (artifact not distributed in this public snapshot),
  [implementation acceptance](implementation-acceptance.md),
  terminal run record (artifact not distributed in this public snapshot), runtime identities (artifact not distributed in this public snapshot).
- Original result JSON is in
  `/tmp/spectral-experiment-artifacts/spectral-component-utility-20260910.4Ze0xq/acquisition-001/results.json`,
  SHA256`3f685f592e0cf3dfd6a761fc86c8abea8af25aa4744cc8b9e1aeae8f53946096`.

Twelve native parents: three seeds×prior training without/with translation×
warmup100/final56304. At each, two fixed translated action batches produce
paired raw/native updates from the **same** weights, Adam moments and observer.
Raw bypasses observation/filtering; native observes and filters once. No
training continues. Decay-only and actual full/tenth materialized path points
give144 endpoint readouts. All25 translations are evaluated on a fixed
training panel; original/per-view true-label prediction is evaluated separately
on128 nontraining examples. The two draws are averaged within each parent,
not treated as independent seeds. Every action is translated, even at a
none-trained parent: there it is a hypothetical onset-of-augmentation action.

The objective decomposition is `L = S + F + C`:

- S: softened-target CE of mean logits, with target .1×true+.9×uniform.
- F: signed force from the fixed label assignment relative to that target law.
- C: label-independent Jensen gap across view logits (the view-consistency
  penalty). Lower C alone is not better clean prediction or a safety result.
- L: assigned-label CE averaged over views, computed directly.
- H_O/H_T: true-label CE on nontraining original images/all25 views respectively.

Every number below is an objective decrease: **before minus after**. Positive
means that named loss fell, not that learning was necessarily useful.
`D = native decrease − raw decrease`; positive H_O D favors native clean CE.
All values are nats. Finite effects/contrasts with magnitude≤1e−8 and linear
utilities≤1e−10 are marked numerically small in the detailed presentation,
not interpreted as sign evidence or equivalence margins. No step was rescaled
after results to obtain a larger effect.

## Primary: translated final parents, actual full step

Seed labels171/172/173 abbreviate202609171/202609172/202609173.

| Seed | Raw H_O decrease | Native H_O decrease | D_HO | Raw C decrease | Native C decrease | D_C |
|---|---:|---:|---:|---:|---:|---:|
| 171 | −5.463519e−4 | −5.397564e−4 | +6.595461e−6 | −9.977187e−6 | −1.002014e−5 | −4.295152e−8 |
| 172 | +1.823412e−3 | +1.811515e−3 | −1.189737e−5 | −7.950385e−5 | −7.935308e−5 | +1.507674e−7 |
| 173 | +5.608214e−3 | +5.599399e−3 | −8.814843e−6 | −1.269678e−4 | −1.265715e−4 | +3.963079e−7 |

Raw C progress is negative in every seed. Therefore the criterion “raw learns
useful consistency which filtering removes” is not met here. The two seeds
with a clean-CE cost from filtering instead have a slightly better C outcome.
The first seed has the opposite ordering. No consistent clean/consistency
benefit or harm jointly follows from keeping the filter at these final states.
These are two-draw averages: seed172's full-step C contrast changes sign between
its two draws (−2.13172e−7,+5.14707e−7 nats). Each primary clean-CE contrast sign
does agree across its two draws. The means are not uniformly per-draw effects.

Linear D_HO values are+5.400142e−6,−1.166113e−5,−8.530495e−6; linear D_C
values−5.739529e−8,+1.431208e−7,+3.440208e−7. Actual tenth-path finite D_HO
is+5.461135e−7,−1.173593e−6,−8.566616e−7. Tenth D_C is−5.618656e−9
(numerically small),+1.462257e−8,+3.483094e−8. These preserve the broad
mixed ordering, but do not turn tiny effects into a strong mechanism claim.

## All stage/training-mode cells remain visible

Full-step D values below, with the same three seeds in every cell:

| Prior training | Stage | Seed | D_HO | D_C | D_F |
|---|---|---|---:|---:|---:|
| none | warmup100 | 171 | −1.119730e−4 | +6.614885e−6 | −1.127422e−5 |
| none | warmup100 | 172 | −4.281549e−5 | +2.144466e−6 | −3.026912e−7 |
| none | warmup100 | 173 | −9.705016e−5 | +2.616498e−6 | −1.144771e−6 |
| none | final56304 | 171 | +1.7093e−5 | −2.6293e−7 | +6.0488e−7 |
| none | final56304 | 172 | −4.4576e−6 | −1.7093e−7 | −1.8286e−6 |
| none | final56304 | 173 | −1.4020e−5 | −7.6238e−8 | −2.3424e−6 |
| translate | warmup100 | 171 | −8.4682e−5 | +2.8218e−6 | +1.3073e−6 |
| translate | warmup100 | 172 | −5.8751e−5 | +1.9912e−6 | −9.8374e−6 |
| translate | warmup100 | 173 | −2.0095e−5 | +1.4364e−6 | +3.7796e−8 |
| translate | final56304 | 171 | +6.595461e−6 | −4.295152e−8 | +1.680995e−6 |
| translate | final56304 | 172 | −1.189737e−5 | +1.507674e−7 | +2.199367e−6 |
| translate | final56304 | 173 | −8.814843e−6 | +3.963079e−7 | +9.904681e−7 |

At warmup, keeping the filter gives **better relative C but worse relative
H_O in both prior-training settings and all three seeds**. Better relative C
can mean either a larger decrease or a smaller increase; it is not always
absolute consistency improvement. The same ordering persists on the actual
tenth paths. These are related states, not six independent seed replications.
They provide a concrete local counterexample to treating lower consistency
penalty as interchangeable with useful clean learning.

At unaugmented final parents, C is relatively worse under native in all three
full-step seed averages, while clean CE is mixed. Raw C itself improves only
for171, where raw clean CE worsens. Thus that secondary cell also fails to
supply a uniform lost-useful-consistency account. No favorable cell replaces
the registered translated-final primary.

All six objectives, absolute raw/native effects, derivatives, actual tenth
paths, decay/data accounting, mean-logit true/uniform terms, original/per-view
accuracy/CE and actually-wrong-subset counts are retained in audit.json under
`checked_parents` and `independent_summary`. The detailed independent
[interpretation](interpretation-analysis.md) accompanies this report; no metrics
are selected by sign. It also reports that primary L deteriorates under both
policies in all three seeds: native's positive D_L is less deterioration, not
absolute assigned-loss progress. S mixes true-target and uniform-target terms;
its seed173 advantage hides worse true-target mean-logit CE. Neither S nor F
alone is a clean-learning or wrong-memorization measure.

## Label fitting and movement: avoid substituting proxies

Primary D_F is positive in all three seeds, but actual wrong-subset assigned
original-image CE changes are mixed: native-minus-raw fitting progress is
−2.011719e−6,+2.686962e−6,+8.081316e−7. Wrong-subset assigned accuracy is
identical between the two choices in all three primary seed averages, as is
original clean accuracy. This does not establish equivalence; CE detects
small changes without an argmax change.

At warmup in the none-trained parents, F progress is lower with native in all
three seeds, yet wrong-subset assigned CE fitting is higher. F is a signed
full-panel component, not an isolated wrong-label memorization score. S/C and
the evaluated view/subset also matter. There is no consistent final-state
demonstration of reduced realization fitting with preserved clean progress.
Across both warmup training settings and both path sizes, native's relative
wrong-subset CE moves toward the assigned wrong targets and away from the true
targets, for both original and mean-per-view evaluation. This is an adverse
local ordering, not a claim that the earlier native trajectory lacks useful
learning. All actual wrong-subset values remain in the independent note.

The actual full-step norms are similar, not zero. Primary raw/native norms
are0.114619/0.114423,0.118928/0.118716,0.112667/0.112731. Native motion is not
uniformly smaller. This shared-parent/inherited-Adam experiment does not
isolate direction from step magnitude or diagnose why accumulated trajectories
diverge. Small present-step differences cannot rule out historical effects.

## Implication and next work

The earlier strong-regime result remains intact: native learns useful clean
structure under heavy corruption without augmentation, while the raw-plus-
augmentation trajectory performs better and the combined native-plus-
augmentation trajectory is worse. See [the completed strong report](../2026-09-10-spectral-strong-augmentation/results.md).
The new contribution is a boundary on explanation: this local test does not
identify suppression of useful consistency learning as its cause.

Next consolidate the signed-update evidence with the earlier learning curves
and observer-history findings. Distinguish accumulated state/history from the
effect of switching the filter for one update before choosing another study.
Do not retune this probe, move the primary state, rerun either completed unit,
or claim a generic failure of the original idea. No new acquisition is selected
by this report; the broader autonomous research goal remains active.
