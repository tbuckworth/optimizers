# Component utility: independent scalar interpretation

## Conclusion and evidence boundary

The primary translated-final comparison does **not support the specific account that native filtering removes useful consistency learning**. Raw Adam increases the consistency objective C in all three seed averages. Native is slightly worse on original clean CE in two seeds and slightly better in one; in the two adverse seeds it actually reduces the consistency deterioration. This is a local objective tradeoff, not evidence that consistency improvement mediated the earlier trajectory gap.

There is nevertheless a coherent adverse signal at warmup: relative to raw, native gives worse original and translated clean CE in all six parents, with lower assigned-wrong-label CE and higher true-label CE on the wrong subset. These are relative differences, not absolute wrong-label fitting in every parent: for example, at none/h100/seed 173 both policies unfit the wrong targets, but native unfits them less. Conversely, final-state contrasts are mixed, and the unaugmented native trajectory's substantial learning remains real. No single component explains all cells.

Sources are the completed [audit](audit.json), the producer's scalar results JSON (artifact not distributed in this public snapshot), the frozen [protocol](protocol.md), and the [prior strong-study report](../2026-09-10-spectral-strong-augmentation/results.md). This review read only JSON scalar metrics/metadata and documentary reports: no saved arrays, checkpoints, model evaluation, or audit rerun. The protocol's prospective status wording is historical; the completed audit is PASS, with 50,388 checks, 12 parents, 24 paired draws, 48 raw/native actions, and 144 endpoints including decay controls. Its stated boundary is saved-array arithmetic/provenance, not model, autograd, or observer-history replay.

Verified inputs:

- Audit SHA-256: `e16a7774b853060d80a3393d1bc0626b2b8965a13b451552989356a18cbb3971`.
- Producer results SHA-256: `3f685f592e0cf3dfd6a761fc86c8abea8af25aa4744cc8b9e1aeae8f53946096`.
- Audit `independent_summary` and producer `summary` have identical structure and maximum absolute numeric difference 6.67e−16.

## Estimand and units

For objective J, finite improvement is E_J = J(parent) − J(endpoint), linear utility is U_J = −∇J(parent)·(actual rounded endpoint − parent), and the paired contrast is D_J = E_J(native) − E_J(raw). Positive means more decrease of that named objective, not necessarily better learning. All numbers in the tables are **micro-nats (10⁻⁶ nats)**. Seed suffixes 171/172/173 mean 202609171/172/173. Each number first averages the two action draws within its saved parent; those draws are not independent seeds.

S is the softened-label mean-logit CE, F the fixed-label residual term, C the nonnegative view-consistency term, and direct assigned-label loss L = S + F + C. H_O is original true-label CE on the 128-example reporting panel; H_T is mean per-view true-label CE there. The component panel has 256 training examples and the full 25-view translation grid. Its actually wrong-label subsets contain 205/203/215 examples. In particular, F is not wrong-subset CE, S is not clean CE, and lower C does not guarantee a correct prediction.

## Primary translated-final result

All parent steps below are 56,304. The tenth path is an actual rounded FP32 affine displacement, not a fresh Adam step at a lower learning rate.

| Seed | Path | Raw E_HO | Native E_HO | D_HO finite | D_HO linear | Raw E_C | Native E_C | D_C finite | D_C linear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 171 | 1 | −546.351894 | −539.756434 | +6.595461 | +5.400142 | −9.977187 | −10.020139 | −0.042952 | −0.057395 |
| 172 | 1 | +1823.412024 | +1811.514656 | −11.897367 | −11.661125 | −79.503848 | −79.353080 | +0.150767 | +0.143121 |
| 173 | 1 | +5608.213952 | +5599.399109 | −8.814843 | −8.530495 | −126.967806 | −126.571498 | +0.396308 | +0.344021 |
| 171 | 0.1 | −43.771899 | −43.225785 | +0.546114 | +0.541152 | −0.328408 | −0.334026 | −0.005619* | −0.005770 |
| 172 | 0.1 | +193.343685 | +192.170092 | −1.173593 | −1.167746 | −7.299164 | −7.284541 | +0.014623 | +0.014304 |
| 173 | 0.1 | +558.383293 | +557.526631 | −0.856662 | −0.849214 | −10.976195 | −10.941365 | +0.034831 | +0.034348 |

At full scale, the three-seed mean D_HO is −4.705583 micro-nats, versus raw's mean improvement +2295.091360 micro-nats; D_C is +0.168041 micro-nats, versus raw's mean deterioration −72.149614 micro-nats. These very small paired differences are not the large accumulated trajectory gap. Seed 171's clean advantage is less damage, not clean improvement. Seeds 172/173 retain positive clean progress, but slightly less under native.

Both draws agree on each primary seed's H_O contrast sign. C is less stable: seed 172's full-scale draw contrasts are −0.213172 and +0.514707 micro-nats, averaging positive. Seed 171 is negative in both draws; seed 173 positive in both. Thus even the two-draw consistency result should not be described as uniformly positive for seed 172.

## All eight cells, without selecting a favorable component

These are finite native-minus-raw contrasts. `h100` is the warmup parent and `final` the 56,304-step parent. An asterisk marks |D| ≤ 0.01 micro-nats, the registered 10⁻⁸-nat small-effect guard, not an equivalence margin.

| Parent augmentation / step / path | Seed | D_S | D_F | D_C | D_L | D_HO | D_HT |
|---|---|---:|---:|---:|---:|---:|---:|
| none / h100 / 1 | 171 | +1.999602 | −11.274219 | +6.614885 | −2.659733 | −111.973046 | −60.157441 |
| none / h100 / 1 | 172 | +2.051855 | −0.302691 | +2.144466 | +3.893629 | −42.815492 | −29.442767 |
| none / h100 / 1 | 173 | −2.703480 | −1.144771 | +2.616498 | −1.231752 | −97.050156 | −63.804869 |
| none / h100 / 0.1 | 171 | +0.173180 | −1.111941 | +0.647790 | −0.290971 | −11.209788 | −6.057732 |
| none / h100 / 0.1 | 172 | +0.135328 | −0.050709 | +0.214241 | +0.298860 | −4.584181 | −3.063551 |
| none / h100 / 0.1 | 173 | −0.352760 | −0.086065 | +0.246317 | −0.192508 | −10.011536 | −6.496144 |
| none / final / 1 | 171 | +0.308315 | +0.604880 | −0.262925 | +0.650270 | +17.092673 | +10.704328 |
| none / final / 1 | 172 | −0.535924 | −1.828555 | −0.170928 | −2.535407 | −4.457638 | −2.252676 |
| none / final / 1 | 173 | +0.009134* | −2.342443 | −0.076238 | −2.409548 | −14.019898 | −8.780655 |
| none / final / 0.1 | 171 | +0.021336 | +0.062633 | −0.029496 | +0.054472 | +1.609259 | +1.033952 |
| none / final / 0.1 | 172 | −0.058212 | −0.182758 | −0.017492 | −0.258462 | −0.389487 | −0.271407 |
| none / final / 0.1 | 173 | +0.002929* | −0.233743 | −0.007278* | −0.238092 | −1.402140 | −0.839997 |
| translate / h100 / 1 | 171 | +1.056403 | +1.307267 | +2.821756 | +5.185426 | −84.682067 | −61.734093 |
| translate / h100 / 1 | 172 | +4.373383 | −9.837373 | +1.991229 | −3.472761 | −58.751171 | −33.324580 |
| translate / h100 / 1 | 173 | +0.662272 | +0.037796 | +1.436406 | +2.136473 | −20.094797 | −18.760813 |
| translate / h100 / 0.1 | 171 | +0.101986 | +0.222091 | +0.269863 | +0.593940 | −8.039862 | −6.131269 |
| translate / h100 / 0.1 | 172 | +0.346813 | −0.918262 | +0.190540 | −0.380908 | −5.929052 | −3.514459 |
| translate / h100 / 0.1 | 173 | +0.006424* | +0.016299 | +0.133894 | +0.156617 | −2.140380 | −1.914557 |
| translate / final / 1 | 171 | −0.570795 | +1.680995 | −0.042952 | +1.067249 | +6.595461 | +0.412004 |
| translate / final / 1 | 172 | −0.012190 | +2.199367 | +0.150767 | +2.337944 | −11.897367 | −1.596918 |
| translate / final / 1 | 173 | +0.426028 | +0.990468 | +0.396308 | +1.812804 | −8.814843 | −10.780403 |
| translate / final / 0.1 | 171 | −0.061695 | +0.173522 | −0.005619* | +0.106209 | +0.546114 | +0.023098 |
| translate / final / 0.1 | 172 | −0.009256* | +0.235382 | +0.014623 | +0.240749 | −1.173593 | −0.182076 |
| translate / final / 0.1 | 173 | +0.027952 | +0.097484 | +0.034831 | +0.160267 | −0.856662 | −1.091569 |

The important qualifications are:

- At h100, C is relatively better while H_O and H_T are worse in all six parents at both scales. Absolute raw C improves only in seeds 171/172 for none and seed 172 for translate; elsewhere native reduces deterioration. Neither pattern is loss of useful consistency improvement under the protocol's joint criterion.
- At unaugmented final parents, C is relatively worse in all seeds; H_O/H_T are better in seed 171 and worse in 172/173. But raw C already deteriorates in 172/173, so those clean-adverse cases again do not demonstrate removed consistency progress.
- At translated final parents, native has positive D_F and D_L in every seed at both scales, but full-step L worsens under both policies: raw E_L = [−130.055424, −948.456179, −460.892936], native = [−128.988175, −946.118235, −459.080132]. The advantage is less deterioration, not noisy-loss learning on this disjoint panel. F itself decreases in seed 171 but increases in 172/173; native accentuates the former and attenuates the latter. A single-view action with inherited Adam moments need not decrease the disjoint 25-view panel loss.
- S is genuinely mixed. Since S = 0.1 S_true + 0.9 S_uniform, its full-step primary contrasts conceal opposing parts: D_Strue = [+2.663428, −0.747596, −12.622015] and D_Suniform = [−0.930153, +0.069522, +1.875811]. Seed 173's positive D_S is therefore not a clean-label benefit. S_true here is CE of mean logits on the training panel, not H_T.

## Actual wrong-label fit, not F alone

The next table uses only the actually wrong subset. Each entry is the seed-ordered triple (171, 172, 173) of native-minus-raw CE improvement. Positive assigned-label values indicate relatively more fitting of wrong targets; positive true-label values indicate relatively more correct-label progress. These are not the same desirability direction.

| Parent / step / path | Wrong assigned CE: original | Wrong true CE: original | Wrong assigned CE: 25-view mean |
|---|---|---|---|
| none / h100 / 1 | +30.756621, +14.962488, +32.709210 | −116.177103, −47.212802, −120.671321 | +14.676156, +10.746875, +17.685495 |
| none / h100 / 0.1 | +2.937659, +1.391791, +3.363509 | −11.317185, −4.988489, −12.220567 | +1.448896, +0.992296, +1.670502 |
| none / final / 1 | −2.601682, −4.084717, −3.868456 | +7.153097, −8.879675, +1.836365 | −1.812870, −1.533588, −1.423282 |
| none / final / 0.1 | −0.318516, −0.423877, −0.363016 | +0.727509, −0.797539, +0.207896 | −0.192084, −0.158209, −0.142592 |
| translate / h100 / 1 | +29.945139, +4.355224, +6.931972 | −78.752417, −66.478657, −35.977089 | +24.870545, +5.090854, +9.063238 |
| translate / h100 / 0.1 | +3.224122, +0.630090, +0.430287 | −7.877720, −6.551533, −3.566989 | +2.573394, +0.512411, +0.844683 |
| translate / final / 1 | −2.011719, +2.686962, +0.808132 | +5.220144, −12.608128, −10.901320 | −0.103424, +0.573676, +2.938937 |
| translate / final / 0.1 | −0.247778, +0.259096, +0.073385 | +0.503415, −1.307037, −1.010510 | −0.006268*, +0.068066, +0.268412 |

The adverse warmup CE ordering relative to raw holds also for true-label 25-view CE in all six parents. However, native's original wrong-target accuracy is exactly identical to raw in every parent and path after averaging, and the individual endpoints also have identical counts. On translated-final parents, 25-view wrong-target accuracy contrasts at full scale are [0, +0.009852, 0] percentage points; all are zero at tenth scale. Original clean reporting accuracy contrasts are zero there at both scales. Discrete accuracy cannot resolve these tiny CE differences.

At translated final states, even the two positive wrong-target CE contrasts describe less *unfitting*: raw original wrong-target improvements are [+188.483780, −1265.472585, −923.839577] micro-nats, and native [+186.472061, −1262.785622, −923.031445]. Seed 171 has less actual wrong fitting under native, despite its positive D_F. This directly rules out equating every F advantage with additional wrong-label memorization.

## Linearity, scale, and interpretation

All 144 seed-averaged contrasts across eight cells and six objectives retain their arithmetic sign between finite and linear readouts. Six finite contrasts are within the registered small-effect guard (marked above); none of the corresponding 144 linear contrasts is within its 10⁻¹⁰-nat guard. These are descriptive sign checks, not significance tests. Full-versus-tenth directions also persist, but the tenth check does not establish perfect linearity: for primary seed 171, raw H_O linear utility is −312.835705 micro-nats at full scale versus finite −546.351894, and at tenth −31.284375 versus −43.771899. The paired contrast is much closer than either absolute effect. Raw C deterioration remains negative even in the primary linear readout for every seed, so the central conclusion is not caused solely by full-step curvature.

Common decay cannot explain the paired contrast: both policies have the same rounded decay reference at each scale, so subtracting it leaves D unchanged. It matters for absolute interpretation: primary full-scale decay alone worsens H_O by 8.63–8.84 micro-nats and improves C by 0.159–0.232 micro-nats. Actual native/raw data-displacement norm ratios range from 0.980804–0.992421 at h100 and 0.996936–1.000756 at final parents across the 12 draws at each stage. Thus these final readouts are not a gross step-norm collapse. They also do not isolate direction from magnitude: no norm-matched control was part of this diagnostic, and inherited Adam moments remain active.

The strongest constructive reading is conditional: filtering can improve a consistency component, can attenuate an assigned-loss deterioration, and can sometimes protect clean CE or reduce wrong fitting. The final unaugmented cells preserve several such positives. These observations fit selective restrictions on learning better than a universal failure story. The strongest adverse reading is likewise conditional: relative to raw, native's warmup action consistently gives worse clean-label utility and lower assigned-wrong-label CE; this can mean less unfitting rather than absolute wrong-label fitting. Final translated native actions still slightly sacrifice clean progress relative to raw in two seeds. Neither is a truth oracle, and no observed component supplies a universal explanation.

The earlier strong study remains the trajectory evidence: unaugmented native reached 79.753% final accuracy versus raw 32.000%, with +43.053 percentage points of native progress after h100; selected native accuracy also exceeded raw (82.973% versus 72.487%) despite worse selected CE. In contrast, translated native finished at 66.820% versus translated raw 84.973%, adverse in every seed on both final accuracy and CE. This fixed-parent analysis does not replace those results or explain their accumulated difference. It uses only native-history parents, two action draws, small reused panels, and outcome-informed follow-up states. Even the none-parent action is translated, a hypothetical augmentation onset rather than its actual next training input.

## Scalar-only reproduction

Run from the repository root. This reads only the completed audit JSON, imports no experiment module, and prints the finite/linear tables and wrong-subset CE contrasts without recomputing any scientific readout:

```python
import json
from math import fsum
from pathlib import Path

path = Path('output/2026-09-10-spectral-component-utility/audit.json')
a = json.loads(path.read_text())
assert a['status'] == 'PASS'
for cell in a['independent_summary']['cells']:
    tag = (cell['augmentation'], cell['step'], cell['fraction'])
    for row in cell['seed_rows']:
        for key in ('S', 'F', 'C', 'L', 'H_O', 'H_T'):
            v = row['objectives'][key]
            print(tag, row['seed'], key,
                  {name: value * 1e6 for name, value in v.items()})
for parent in a['checked_parents']:
    for fraction in (1.0, 0.1):
        for view in ('wrong_original', 'wrong_per_view'):
            for label in ('assigned', 'true'):
                baseline = parent['baseline_metrics']['I'][view][label]['ce']
                effects = {}
                for policy in ('raw', 'native'):
                    es = [e for e in parent['endpoints']
                          if e['fraction'] == fraction and e['policy'] == policy]
                    assert len(es) == 2
                    effects[policy] = fsum(
                        baseline - e['metrics']['I'][view][label]['ce']
                        for e in es) / 2
                print(parent['seed'], parent['augmentation'], parent['step'],
                      fraction, view, label,
                      (effects['native'] - effects['raw']) * 1e6)
```
