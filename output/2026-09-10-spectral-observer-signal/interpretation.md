# Observer signal: the rare grouping boundary is already visible at delivery

Codex — Spectral Optimizer Investigation, 10 September 2026.

**Post-hoc saved-vector diagnostic on three reused, outcome-informed parents.**
The incoming block gradient has a positive inner product with the true rare
probe in all six Clean/Diffuse cases. Diffuse has larger rare inner products,
but lower rare-input cosines. Thus these cases do not exhibit an absent or
reversed *net rare input dot*; they do exhibit weaker directional alignment
under Diffuse. Grouping's rare contrast is positive in every Clean parent
and negative in two Diffuse parents already at filter delivery. Those six
signs agree with the previously audited Adam and heldout-loss contrasts.

This locates an adverse **relative** rare effect before Adam for those two
Diffuse parents. It does not make their absolute delivered rare directions
harmful: all native rare delivered dots remain positive. Nor does it show
that Adam generally preserves order, that accessibility causes utility, or
that this local contrast mediates the earlier training endpoint.

## Evidence, population and units

The numerical source is the archived [result JSON](results/result.json),
SHA256 `20d0514e3e6d5b0453d781957e38ac549c4c8269892657c5b76ee594e3d8cb2c`.
Its hash was checked before scalar interpretation. The
[protocol](protocol.md), [implementation](implementation.md),
[source review](source-review.md), preflight (artifact not distributed in this public snapshot) and
admission (artifact not distributed in this public snapshot) define the fixed analysis. The full
[48-row action table](results/all-actions.csv) complements the rounded
tables below; the JSON also contains every control contrast and numerical
consistency record.

Seeds throughout are **202609121 / 202609122 / 202609123**, in that order.
Clean and Diffuse share these three early parents, true-label probes and
example occurrences. They are not six independent replicates. The rare
probe is a fixed 32-example training-gradient average; the common probe is
a fixed 54-example stratified training-gradient average. Neither is a
heldout gradient or a decomposition into the rare examples' contribution
to the action input. Common heldout loss is the nine-class macro average.

Let `q` be a true-label probe, `g` the incoming averaged block gradient and
`h_s` the saved O151 delivered action. Within each cell, native and raw actions
start from the same `g`; the zero action delivers zero. All actions use the same
copied model/Adam state. This deliberately smoothed
block mean is not a newly sampled next minibatch. Define:

| Quantity | Definition | What a positive value means |
| --- | --- | --- |
| Input B | `qᵀg` | Alignment useful for a hypothetical sufficiently small step along `−g` on this training probe |
| Delivered F | `qᵀh_s` | The corresponding gradient-descent alignment after delivery |
| Filter change K | `qᵀ(h_s−g)` | Delivery increases that signed dot relative to raw input |
| Actual-step J | `−qᵀΔθ_s` | First-order helpfulness of the saved actual Adam displacement on the probe |
| Heldout U | `CE_before−CE_after` | Actual finite heldout group-loss improvement |

`D_filter`, `D_Adam` and `D_U` are Grouped minus Interleaved differences in
F, J and U. F/K are gradient inner products; J is a first-order loss
quantity; U is a finite CE change. Their magnitudes are not interchangeable,
and no conversion-efficiency or retained-useful-fraction ratio is estimated.

## 1. Input: positive rare dot, lower Diffuse orientation

| Cell / probe | B, all seeds | Input cosine, all seeds |
| --- | --- | --- |
| Clean / rare | +2.200128 / +2.551284 / +2.145365 | 0.385147 / 0.447243 / 0.430507 |
| Diffuse / rare | +4.769761 / +4.612978 / +5.276975 | 0.161639 / 0.178099 / 0.169654 |
| Clean / common | +0.089687 / +0.120573 / +0.064709 | 0.278376 / 0.408934 / 0.305274 |
| Diffuse / common | −0.159239 / −0.200426 / −0.101145 | −0.095680 / −0.149710 / −0.076448 |

Mean rare B is 2.298926 in Clean versus 4.886571 in Diffuse, while mean
rare-input cosine falls from 0.420966 to 0.169797. Input norms rise from
`0.377108 / 0.402988 / 0.315704` to
`1.948029 / 1.829769 / 1.970522`. The rare probe norms are unchanged across
cells (`15.148033 / 14.155438 / 15.784842`); so are common probe norms
(`0.854343 / 0.731655 / 0.671420`). A larger raw dot must not be relabeled
better orientation or a better optimizer step.

The matched label-input contrast `qᵀ(g_Diffuse−g_Clean)` is
`+2.569633 / +2.061694 / +3.131610` for rare and
`−0.248926 / −0.320999 / −0.165854` for common. This is an effect of changing
assigned labels on the total input at fixed model/examples. Rare labels
themselves are uncorrupted. It is not evidence that the rare examples
contributed more signal, or that corrupted labels are generally beneficial.
The common input dot actually reverses sign in every parent.

## 2. Delivery: greater accessibility does not fix the sign of the contrast

All entries below are gradient inner products, not CE changes.

| Cell / probe | Interleaved F | Grouped F | Interleaved K | Grouped K |
| --- | --- | --- | --- | --- |
| Clean / rare | 2.061640 / 2.292026 / 1.876101 | 2.190203 / 2.542877 / 2.128154 | −0.138488 / −0.259258 / −0.269264 | −0.009925 / −0.008407 / −0.017211 |
| Diffuse / rare | 4.706770 / 4.653614 / 5.415138 | 4.750791 / 4.591974 / 5.278159 | −0.062991 / +0.040636 / +0.138162 | −0.018970 / −0.021004 / +0.001184 |
| Clean / common | 0.075288 / 0.094538 / 0.047442 | 0.076149 / 0.097367 / 0.049330 | −0.014399 / −0.026035 / −0.017267 | −0.013538 / −0.023206 / −0.015379 |
| Diffuse / common | −0.159774 / −0.208636 / −0.110263 | −0.163616 / −0.208957 / −0.111223 | −0.000536 / −0.008211 / −0.009118 | −0.004378 / −0.008532 / −0.010078 |

Raw has F = B and K = 0; zero has F = 0 and K = −B. Zero's cosine is
undefined, not zero. Native F is positive for rare in all cases, positive
for common in Clean, and negative for common in Diffuse. Native K is
negative throughout Clean and for Diffuse common; Diffuse rare K is mixed.
Thus the filter is not uniformly preserving or reducing a scalar amount of
"useful signal." Signed cross terms can increase even when a filter's
geometric retention is lower.

The [previous audited observer study](../2026-09-10-spectral-observer-pathway/interpretation.md)
found rare squared-norm accessibility rising from roughly 85–91% to 98–99%
in Clean and 26–35% to 97–99% in Diffuse. The present diagnostic adds the
actual input-to-probe cross terms. In Diffuse parents 122 and 123,
Interleaved F already exceeds raw B; Grouped F is closer to raw B but lower
than Interleaved F. Increased accessibility of a possible rare direction
therefore need not increase alignment of this delivered block input.

## 3. The primary rare contrast agrees across stages; common does not always

For visibility, J and U contrasts below are in **millinats/example**
(`1000 ×` their saved loss quantities); D_filter is unscaled.

| Cell / probe | D_filter, all seeds | 1000 D_Adam, all seeds | 1000 D_U, all seeds |
| --- | --- | --- | --- |
| Clean / rare | +0.128563 / +0.250851 / +0.252053 | +1.370289 / +4.277369 / +3.641038 | +1.414117 / +4.463830 / +3.399293 |
| Diffuse / rare | +0.044021 / −0.061640 / −0.136978 | +0.652785 / −0.982226 / −1.782256 | +0.752069 / −0.962077 / −1.638440 |
| Clean / common | +0.000860 / +0.002829 / +0.001888 | +0.003932 / −0.055435 / +0.025349 | +0.001719 / +0.001346 / −0.003959 |
| Diffuse / common | −0.003842 / −0.000321 / −0.000959 | −0.032580 / −0.009665 / −0.011015 | −0.000560 / +0.000709 / −0.003878 |

Mean rare D_filter is `+0.210489` in Clean and `−0.051532` in Diffuse;
their sample SEs are `0.040964` and `0.052494`. Mean rare D_U is
`+0.003092413` and `−0.000616150` nats/example, with sample SEs
`0.000893648` and `0.000711426`. These are descriptive three-parent
summaries, not independent-population or endpoint confirmation. Individual
nonzero primary filter signs clear the frozen numerical consistency bounds;
that numerical resolution is not statistical certainty.

## 4. Absolute steps and controls remain essential

The following are absolute action outcomes, not Grouped-minus-Interleaved
effects. All columns are ordered seed vectors in **millinats/example**.

| Cell / action | Rare 1000 J | Rare 1000 U | Common 1000 U |
| --- | --- | --- | --- |
| Clean / Interleaved | −7.479 / +2.835 / +20.518 | −13.715 / +3.499 / +22.387 | +2.497 / +3.847 / +1.952 |
| Clean / Grouped | −6.109 / +7.112 / +24.159 | −12.301 / +7.962 / +25.786 | +2.499 / +3.848 / +1.948 |
| Clean / raw | −6.757 / +7.452 / +24.089 | −12.908 / +8.158 / +25.710 | +2.658 / +4.029 / +2.110 |
| Clean / zero | −37.442 / −30.023 / −12.592 | −44.283 / −31.039 / −10.354 | +2.135 / +3.063 / +1.623 |
| Diffuse / Interleaved | +10.789 / +31.717 / +48.629 | −0.322 / +33.703 / +48.445 | +0.676 / +1.950 / +0.386 |
| Diffuse / Grouped | +11.442 / +30.735 / +46.847 | +0.430 / +32.741 / +46.807 | +0.675 / +1.951 / +0.382 |
| Diffuse / raw | +11.351 / +30.930 / +47.253 | +0.168 / +32.855 / +47.233 | +0.813 / +2.093 / +0.498 |
| Diffuse / zero | −37.442 / −30.023 / −12.592 | −44.283 / −31.039 / −10.354 | +2.135 / +3.063 / +1.623 |

Clean parent 121 has positive rare F under every nonzero action but
negative actual rare J and U. Grouping reduces damage there; it does not
make that action absolutely helpful on rare loss. In parents 122 and 123,
Clean grouping improves an already useful rare step. Diffuse Grouped has
positive rare U throughout, despite losing the relative comparison in two
parents. Diffuse Interleaved parent 121 has positive training-probe J and
true-label training rare U, yet a small negative heldout rare U. Probe
membership and finite-step effects are not separated by this evidence.

The common probe gives another clear distinction: all Diffuse nonzero F
values are negative, while all corresponding actual J and heldout U values
are positive. Gradient-descent intuition about the incoming action alone
does not describe the carried-state Adam step. Zero-gradient Adam itself
has nonzero movement and helpful common J/U, but harmful rare J/U, in every
parent. It is an active baseline, not a no-update control. `J_s−J_zero`
is a contrast through the same nonlinear Adam map, not an additive
decomposition of momentum and the current gradient.

All nonzero actions beat zero on rare J/U in every case. On common J/U,
they beat zero in Clean but lose to it in Diffuse. Raw beats both native
schedules on common heldout CE in every case. Raw also beats Clean
Interleaved on rare J/U throughout, while Grouped-versus-raw rare utility
is mixed in both cells. These controls preclude a blanket native-superiority
or cost-free noisy-label benefit claim.

Every rare training and heldout accuracy remains **0% before and after**
every action; this diagnostic adds no recognition rescue. Nonzero Diffuse
actions also improve assigned-target and actually-wrong-target training CE
in every parent. Wrong-target U is respectively
`0.017299 / 0.009351 / 0.013308` (Interleaved),
`0.017327 / 0.009360 / 0.013336` (Grouped), and
`0.017043 / 0.009038 / 0.013025` (raw), versus
`−0.018574 / −0.022900 / −0.027357` (zero).
These are target-fitting readouts, not rare-probe-specific quantities; their
duplication across probe joins must not be counted as independent evidence.
The result does not demonstrate suppression of wrong-label fitting.

## Scope and provenance

This was one saved-vector analysis, not a new training run. The analyzer
loaded 21 pinned archives once and reports 12 input alignments, 24 native
delivered alignments, 24 native filter changes, 48 logical action/probe
joins, 12 primary grouping and 72 total control contrasts, plus six label
input contrasts. FP64-before-difference products were compared with
compensated summation and checked algebraic identities under fixed
operational bounds. Exact-zero control identities should not be confused
with uncertain scientific nonzero signs.

The new JSON's PASS is **source-reviewed execution with internal compensated
numerical consistency checks, not a new independent scientific audit**.
Its J/U are joins from the earlier independent PASS audit, SHA256
`d9b05160c9a58952d4f0899699f79de2ee2d919c5319f2bb96d687ea448d0d35`;
those old model/gradient/loss readouts were not rerun. This interpretation
read saved scalar JSON only, with no tensor loading or new products.

Source freeze: `ec3285c4fc9f0991ac0307e70cde726a1861b2de`.
The completion receipt (artifact not distributed in this public snapshot), SHA256
`5951b22e03308117280efde69bd06a0fea1cb15956bc64aaa2fb58c8c7d683af`,
records 8.389 seconds and process high-water RSS 536,060 KiB under the
admitted one-CPU, 4 GiB/no-swap, 180-second service. The archived
manifest (artifact not distributed in this public snapshot) hash is
`5b3723ec825f9ee13f556d29ece6e60db96067c26c13a00377e66fca01ce2dce`.
The original run remains at
`/tmp/spectral-experiment-artifacts/spectral-observer-signal-20260910.PXKMpV/analysis-001/`.

The positive steelman is a concrete, signed local pathway: Clean grouping
increases rare delivered alignment and improves the same-state rare step
in all three parents, with two absolute benefits and one damage reduction.
The negative steelman is equally concrete: the much larger Diffuse gain in
rare-direction accessibility does not imply an increase in delivered rare
alignment or relative rare utility, and common/raw/zero controls expose
costs. The full observer intervention includes EMA-weighted mean, ordering,
centered finite-rank history and native direction/gain; these products do
not isolate population covariance or an Adam submechanism. Three reused
parents, finite training probes and one copied-state step per action cannot
establish endpoint mediation, calibrated predictions, semantic selection,
safety benefit or production utility.
