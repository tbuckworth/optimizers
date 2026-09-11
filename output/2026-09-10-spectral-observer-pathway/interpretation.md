# Matched-parent observer pathway: a useful local effect, with a sharp boundary

Codex — Spectral Optimizer Investigation, 10 September 2026.

The strongest mechanistic boundary is equally informative. Grouping greatly
increases accessibility of the true rare probe gradient under **both** label
conditions, yet that accessibility does not reliably improve the Diffuse
action's rare utility. Retaining a helpful possible direction and taking a
helpful actual step are different properties.

## Evidence and sign convention

The sole numerical source is the independently checked
audit JSON (artifact not distributed in this public snapshot),
SHA256 `d9b05160c9a58952d4f0899699f79de2ee2d919c5319f2bb96d687ea448d0d35`,
verified before reading. It reports PASS, zero errors and 33,032 checks:
six cases, 12 histories, 600 stream gradients, six oracle gradients,
21 physical steps and prediction pairs, and 24 logical readouts. This note
uses only its `checked_cases`, `checked_histories`, `checked_readouts` and
`independent_summary`, plus light arithmetic on their saved scalars. No
scientific tensors, predictions or models were loaded, and no acquisition,
inference, differentiation, observer replay or audit was rerun.

Interpretation follows the prospective [plan](interpretation-plan.md) and
[corrected mathematical note](mathematical-readout.md). Define useful CE change
as `U = CE_before − CE_after`; positive is helpful. The primary contrast is
`D = U_grouped − U_interleaved`. Common means the nine-class macro average;
rare means digit 8. Seeds below are always **202609121, 202609122, 202609123**.
Clean and Diffuse reuse these same three parents; they are not six replicates.

## Absolute usefulness and the primary contrast

The complete primary held-out absolute results are below. Entries are ordered
three-seed vectors in **millinats/example** (`1000 × U`), rounded for display.
The unrounded source remains authoritative.

| Cell / delivery | Rare U, seeds 121 / 122 / 123 | Common U, seeds 121 / 122 / 123 |
| --- | --- | --- |
| Clean / Interleaved native | −13.715 / +3.499 / +22.387 | +2.497 / +3.847 / +1.952 |
| Clean / Grouped native | −12.301 / +7.962 / +25.786 | +2.499 / +3.848 / +1.948 |
| Clean / raw | −12.908 / +8.158 / +25.710 | +2.658 / +4.029 / +2.110 |
| Clean / zero-gradient Adam | −44.283 / −31.039 / −10.354 | +2.135 / +3.063 / +1.623 |
| Diffuse / Interleaved native | −0.322 / +33.703 / +48.445 | +0.676 / +1.950 / +0.386 |
| Diffuse / Grouped native | +0.430 / +32.741 / +46.807 | +0.675 / +1.951 / +0.382 |
| Diffuse / raw | +0.168 / +32.855 / +47.233 | +0.813 / +2.093 / +0.498 |
| Diffuse / zero-gradient Adam | −44.283 / −31.039 / −10.354 | +2.135 / +3.063 / +1.623 |

The primary paired summary, in **nats/example**, is:

| Cell / group | D, all three seeds | Mean | Sample SD | Sample SE | Positive / negative |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean / rare | +0.001414117 / +0.004463830 / +0.003399293 | +0.003092413 | 0.001547843 | 0.000893648 | 3 / 0 |
| Clean / common | +1.718502e−6 / +1.346196e−6 / −3.958719e−6 | −2.980069e−7 | 3.175730e−6 | 1.833509e−6 | 2 / 1 |
| Diffuse / rare | +0.000752069 / −0.000962077 / −0.001638440 | −0.000616150 | 0.001232227 | 0.000711426 | 1 / 2 |
| Diffuse / common | −5.603472e−7 / +7.093005e−7 / −3.877599e−6 | −1.242882e−6 | 2.368396e−6 | 1.367394e−6 | 1 / 2 |

Clean's favorable rare contrast is not solely a damage-reduction result:
parents 122 and 123 obtain positive absolute rare U, improved further by
grouping. Parent 121 instead gets reduced damage. Grouped rare mean U is
`+0.007149183 ± 0.011002411` sample SE, versus Interleaved
`+0.004056770 ± 0.010425559`. The tiny mixed common contrast is retained as
signed evidence, not declared equivalence or zero cost.

In Diffuse, Grouped achieves positive rare and common U in all three parents.
Its rare mean is `+0.026659243 ± 0.013728841` SE. Nevertheless, Interleaved
has a slightly larger mean rare U, `+0.027275393 ± 0.014440101`; grouping
hurts the relative rare result in parents 122 and 123. Thus both a useful
absolute noisy-label step and an adverse grouping contrast are true.

Every rare held-out and training accuracy remains zero, before and after all
readouts. Initial held-out rare CE is 8.5673, 9.4169 and 8.4546; these one-step
changes are not rare recognition rescue. True-label training CE broadly
preserves the rare ordering: Clean grouping helps all three, Diffuse helps
only parent 121. Diffuse Interleaved's first parent improves training rare CE
despite its small adverse held-out rare U. Assigned and actually-wrong
training CE improve under all nonzero Diffuse actions; mean wrong-label U is
0.013320/0.013341/0.013035 for Interleaved/Grouped/raw. This local readout does
not demonstrate suppression of wrong-label fitting.

## The historical action difference survives self-inclusion

The common-gradient mean discrepancy is only 1.25e−8–3.58e−8 in norm, versus
fixed bounds 1.34e−5–4.64e−5; every case passes. This is numerical admission
of the common unweighted input, not equality of observer histories.

| Cell / seed | History action distance, O150 → O151 | Action cosine, O150 → O151 |
| --- | --- | --- |
| Clean / 121 | 0.053643 → 0.053628 | 0.988178 → 0.988184 |
| Clean / 122 | 0.059275 → 0.059259 | 0.987584 → 0.987591 |
| Clean / 123 | 0.056518 → 0.056472 | 0.980292 → 0.980324 |
| Diffuse / 121 | 0.044091 → 0.043998 | 0.999743 → 0.999744 |
| Diffuse / 122 | 0.028630 → 0.028617 | 0.999877 → 0.999877 |
| Diffuse / 123 | 0.051251 → 0.051213 | 0.999660 → 0.999661 |

O150 is descriptive; only O151 is delivered. The common final observation
barely changes these distances, rather than erasing the history pathway.
Clean delivered norms are Interleaved `0.348249, 0.373259, 0.281331` versus
Grouped `0.349327, 0.377310, 0.286053`. Diffuse norms are approximately
`1.9443, 1.8260, 1.9667` for either schedule. Both direction and gain remain
parts of the intervention; it is not norm matched.

At O151, rare oracle **squared-norm retention** changes as follows:

| Cell | Interleaved, all seeds | Grouped, all seeds |
| --- | --- | --- |
| Clean | 91.44% / 84.54% / 90.20% | 99.13% / 98.34% / 99.15% |
| Diffuse | 34.78% / 26.28% / 31.87% | 98.64% / 97.17% / 98.49% |

All numerical bases retain rank 32; no identity fallback supplies this result.
This is strong evidence of changed rare-gradient accessibility. But the
Diffuse delivered input itself retains about 99.6% of its energy under either
observer. Being able to pass an oracle rare direction is not equivalent to
receiving that direction in this assigned-target block mean. The large
accessibility gain with mixed/adverse rare utility directly limits an
accessibility-only explanation. It does not show the inaccessible component
was harmful, or identify which retained component caused the loss difference.

## Actual Adam movement and oracle utilities

Native adaptive displacement has **84.84–95.39% of its squared norm outside
the corresponding delivered observer's numerical span**. Incoming projection
is therefore not confinement of the actual parameter step. This reflects the
combined coordinatewise Adam mapping and inherited state; the fraction is not
a measured causal contribution of momentum, adaptation or any one mechanism.
Native total step norms lie between 0.04714 and 0.04928. Raw is larger in every
case (0.04974–0.05184), while zero-gradient Adam still moves 0.04352–0.04705.
The latter is an essential active reference, not a no-update baseline.

For the 32-example rare true-label oracle, Grouped-minus-Interleaved
`−g_probe · actual_displacement` is `+0.0013703, +0.0042774, +0.0036410`
in Clean and `+0.0006528, −0.0009822, −0.0017823` in Diffuse. These signs
agree with the corresponding primary rare contrasts. Absolute rare oracle
utility is negative in Clean parent 121 and positive for the other two;
it is positive for both native schedules in all Diffuse parents. Zero's
rare oracle utility is negative in every parent. The common oracle utility
is positive for every physical readout, but its paired grouping signs are
mixed in Clean and negative throughout Diffuse.

These are useful signed diagnostics, not separately measured held-out
gradients. In particular, positive Diffuse oracle utility for Interleaved
parent 121 coexists with adverse held-out rare CE. Different probe/population
membership and finite-step nonlinearity remain possible reasons; this record
does not apportion them. As the corrected mathematical note stresses,
differentiability alone gives an `o(||Delta||)` remainder, not an automatic
quadratic error bound across a finite ReLU step.

## What this adds, and what remains open

The favorable Clean result establishes more than geometric motion: changing
the full observer history can improve a common-input finite useful step at
fixed model and Adam states. The Diffuse result shows why that mechanism
cannot be summarized as “grouping makes rare directions accessible, therefore
rare learning improves.” Accessibility rises much more there without a
consistent relative utility benefit, and raw/zero expose important costs.

These are three reused, outcome-informed early parents and a deliberately
smoothed block-mean input, not ordinary next-minibatch continuation. The
observer intervention includes exponentially weighted means, centered history,
finite-rank state and numerical ordering, as well as native gain and direction.
It does not isolate population covariance, establish semantic selection or a
safety benefit, explain the completed step-2000 batching result, or establish
long-run mediation. It supplies a concrete local pathway and an experimentally
observed boundary for the paper, while leaving those stronger claims open.
