# J-Lens: a useful contrast is not yet a portable concept label

Codex — Spectral Optimizer Investigation · 11 September 2026

## Bottom line

The idea has produced useful local results. The best-supported interpretation
is a **description of a signed contrast in a specified activation population**,
not a discovered semantic cluster or a label that must work on arbitrary prose.
The natural-text add-on test does not show a reliable overall benefit over
examples. Its near-tie also hides consistent useful and adverse cases.

## What the experiments actually support

| Completed setting | Result | Defensible reading |
|---|---|---|
| New action words, fixed observation/provision forecast | PC4:16/16 at the local verb and16/16 at sentence end | A specific direction-based hypothesis made successful new-content predictions; these are forecasts, not reader accuracies |
| Readers on that known positive panel | Tokens116/128; examples98/128; PC4:32/32 vs26/32 | Local communication benefit on selected content; PC4 has one cohort win and one tie |
| New pre-frozen authored panel | Tokens108/128; examples106/128; PC4:26/32 in both arms | Local usefulness survives, but primary added benefit does not; all four PC4 readers make identical three errors |
| Earlier Wikipedia panel | Tokens50/96; examples48/96; PC4:19/24 vs12/24 | One secondary direction-specific positive, not a convincing all-direction advantage |
| Latest Wikipedia add-on panel | Examples+tokens74/128; identical examples alone75/128 | No demonstrated overall incremental value; different question from replacing examples with tokens |

Direct completed reports: [new-action forecast](../output/2026-09-11-jlens-new-verb-transfer/results.md),
[selected readers](../output/2026-09-11-jlens-verb-reader-comparison/results.md),
[fresh authored readers](../output/2026-09-11-jlens-fresh-authored-comparison/results.md),
[earlier external panel](../output/2026-09-11-jlens-independent-content/results.md),
[latest add-on result](../output/2026-09-11-jlens-natural-addon/results.md).
Do not pool these denominators: they reuse directions, templates and related
readers, have different endpoints, and ask different questions. These are small
single-model panels, not multi-model or independent-trial confirmation.

## The near-tie is not an absence of case differences

There are64 distinct direction/pair cases in the latest panel, each assessed
in two reader groups. Comparing examples+tokens with examples alone gives:

| Pattern across the two existing groups | Cases |
|---|---:|
| Higher credit in both | 5 |
| Lower credit in both | 6 |
| Opposite differences | 1 |
| Difference in one group only | 7 |
| Same credit in both | 45 |

Of the45 unchanged cases,27 are correct for all four readers,16 wrong for all
four, and2 have both arms correct in one group and wrong in the other. This
accounts for every case. Across128 group/direction/pair cells there are15
higher,16 lower and97 unchanged credits. "Both groups" means the two existing
groups, **not a new replication**. Arms use different readers: the contrasts
do not by themselves isolate the causal effect of a particular token.

A concrete useful candidate is PC4/N08: both tokens+examples readers choose
the dangling-else programming problem over performance portability correctly;
both examples-only readers choose the reverse. The original positive list
contains the exact token ` glitches`, whereas the raw examples are a goalkeeper
blocking and a recipe combining. That is a plausible extra association, not
observed reader reasoning or a validated general bug detector.

There are equally important counterexamples. On PC4/N04 both examples-only
readers correctly rank Code Club above Floyd's triangle, while both combined
readers reverse it. On PC4/N05 both combined readers rank psychology-of-
programming research above coding classes incorrectly. An observation/provision
reading could plausibly contribute to those errors, but that is retrospective.
On PC4/N14 all four readers misorder the cooking pair despite a gain on that
same pair for PC1. A single reusable semantic explanation is not established.

The [complete accounting](../output/2026-09-11-jlens-usecase-synthesis/analysis.json)
is reconstructed directly from the [frozen individual judgments](../output/2026-09-11-jlens-natural-addon/graded/grades.json),
not a new grade. An independent Astra report inventories all64 cases, exact
reference tokens, raw examples and prefixes: worker commit
`e53494f575bb5ecac44e2d1f7e5b30fb0f212739`, report SHA
`4a213cf2ab0fab84e98ca9b1201ccb0a32b2bf470b6d81e14c0144d6a37fe51c`.
Its separate broader-PCA recommendation is a different option from the
fixed-score calibration proposed below; those must not be conflated.

## Why a correct global description can fail within a topic

Let h be an activation vector, c the original centering vector, u a fixed unit
direction, and L a fixed linear vocabulary-readout surrogate. Define:

```text
z = uᵀ(h − c)                     the score we ask readers to predict
d = Cov(h,z) / Var(z)
  = C u / (uᵀ C u)                population activation pattern per score unit
slope of Lh on z = Ld             provided Var(z) > 0
```

If u is an exact eigenvector of this population's C, with positive eigenvalue,
then d=u and the slope is Lu. However, for a topic/group G=g:

```text
C_g = Cov(h | G=g)
d_g = C_g u / (uᵀ C_g u)
slope of Lh on z within group g = Ld_g

C = E[C_G] + Cov(E[h | G])
```

Being an eigenvector of pooled C does not make u an eigenvector of every C_g.
Changing the common centering vector c changes the intercept, not this slope
or a pairwise score gap. Changing relative group means can change pooled C.
These are per-axis least-squares slopes with an intercept, **not** necessarily
conditional expectations or joint partial-regression coefficients controlling
the other PCs. Covariance stationarity is relevant even when the model and
the axis do not change.

### An exact four-point example

Let G and Z be independent, equally likely ±1. No model is involved.

```text
h = (√3 G + Z, GZ)
C = diag(4,1)                 u = (1,0), the unique leading eigenvector
z = h₁                        L = (1,2), so t = Lh = h₁ + 2h₂

Pooled:      d = (1,0)        slope(t on z) = +1
Within G=+1: d = (1,+1)       slope(t on z) = +3
Within G=−1: d = (1,−1)       slope(t on z) = −1
```

The global description is exactly correct, yet one within-group relationship
has the opposite sign. This proves a possibility; it does **not** identify why
any of our readers erred. No within-topic covariance or readout slope was
measured in this synthesis. An illustrative plot is labeled separately from
the saved model-result plot.

### What this does and does not say about J-Lens

The real J-Lens applies normalization after its learned linear transformation.
The Lu argument concerns a bias-free, unnormalized linear surrogate and the
associated direction-token ordering under ideal RMS normalization. It is not
an identity for normalized example logits, probabilities or local causal
effects. Human-readable top tokens are yet another compression; a reader must
then infer scores from text. Each link can be lossy.

The original J-Lens averages downstream Jacobians over prompts and positions;
its method and limitations distinguish vocabulary readouts from full relational
descriptions. Our activation-PCA extension is not its original experiment.
[Primary J-Lens article](https://transformer-circuits.pub/2026/workspace/index.html).

The distinction between an extraction filter and its population activation
pattern is established in Haufe and colleagues' work. Their author poster
gives A=Σx W Σs⁻¹; our d is its one-score least-squares form. This is an
application of existing linear-model reasoning, not a new theorem or a
neuroscience-style localization claim about a transformer.
[Primary author poster](https://f1000research-files.f1000.com/posters/docs/263125503).

Testing whether an explanation predicts held-out behavior is also established
methodology, including generation, simulation and scoring in
[OpenAI's automated-interpretability repository](https://github.com/openai/automated-interpretability).
Our open question is the utility of this specific population-conditioned display.

## The next discriminating question: recalibrate the description, not the target

Keep all four original u vectors, score signs, model, layer and observation
rule fixed. On a new, document-disjoint calibration population, estimate:

```text
d_j = C_cal u_j / (u_jᵀ C_cal u_j)

Original display:    J-Lens(+u_j), J-Lens(−u_j)
Calibrated display:  J-Lens(+d_j), J-Lens(−d_j)
Evaluation target in BOTH arms: z_j = u_jᵀ(h − c)
```

This notation uses the actual normalized lens for the proposed display; the
linear-surrogate algebra motivates it but does not guarantee improved tokens.
In exact arithmetic u_jᵀd_j=1: the calibrated pattern describes covariation
with the same coordinate, not a replacement score. It can include orthogonal
components. It is observational and should not replace u for an intervention.

Before new data, fix the calibration/evaluation population and document split,
token role, sample budget, centering/precision, variance floor and failure policy,
normalization implementation, identical example panels and12 tokens per pole.
Use the same held-out score questions in both arms and retain all four axes as
primary. New readers must be locked before labels as before; no selection of
axes, articles or descriptions by held-out outcomes. A pooled natural-text
calibration and topic-conditioned calibration are different estimands: choose
one in advance, not whichever works afterward.

The useful outcome would be a repeated held-out prediction gain over the
original display with the same examples, not just more appealing words or
closer fit to calibration states. No gain would leave the algebra intact but
limit this proposed remedy. Do not respond by repeating similar panels until
positive. Refit-PCA is a separate, broader future question because it changes
the measured coordinates themselves.

## Feasibility: no full covariance matrix needed for fixed axes

For activation width m, k fixed directions U and each frozen-model state h_t,
compute z_t=Uᵀ(h_t−c). Store means, the m×k cross-scatter M_hz and k score
scatters. This concerns layer activations, not a parameter-update covariance:

```text
δh = h_t − mean_h_(t−1)           δz = z_t − mean_z_(t−1)
mean_h_t = mean_h_(t−1) + δh/t
mean_z_t = mean_z_(t−1) + δz/t
M_hz_t = M_hz_(t−1) + ((t−1)/t) δh δzᵀ
M_zz,j,t = M_zz,j,(t−1) + ((t−1)/t) (δz_j)²
d_j = M_hz[:,j] / M_zz,j          when the denominator is positive
```

The sample-covariance denominators cancel. This is an exact centered online
cross-moment identity in exact arithmetic, with O(mk) extra storage and work
per state, excluding model-forward cost. It does not learn new leading
eigenvectors, approximate an entire covariance matrix, or estimate joint
partial slopes. Model parameters never update. The present helper checks
this recurrence only on the four-point mathematical example, not on model data.

## Confidence, scope and safety

High confidence in the algebra and saved arithmetic; provisional confidence
in the local use-case interpretation; no evidence yet that calibration improves
usefulness. The latest empirical panel has16 pairs, four same-family AI readers,
one small frozen model/layer and an unbalanced source frame (no football).
Reader/arm and context effects remain. Its pre-model runtime failure and explicit
first-pass amendment are preserved. Stored transport prompts are encrypted, so
exact first responses/settings were checked but independent plaintext dispatch-
byte verification is unavailable. No claim of equivalence, distinct semantic
clusters, alignment benefit or deployment safety follows.

This route remains a frozen-model interpretation question. It avoids making
training speed the objective and does not revive deferred optimizer work.
It could clarify what a direction-based explanation means; broader practical
or safety value would still need separate evidence.
