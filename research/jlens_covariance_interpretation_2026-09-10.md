# What does a covariance direction mean through J-Lens?

Codex — Spectral Optimizer Investigation · 10 September 2026

Status: **conditional mathematics and source interpretation**, not a new
experiment, causal diagnosis or novelty claim. Open the readable HTML (artifact not distributed in this public snapshot).

The useful interpretation is a **population contrast**: a direction can
summarize how vocabulary-associated features vary across examples. It is not
itself a representative example, and it need not identify a separate cluster.
This gives the user's idea a precise mathematical foundation without
requiring every recognizable token list to be correct.

## 1. What the completed comparison already establishes

In the [fresh-text test](../output/2026-09-10-jlens-fresh-content/results.md),
signed-direction token lists supported 11/12, 9/12 and 9/12 correct orderings
on the first three axes, but only 2/12 on the fourth. Overall, direction
tokens scored 31/48, selected-example tokens 33/48 and selected-example text
34/48. Direct-token and raw-example choices coincide on all 36 first-three-axis
targets, not just their totals. See the [immutable grade](../output/2026-09-10-jlens-fresh-content/graded/grades.json)
and [saved-output audit](../output/2026-09-10-jlens-fresh-content/result-audit.md).

That is useful small-panel correspondence, not just attractive vocabulary.
It does not show an overall practical advantage over inspecting selected
examples. All arms use the same PCA geometry; the comparison is not PCA
versus no PCA. There are 24 authored texts / 12 shared pairs within four
familiar topics and same-model-family readers—not 144 independent cases.
Nothing below changes that grade, drops PC4 or explains its cause.

## 2. The exact population-contrast identity

Let h be a residual activation, μ = E[h], and C = E[(h−μ)(h−μ)ᵀ]. Let v be
a unit covariance eigenvector: Cv = λv, λ > 0. Its signed score is
z = vᵀ(h−μ). For **any fixed linear map L**, elementary least squares gives:

```text
Cov(Lh, z) = LCv = λLv
Var(z)    = vᵀCv = λ
β         = Cov(Lh,z) / Var(z) = Lv
prediction of Lh from z = Lμ + zLv
```

Thus decoding v through L gives a slope, while decoding an example h gives
a value. This holds exactly for empirical covariance/regression using the
same fit rows and a consistent covariance divisor. It is a standard linear
regression identity, not a newly discovered theorem.

For an arbitrary saved direction u, the exact coefficient is instead
βᵤ = LCu / (uᵀCu), provided uᵀCu > 0. A nonunit exact eigenvector gives
Lu / ‖u‖². The actual
[protocol](../output/2026-09-10-jlens-fresh-content/protocol.md) saved normalized
float32 directions: their near-eigenvector identity is approximate, while
their actual scoring/decoding identity is explicitly pinned. The formula
with C from the fit population also need not hold for a changed fresh
population C′; there the coefficient is LC′u / (uᵀC′u).

An exemplar decomposes as h = μ + zv + r, with vᵀr = 0. Its linear readout
contains **Lμ + zLv + Lr**. Removing mean and other components can make a
population contrast cleaner than an exemplar. Conversely, those components
may carry information that an isolated direction omits. Centering cancels
from pairwise score differences, but not from this interpretation.

## 3. What normalization and the implementation change

The original method transports a residual through a fixed average Jacobian,
then applies the final normalization and vocabulary head. Its ranked-token
readout and its sparse nonnegative concept decomposition are different
operations; this pilot used ranked tokens, not that decomposition.
[Primary method description](https://transformer-circuits.pub/2026/workspace/index.html).

The pinned local transport (artifact not distributed in this public snapshot)
implements y = Jh. Its unembed (artifact not distributed in this public snapshot)
casts to the model dtype, applies final normalization, then the head.
The installed Qwen3.5 RMSNorm (artifact not distributed in this public snapshot)
has a diagonal gain D = diag(1 + weight), no additive bias; the language head
is also bias-free (same source, lines 1699 / 1810). In real arithmetic:

```text
s(y) = √(‖y‖²/d + ε) > 0
N(y) = Dy / s(y)
vocabulary logits at v = W N(Jv) = (WDJv) / s(Jv)
L = WDJ
```

Within a single readout, positive scalar normalization and softmax preserve
token ordering. Consequently the ranked direction list ranks the entries of
Lv: the slopes of the **unnormalized linear surrogate** Lh. Negative v
reverses those ideal logits. This is not a statement that softmax
probabilities negate, nor a license to flip a failed axis after evaluation.
Token-dependent biases would break the simple slope-ranking equivalence;
finite-precision casts can affect close ranks in the actual implementation.

Two important distinctions follow. First, the denominator changes from
example to example, so this is not regression of actual normalized example
logits. Second, applying the lens to v is not taking the lens's derivative
along v at an example. For a perturbation δ of y:

```text
DN(y)[δ] = Dδ/s(y) − Dy (yᵀδ)/(d s(y)³)
derivative of lens logits at μ along v = W DN(Jμ)[Jv]
```

Normalization substantially removes input magnitude, so a clean token list
does not reveal eigenvalue, explained variance or predictive reliability.
It exactly removes magnitude for ε = 0; with ε > 0 this is approximate.

The checked fitting source (artifact not distributed in this public snapshot)
sums derivatives over valid downstream target positions and averages valid
source positions, then prompts. That is not necessarily the final-prefix-
position-to-final-prefix-position Jacobian relevant to a particular example.
The source checkout is pinned to 581d398613e5602a5af361e1c34d3a92ea82ba8e;
the saved lens's entire fitting history is not certified by this inspection.
Our algebra is valid for any fixed J—even one that is a poor local causal
approximation. It does not validate that approximation or model behavior.

## 4. When a clear direction is also reliable

For one scalar surrogate score t = aᵀh, the eigenvector identity gives:

```text
slope of t on z = aᵀv
R² = λ(aᵀv)² / (aᵀCa)                 [when aᵀCa > 0]
residual variance = aᵀCa − λ(aᵀv)²
```

The signed list exposes slope ordering, not this residual variance. A
large share of activation variance λ / tr(C) need not be a large share of
variation in a particular vocabulary score. Even a correct conditional
mean E[t|z] does not guarantee correct individual pair ordering.

Here are two **analytic constructions**, not fitted explanations of PC4.
Take h = (z,r), with independent equiprobable z = ±1 and r = ±1/4. Then
C = diag(1,1/16); its unique top PC is v = (1,0), with a substantial eigengap.
Set J = I, D = I and use two token rows +aᵀ and −aᵀ:

| Property | Useful semantic direction | Readable but unreliable proxy |
| --- | --- | --- |
| a | (1,0) | (1,8) |
| Score t | z | z + 8r |
| Decoded signed direction | positive / negative token | the same positive / negative token |
| E[t\|z] | z | z |
| R² | 1 | 1/5 |

In the first construction, if z is a semantic quantity the tokens express
and new wording preserves it, an informed reader can order fresh examples
perfectly. This is a genuine useful regime, not just absence of a confound.

In the second, compare hₐ = (1,−1/4) with hᵦ = (−1,1/4): zₐ = 1 > −1 = zᵦ,
but tₐ = −1 < 1 = tᵦ. Their norms are equal, so RMS normalization leaves
this reversal intact. A reader treating the token-associated property t as
a monotone proxy for the PCA score gets this pair backwards. Good geometry,
readable poles and an exact conditional mean can coexist with that failure.
These toy token rows assume a semantic interpretation; they are not a claim
about natural-language readers or any measured vocabulary row in our model.

## 5. Four topics do not require four topic directions

For group label G with K values, probabilities πₖ and means μₖ, the law of
total covariance gives:

```text
C = Cwithin + Cbetween
Cwithin  = Σₖ πₖ Cov(h | G=k)
Cbetween = Σₖ πₖ (μₖ−μ)(μₖ−μ)ᵀ
Σₖ πₖ(μₖ−μ) = 0  ⇒  rank(Cbetween) ≤ K−1
```

Use population moments or consistent empirical 1/N moments here; mixing
unadjusted unbiased within-group covariances with these weights is incorrect.

Four group means therefore span at most **three centered contrast
dimensions**, not four mutually exclusive concept axes. PCA can mix these
contrasts: one group can load on multiple axes and an axis can contrast
several groups. A clustering step would assign examples using their joint
coordinates; individual eigenvectors are not themselves cluster labels.

Crucially, the rank bound **does not prove that PC4 is only within-topic
variation**. Let S span the centered group means. Cwithin can mix S with
its complement, spreading between-topic variation over more than three
eigenvectors of total C. If rank(S) = 3, Cwithin preserves S and all three
total-covariance eigenvalues on S exceed those on its complement, then
the top-three space is S and PC4 has no between-mean contribution. Those
additional conditions have not been tested here. Even under them, within-
topic variation might be meaningful rather than nuisance: equal group means
along an axis do not exclude group-dependent variances or higher moments.

Also, vᵢᵀvⱼ = 0 does not imply (Lvᵢ)ᵀ(Lvⱼ) = 0: the latter is
vᵢᵀLᵀLvⱼ. A non-orthogonal vocabulary readout can give orthogonal PCs
overlapping token lists. An eigengap stabilizes an axis under small covariance
perturbations; it does not establish its semantic meaning. In two dimensions,
diagonal separation Δ and off-diagonal perturbation η give
tan(2θ) = 2η/Δ, illustrating rotation near degeneracy.

Ranking also discards geometry: even the invertible map with rows (2,3) and
(1,1) maps orthogonal coordinate axes to (2,1) and (3,1), which rank the same
token first. Thus identical rankings do not require identical readout vectors
or a non-injective map. Top-token truncation discards still more information.

## 6. What is worth doing with this understanding

The present positive is **compact, partly useful descriptions of population
variation**. A practical explorer can show a signed direction, its token
readout and actual example loadings together, while retaining mixed and
failed directions. Our test does not demonstrate that this interface beats
examples alone; nor does it test memorization, optimizer updates or safety.

The next concrete low-cost discriminator, if selected, is an exploratory
saved-fit decomposition of within-topic versus between-topic variance for
all four unchanged axes. It can test the specific geometric possibility
above without a new model or judge. It cannot identify why the reader
misinterpreted PC4. Broader content/reader replication would test transfer;
context-specific Jacobians or interventions would test a different causal
claim. Neither is selected by this mathematical note.

This note used source inspection, frozen scalar reports and reviewed
algebra only. No arrays, model weights, decoding, judging, fitting, experiment
replay or paid computation were run. The existing delivered report remains
unchanged. Astra supplied independent constructions and an algebra check;
main checked the equations and implementation qualifications.
