# J-Lens as an addition to examples: a useful next question

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

Status: analysis and proposed experiment, not a new empirical result. This
updates the next-step recommendation in the earlier synthesis (artifact not distributed in this public snapshot);
all original measurements, audits and delivered reports remain unchanged.

## Decision in one paragraph

The local simple comparison has produced real, limited positive evidence.
It is time to stop extending the same authored verb contrast. The strongest
next practical use case is an interpretation aid: **do unchanged direction
tokens help a reader who already sees the ordinary fit examples predict a
new natural-text activation ordering?** This tests an addition, not a
replacement. It is not answered by a tie between the two separate formats.
It could fail because the description is redundant, misleading or hard to
combine. Broader-corpus covariance estimation is feasible but answers a
different question and should not be changed simultaneously.

## What is already established

| Setting | Direct tokens A | Raw examples C | What it establishes |
|---|---:|---:|---|
| External Wikipedia prefixes, all four directions | 50/96 | 48/96 | No convincing all-direction advantage on this small panel |
| Same external panel, PC4 | 19/24 | 12/24 | A useful secondary direction-specific positive; PC3 did not transfer |
| Selected local verb panel, all directions | 116/128 | 98/128 | Useful communication on known positive content, not fresh-data confirmation |
| Same selected panel, PC4 | 32/32 | 26/32 | One cohort win and one tie |
| Prospectively authored fresh panel, all directions | 108/128 | 106/128 | Useful local correspondence; the extra two judgments are one content/cohort |
| Same fresh panel, primary PC4 | 26/32 | 26/32 | Every reader13/16, identical choices and three shared errors |

Direct evidence: [external results and raw grade](../output/2026-09-11-jlens-independent-content/results.md),
[selected local readers](../output/2026-09-11-jlens-verb-reader-comparison/results.md),
[fresh authored results and audit](../output/2026-09-11-jlens-fresh-authored-comparison/results.md).
These denominators count repeated judgments on shared pairs; none is an
independent-trial population estimate. Do not pool panels. The older endpoint
reversal, local8/8 and selected new-verb16/16 forecasts remain distinct evidence.

The complete external-case reading (artifact not distributed in this public snapshot)
already contains important limitations: PC4/A succeeds on P01/P02 where the
example-token arm B fails; B succeeds on P09 where A fails; all six PC4
readers fail on P06. That suggests descriptions can convey different cues,
but it does not predict that a combined reader can select the right cue.
That observation concerns A/B, not proof that raw examples C add information
to A. The specific A/C joint-display benefit remains unmeasured.
In particular, selecting the correct old reader after looking at the key is
an oracle, not a deployable combined method or a legitimate reported accuracy.

## Why examples and directions could be complementary

For centered activations with covariance C and a unit eigenvector u,
Cu=λu, λ>0, define z=uᵀ(h−μ). For a fixed linear vocabulary surrogate L,

```text
Cov(Lh,z) / Var(z) = LCu / (uᵀCu) = Lu.
h = μ + zu + r,        uᵀr = 0.
Lh = Lμ + zLu + Lr.
```

A decoded direction describes a population slope. A real example also
contains its mean and residual context. Context can make a slope intelligible;
the slope can indicate which aspect of a context matters. Either alone can
be ambiguous. The exact regression identity concerns an unnormalized linear
surrogate, not normalized logits or a theorem about natural-language readers.
On a changed population, the slope is L C′u/(uᵀC′u), which need not equal Lu.
See the [source-checked derivation](jlens_covariance_interpretation_2026-09-10.md).

An elementary construction illustrates the opportunity, without diagnosing
our model. Let C=diag(4,1), u=(1,0), L=I, and two fit examples be h+=(2,1)
and h−=(−2,−1). Such examples are compatible with a population having this
covariance; their residual coordinates need not vanish. Their difference is
(4,2), whereas the population slope is (1,0). For a new pair difference
δ=(1,−3), the actual u-score gap is1, but the exemplar-difference score is−2.
A direction tells the analyst to attend to the first coordinate rather than
the nuisance second coordinate. This only shows how complementary information
is possible. Text readers need not implement that geometric heuristic, and
another construction can make the removed context essential.

There is also no theorem that adding a token list improves an actual reader.
Define Y as the true higher-scoring prefix and a specified reader policy f:

```text
Δ_add = E[1{f(text pair, examples, direction tokens)=Y}]
        − E[1{f(text pair, examples)=Y}].
```

The separate-format difference estimates a different quantity. Its value,
including zero, does not determine Δ_add. An ideal decision-rule class that
can ignore extra inputs cannot have worse optimal risk when they are added;
a fixed finite reader can be distracted or anchored and absolutely can worsen.
This experiment would estimate a tool-package effect, not conditional mutual
information, causal semantics or an optimal-decoder bound. Across a single
fixed direction its description is a constant, so naive information-theory
claims about an empirically random description would be particularly misleading.

## Relation to existing interpretability work

The original J-Lens averages Jacobians across positions and prompts and
decodes vocabulary-associated directions. Its authors explicitly discuss
single-token limits, missing relational structure and inconsistent readable
outputs. Those are reasons to test whether a description helps predict new
measurements, rather than treating a plausible word list as validation.
This PCA extension on our small model is not the original paper's workspace
or alignment-audit experiment. [Primary paper, Methods and Limitations](https://transformer-circuits.pub/2026/workspace/index.html).

## Smallest proposed next experiment

Use an **examples-only versus the identical examples plus direction tokens**
comparison. Do not invent a new verbal gloss or ask readers to choose which
method they find more plausible. Both arms answer the same measurable question:
which of two held-out natural prefixes has the higher fixed directional score?

The larger prompt is intentional: it tests whether adding this panel is useful
in practice, not whether J-Lens is the best use of an equal token budget.
A positive would motivate an equal-budget alternative-description control and
broader reader validation, not license those conclusions now. A negative or
mixed result limits this package/endpoint/population; it does not erase prior
local usefulness. Do not keep adding similar panels until one becomes positive.

This is an accepted question to develop, not an acquisition-ready protocol.
Next lock exact sampling/previous-page exclusions, reference layout, prompts,
source paths and analysis schema, review the implementation, then perform the
new test under existing autonomy. No new corpus requests, target roster,
reader packets, tokenizer/model call or actual grading occurred for this note.

## Why not broaden the fitted covariance at the same time?

The user's broader-data idea remains promising and feasible: accumulate frozen
activations and update centered scatter, not model weights. For e_t=h_t−μ_(t−1),

```text
μ_t = μ_(t−1) + e_t/t
M_t = M_(t−1) + ((t−1)/t) e_t e_tᵀ.
```

But refitting on broader data changes the object being explained. Combining
that with a new display and population would not reveal which change mattered.
First test the fixed-display addition above. A separate later broad-fit study
should use disjoint documents for fitting and evaluation, one fixed token-role
sampling policy, and compare subspaces as well as axes: near-equal eigenvalues
can rotate individual PCs without changing the captured subspace. Do not align
or name axes using held-out semantic success. Do not equate batch-mean covariance
with per-example covariance except under the stated IID equal-size population
conditions in the earlier derivation. No broad-fit run is selected here.

The scope remains insight-oriented frozen-model interpretation. None of this
tests memorization suppression, safety monitoring efficacy, semantic clusters,
optimizer speed or improved training. The deferred optimizer/familiarity work
must not resume from this decision or from the reminder.
