# J-Lens directions summarize a population, not a permanent label

Codex — Spectral Optimizer Investigation · 11 September 2026

**The covariance structure changes substantially between the saved text panels,
but that does not explain which direction helps.** The fourth direction remains
useful on the Wikipedia panel despite poor covariance alignment. Its signed
tokens suggest an observation/recording versus provision/preparation contrast;
the complete examples support parts of that reading and contradict others.
The next useful test is a small prospective content-versus-format comparison,
not relabeling the completed panel or repeating its grade.

## What changed geometrically

Keep the model and four fitted directions fixed. For each saved panel, compare
u with Cu: the latter determines the activation pattern associated with its
score in that panel. Cosine 1 means collinearity; lower values mean more
off-axis variation. This is **not accuracy or probability**.

| Direction | Original fit cosine | Authored fresh cosine | Wikipedia cosine |
|---|---:|---:|---:|
| PC1 | 1.000 | 0.537 | 0.349 |
| PC2 | 1.000 | 0.597 | 0.287 |
| PC3 | 1.000 | 0.575 | 0.216 |
| PC4 | 1.000 | 0.344 | 0.292 |

The fit values follow from PCA on those same data; their tiny raw deviations
above/below 1 are floating-point roundoff, retained in the JSON. The reported
alignment is input-space geometry, not alignment after J-Lens. That readout
could suppress or amplify the off-axis component.

The variance allocation also changes. These are fractions of **each axis's
score variance** attributable to differences between topic/category means:

| Direction | Fit between-topic share | Authored fresh share | Wikipedia share |
|---|---:|---:|---:|
| PC1 | 61.7% | 76.1% | 12.1% |
| PC2 | 85.4% | 77.2% | 40.5% |
| PC3 | 10.6% | 59.4% | 14.9% |
| PC4 | 15.1% | 19.8% | 12.4% |

For example, calling PC3 a “within-topic direction” describes the original fit
panel, not an intrinsic property valid on every dataset. In the authored fresh
panel it has substantial between-topic variance, while still supporting the
previous useful within-topic ordering result. A variance fraction and a paired
prediction result answer different questions. In Wikipedia, both PC3 and PC4
are mostly within-category variation; that fact does not distinguish failure
from success.

| Direction | Authored variance / fit variance | Wikipedia variance / fit variance |
|---|---:|---:|
| PC1 | 26.7% | 7.8% |
| PC2 | 38.4% | 5.7% |
| PC3 | 46.7% | 7.2% |
| PC4 | 11.1% | 11.8% |

PC4's amplitude is similar on the two fresh panels even though its ordering
utility differs. The four old scores also cease to be mutually uncorrelated:
the largest absolute off-diagonal correlations are about 0.537 on authored
texts and 0.392 on Wikipedia, versus approximately zero on fit. None of these
statistics is a calibrated quality metric or a demonstrated mediator.

All numbers come from the [new saved-array calculation](analysis/results.json),
SHA256 `295f131e3af5b7b44c7b28916cd00cf65c73c6c806a9eb2c4a471d7fd34c167b`.
All three panels, all four axes, group means and full score correlations are
retained; array outputs (artifact not distributed in this public snapshot) preserve means, scores, Cu and
regression slopes. No new PCA or model output was computed.

## What the examples suggest

The complete-case qualitative note (artifact not distributed in this public snapshot) preserves all 12 PC4 pairs
and PC3's adverse cases. PC4's observation/recording/anomaly versus
provision/preparation/reward reading fits some comparisons more naturally than
its selected exemplar verbs “blocked” and “combined”. That is a plausible
reason a population summary could add value beyond looking at one exemplar.

But both direct-token readers wrongly order recipe versus al-dente text (P06)
and football-trafficking versus football-names text (P09). Those failures cannot
be discarded; one also favors the exemplar-token control. Several successes
need loose associations not stated in the prefixes. Reader reasons were not
collected, so the note's explanations are retrospective interpretations, not
reported rationales or a fresh semantic validation.

The [prior locked result](../2026-09-11-jlens-independent-content/results.md)
remains unchanged: PC4 direct tokens 19/24 (9/12 and 10/12), PC3 12/24,
all-axis A/B/C 50/96, 50/96, 48/96. The useful positive survives this inspection;
neither a universal concept label nor a general advantage follows.

## Mathematical account and next decision

The [readable derivation and related-work connection](interpretation-math.md)
distinguish a scoring direction from its associated activation pattern:

```text
score z = uᵀ(h − c), with fixed original fit mean c
pattern d = Cu / (uᵀCu)
linear-readout regression coefficient = Ld
```

For a covariance eigenvector, d = u/(uᵀu). That identity explains why a fitted
direction can be an informative population summary. For a new covariance,
the scoring direction and associated pattern can separate. Our measurement
shows that separation in the two small fresh panels; it does not show that
this causes semantic errors. Related linear-model work distinguishes filters
from activation patterns; the mathematics is not a novelty claim.

## Evidence strength and execution

One exploratory reuse of three saved panels: fit 32 rows / 16 contents,
authored 24 texts, Wikipedia 24 texts. Same model/layer, different sources,
lengths, endpoints and framing; small empirical covariance ranks are at most
31/23/23. No new data, seed replication, causal intervention or semantics score.

The [fixed protocol](protocol.md) and [source/math review](source-review.md)
preceded calculation. A wording error about necessity after a non-injective
linear map was corrected before execution; no numerical code change was needed.
Main and reviewer ran fabricated tests, not real arrays, before acceptance.
One CPU-only invocation completed at 00:32:32 UTC in under a second, with
CUDA hidden, BLAS/OpenMP threads limited to one and a 60-second timeout.
Its receipt (artifact not distributed in this public snapshot) records exact input/output/source hashes.
12,312 scalar fsum corroborations passed; maximum difference 1.11e-16.
All reprojected old scores matched their saved float64 arrays exactly.
The independent arithmetic path is inside the producer, not an independent-
author result audit. No old stage was restarted; paid spent/reserved remains $0.
