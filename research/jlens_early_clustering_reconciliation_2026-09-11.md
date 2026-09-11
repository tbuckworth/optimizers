# Restoring the early J-Lens branch's joint-clustering positive

Codex — Spectral Optimizer Investigation · 11 September 2026.

**Correction:** my recent central synthesis concentrated on interpreting
individual directions and omitted the earlier joint-clustering experiments.
The omission understated the affirmative evidence. The raw results were
always preserved; this note restores them without rerunning anything.

## What worked

Four activation-PC coordinates retained substantial topic-group structure
on held-out authored examples in two panels. PCA and KMeans used fit rows
only; held-out rows were assigned afterward. The result does not depend on
interpreting each eigenvector as a separate cluster.

| Panel | Activation ARI, layers 11 / 17 | Residual loss-gradient ARI, layers 11 / 17 | Activation random-four mean ARI |
|---|---:|---:|---:|
| Initial 72-sentence pilot: 48 fit, 24 held out | 0.7273 / 0.7037 | 0.0867 / 0.0640 | 0.1403 / 0.2143 |
| Meaningful-completion panel: 32 fit rows, 16 held out | 0.6009 / 0.8205 | 0 / 0 | 0.1685 / 0.2256 |

ARI is adjusted Rand index: agreement between predicted group assignments
and intended topic labels, adjusted for chance grouping. It is not an
accuracy percentage. KMeans used the specified number of groups, four;
this was not discovery of how many groups exist or graph-Laplacian clustering.
The random controls average32 four-dimensional projections on the same
examples, not32 independent datasets.

The separate **supervised nearest-centroid** classifier got22/24 at both
layers initially, and14/16 and15/16 in the completion panel. Fit topic labels
construct its centroids. These counts must not be called unsupervised
clustering accuracy. In the completion panel, full-vector classification
was15/16 at both layers and full-vector ARI0.5614/0.8205. Thus four dimensions
preserve useful structure; they do not improve supervised accuracy here.
The modest layer11 ARI increase and layer17 tie are retained without a
general clustering-superiority claim.

Direct evidence:
initial report (artifact not distributed in this public snapshot),
initial metrics (artifact not distributed in this public snapshot),
completion report (artifact not distributed in this public snapshot),
completion metrics (artifact not distributed in this public snapshot).
Metric SHA256 values are245705ade07879ec6ba2c8fd328faabbd4bda18bdc6c9f76033bce69cff4b6a0
and6015b8762bb071c129a3e64dc77db3a94f07b1b09ce6eb411a10a32816553aab.

## What it does not establish

The clustering calculation never uses J-Lens token descriptions. This is
positive evidence for low-dimensional activation geometry, not evidence that
the decoder improves grouping or explains every group correctly. The later
word-description comparisons answer a different question and do not erase it.

Both panels use one frozen small model and dependent layers. The initial
topics have distinctive vocabulary and partly differing styles. The completion
panel crosses24 prefix/completion contents with two frames: its16 held-out
rows represent eight contents twice. Topic structure survives that particular
framing change, not all linguistic controls or unrestricted natural text.

The initial gradients differentiate a common next-token target; the later
gradients differentiate meaningful teacher-forced reference-completion loss.
Both are gradients with respect to the final-prefix residual, not full model
parameters. Topic clustering is weak for these gradients, including zero ARI
in the completion and first-token-gradient controls. That does not establish
that gradients contain no useful information for a different question.

## Relation to the new user request

The user's latest proposal filters a many-example **aggregate** gradient
through a learned subspace, then interprets the resulting vector. Neither
clustering individual example coordinates nor decoding PCs one at a time
tests that proposal. The new brief (artifact not distributed in this public snapshot)
keeps the exact P_k mean operation and the activation analogue in scope.
Earlier gradient clustering negatives are relevant context, not grounds to
skip that distinct test; activation clustering positives are not proof it wins.

Main reviewed both reports, the metric summaries and the complete existing
analysis scripts. Independent Astra reviewed exact fit/test/label boundaries,
all baseline summaries and acquisition objectives; worker commit5086e3ab829595403a28f0ceb4cc2b3584c2a8e6
contains its source-bound review (artifact not distributed in this public snapshot).
This is read-only corroboration, not an independent experimental replication.
