# Fresh-text J-Lens comparison

Codex — Spectral Optimizer Investigation · 10 September 2026

**There is useful correspondence on three directions, and a conspicuously
misleading fourth direction.** Direct covariance-direction readouts predict
31/48 fresh-text orderings correctly; individual-example readouts score33/48
and the raw fit examples34/48. This establishes a small positive interpretive
demonstration, not an overall advantage over examples or four semantic clusters.

## What was tested

For each of four fixed layer11 activation-covariance directions, three fresh
isolated readers predicted which of two new prefixes would have the higher
signed projection. The [prospective protocol](protocol.md) fixes all inputs,
fit-only reference selection, pairing, counterbalanced allocation and scoring.
The same24 new authored prefixes form12 disjoint cross-topic pairs, tested
on every direction. These are familiar astronomy/cooking/football/programming
topics with new wording, not natural-corpus or new-topic generalization.

A displays J-Lens token lists for the two signed directions. B displays
J-Lens lists for the maximum/minimum fit examples, using raw h at both ends.
C displays those exact two fit prefixes. B/C use the same covariance geometry
to select examples: this is direction decoding versus interpreting selected
examples, not PCA versus no PCA. A/B each provide two ranked12-token lists;
C provides two intact prefixes. There is no equal-information-budget claim.

The mean/basis came from the existing32 fit activations. New actual rounded
U32 directions were saved and used consistently for decoding and projection;
no PCA or old-fit forward was repeated. A single frozen-model run decoded15
reference vectors, saved/hash-locked their outputs, then forwarded the24 new
prefixes. The [acquisition review](acquisition-review.md) records22.0199seconds,
unchanged parameters and no training, gradients or paid compute.

## All primary results

Every count below is reconstructed from [the immutable grade](graded/grades.json).

| Direction | A: direction tokens | B: exemplar tokens | C: raw exemplars | Always FIRST / SECOND |
| --- | ---: | ---: | ---: | ---: |
| PC1 | 11/12 | 9/12 | 11/12 | 7/12 / 5/12 |
| PC2 | 9/12 | 8/12 | 9/12 | 4/12 / 8/12 |
| PC3 | 9/12 | 9/12 | 9/12 | 6/12 / 6/12 |
| PC4 | 2/12 | 7/12 | 5/12 | 7/12 / 5/12 |
| **All four** | **31/48 (64.6%)** | **33/48 (68.8%)** | **34/48 (70.8%)** | **24/48 / 24/48** |

No exact ties, exclusions or extra trials. Small nonzero gaps remain fully
scored and every signed/absolute gap is retained in the grade. A−B is−2
correct comparisons, A−C is−3; by axis those differences are(+2,+1,0,−5)
and(0,0,0,−3). The fourth direction accounts for A's net deficit; it is not
dropped, inverted or relabelled after seeing the result. The [saved-output
audit](result-audit.md) further confirms that A/C choices coincide on all36
PC1–PC3 targets, not just in their aggregate counts.

## Interpretation and limits

The first three direction readouts are informative about fresh measurements
in this small panel. Their11/12,9/12,9/12 counts are more compelling than
judging token-list fluency alone. PC2's9/12 is nevertheless only one correct
answer above its better constant-position baseline. The earlier apparently
recognizable PC4 readout remains actively misleading here. Its2/12 outcome
shows why decoded vocabulary is not a reliable semantic certificate.

Raw examples tie direct decoding's per-axis totals on PC1–PC3 and beat it
on PC4. Thus this experiment does not demonstrate extra practical utility
from direction-token decoding over inspecting PCA-selected fit exemplars.
That comparison does not erase the positive correspondence. It also does
not identify whether PC4 fails because of lens approximation, decoding a
standalone direction, reader interpretation, lexical transfer, or a mixture.

There are24 unique texts/12 pairs, not144 independent observations. All
directions share those texts; each arm–axis cell has one same-model-family
reader and no human replication. Counterbalancing reduces arm/rater
confounding but not rater-by-axis interaction. Texts are hand-authored within
four familiar topics. Formats reveal parts of the method. Do not attach a
population confidence interval, significance claim, four-cluster finding,
causal explanation or optimizer/safety conclusion to these counts.

## Provenance and immediate next step

Main read the full acquisition/judging implementations and reran their
fabricated tests; the worktree checked saved array identities and exact
reference/input exports. [Judging acceptance](judging-review.md) and
rater transport (artifact not distributed in this public snapshot) document isolation and limitations.
All144 returned choices were locked in commit
`92179228bde2be856726169e76f1246749bb38bc` before the single grading call.
The code verified their exact committed bytes before opening the scalar key.

- Grade SHA256:`1ace0f90f0d8f539094ece65e6acce1517ea2159ef80f21ea143cfc086a7aa29`.
- Packet manifest:`2be4392f32e800c2f8c4cadbf0dcc04759c01bc2caacfc6565d29e41ef445ef4`.
- Response lock:`957f3d2846a6d92a76516ec54332b7668166c5a0fe257c5827caa2908bfb6364`.

All scientific handles are complete and consumed. The saved-output aggregate
check passed; immediate work is delivery of the inspected plot-led HTML report,
not another model run. A stronger later test would need independent content and reader
replication; it is not necessary to withhold this useful limited result.
