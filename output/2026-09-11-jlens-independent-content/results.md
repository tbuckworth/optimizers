# J-Lens on independently authored text: useful fourth direction, limited transfer

Codex — Spectral Optimizer Investigation · 11 September 2026

**A useful positive survives on external text, but not on the direction that
motivated this follow-up.** The unchanged fourth-direction token description
predicted 19/24 orderings, from 9/12 and 10/12 for its two fresh AI readers. They
selected the same underlying prefix on 11/12 pairs. The previously promising
third direction scored 12/24, from 5/12 and 7/12. Across all four directions,
the three description methods were near the 50% reference. This is a small
source-transfer pilot, not evidence of generally reliable semantic clusters.

## Complete comparison

Every cell pools two fresh AI readers on the same 12 fixed pairs, presented in
opposite orders. There are 24 distinct texts, not 288 independent observations.
A uses direct signed-direction tokens; B uses tokens from selected old fit
examples; C shows those raw fit-example prefixes. All descriptions, four
directions/signs, model/layer and pairing rules were fixed before measurement.

| Direction | A: direction tokens /24 | B: example tokens /24 | C: example text /24 |
|---|---:|---:|---:|
| PC1 | 10 | 11 | 12 |
| PC2 | 9 | 13 | 12 |
| PC3 | 12 | 13 | 12 |
| PC4 | 19 | 13 | 12 |
| All directions /96 | 50 | 50 | 48 |

For every axis/arm, always-FIRST and always-SECOND each obtain 12/24 by
construction; the all-axis reference is 48/96. No exact ties, tiny-gap
exclusions, response repairs, axis drops or sign changes occurred.

| Direction | A: reader1 / reader2, each /12 | B, each /12 | C, each /12 |
|---|---:|---:|---:|
| PC1 | 4 / 6 | 6 / 5 | 7 / 5 |
| PC2 | 5 / 4 | 7 / 6 | 7 / 5 |
| PC3 | 5 / 7 | 7 / 6 | 5 / 7 |
| PC4 | 9 / 10 | 7 / 6 | 6 / 6 |

Here “reader1/2” denotes the two cohorts, not one identical reader assigned
every cell. The exact six-agent allocation is in the [protocol](protocol.md).

| Direction | A: same chosen prefix /12 | B /12 | C /12 |
|---|---:|---:|---:|
| PC1 | 8 | 9 | 10 |
| PC2 | 9 | 9 | 8 |
| PC3 | 10 | 9 | 10 |
| PC4 | 11 | 11 | 10 |

All three arms agree across their two readers on 38/48 targets. Agreement is
not accuracy: for PC3/A, both readers are right on 5 pairs and wrong on 5,
with 2 disagreements. For PC4/A, both are right on 9, wrong on 2, and disagree
on 1. Thus PC3's weak transfer is not simply wholesale disagreement, and
PC4's positive is not driven by just one reader.

## What this changes—and what it does not

The prior [within-topic authored-text result](../2026-09-10-jlens-within-topic/results.md)
was PC3/A 11/12 and PC4/A 6/12, with one reader per cell. The new panel does
not reproduce that pattern: the successful direction changes. Preserve both
results rather than treating PC3 as generally validated or PC4 as universally
misleading. The PC4 point result exceeds both simple description controls on
this panel, but was not the motivating selected hypothesis; all four axes
must accompany it. No corrected significance or population reliability is claimed.

The source is mechanically selected Wikipedia category pages, six per root.
Each prefix is exactly the first 16 whitespace-separated words, unedited beyond
normalizing whitespace. Category membership is a sampling stratum, not an
independently verified fine-grained semantic label. Some excerpts end mid-sentence
or contain formatting remnants; those are retained. Source, length, grammatical
endpoint and reader/order design changed together. This is not a controlled
causal comparison with the older authored panel, and pretraining exclusion is
not certified.

The [population-contrast mathematics](../../research/jlens_covariance_interpretation_2026-09-10.md)
supplies a possible interpretation, not a diagnosis: under a new covariance C,
the regression contrast for a fixed direction u is LCu/(uᵀCu), not necessarily
Lu. Whether covariance shift, endpoint cues, lexical content or another cause
explains this pattern remains untested. PCA directions remain continuous
contrasts, not recovered clusters; no optimizer, causal mechanism or safety
benefit has been validated by this study.

Next inspect the saved successes/failures and covariance geometry to understand
what PC4 is tracking and why usefulness changes, before selecting another run.
No new model training, experiment restart or broader optimizer work is selected.

## Evidence and execution

- [Raw grade](graded/grades.json), SHA256 `5495950b0666d4365d146af2860c757dd6df87e6409b57d6861d3640d3099231`.
- Response lock (artifact not distributed in this public snapshot), committed `d1abdde61a5448338d1d75788fcfc29f9197851f`
  before the one 00:03:13 UTC grade. Six fresh history-free AI readers; no retries.
  Delivery record (artifact not distributed in this public snapshot) and exact prompts retain the transport
  details. Main separately checked the six prompt hashes/decoded packets/common
  wrapper. Instruction-level tool isolation is not physical removal of tools.
- Saved-measurement audit (artifact not distributed in this public snapshot) PASS: 96 scalar fsum
  projections, 48 exact gaps and ordering signs, 96 JSON scores, 48 JSON gaps and 24
  exact input rows. Maximum projection difference 1.11×10⁻¹⁶; no model replay.
- [Separate choice-summary arithmetic audit](result-audit.md) PASS: 21,082
  checks match all 288 item records, exact score/gap values, summaries, paired
  differences and reader agreement. This is independent arithmetic, not an
  independent-author review or proof of reader isolation. Main reviewed the
  full audit source before its single execution. Main also reviewed the full
  grading code and independently ran its 23 fabricated tests before execution.
- Execution record (artifact not distributed in this public snapshot) documents the single 24-forward run,
  source-collector continuation correction, all immutable handles, and zero
  paid spend. Model weights, references, PCA and signs were not updated.
- [Source attribution](attribution-review.md) preserves both additional notices
  identified in the fixed 72-URL screen. Collection/licensing caveats are separate
  from scientific selection; no source page was substituted for quality.
