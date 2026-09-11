# J-Lens: useful directions need not be topic labels

Codex — Spectral Optimizer Investigation · 10 September 2026

## What was measured

One new CPU calculation decomposed the 32 **already saved** fit-score rows
along all four unchanged canonical directions. These are 16 authored content
prefixes, each plain or prefixed with “A brief factual note.” No model,
activation reprojection, PCA, eigenvector change, decoding or new judgment.
See the [fixed exploratory protocol](protocol.md) and [implementation review](implementation-review.md).

Raw evidence in the J-Lens worktree:
results.json (artifact not distributed in this public snapshot),
receipt (artifact not distributed in this public snapshot).
Result SHA256: `aee0a38a1114d6e81edb437c18ee9c12769a36f3560c4b362bc9a25173303735`.
The primary call completed at22:27:30UTC with exit0; its attempt is consumed.

## All four directions

Each row sums to 100% before displayed rounding. These are shares of variance
**within each axis**, not shares of all 1,024-dimensional activation variance.

| Direction | Between topics | Content within topic | Framing within content | Prior fresh-text direct-token accuracy |
| --- | ---: | ---: | ---: | ---: |
| PC1 | 61.67% | 38.13% | 0.20% | 11/12 |
| PC2 | 85.35% | 14.37% | 0.27% | 9/12 |
| PC3 | 10.56% | 88.99% | 0.45% | 9/12 |
| PC4 | 15.10% | 84.09% | 0.81% | 2/12 |

The last column is copied from the [previously completed fresh-text test](../2026-09-10-jlens-fresh-content/graded/grades.json),
not a newly scored outcome. Fit geometry and fresh ordering are different
populations/measurements. No cross-axis correlation or causal model is fit
to these four observations.
The earlier fresh test used cross-topic pairs; useful prediction of two
examples within the same topic remains untested. These fit variance shares
do not establish that the prior prediction success used finer-than-topic
semantics rather than the remaining topic-mean contrast.

Population score variances V are 0.2844511, 0.2525716, 0.1890012 and
0.1650829. The largest absolute V−B−Q−F residual is below 1e−16; the paired
framing identity F = mean((note−plain)²)/4 agrees within 3e−19. All 32
score rows, four topic means, sixteen content means and signed framing
differences are retained in the raw result, not just selected endpoints.

## Interpretation

This rules out a simple description of the measured fit geometry as “three
predominantly topic-mean axes, then a framing axis.” PC3 itself is chiefly
within-topic content variation yet supplied useful fresh-text predictions.
So within-topic variation should not be equated with meaningless noise or
memorization. PC4 has a similar coarse variance allocation and a different
interpretive outcome; this allocation alone does not distinguish them.

The [mathematical rank bound](../../research/jlens_covariance_interpretation_2026-09-10.md)
still holds: four topic means span at most three centered contrasts. But it
does not require those contrasts to be the first three total-covariance PCs.
We did not measure full-subspace invariance or covariances outside these four
directions. Likewise, the small preamble contribution bounds this particular
framing change; it is not universal formatting robustness or a guarantee that
a small component cannot affect close comparisons.

The useful candidate remains a compact view of **finer population contrasts**,
shown alongside actual example loadings. The semantic nature of those
contrasts is not established by this decomposition. The prior fresh test
still shows no overall advantage over inspecting PCA-selected examples.

## Evidence quality and continuation

This is one reused fit panel at one residual layer in one frozen model.
Content and topics were authored, the same rows fit PCA, and this analysis
was motivated after seeing readouts and fresh scores. It is exploratory,
not independent confirmation, a semantic cluster discovery, optimizer result
or safety finding. Sixteen contents with two framings are not 32 independent
content samples. “Framing” includes content-dependent effects of the preamble.

The [independent audit](result-audit.md) checks exact saved-score/metadata
identity and scalar arithmetic separately from the producer. Report all four
axes, preserve the failed fourth and do not rerun the completed acquisition,
judging or decomposition. The immediate continuation is an inspected,
plot-led HTML update explaining this distinction—not additional model work.
