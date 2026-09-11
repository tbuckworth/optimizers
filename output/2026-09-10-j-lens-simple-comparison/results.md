# J-Lens simple comparison

J-Lens matched 3 of 4 PC endpoint pairs; the plain lens matched 2. This small exploratory comparison does not establish an advantage: random axes also scored 3/4 versus 2/4. PC3 supplies the extra PC match; both methods miss PC4.

The fixed layer-11 activation comparison used four PCs, both signs, all 12
saved tokens per sign, and the same held-out extrema under both lenses.
Two fresh blinded Astra raters judged one method per axis, counterbalanced
within PC/random families. Responses were locked in `fc185c7210b5cd4c611a524c4490a4c204056aaa` before grading.

| Axis | J-Lens | Plain lens |
|---|---|---|
| PC1 | Match (high) | Match (low) |
| PC2 | Match (high) | Match (medium) |
| PC3 | Match (high) | Miss (low) |
| PC4 | Miss (medium) | Miss (low) |
| random1 | Match (low) | Miss (low) |
| random2 | Miss (low) | Miss (low) |
| random3 | Match (low) | Match (low) |
| random4 | Match (low) | Match (low) |

PC: 3/4 versus 2/4; paired difference +1 correct axis. Random: 3/4 versus 2/4;
paired difference +1. Confidence is descriptive and does not change grading.
No same-content/different-framing extrema occurred; none were replaced.

Four dependent PCs; eight authored contents in two framings; two fresh Astra raters of the same model, one judgment per item, no human validation. The task matches geometric extrema, not independently established concepts. Random-axis correspondence is conditional on reconstructing the original seed/QR basis; its original numeric environment was not recorded. No new model inference or classification experiment.

This tests limited readout-to-example correspondence. It does not demonstrate
that PCs beat individual vectors, identify four distinct concepts, or improve
classification. Recognizable tokens alone were not the outcome criterion.

[Exact per-card grades and validation](grades.json) · [Fixed protocol](comparison-decision.md) ·
Rater 1 locked responses (artifact not distributed in this public snapshot) · Rater 2 locked responses (artifact not distributed in this public snapshot) ·
Full visual report (artifact not distributed in this public snapshot) · [Grader/plotter](../../scripts/report_j_lens_simple_comparison.py).
The grader checked all 16 cards, rater allocations, allowed responses,
polarity keys and correspondence to pinned saved readouts/prefixes. It did not
open feature/analysis archives. Figures decode no new directions.
