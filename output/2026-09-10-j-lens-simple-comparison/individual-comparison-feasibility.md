# Individual-example comparison: feasible as a qualitative display

**Recommended next action:** add one qualitative comparison beside the existing
four-PC display: each signed PC pair versus the two individual J-Lens readouts
for its already fixed held-out endpoint prefixes. Keep activation layer 11 and
all four PCs. This needs only retained JSON; do not add a numerical score.

The common practical question is: **“What distinction between these same two
examples can I read from these lists?”** Both arms can receive the same two
prefixes and two lists of 12 ranked token strings. A reader could express each
arm's contrast in at most 20 words, with no rating or correctness claim.
That equalizes displayed evidence and answer length. It does **not** equalize
the underlying information: PCs summarize 32 fit rows, whereas the individual
arm directly decodes the two held-out examples. Also, 24 model-token strings
are not necessarily 24 words. Do not label this a fully matched information
budget or a test of held-out generalization.

## Exact proposed rows

Use the existing endpoint choices in [locked grades](grades.json), without
selecting a nicer example, changing framing, or recalculating a score.

| Spectral lists | Individual list for positive endpoint | Individual list for negative endpoint |
|---|---|---|
| `PC1+`, `PC1-` | `astronomy-4-note` | `cooking-4-note` |
| `PC2+`, `PC2-` | `football-4-plain` | `programming-4-plain` |
| `PC3+`, `PC3-` | `programming-4-plain` | `football-5-plain` |
| `PC4+`, `PC4-` | `cooking-4-note` | `programming-5-plain` |

Every list is available in `activation_11_jlens` in
[readouts.json](../2026-09-10-j-lens-completions/readouts.json). “Negative
endpoint individual” means the raw activation of the example at the negative
PC end, **not** negating that example's activation. Preserve all 12 strings,
rank order, multilingual fragments and visible whitespace. Across four rows,
each arm displays eight lists; repeated examples remain visibly repeated.

This would expose a useful distinction directly: the saved individual readout
for `programming-4-plain` starts with “implementing”, “implementations”,
“implementation”; `football-5-plain` starts with “lifted”, “lift”, “lifts”.
Their corresponding PC3 lists instead begin with testing/evaluation on one
sign and wrapping/bundling-related Chinese strings on the other. That is a
descriptive difference between population directions and local examples, not
a demonstrated benefit of either representation.

## Retained individual roster and missing controls

The exact 16 individual IDs are the following eight stems with **both**
suffixes `-plain` and `-note`:

| Topic | Stems |
|---|---|
| Astronomy | `astronomy-4`, `astronomy-5` |
| Cooking | `cooking-4`, `cooking-5` |
| Football | `football-4`, `football-5` |
| Programming | `programming-4`, `programming-5` |

Dataset (artifact not distributed in this public snapshot) identifies all of these
as held out; the note framing prepends `A brief factual note:\n`. These IDs have
individual readouts under both lenses at layers 11/17 for activations,
completion-loss gradients and first-token gradients. Only activation layer 11
J-Lens is needed for the proposed display. Individual activation directions
are normalized raw `h`, while individual gradient directions use `-g`.
See the direction/readout loop in
[acquisition source](../../experiments/j_lens_completions.py).

No fit-individual readouts were retained: the missing fit IDs have topic
astronomy/cooking/football/programming, index 0–3, and both framings. There are
also no centered-individual, rank-four-reconstructed-individual, per-topic mean,
or negative-individual activation readouts. The saved global mean is not an
individual-example baseline. Thus retained readouts cannot provide a comparison
where both a spectral summary and exemplar summary are constructed from the
same fit inputs and then explain new held-out content. That would require
additional retained readouts or new decoding; it is not authorized or needed
for this qualitative addition.

The existing study was already inspected and judged; this follow-up is
outcome-informed. Shared endpoint examples, one model and eight authored
held-out contents limit interpretation. Equal list counts alone cannot support
a claim that spectral summaries outperform inspecting examples. No new model,
array access, PCA, scoring, judging or figure generation was performed for
this feasibility note.
