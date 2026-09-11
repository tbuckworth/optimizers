# Natural-text J-Lens add-on: no overall improvement in this pilot

Codex — Spectral Optimizer Investigation · 11 September 2026

## Finding and implication

Adding the original J-Lens direction tokens to the same raw examples produced
**74/128 correct predictions (57.81%), versus75/128 (58.59%) with examples alone**.
The two reader groups were a tie and a one-answer loss. The predeclared
incremental-usefulness criterion is not met. This is evidence against a clear
benefit of this particular display package on this particular new panel, not
proof of equivalence or that J-Lens cannot be useful.

All numerical claims below come from the [immutable grade](graded/grades.json)
and [independent full audit](audit.json). Grade SHA256
`4bae97cd90fb3de288a56624d919e57e739c90add2ac17519df5593a702dfe63`;
audit SHA256 `167131f312aa1662e792cb6cf3f61aa415b7f7abfc19e58b7d02e6e4a3408033`.
This is a provisional single-model experiment with four fresh AI readers,
not four independent model families or256 independent data samples.

## The meaningful simple comparison

E readers received the original positive/negative fit examples. EA readers
received exactly those examples plus the original12-token lists for each
pole of the same direction. No new gloss, selected replacement example,
decoder, sign flip or covariance refit. The question was which of two new
prefixes has the higher measured value on that direction, not which topic
the text belongs to or whether it is a true semantic cluster.

There were32 mechanically sampled Wikipedia prefixes,16 disjoint within-
category pairs and four original axes. Each input was the first16 whitespace-
delimited words, captured at its final actual model subtoken. The source frame
was6 astronomy,4 cooking and22 programming texts, with no remaining football
articles. Candidate pairs were fixed before eligibility; one redirect rejected
its whole pair, without re-pairing survivors. The corpus was frozen before
measurement and reader calls. More natural does not mean representative or
unseen during pretraining.

The frozen Qwen3.5-0.8B model was not trained. At post-block11, for direction u,
the measured score was s(x)=(h(x)−μ)ᵀu, with saved feature/direction precision
and float64 scalar arithmetic specified in the [protocol](protocol.md).
For two texts, Δ=s(x_left)−s(x_right); its sign determines the correct ordering.
Every nonzero gap was retained, however small. An exact zero would receive
half credit, but there were no exact zeros here. This target is a coordinate
of a frozen model, not external semantic ground truth.

Four history-free Astra/high readers made64 first-response choices each:
one E and one EA reader per group. Target order was matched within groups
and reversed between groups. All256 responses and the exact five response/
lock blobs were committed at `2c41b3be5a4a868d069ee55127c625abb7df1b91`
before any scientific-score/private-map/array inspection. No reader was
repaired, repeated or replaced. See reader delivery (artifact not distributed in this public snapshot).

## Primary result and controls

| Scope | Examples only E | Same examples + tokens EA | EA−E |
| --- | ---: | ---: | ---: |
| Combined |75/128|74/128|−1|
| Reader group1 |37/64|37/64|0|
| Reader group2 |38/64|37/64|−1|

The criterion required a positive EA−E difference in **both** groups and
each EA reader to exceed its own stronger realized always-FIRST/SECOND rule.
Both EA readers beat that35/64 control with37/64, as do the E readers, but
neither group has positive incremental gain. Opposite target orders make
the pooled fixed-position rules64/128, or50%; individual controls are54.69%,
not50%. The criterion is a descriptive pilot rule, not a significance test.

## All directions, including adverse results

| Direction | E | EA | EA−E, group1/group2 |
| --- | ---: | ---: | ---: |
| PC1 |19/32|21/32|+2 /0|
| PC2 |15/32|15/32|0 /0|
| PC3 |21/32|20/32|0 /−1|
| PC4 |20/32|18/32|−2 /0|

PC4 was secondary here; its earlier positives do not justify making it
the primary axis after seeing this result. None of the four directions has
a positive incremental difference in both groups.

Across128 matched group/direction/pair cells, EA has higher credit in15,
lower in16 and the same in97. Same credit includes both-correct and both-wrong;
it does not mean a zero model-score gap. The two EA readers choose the same
underlying text in60/64 cases, versus55/64 for E. Greater consistency does not
establish greater accuracy, and different readers still confound causal
attribution of individual changes to the added tokens.

| Source category | Unique pairs | E | EA |
| --- | ---: | ---: | ---: |
| Astronomy |3|14/24|14/24|
| Cooking |2|6/16|12/16|
| Programming |11|55/88|48/88|
| Football |0|not sampled|not sampled|

The cooking positive is worth retaining, but it is only two pairs reused
across four axes and two readers. It is not16 independent examples or an
established domain-specific effect. Programming contributes the opposing
negative. No category is removed or reweighted to turn the primary result
positive. The complete grade preserves all256 item decisions, signed gaps,
reader/group/category/axis splits, position controls and agreement rows.

## Relation to the useful version of the idea

The [earlier local action-word forecast](../2026-09-11-jlens-new-verb-transfer/results.md)
remains16/16 in the predicted PC4 direction. The subsequent saved-panel
[reader comparison](../2026-09-11-jlens-verb-reader-comparison/results.md)
was116/128 for direction tokens versus98/128 for raw examples, with one
PC4 group win and one tie. A [fresh authored panel](../2026-09-11-jlens-fresh-authored-comparison/results.md)
then gave108/106 overall and the same13/16 PC4 answers/errors for every reader.
These positives and limits are not overwritten or pooled with today's data.

The coherent present account is **conditional interpretive usefulness, without
a demonstrated general add-on advantage**. A covariance eigenvector describes
a signed population contrast, not necessarily a semantic cluster; decoding
that contrast and showing examples can help locally without telling a reader
how arbitrary later prefix states are ordered. Today's experiment tests the
combined display directly; the earlier separate-format ties did not answer it.

The most defensible next work is to consolidate the successful and unsuccessful
saved cases into a concrete use-case explanation. Broader-corpus covariance
remains a distinct possible study with disjoint fitting/evaluation, not a
silent change or automatic retry here. No new panel, optimizer experiment,
training, paid judge or larger fit is selected by this negative result.

## Verification and limitations

- One saved-array checker independently recomputed128 projections and64 gaps;
  maximum scalar error2.22×10⁻¹⁶, below the predeclared10⁻¹² tolerance. It loaded
  no model/tokenizer. Worker evidence is under
  `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-natural-addon/checked/`.
  Main's separate [auditor](audit_saved.py) checks all256 item joins and every
  summary without importing the grader. These are arithmetic/provenance
  checks, not independent human semantic validation.
- A distro Python/glibc update caused the original admission to fail before
  any model call. That failure is preserved. A separately frozen and reviewed
  runtime amendment (artifact not distributed in this public snapshot) allowed the first actual pass with
  exact new runtime pins and all inputs/scientific criteria unchanged. No
  equivalence across runtime patches is claimed. Successful forward receipt
  SHA`f7da25d6a17ec12c679a60cd658b2d009484f00e73b914af2dcb6f8a4896f408`.
- Reader transcripts verify exact first finals plus terminal newline, one call
  each, Astra/high and no tool messages. Saved prompt hashes are fixed, but
  incoming and dispatched message payloads are encrypted in stored logs;
  independent plaintext equality of the transmitted prompts is unavailable.
  This is an explicit provenance limitation, not a proven mismatch. Isolation
  was instruction-level, not physical tool disabling.
- One small model/layer, four same-family AI readers,16 pairs and an unbalanced
  catalogue do not establish general superiority, equivalence, semantic
  clusters, causal features, optimizer performance or safety effects. EA also
  has more tokens: the comparison intentionally tests the actual add-on package,
  not equal information or reading time.
- Attribution (artifact not distributed in this public snapshot) and the [additional translation credit](attribution-review.md)
  accompany the excerpts. No illustrations copied. The bounded notice screen
  is not a rights certification. Paid spent/reserved remains$0/$0 of$100.

All data collection, tokenization, model, reader, lock, checker and grading
stages are now consumed. Continue from saved evidence; never replay them.
