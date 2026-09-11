# A simple J-Lens comparison

Status: source/readout audit and fixed proposal; **no semantic score has been
calculated**. Provisional, reused single-model authored-panel evidence. Source
worktree HEAD: `14353f3b0220d35b97f44d34bac2eb6c6f81dec4`.

## Decision

Use activation covariance at the already fixed **layer 11**, all four PCs.
Ask whether their token readouts identify the positive versus negative held-out
examples better with J-Lens than with plain unembedding. This tests the lens's
interpretive correspondence for exactly the same individual eigenvectors.
Keep four matched random axes as a nuisance control. Do not run another topic
classifier or select the cleanest-looking axis/layer.

The smallest useful result is a four-row display: each PC's complete positive
and negative token lists under both lenses beside its mechanically selected
held-out endpoints. A blinded matching score can accompany it only if judgments
are obtained without access to the geometric answers or the other lens's answer.
An unblinded executor reading these already reviewed outputs is not a blinded
judge. If independent judgments are unavailable, publish the comparison as a
descriptive display without an invented semantic-quality number.

## What is actually retained

[readouts.json](../2026-09-10-j-lens-completions/readouts.json) has 12 groups:
activation, completion gradient and first-token gradient; layers 11 and 17;
J-Lens and plain. Each group has exactly the same **33 named directions**, each
with all 12 top token IDs, decoded strings, ranking scores and input norms.

| Direction family, per kind/layer | Available under both lenses | Interpretation |
|---|---:|---|
| Held-out individuals | 16 directions | Raw full residual vector for each input; two framings of eight contents |
| Fit mean | 1 direction | Global fit mean; not a topic-specific mean |
| Covariance eigenvectors | 4 axes, both signs | Each PC decoded individually, not the joint four-score classifier |
| Random controls | 4 axes, both signs | One fixed QR draw; not the 32 projection draws in classification metrics |

Each direction is normalized before either lens. The plain control skips
`lens.transport`; both then call the same `model.unembed`. Individual/mean
gradient directions use minus the loss gradient; PC/random directions retain
both signs. See [acquisition source](../../experiments/j_lens_completions.py),
especially `directions`, `names`, the normalization, and the two-method loop.

There is **no separate full-vector decoder** beyond those individual raw-vector
readouts. The full-vector classifier in [analysis source](../../scripts/analyze_j_lens_completions.py)
is a different calculation. No fit-individual readouts, centered-individual
readouts, reconstructed rank-four-individual readouts, topic-mean readouts,
negative individual/mean readouts, or full vocabulary logits were retained.
Do not describe any of those as available matched baselines.

## Fixed roster and metric before evaluation

- Feature/layer: `activation_11` only. Axes: `PC1`–`PC4`, with `random1`–`random4`
  retained as a separate control family. Methods: `jlens`, `plain`.
- Candidate contents: all 16 `heldout` rows (both framings of eight contents) in
  dataset.json (artifact not distributed in this public snapshot): astronomy-4/5,
  cooking-4/5, football-4/5, programming-4/5. Show **prefixes only**. These
  activations occur before the completion; completion text would add information
  unavailable to the activation and change the interpretation question.
- For each axis choose the candidate with greatest and smallest signed score,
  with lexicographic ID tie-breaking. If both extremes are different framings
  of the same content, retain and flag the item; never substitute a nicer pair.
  PC scores come directly from
  analysis_arrays.npz (artifact not distributed in this public snapshot).
  Random scores are saved-feature arithmetic: `(h - fit_mean) @ q`, using seed
  `20260910` and the QR recipe in the acquisition source. Original random basis
  vectors and interpreter/NumPy/build metadata were not saved in acquisition
  provenance. The current interpreter, NumPy version/build and platform will
  be recorded. Identical basis reproduction cannot be certified from seed and
  recipe alone: NumPy's [stream compatibility](https://numpy.org/doc/stable/reference/random/compatibility.html)
  is environment-dependent, and [QR](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html)
  uses LAPACK. Random-control correspondence is therefore explicitly secondary
  and conditional on reproduction of the original basis. Stored-score PC
  primaries are unaffected. No new model execution or PCA refit is needed.
- One matching card contains that axis's positive and negative **complete
  12-token lists** under one lens plus its two endpoint prefixes in randomized
  order. Randomize the two token-list positions too. Hide method, PC/random
  identity, topic labels, signed scores, token-logit scores and source IDs.
  Display strings faithfully; escape whitespace rather than delete fragments or
  translate only selected tokens. Fixed packet seed: `20260911`.
- Fixed prompt: “Match the two token lists to the two prefixes. Which assignment
  best reflects their meaning? Choose A1_B2 or A2_B1 even if uncertain; report
  confidence as low, medium or high.” No abstention; confidence is descriptive
  and never changes the primary score. The answer key comes exclusively from held-out
  geometric extrema and randomized positions, never from topic names or a
  hand-written token dictionary.
- There are **8 PC cards** (4 axes × 2 methods), and 8 separate random cards.
  Primary descriptive metric: correct polarity assignments out of four for
  each method, paired difference, and the four per-axis outcomes. Random cards
  use the same metric, reported separately. Forced-choice chance is 1/2 per
  card under a no-information guessing rule. No p-value or population CI.
- Two fresh Astra raters receive eight cards each, ordered by anonymous card
  ID, exactly one method per axis. Rater 1 receives odd PC J-Lens/even PC plain
  and odd random plain/even random J-Lens; rater 2 receives the complement.
  Each therefore judges two PCs and two random axes per method. No rater sees
  the counterpart for an axis: shared endpoint options permit answer transfer.
  Both return locked JSON choices/confidence before answer keys are released.
  Give only public card contents in their messages; no tools, files, prior
  context, answer-key source, axis interpretations, or archive access. This is
  two raters of the same model, with no per-item replication or human validation.
  Counterbalancing avoids complete method/rater confounding but does not remove
  every rater-by-axis effect.

## Packaging checks and source freeze

[Packet builder](../../scripts/package_j_lens_simple_comparison.py) is import-inert.
Its `fixture` stage uses fabricated data only. The explicit `build` stage first
rejects any existing packet target, verifies fixed expected SHA-256 input pins
before loading, and verifies them again after loading. It checks the exact
16 held-out IDs/order, unique 33 readout names and finite array values. Scientific
loading is limited to saved arrays/readouts; no original analysis is rerun.
Public cards contain anonymous IDs, instructions, token lists and prefixes
only. A separate private key retains axis/lens/signs, answer, endpoint scores,
same-content flags, rater assignments and provenance. Do not expose the private
key to raters. Main runs one packaging invocation after source review.

## Individual/full-vector and mean controls: audited, not added to this test

The saved individual readouts and the one mean exist under both methods.
Their
input attribution task differs from axis pole matching, so **do not combine
their scores or claim PCs outperform individuals** from the pole task.
No separate individual-prefix matching task is included in this comparison.

## Limits that stay beside the result

This is an outcome-informed evaluation design on previously inspected readouts,
not preregistered fresh confirmation. Four PCs are correlated summaries of the
same fit set; only eight unique held-out contents are available. Selecting
extrema makes a limited, favorable correspondence test; success does not imply
all held-out scores are interpretable or the four axes are distinct concepts.
Random axes may also carry semantic information after the lens, so they are
not guaranteed to behave like pure chance. Token familiarity alone is not
correspondence. Multilingual/fragmented rankings and an English-only reader can
affect judgments. The plain control is exactly this saved normalized residual
unembedding, not a comparison to every tuned/interpretable lens.

Generic lexical overlap could be computed without hand-crafted topic words,
but it would mainly measure shared spelling, miss synonyms and multilingual
readouts, and reward copying prefix verbs. Do not substitute it for semantic
interpretation. Geometry-based answer keys plus blinded semantic matching
avoid that tautology while remaining a small exploratory check.