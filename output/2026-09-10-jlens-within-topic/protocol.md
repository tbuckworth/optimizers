# J-Lens within-topic directional ordering

Codex — Spectral Optimizer Investigation · 10 September 2026

## Question and status

Can the unchanged J-Lens reference descriptions help a reader order two
prefixes on the **same authored topic** by their score on a fixed direction?
The earlier cross-topic result and saved-fit variance decomposition motivate
this follow-up. This protocol is fixed before collecting its responses, but
the measured texts and earlier outcomes have already been used. It is an
**exploratory reused-data comparison**, not a fresh-sample confirmation.

The user has prioritized a meaningful simple J-Lens comparison. No optimizer
experiment, new activation acquisition, refit, decoding or model run is part
of this task. Existing completed studies and their payloads remain immutable.

## Fixed inputs and pairing

Use all 24 prefixes, in their original order, from
`../2026-09-10-jlens-fresh-content/dataset.json`, SHA256
`c655a5531af3215a29e332ddc6b23ac4854b0bffdf91b8ad5e8a045748ff7cd9`.
For each topic (astronomy, cooking, football, programming), pair IDs 0 with 1,
2 with 3, and 4 with 5. The 12 disjoint pairs in `pairs.json` use every prefix
once. This adjacent-ID rule is not chosen using score gaps or pair semantics.

Keep PC1–PC4, their signs, and the canonical U32 directions unchanged. Use
the existing worktree files under
`output/2026-09-10-j-lens-fresh-content/`:

- `references/interpretations.json`, SHA256
  `0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`;
- `fresh-features/scores.json`, SHA256
  `f1e4d2aa1ace39d9d482681a3e672f60c279348cf62884042a9ee93409a19a46`.

The worktree is `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree`.
The scalar key has been hash-checked, not reparsed for this follow-up before
protocol freeze. The researchers have seen earlier aggregate and item-level
results: this is not a claim of globally unseen measurements or preregistration.

## Three unchanged reference arms

A: top 12 token strings from each signed direction through J-Lens.
B: top 12 token strings from J-Lens on the high/low fit examples.
C: the intact high/low fit prefixes themselves.

Preserve exact strings, empty/whitespace token strings, duplicates, and the
same fit-selected examples. These are different practical descriptions, not
equal-information representations. A useful absolute correspondence is worth
reporting even if A does not beat B or C.

## Reader allocation and packaging

Three new isolated Codex reader agents, no conversation history, each receive
one public packet with 4 anonymous blocks of 12 comparisons (48 choices).
They see only the assigned references and the two prefixes, not axis/arm
names, score values, truth labels, old results, or the private map. No tools,
other files, outside research or contact with other readers. No new paid API.

Fixed allocation across PC1, PC2, PC3, PC4:

| Reader | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| rater1 | B | C | A | B |
| rater2 | C | A | B | C |
| rater3 | A | B | C | A |

Use local `random.Random(20260913)` and record the exact Python version and
source hash. For each axis/pair the FIRST/SECOND presentation is the same in
all three arms. Randomize public block/item labels and order using that one
stream. Do not regenerate to improve balance or results.

Prompt: “Using only the two reference descriptions, which of the two prefixes
should have the higher value on this direction? Choose FIRST or SECOND even
if uncertain.” Require exactly one FIRST/SECOND choice per item, no rationale
or confidence field. Return one JSON response using schema
`jlens_within_topic_responses_v1` and the exact packet ID.

One response per reader. Preserve each returned JSON as received (terminal
newline may be added). Validate and seal all 144 choices, then commit their
exact bytes and lock before reading the scalar key for grading. No repair,
retry, rerating or sign reversal after outcomes. An invalid/incomplete return
is an operational failure to report, not a license to choose a better return.

## Outcome and interpretation

Truth is the sign of saved score(first) minus score(second). Exact equalities
receive 0.5 credit for either choice; retain every nonzero gap, however small.
Keep all 48 axis/pair targets and all 144 arm-specific predictions. Report
correct/total by axis and arm, pooled credits out of 48, both constant-position
baselines, A-minus-B and A-minus-C paired credit differences, and every gap.
No exclusion, axis selection, relabeling, p-values or population intervals.

Within each pair the supplied coarse topic is constant. Success could show
ordering beyond that label alone on this panel, but does not isolate semantic
mechanism, rule out lexical/syntax cues, validate arbitrary topic groupings,
or establish natural-corpus generalization. Failure could reflect inadequate
descriptions, difficult small gaps, reader limitations, or a poorly readable
axis; this test alone does not distinguish them. There are 12 new pairings of
24 reused texts, 4 dependent directions, and one reader per arm/axis—not 144 independent
examples or a reliable ranking of reader populations. Preserve all positives,
negative results, and the cross-topic result without promoting one to a broad
claim. No causal, safety, clustering, or optimizer benefit is tested here.
The references remain global fit extrema, not topic-specific explanations.
Cross- versus within-topic results also change pairings and readers, so their
difference is not a clean causal effect of removing topic information.

## Implementation boundary

Copy the earlier import-inert judging module into this new directory, changing
only pairing validation, seed/allocation/prompt, new-stage schemas and input
pins. Retain the acquisition reference/score schemas because those files are
unchanged. Fabricated-only tests must cover exact adjacent pairing, rejection
of other within-topic pairings and cross-topic pairings, all 144 allocations,
score-independent packaging, strict responses, tie/small-gap handling, and
exclusive stages with a committed response lock before score access.

The current [Python JSON contract](https://docs.python.org/3.12/library/json.html)
supports duplicate-key inspection and non-finite-value rejection; retain the
existing strict decoder and size caps. The [exclusive file-creation contract](https://docs.python.org/3.12/library/functions.html#open)
supports fail-if-existing stage outputs. The [random reproducibility notes](https://docs.python.org/3.12/library/random.html#notes-on-reproducibility)
warn that algorithms may change between Python versions; a seed alone is not
a cross-version guarantee, hence exact interpreter/source recording.

Freeze protocol and implementation before one packaging call. The subsequent
seal and grade stages each use separate exclusive output directories. A
separate result check may verify scalar arithmetic without rerating. No GPU,
training, new measurements, paid cloud reservation, or completed-stage restart.
