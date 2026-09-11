# J-Lens within-topic comparison: a useful third-direction signal

Codex — Spectral Optimizer Investigation · 10 September 2026 UTC

**Direct J-Lens direction descriptions predicted 11 of 12 same-topic
orderings on PC3.** Example-token descriptions got 9/12; raw fit-example text
got 8/12. This is a concrete useful correspondence in this authored panel,
not merely a coherent-looking word list or a topic-classification result.
The fourth direction remains a counterexample to general reliability:
direct descriptions got 6/12, versus 8/12 and 10/12 for the example arms.

This is exploratory reuse of existing measurements, not fresh-sample
confirmation. Every direction and every pair was retained. Overall direct
tokens tie example tokens and trail example text, so no general advantage is
established. A is direct signed-direction tokens; B is fit-example tokens;
C is intact fit-example text. All arms use the same axis/pair targets.

| Direction | A: direction tokens | B: example tokens | C: example text | Always FIRST | Always SECOND |
|---|---:|---:|---:|---:|---:|
| PC1 | 7/12 | 6/12 | 8/12 | 6/12 | 6/12 |
| PC2 | 7/12 | 8/12 | 7/12 | 3/12 | 9/12 |
| PC3 | 11/12 | 9/12 | 8/12 | 8/12 | 4/12 |
| PC4 | 6/12 | 8/12 | 10/12 | 8/12 | 4/12 |
| All four | 31/48 | 31/48 | 33/48 | 25/48 | 23/48 |

Source: the [complete frozen grade](graded/grades.json), SHA256
`186cb971b9a52b2d47bcf586a0a3e38be453c18bc2c67a66ffe77d0a66f1582f`.
The 50% random-guess expectation is not a significance threshold. The two
constant-position results above are recorded controls, not an oracle policy
allowed to choose its better side after seeing each axis's outcomes.
No exact score ties occurred. All nonzero gaps, including very small ones,
remain in the results; no sign flips or exclusions were made.

## What is useful here?

The unchanged positive PC3 list contains testing/evaluation/verification
terms. Its reader correctly favored, for example, the static-analysis prefix
over the deployment prefix and the lunar-recording prefix over the black-hole
prefix. These are illustrations selected after the result, not new metrics or
proof of a semantic mechanism. The single PC3 error favors the open-source
patch prefix over database migration; the measured ordering goes the other
way. The full dataset (artifact not distributed in this public snapshot),
pair roster (artifact not distributed in this public snapshot) and all 144 scored choices preserve every case.

PC3's success is not just distinguishing the four supplied coarse topic
labels: the two members of each pair have the same label. Topic-mean
subtraction therefore cancels exactly within each pair. This does not rule
out useful lexical, syntactic, subtopic, or other non-semantic cues. It also
does not establish that PC3 is a single concept or a cluster.

The negative result matters constructively. PC4's direction description is
not useful by this ordering measure here, despite a 10/12 raw-example result.
That motivates caution about the description/reader interface, rather than
concluding that the underlying direction has no readable structure. Reader
identity is confounded with arm within each axis, so this cannot isolate the
cause of the arm difference.

## How this relates to earlier findings

The [previous cross-topic test](../2026-09-10-jlens-fresh-content/results.md)
found direct scores 11,9,9,2 out of12. These new within-topic scores are
7,7,11,6. The overall direct total happens to be 31/48 in both tasks; the
axis-level pattern differs. These are different pairings, readers, allocations
and presentation swaps, not a causal experiment removing only topic labels.

The [saved-fit geometry](../2026-09-10-jlens-fit-geometry/results.md) found
both PC3 and PC4 chiefly vary among contents within topics. The current
comparison directly tests readability of new within-topic pairings; it shows
why that geometric fact alone was insufficient to predict useful descriptions.
The [population-regression interpretation](../../research/jlens_covariance_interpretation_2026-09-10.md)
supplies a mathematical steelman for the positive result, but neither the
regression identity nor these reader choices establish a causal mechanism.

## Execution and provenance

Protocol/pairs were fixed atf2c9aa3. Main accepted the 16 fabricated tests and
full source diff. A pre-packet wording-only clarification was committed
at7e7561d; the exact final [protocol](protocol.md) SHA256 is
`25b632fc7afa5a291a007ba3bf12ac9036bda9a0c5c0a6e81705f62701950c3d`.
The final [judging source](judging.py) SHA256 is
`3b6148b835001fa9a088ad5f70c779a12d4993966bef3d98016bf1b8de0d3f2b`.
See [design](design-review.md) and [implementation](implementation-review.md)
reviews for their exact scope, including the original and amended pins.

One package ran at22:56:43UTC, using Python3.12.3 and seed20260913. Manifest
SHA256 `8fb30df285f757005193e1af27b213049f046e64baad65c13edf99bbca129ab3`,
commit3401e54. Three new isolated agents `/root/within_topic_rater_1`,
`/root/within_topic_rater_2`, `/root/within_topic_rater_3` each received only
their exact public JSON packet with no conversation history or tool use.
Each returned exactly 48 choices once. Main saved the returned JSON unchanged
apart from a terminal newline; the seal copied those exact bytes.

All 144 choices and the response lock (artifact not distributed in this public snapshot) were committed
at`f92d70d1dc6eefce7720b40c669701e8ded7df2f` before grading. Lock SHA256
`d7227b18e4a0e079860e0415ebff833361c0240e2df6cddb10bcad6ad576e1ea`.
One grade began23:02:39UTC, exit0, committed at33279f2. It verified exact
committed response bytes, source and all inputs before scalar-key access.
The key is the unchanged old `fresh-features/scores.json`, SHA256
`f1e4d2aa1ace39d9d482681a3e672f60c279348cf62884042a9ee93409a19a46`.
All attempts are consumed; no repeated judgment, grade or acquisition.

## Confidence and next step

This is a small single-model/layer result: 24 hand-authored measured texts,
12 new disjoint pairs reused across four directions, and one same-family
Codex reader per arm/axis. Researchers had seen earlier outcomes before
designing the task. The descriptions are global fit extrema, not tailored
within-topic descriptions; list length does not equalize information. There
are not 144 independent cases. No p-values, population intervals, general
method ranking, causal J-Lens fidelity, safety or optimizer benefit is claimed.