# Independent-text J-Lens repeatability pilot

Codex — Spectral Optimizer Investigation · 11 September 2026 BST

## Question and current boundary

Does the useful PC3 within-topic ordering signal repeat on independently
written text and with more than one reader per description? Keep the same
four canonical directions and all three existing reference arms. The aim is
positive usefulness and its limits, not selecting another favorable subset.

The earlier [within-topic result](../2026-09-10-jlens-within-topic/results.md)
used 24 authored prefixes and one reader per arm/axis. All its stages and
delivery are complete. The current design is outcome-informed by that work,
but new article selection and reader rules are fixed before obtaining the
new text panel or measuring its activations. This is a small prospective
transfer pilot, not a population estimate or independent research-team study.

This freeze admits source collection after collector review. It does not yet
admit a new model run: the forward-only adapter, resource check and final
dataset/measurement pins require a separate implementation acceptance.
No optimizer, training, paid cloud, paid judge, lens refit or repeated decoding.

## Mechanical text selection

Source: English Wikipedia, with these four category roots, in this order:

| Topic label | Category |
|---|---|
| astronomy | Category:Astronomy |
| cooking | Category:Cooking |
| football | Category:Association football |
| programming | Category:Computer programming |

Before this freeze, main made one metadata-only reachability probe for the
astronomy root. It returned53namespace-0 members, no continuation/error;
only the count was displayed. No candidate article extract has been inspected.
A web-search result incidentally displayed the general Astronomy lead while
looking for API documentation; no exclusion or preferential selection of that
article is permitted. This is not a claim that researchers have never seen
any Wikipedia text.

Obtain and retain a fresh complete category-member response for each root,
namespace0, page type, cmlimit500. Require no continuation and at least6members;
otherwise stop source assembly and document the limitation before any model
run. Do not silently sample the first alphabetical page or traverse subcategories.

Sort candidates independently in each topic by the hexadecimal SHA256 of
UTF-8 `20260914|<topic>|<pageid>` (decimal page ID), breaking any hash tie by
page ID. This is a deterministic pseudorandom ordering of the captured
catalogue, not a probability sample of all prose on the topic. Inspect
candidates only in this order until6eligible distinct pages per topic have
been obtained, with a hard bound of20candidate requests per topic.

Request each candidate by page ID with `prop=extracts|revisions|info|pageprops`,
plain-text introductory extract with exchars1200, revision IDs/timestamp,
and full URL. No redirect resolution. Eligibility requires namespace0, no
missing/invalid/redirect/disambiguation marker, at least16whitespace-separated
words in the returned extract, a valid page ID/title/URL/revision ID, and a
new page ID and prefix not selected earlier. Do not reject for odd vocabulary,
topic mismatch, desired semantic theme, predicted PC score or apparent quality.
The service may return slightly more than1200characters; that request setting
is not an eligibility cutoff (the2MiB response cap remains). Record every
skip and exact reason. Cross-topic duplicates belong to the
first root in the fixed topic order; later roots continue down their list.

The prefix is exactly the first16words joined by single ASCII spaces, using
Python Unicode `str.split()` on the returned extract. No sentence selection,
paraphrasing, punctuation repair, added preamble, instruction/chat wrapper,
verb filtering, or manual shortening. It may cut a sentence mid-thought;
the task is prefix ordering, not document comprehension. Empty/short extracts
are technical exclusions, not permission to choose a nicer passage.

Keep24rows, six per topic, with IDs `<topic>-wiki-0` through `-5` in accepted
rank order. Pair0–1,2–3,4–5 within each topic, giving12disjoint pairsP01–P12.
All24pages and12pairs are retained once the dataset is frozen. The six
examples per category need not represent distinct subtopics or independent
authors, and Wikipedia may already occur in the model/readers' pretraining.
Pairs are within the same source category, not independently verified
same-semantic-topic pairs. Keep that distinction in reporting.
“Independent text” here means externally authored and separate from the
investigation's fit/hand-written panels, not certified unseen training data.

## Attribution and source integrity

Retain the API request parameters, retrieval times, exact response bytes and
SHA256, selected IDs/titles, reported revision IDs/timestamps, source URLs and
the derived16word prefixes. The exact captured bytes define the source
snapshot. A revision number reported alongside an extract is not an atomic
historical reconstruction guarantee; do not claim that more strongly.

Attribute each excerpt to its Wikipedia article/contributors via its page
and history links, supply the CC BY-SA4.0 license link and identify the
whitespace normalization/truncation. Preserve any applicable additional
attribution notice found during the source check; do not silently relicense
the repository as a whole. Publish excerpt-derived dataset/display content
under those source terms, with code and research analysis distinguished.
No article illustrations or other media are needed.
Before any redistribution of excerpts, check the fixed selected pages and
their attribution notices (including applicable footer/history/talk-page
notices). The collector's extract/revision/info/pageprops API response alone
does not certify those notices. Record the check and preserve requirements;
an unresolved rights issue limits publication, not permission to substitute
a scientifically more favorable article or silently change the dataset.

The TextExtracts service has documented formatting/empty-extract limitations;
we do not use its discouraged sentence-count option. Record source oddities
instead of assuming clean natural prose. This is a small external research
consumer, not a new feature running in Wikimedia production.

## Fixed model-side question

For each prefix x and each unchanged canonical U32 direction u, measure the
same layer11 last-input-position residual feature and signed centered score
`z(x) = uᵀ(h(x) − μ)` using the original fit mean. No covariance recomputation,
new eigenvectors, new topic means for scoring, reorientation, fit-example
selection, normalization change, or new J-Lens decoding.

Use the same cached frozen Qwen3.5-0.8B model/revision, tokenizer configuration
and forward path as the previous acquisition. Exact truncation/token limits
must be resolved in the separate source review before model admission. In
particular, do not silently cut a16word prefix to fit an old8word assumption.
Save actual input tokens/masks and extracted features so checks need no rerun.
The moved text distribution is intentional; failure could reflect transfer
limits rather than absence of readable directions in the original regime.

Reference arms remain A signed-direction top12tokens at each pole, B top12
tokens of the original selected high/low fit examples, C their intact fit
prefixes. Keep exact strings/order/whitespace and all four axes. These global
references are not tailored to the new topics or excerpts, and equal token
list length is not equal information across arms.

## Six new isolated readers: two judgments per arm/axis

Each reader receives one anonymous48choice packet: four direction blocks,
each with the same12new text pairs and one assigned reference arm. All six
packets are constructed and committed before any reader is invoked. No
conversation history, tools, corpus/source metadata, private maps, scores,
old results, other readers' responses, axis or arm names in reader context.

| Reader | PC1 | PC2 | PC3 | PC4 | Presentation cohort |
|---|---|---|---|---|---|
| rater1 | A | B | C | A | 1 |
| rater2 | B | C | A | B | 1 |
| rater3 | C | A | B | C | 1 |
| rater4 | A | B | C | A | 2 |
| rater5 | B | C | A | B | 2 |
| rater6 | C | A | B | C | 2 |

Use local `random.Random(20260915)`, recording exact interpreter/source.
Cohort1 chooses a fixed swap for each axis/pair, common across A/B/C.
Cohort2 uses its exact opposite, also common across arms. Randomize labels
and ordering without rerolling. Thus every fixed target is presented both
ways within each arm: alwaysFIRST and alwaysSECOND each get exactly half
credit over the two readers (including half credit on exact score ties).
This removes aggregate position imbalance by design, not by choosing a
post-hoc better-side baseline.

Prompt unchanged: “Using only the two reference descriptions, which of the
two prefixes should have the higher value on this direction? Choose FIRST or
SECOND even if uncertain.” Require one FIRST/SECOND response per item, no
confidence or explanation fields. One response per fresh agent; no rerating,
repair after outcomes, or reader selection. Invalid/incomplete responses are
reported as operational failures, not silently replaced. All288exact choices
and response lock must be committed before reading the new scalar key for
grading. No new paid API. Model-family sampling/configuration limits remain.

## Outcomes and decision

Keep all48axis/pair targets and all288reader/arm choices. Grade by sign of
the saved first-minus-second score; exact ties receive half credit, every
nonzero gap remains. Report all four axes and three arms, each reader's
12choice result, pooled24choice credit per arm/axis, pooled96choice credit
per arm, both constant-position controls, paired A-minus-B/A-minus-C credits,
and reader agreement after mapping choices back to the underlying prefixes.
Two readers disagreeing is not a majority-vote success or a target exclusion.
Because the two readers also see opposite orders, disagreement measures a
mixture of reader variation and order sensitivity; this design does not
separately identify those components.

The previously highlighted PC3 is the motivating transfer question, not the
only axis measured or reported. A useful positive need not defeat every
control; a general advantage needs broader evidence than this pilot.
No p-values, population confidence intervals, treating288choices as288
independent texts, or causal/safety/optimizer conclusions. Reader repetition
addresses one-reader fragility, not independent model-family or human validity.

If descriptions predict meaningful orderings across the new texts/readers,
preserve that positive and its scope. If performance is weak or inconsistent,
distinguish description transfer, source/length shift and reader instability
using the full retained record before deciding whether any focused refinement
is worthwhile. Do not refit the current data into a success or treat one
failed transfer as a disproof of the original useful correspondence.
