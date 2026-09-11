# Saved verb-position reader comparison

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

## Question and status

Can fresh readers use the existing direction-token descriptions to predict
the saved local ordering, and do they do better than the unchanged fit-example
prefixes on the same targets? The previous [fixed-gloss result](../2026-09-11-jlens-new-verb-transfer/results.md)
already establishes useful 16/16 signed prediction on the selected PC4 panel.
That positive does not require a baseline win to count as useful.

This is a separate **exploratory saved-target reader comparison**. Researchers
know the measured outcomes and selected this panel after its positive result.
Reader rules are fixed before new responses, not before the old model scores.
No new-data confirmation or general population estimate. No old experiment
will be restarted; no model forward, tokenization, PCA, decoder, training,
paid API or optimizer/familiarity branch is needed. Existing user autonomy
covers this J-Lens-only task without another approval gate.

## Fixed inputs and target boundary

Use all32 input rows and16 pairings of the completed new-action-word panel:

- Main `../2026-09-11-jlens-new-verb-transfer/dataset.json`, SHA
  `fd4605ab7a54ad17dd6aed27492f1fee61f6924c5515583c0f911b547d484d00`.
- Main `../2026-09-11-jlens-new-verb-transfer/pairs.json`, SHA
  `735b9b88c19654fa38251cee0221dc65dd523feb24864dce139a25b390f683c7`.
- Worker `output/2026-09-11-j-lens-new-verb-transfer/token-preflight/tokens.json`,
  SHA `d7ae3f7e9494d3e833c258b9b847a055bbafd0c22bcca0c003739fd23f522179`;
  preflight receipt SHA `03b61d06e886a114ce4fe31e2235b1b24e296a9951024437ad9cc0fd4360d092`.

Worker root: `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree`.
Its source/measurements are frozen; main checks them, not changes them.

For each row, the reader target is exactly `text[:verb_span[1]]`, with no
punctuation addition, paraphrase or later words. Verify this endpoint against
the saved verb capture's character offset, row identity and preflight receipt.
Active prefixes have three words, passive prefixes four. This retains the
complete verb even when its final selected token is only `cribed` or `ized`.
Do not show the active object or passive actor after the captured verb, and
do not rerun the tokenizer or shortened model inputs. Prefixes describe the
causally available text at the already measured state.

Keep original nv01–nv08 content/template pairings. Full rows/spans and O/P IDs
remain in private provenance only; public items contain anonymous IDs and the
two truncated strings. All16 pairings and allfour axes remain, no tiny-gap
exclusion or outcome-based substitution. Only the verb role is judged; the
sentence-final role remains an already reported descriptive measurement.

## Two unchanged fit-only reference arms

Use worker `output/2026-09-10-j-lens-fresh-content/references/interpretations.json`,
SHA `0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`.

A: the exact twelve decoded token strings at each signed direction pole.
C: the exact original high/low fit-example prefixes. Do not add arm B here.
Preserve whitespace, newlines, duplicates, empty fragments and Unicode.
No translations, new verbal gloss, new extrema selection or tailored examples.
In particular, PC4's old examples remain goalkeeper/blocked and recipe/combined.
Readers receive positive/negative reference descriptions, not researchers'
observation/provision label or the successful signed forecast.

Both arms are restricted to original fitting information; their representations
contain different information by design. This is not an information-equivalence,
equal-length or PCA-versus-no-PCA test. Both use the same fixed directions.

## Four isolated readers, two judgments per arm/axis

Each reader receives one anonymous packet of four blocks × sixteen pairs =
64 choices. No conversation history, tools, source paths, private metadata,
scores, prior results, other responses, explicit axis or arm names. Format
differences between token lists and raw examples are necessarily visible.

| Reader | PC1 | PC2 | PC3 | PC4 | Cohort |
|---|---|---|---|---|---|
| rater1 | A | A | A | A | 1 |
| rater2 | C | C | C | C | 1 |
| rater3 | A | A | A | A | 2 |
| rater4 | C | C | C | C | 2 |

Packets are arm-homogeneous: an example-reference reader never sees token
lists from another axis. This replaces the initial crossover allocation
before any packet or reader run, after review identified cross-block leakage
on repeated targets. Readers can still borrow information across axes within
their own representation. Arm differences remain confounded with reader
variation; the two readers per arm are descriptive replication, not a
within-reader causal comparison of representations.

Use a single local `random.Random(20260916)`, recording exact Python version
and source hash. For each axis/pair choose one cohort1 FIRST/SECOND orientation
shared across arms; cohort2 uses its exact opposite. Randomize anonymous block
and item labels and ordering, without rerolls. Each underlying target appears
both ways within each arm, so alwaysFIRST and alwaysSECOND each receive half
of pooled credit. Reader/order effects remain mixed, not separately identified.

Prompt: “Using only the two reference descriptions, which of the two prefixes
should have the higher value on this direction? Choose FIRST or SECOND even
if uncertain.” Use schema `jlens_verb_reader_responses_v1` with exactly packet_id
and 64 `{item_id, choice}` records in addition to schema. No confidence,
explanation, repairs or selected subsets. Readers are fresh Codex agents,
not a new paid endpoint or independent human/model-family validation.

Construct and commit all four public packets/private map before any reader
starts. Record the actual reader handles and supplied packet hashes. Obtain
one response per reader, preserve exact returned JSON bytes (terminal newline
allowed), validate and seal all256choices, then commit the four response copies
and lock before the grader reads the scalar key. Invalid/incomplete output is
an operational failure to preserve, not permission to replace a reader or
repair choices after seeing accuracy. No rerating or majority-vote selection.

## Grading and reporting

Use the unchanged worker `output/2026-09-11-j-lens-new-verb-transfer/forwards/scores.json`,
SHA `431ec80549b0eea2f4b7770f28839696211247ef91275c9349e80b23b430c55b`.
Schema is `jlens_new_verb_transfer_scores_v1` with ordered locations `verb`,
`sentence_end`, each containing PC1–PC4 and all32 row scalars. Validate the
whole schema, but select `verb` only for grading. Scores were corroborated in
the completed checker; no new projection calculation or archive load.

The key is linked by forward receipt
`e985c73109909d293bbc071944fbe67c0daefbd49f9960df9e18a793eac9a07d`,
checked result `d0c841c52bb9947dc45df466fdd5e0a1f9b7ba027c7eae76f1529d17cef96308`
and checker receipt `1a18b75742993f43c962e386c6e9900c26aedb00b1aef14064fed25df3e49983`.
The producer score bytes above are the sole grading key, not rounded tables.

Truth is sign(score_FIRST−score_SECOND). Keep exact nonzero differences;
exact ties give either response 0.5 credit. Report all256 item records,
each reader's16-per-axis credit, 32-per-arm/axis credit, 128-per-arm total,
both constant-position controls, paired A−C credit overall/by-axis/by-cohort,
and reader agreement after mapping choices to underlying prefix IDs. Also
retain active/passive breakdowns; these are two forms of eight contents,
not independent samples. No p-values or population confidence intervals.

Primary focus is PC4's A−C difference out of32, not a newly selected winner.
For descriptive consistency, ask whether each A reader exceeds8/16 and
whether A−C is positive in each cohort. Report partial/mixed evidence too;
these checks are not statistical significance, universal success thresholds
or reasons to erase the earlier16/16 forecast. Other axes and overall totals
remain explicit rather than reselecting the favorable result. An A/C tie
can preserve useful prediction while showing no added benefit on this panel.
No causal, distinct-cluster, natural-corpus or optimizer/safety claim follows.

## Implementation and run boundaries

Adapt the existing inert standard-library packet/lock/grader into a NEW worker
study directory. Never invoke a consumed old stage. Inspect reused code fully,
replace six-reader/three-arm/12-pair assumptions with this exact four-reader/
two-arm/16-pair contract, and bind all input/source hashes. Packaging accepts
no score argument and never reads the key. Preserve exclusive attempt dirs,
strict bounded JSON, exact response blobs and committed-lock checks before
grading. Main must read final source/tests and run fabricated fixtures before
actual packaging. No new infrastructure/dependencies or GPU are necessary.

Fixtures cover exact verb truncation/binding, split-token endpoints, removed
future words, all256 allocations/orientation balance, score-independent packets,
strict64-item responses, exact ties/tiny gaps, active/passive summary counts,
and refusal of repeated/failed stages or grading before a committed lock.
CPU-only, one-thread limits and a60-second command bound for package/seal/grade.
No automatically repeated actual stage. Report the existing paid budget as
spent/reserved $0/$100; subscription reader calls are not charged cloud runs.

Current official [Python JSON](https://docs.python.org/3.12/library/json.html),
[exclusive creation](https://docs.python.org/3.12/library/functions.html#open)
and [random reproducibility](https://docs.python.org/3.12/library/random.html#notes-on-reproducibility)
contracts support bounded strict parsing, fail-if-existing outputs and recording
the interpreter alongside the seed. These are implementation safeguards,
not mathematical evidence that the research hypothesis is correct.
