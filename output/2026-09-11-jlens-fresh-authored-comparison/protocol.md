# Fresh authored J-Lens comparison — protocol before new content

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

## Question

Can the unchanged direction-token descriptions convey useful local score
ordering on new, prospectively fixed content, and do they add value over the
unchanged raw fit-example references? The completed saved-panel result is
useful, but its PC4 added benefit was one win and one tie. It remains evidence
regardless of this next outcome. This test is still hypothesis-guided and
authored, not a random-language benchmark or a clustering experiment.

Main follows the accepted [next decision](../2026-09-11-jlens-verb-reader-comparison/next-decision.md).
No optimizer/familiarity work, training, paid judge or cloud run is selected.
Existing autonomous authority applies; these source/data freezes are execution
safeguards, not new user-approval gates.

## Content authoring before outcomes

Freeze an exact standalone author prompt and lexical exclusion list before
one fresh isolated author receives it. The author gets only: the abstract
observation/recording versus provision/preparation contrast; the requirement
to propose eight matched actor/object pairs with two actions each; grammar/schema requirements;
and alphabetically sorted previous action-lemma exclusions. No direction
tokens, raw fit examples, prior stimulus sentences, scores, earlier results,
reader choices, model identity, axis labels or project history. No tools.
Use `fork_turns="none"` and inherited parent configuration, without overrides.
Record actual prompt/response bytes, hashes and handle. One candidate roster,
not a larger pool selected for apparent promise. Preserve any invalid output.

Exactly eight ordered entries `fa01`–`fa08`, each with seven fields:
`id`, `actor`, `object`, `observation_lemma`, `observation_past`,
`provision_lemma`, `provision_past`. The response wrapper has exactly
`schema: jlens_fresh_author_proposal_v1` and `contrasts` (eight entries).
All six linguistic strings are lowercase ASCII alphabetic single words.
Actors are singular common nouns, objects plural common nouns. Both actions
take that same actor/object plausibly, using a simple-past form that is also
the passive past participle. All sixteen action lemmas must be distinct and
absent from the fixed exclusion list, including declared spelling aliases.
No verb alternatives, explanations, extra examples or confidence fields.

The parent constructs, without paraphrase, exactly these four strings per entry:

```text
The {actor} {observation_past} the {object}.
The {actor} {provision_past} the {object}.
The {object} were {observation_past} by the {actor}.
The {object} were {provision_past} by the {actor}.
```

IDs are `faNN-active-O/P`, then `faNN-passive-O/P`; pair order is `fa01` through
`fa08`, active then passive. Derive exact verb character spans and all16 pair
joins mechanically. Check the complete roster for schema/grammar, repetitions,
stem/alias exclusions and the specified transitive contrast before measurement.
Do not remove an otherwise valid ambiguous case because it appears difficult
or unlike the prior readout. Document semantic overlaps rather than curate
on expected success. If constraints cannot be met, preserve the failure and
resolve it before measuring; no guessed indices or silent replacement.

Lexical inventory must record exact prior text source hashes, row IDs and
surface-form spans plus explicit form-to-lemma mapping. Linguistic mapping
uses declared agent judgment, not a claim of purely algorithmic lemmatization.
Exclusions are fixed before authoring and do not derive from decoder tokens
or scientific scores. They demonstrate only declared prior-input disjointness,
not pretraining novelty, novel semantics or absence of subtoken overlap.

## Fixed target and acquisition scope

Keep cached Qwen/Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, lens/adapter revision
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, original post-block11 state,
BF16/eager/eval/no_grad settings, original source_mean64 and actual U32 axes.
Reference archive SHA `47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa`.
No PCA, mean, signs, decoder or reference reselection. Only one new32-sentence
forward lane and **only the final distinguishing-verb subtoken** capture;
no sentence-final or later-ending condition. The target prefix supplied to
readers is exactly `text[:verb_span[1]]`, retaining the whole verb but no
future object/actor or added punctuation. No additional shortened-input run.

Commit the frozen panel and the decision to run its reader comparison before
any scalar outcomes. Do not examine signs/gaps to decide whether the panel
deserves judging. Acquisition may save results, but keep new scalar contents
out of author/reader contexts and out of selection decisions. A separate
saved-data arithmetic check follows locked judgments; no model replay.

## Fixed reader comparison

Original references are unchanged `interpretations.json` from the completed
fresh-content study, SHA
`0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`.
A uses the original twelve signed direction tokens per pole; C uses original
raw fit prefixes. No arm B. Preserve whitespace, Unicode, duplicate/empty
token fragments and the original PC4 blocked/combined examples. Both are
fit-only, not information-equivalent representations.

Four fresh history-free readers, each64choices over four anonymous16-pair
blocks. Raters1/3 receive only A; raters2/4 only C. Cohort1 is raters1/2;
cohort2 is raters3/4. Use local `random.Random(20260917)`, one recorded stream:
axis/pair orientations matched across A/C in cohort1 and exactly reversed
in cohort2, then shuffled anonymous labels/block/item order without rerolls.
No source paths, model/axis/arm labels, O/P labels, private maps, scores,
author rationale, earlier results or other responses. Only the public packet
and common no-tools/response-schema wrapper, no model/reasoning overrides.
Record actual handles/configuration insofar as exposed; no independent-family
or physically removed-tool claim.

Common question: “Using only the two reference descriptions, which of the two
prefixes should have the higher value on this direction? Choose FIRST or
SECOND even if uncertain.” Response schema `jlens_fresh_authored_responses_v1`,
packet_id and exactly64 `{item_id, choice}` records, FIRST/SECOND only.
Commit all four packets and private mapping before any reader. Retain one
response per allocated reader; no repair, replacement, rerating or majority
selection. Commit all256 response choices and exact-copy lock before the
grader opens the scalar key. Invalid output is an operational failure, not
permission to replace a low-scoring reader.

## Estimand, reporting and limits

For each row/axis, `s = (float64(h32) − source_mean64) · float64(u32)`.
Truth is the exact sign of `s_FIRST − s_SECOND`; exact ties receive0.5 credit,
with no tiny-gap exclusion or rounding before comparison. Primary comparison
is PC4 A−C credit out32. Separately report whether each A reader exceeds8/16
and whether A−C is strictly positive in each cohort; a tie is not a win.
These are descriptive consistency checks, not population significance tests.
Also retain the fixed O−P forecast's all16 signed gaps as descriptive evidence,
without substituting it for reader accuracy or dropping contrary cases.

Report all256 choices, every gap, four axes, both arms, four readers, each
cohort and active/passive form, text-level reader agreement, both fixed-
position controls and paired A−C differences. No best-axis substitution,
sign reversal, pooled old/new accuracy, independent-Bernoulli confidence
intervals, p-values, causal or cluster/safety/capability claim. There are
eight hypothesis-guided contents in two forms, one model/layer and same-parent
AI readers. Arm and reader effects remain mixed; within-arm cross-axis
context is available. A useful A result with an A/C tie remains useful but
does not prove added benefit. A failure limits transfer, not erases prior work.

At creation only protocol/lexical-source preparation is active. No author,
new stimuli, code, preflight, reader, model or grading stage has been run.
