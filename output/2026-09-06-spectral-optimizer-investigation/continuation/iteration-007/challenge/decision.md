# Design challenge disposition — mechanism first

Codex parent, Spectral Optimizer Investigation, 6 September 2026.
Status: **MAJOR_REVISIONS; no implementation, pilot or training approval.**

## Decision

Defer the proposed 16-plus-four whole-policy bundle extension. Specify a smaller
prospective common-state one-step study first. This is a change in research
priority, not a new empirical result or a claim that the deferred study is invalid.
The original [planning decision](../design-decision.md) and
decomposition (artifact not distributed in this public snapshot) remain available as historical proposals.

Three independent concurrent reviewers read the base proposal without seeing
one another's assessments:

- [Assumptions](assumption-analysis.md): the next-action rule is missing;
  fresh randomness is not an independent implementation or untouched test set.
- [Mentor](mentor-review.md): MAJOR_REVISIONS; collect exact state and examine
  immediate AdamW response before more performance bundles.
- Pre-mortem (artifact not distributed in this public snapshot): a final bounded extension is conditionally
  defensible, but overinterpretation and missing replay state are material risks.

The parent accepts the mentor's ordering recommendation. The pre-mortem's
conditional defense is preserved, not rewritten as unanimous rejection.
The assumptions review enumerates eleven assumptions despite saying ten in its
summary; this minor counting error does not affect its substantive advice.

## Information-value check of the deferred extension

Apply these possibilities separately to current-minus-AdamW, lagged-minus-current
and restored-minus-current. They are scenario descriptions, not numerical
classifications to fit after seeing outcomes.

| Possible additional observations | What they add | What they would not change |
|---|---|---|
| All four contrasts positive | Four favorable realized bundles under this generator | No stable population sign or mechanism identified; still need same-state response |
| All four negative | Four adverse realized bundles | No universal failure or cause identified; still need same-state response |
| Mixed, small magnitudes | Additional conditional variability | No equivalence claim; still need same-state response |
| Mixed, large magnitudes | More examples of strong conditional sensitivity | No attribution to initialization, split, corruption or order; still need same-state response |

Thus the immediate next scientific action is unchanged in every scenario. More
MNIST performance seeds are not the priority. This does not assert that no
future decision could justify them, nor automatically reactivate the sweep
after the mechanism study. A later extension needs a new stated information need.

## Resolve now

1. Define a **current-policy-state conditional intervention**, not trajectory
   mediation: identical parameters, Adam moments, step counters, observer,
   next batch and random state before each alternative one-step input.
2. Require complete prospective anchors. Model-only iteration-006 checkpoints
   and iteration-005 parameter hashes cannot be relabeled complete runtime state.
3. Include both reciprocal norm swaps. This supplies a direction-by-input-norm
   comparison at fixed history; it does not match actual AdamW displacement.
4. Separate numerical audit from fresh-draw sensitivity. An independent
   reconstruction checks implementation; a separate seed checks another state
   sample. Neither provides an untouched generalization test.
5. Do not load official test data or select checkpoints in the proposed mechanism
   study. Predeclared anchor times remove own-selector exposure as a within-anchor
   confound; reused MNIST and source-policy conditioning remain limitations.
6. Prove snapshot round-trip and instrumentation neutrality before any neural GO;
   count pilot, primary, audit, serialization overhead and partial artifacts.

See [revised common-state specification](../common-state-design.md) and
implementation feasibility check (artifact not distributed in this public snapshot). Both are proposals
for another design check, not outcomes or launch authorization.

## Defer and accept explicitly

- Factor attribution and matched-plan cross-harness replay are deferred. The
  static source check supports reuse, not an explanation of the historical sign
  reversal. The proposed study is standalone, not pooled with older bundles.
- Raw-AdamW-, lagged- and restored-policy source trajectories are omitted for
  the minimal first conditional test. Broad source-policy claims are forbidden.
- Clean training, scalar whole-training arms, SGD and actual-update matching
  remain separate unperformed interventions, not conclusions of a norm swap.
- A handful of states cannot establish a sign frequency, equivalence, long-run
  denoising or production recommendation. Retain every signed cell and mask.
- If complete capture fails, stop and revise the capture design. The reviewers'
  option to drop anchors applied to the now-deferred performance study; anchors
  are indispensable for the replacement question.

No experiment was started or restarted during this challenge. Prior evidence,
production optimizer, delivered PDF and two-hour timer remain unchanged.
