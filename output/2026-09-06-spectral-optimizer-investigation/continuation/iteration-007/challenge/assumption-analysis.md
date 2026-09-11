# Assumption Analysis

## Summary

## Critical Assumptions (Low confidence, High impact)

### 1. Four primary bundles have enough information value without a decision rule
- **Category**: Methodological
- **Confidence**: Low
- **Currently assumed because**: Four new paired bundles are cheap, the old noisy contrasts reverse sign, and the plan promises to stop afterward. Rejecting the proposed 3-of-4 and 1 pp thresholds is correct, but leaves no rule connecting possible outcomes to beliefs or action.
- **What changes if wrong**: Every possible outcome becomes another descriptive pattern followed by the already planned stop and mechanism study. The run adds four draws without discriminating among the live explanations.
- **How to test**: Before implementation, write an outcome-to-decision table covering concordant positive, concordant negative, small mixed, and large mixed results for all three contrasts. State what belief and next action changes. If none changes, skip the extension and prioritize common-state mechanism work.
- **Relevant evidence**: Seven earlier noisy bundles are already heterogeneous: iteration 004 gives current-minus-AdamW `-2.38 pp` on average, iteration 006 gives `+4.827 pp`, and its separately fixed bundle reverses both lagging signs.

### 2. A single same-harness audit bundle independently checks the scientific comparison
- **Category**: Independence
- **Confidence**: Low
- **Currently assumed because**: Bundle 60011 is frozen separately, run after the primary cells, kept unpooled, and given its own train-before-test boundary.
- **What changes if wrong**: Four extra cells neither validate a population sign nor detect shared implementation error. Calling them an audit overstates what is one additional random-effects draw through the same producer, selector, and test set.
- **How to test**: Define its role before launch. If it is a fresh-draw sensitivity check, predeclare only a side-by-side four-value report. If it is an implementation audit, use an independent producer or replay. Never let its result promote or demote the primary narrative.
- **Relevant evidence**: Iteration 006's fresh bundle demonstrated heterogeneity by changing both noisy signs; one bundle did not adjudicate which sign is stable.

### 3. Static source compatibility is enough to attribute the iteration-004/006 reversal to bundles
- **Category**: Baseline
- **Confidence**: Low
- **Currently assumed because**: Both studies bind the same helper, optimizer, data, nominal current-32 rule, environment, and selectors, with matching recorded hashes.
- **What changes if wrong**: New results cannot extend one exchangeable historical series. Runner ordering, RNG consumption, state capture, or another orchestration difference remains an alternative to aggregate bundle variability.
- **How to test**: Either perform a bounded matched-plan cross-harness trajectory check using parameter and observer hashes, or explicitly treat iteration 007 as standalone under the iteration-006 harness and cite the older reversal only as context.
- **Relevant evidence**: `source-compatibility.md` explicitly says no matched-seed cross-harness replay exists. Source identity rules out obvious configuration drift, not all execution non-equivalence.

## Moderate Assumptions (Medium confidence or Medium impact)

### 4. “Bundle variability” is a sufficiently specific explanation
- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: Each bundle jointly changes initialization, split, replacement labels, auxiliary draws, and minibatch order, matching the intended generator.
- **What changes if wrong**: The study establishes only sensitivity to the joint generator; it cannot identify a causal factor or interaction.
- **How to test**: Name the estimand joint-bundle sensitivity. Defer factor attribution to a crossed or one-factor-at-a-time common-plan study.
- **Relevant evidence**: The compatibility note already records that all sampled factors change together.

### 5. Own-selector test accuracy measures the ordering effect of interest
- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: Earliest strict maximum-validation accuracy is a defensible policy-level checkpoint rule and is where the baseline reversal appeared.
- **What changes if wrong**: Contrasts conflate delivery with different selected exposures and selector noise. They compare whole policies, not delivery order at a matched state or step.
- **How to test**: Keep own-selector accuracy primary, but require fixed step-100 and step-2000 contrasts and selection-step differences beside it. Forbid mechanism language when selectors and fixed steps disagree.
- **Relevant evidence**: Iteration 004 changed hard32-minus-AdamW from `+18.81 pp` at endpoint to `-2.38 pp` at accuracy selection; iteration 006 also found accuracy/CE exposure tradeoffs.

### 6. Reusing the official test set remains adequate for the intended estimand
- **Category**: Data/Resource
- **Confidence**: Medium
- **Currently assumed because**: Training and validation selection finish before test loading, and the plan labels the result adaptive exploratory reuse.
- **What changes if wrong**: Within-run separation does not remove project-level adaptivity: arms, benchmark, endpoints, and question were chosen after repeated official-test results. The fixed 10,000 examples also create shared test-set error across bundles.
- **How to test**: Define the estimand as performance on this fixed test set under newly sampled training bundles. Freeze outcome hierarchy and narrative rules. Reserve population or generalization claims for an untouched benchmark protocol.
- **Relevant evidence**: Both prior reports retain the repeated-test limitation; fresh training streams do not create fresh test examples.

### 7. Three correlated primary contrasts will not invite outcome-driven emphasis
- **Category**: Independence
- **Confidence**: Medium
- **Currently assumed because**: Every signed contrast and bundle value is mandatory, with no p-values or threshold decisions.
- **What changes if wrong**: Shared arms and selectors make the contrasts dependent, while multiple selectors and endpoints create narrative degrees of freedom. A favorable member can become the headline despite adverse companions.
- **How to test**: Freeze this hierarchy: baseline variability first, delivery ordering second, norm restoration third. Put one all-outcomes table before prose and explain every selector/endpoint disagreement. Do not treat contrasts as independent corroboration.
- **Relevant evidence**: Iterations 004 and 006 show that endpoint, accuracy selection, and CE selection can support different rankings.

### 8. Omitting clean and scalar arms preserves the intended conclusions
- **Category**: Scope
- **Confidence**: Medium
- **Currently assumed because**: Four existing clean bundles give concordant adverse lagging contrasts; scalar controls track AdamW-like memorization and fail to match actual displacement.
- **What changes if wrong**: The study cannot estimate a contemporaneous noise-by-policy interaction or use clean behavior as a drift sentinel. It also cannot revisit attenuation, though scalar arms still would not isolate direction from magnitude under AdamW.
- **How to test**: Keep both omissions now only if all claims are noisy-only and whole-policy. Do not call a result noise-specific. Reintroduce clean arms for an interaction question and scalar/reciprocal arms only after a same-state diagnostic specifies the mediator.
- **Relevant evidence**: Clean lagging was adverse in all four prior bundles; scalar controls neither reproduced endpoint preservation nor matched hard-filter update norms.

### 9. Saving full common-state anchors is not worth the implementation burden
- **Category**: Data/Resource
- **Confidence**: Medium
- **Currently assumed because**: State capture is outside the stability estimand, old checkpoints lack moments, and serialization creates a new correctness surface.
- **What changes if wrong**: A prospective chance to retain exact AdamW and observer state is lost, forcing new training or non-exact reconstruction before the already ranked mechanism study.
- **How to test**: Add a pre-pilot size/time and round-trip check for a minimal set: current32 in each primary bundle after steps 100, 999, and 1999, before the next update. Store model, AdamW, observer, RNG, plan/batch pointer, and source/environment bindings. Require save-on/save-off trajectory identity and exact reload.
- **Relevant evidence**: Iteration 006 saved models but not moments; iteration 005 lacks parameters and moments. The theory note makes historical moments necessary for exact same-state AdamW responses.

### 10. Current-policy anchors can later support broad mechanism claims
- **Category**: Scope
- **Confidence**: Medium
- **Currently assumed because**: Twelve current32 anchors would make exact raw/current/lagged/restored one-step branches cheap across phases and bundles.
- **What changes if wrong**: Results remain conditional on states created by current32; they cannot identify a trajectory mediator or establish behavior on lagged, restored, clean, or other-model states.
- **How to test**: Label the later estimand current-policy-state conditional response. If symmetry becomes essential, prospectively capture lagged-policy anchors too; never reconstruct absent state.
- **Relevant evidence**: `predictable-projection-theory.md` proves conditional fixed-state identities and explicitly provides no AdamW trajectory or generalization theorem.

## Background Assumptions (High confidence, Low impact)

### 11. The recipe fits the proposed resource envelope
- **Category**: Data/Resource
- **Confidence**: High
- **Currently assumed because**: Forty-two iteration-006 cells passed on the same RTX 3090, and the proposed arms are a subset of that implementation.
- **What changes if wrong**: A gate failure stops with partial evidence retained; it does not alter a scientific result because retry and seed replacement are forbidden.
- **How to test**: Use the required runtime-only pilot and include anchor serialization overhead if anchors are accepted.
- **Relevant evidence**: Iteration 006 completed 36 cells in 622.9 harness seconds and six in 108.2 seconds.

## Assumption Dependency Map

`Static compatibility (3)` supports interpreting `joint-bundle variability (4)` across old and new runs. If 3 fails, iteration 007 can remain standalone but cannot explain the historical reversal.

`Information value (1)` depends on honest interpretation of `audit bundle (2)`, `test reuse (6)`, and `contrast hierarchy (7)`. If any fails, more cells increase apparent evidence more than actual scope.

`Own-selector estimand (5)` and `omitted controls (8)` restrict the study to noisy whole-policy performance; they cannot support direction/magnitude or noise-interaction conclusions.

`Anchor value (9)` depends on exact capture and reload; `anchor scope (10)` limits what a successful replay means. Neither is required for the descriptive 20-cell result.

## Recommendations

### Resolve now, before implementation or launch review

1. Add the outcome-to-decision table. If plausible outcomes do not alter beliefs or next work, do not run.
2. Rename bundle 60011 a separate fresh-draw sensitivity check unless an independent implementation is used; retain unpooled reporting and its own test boundary.
3. Perform matched-plan cross-harness equivalence or declare the new study standalone; source compatibility alone cannot assign the old reversal to bundles.
4. Freeze a finite-test-set, adaptive, noisy-only, whole-policy estimand and outcome hierarchy. Put all bundle values and endpoint disagreements ahead of the headline.
5. Save the 12 minimal current32 anchors only if the pilot proves exact round-trip and instrumentation neutrality within existing caps. Failure removes anchors, not the study.

### Defer deliberately

- Defer clean arms unless the question becomes a prospective noise interaction or clean drift check.
- Defer scalar and reciprocal training arms until same-state displacement results justify them.
- Defer factor attribution to a design that varies initialization, split, corruption, and batch order separately.
- Defer population, sign-frequency, equivalence, transfer, or production-default claims to a fresh benchmark and untouched test protocol.

### Acceptable risks

This review does not approve implementation, a pilot, or launch.
