# Revised Assumption Analysis

## Summary

The replacement design is materially more informative than another whole-policy seed extension: exact same-state branches can describe immediate AdamW responses without pretending to recover missing historical moments. Its main remaining risk is interpretive, not computational. The proposed 2x2 grid identifies finite responses to four constructed gradients at current32-generated states; it does not separately identify general direction and magnitude mechanisms, and several outcome rules still need operational definitions before implementation.

## Critical assumptions

### 1. The 2x2 swap identifies “direction” and “norm” as general mechanisms
- **Category**: Theoretical
- **Confidence**: Low
- **Currently assumed because**: Current, reciprocal, restored, and lagged fill the algebraic grid `{current, previous direction} × {current, previous input norm}` while holding parameters, moments, and observer state fixed.
- **What changes if wrong**: The grid still gives exact within-anchor finite contrasts, but only between two gradient-derived directions and two gradient-derived norms. It does not identify direction or magnitude as a trajectory mediator, and AdamW can transform an input-norm swap nonlinearly through historical moments, epsilon, and coordinatewise denominators.
- **How to test**: Rename the estimand “conditional 2x2 delivered-gradient response.” Define every cell as unit direction times target input norm, and state that the factors are pre-Adam interventions. Never describe the resulting displacement or loss contrasts as separate causal mechanisms beyond the sampled grid.
- **Relevant evidence**: Iteration 006 showed that matching delivered-gradient norms did not match decay-subtracted AdamW displacement; the predictable-projection note supplies no AdamW descent or mediation theorem.

### 2. Each anchor supplies enough factor leverage to interpret the 2x2 contrasts
- **Category**: Methodological
- **Confidence**: Low
- **Currently assumed because**: Current and previous projections usually differ in both angle and norm in iteration 006 averages.
- **What changes if wrong**: If directions are nearly collinear or `nc` and `nl` are nearly equal, the corresponding direction or norm contrast has negligible treatment separation. A small loss difference would then be uninformative, not evidence against that factor. Means across anchors would mix different treatment doses.
- **How to test**: Predeclare a leverage table per anchor: `nc`, `nl`, their ratio/difference, current/previous unit-direction cosine, and delivered-vector distances. Mark factor conclusions unavailable when a level is undefined or indistinguishable at the frozen arithmetic tolerance. Report response contrasts jointly with their realized treatment separations; do not average them as equal-dose effects.
- **Relevant evidence**: Iteration 006's retention and projection cosines varied strongly by condition and state; averages do not guarantee leverage at each proposed anchor.

### 3. One bundle-specific 256-example auxiliary probe supports the primary comparison across bundles
- **Category**: Data/Resource
- **Confidence**: Low
- **Currently assumed because**: A fixed held-out clean probe avoids checkpoint selection and is cheap enough for CPU-float64 gradients at every anchor.
- **What changes if wrong**: Across bundles, both source state and primary evaluation examples change. Bundle variation in delta CE therefore combines state, training plan, and probe sampling. Reusing the same probe across four times induces dependence and may make apparent temporal structure probe-specific.
- **How to test**: Either evaluate the full 5,000-example auxiliary pool if the pre-pilot resource check permits, or add a second prospectively fixed disjoint auxiliary probe and report both without selecting between them. If neither is affordable, define the primary estimand as response on the exact fixed 256 examples and call bundle summaries joint state-and-probe sensitivity.
- **Relevant evidence**: The revised design correctly avoids official-test reuse, but its stream-6 probe is redrawn with every bundle and is not a common evaluation sample.

## Moderate assumptions

### 4. Current32-generated states answer the motivating optimizer question
- **Category**: Scope
- **Confidence**: Medium
- **Currently assumed because**: Current32 is the policy involved in both historical reversals, and its exact full state can now be saved prospectively.
- **What changes if wrong**: Results can show how alternative next gradients behave on current32 states only. They cannot explain how AdamW-, lagged-, or restored-generated states respond, nor whether repeated counterfactual steps would change learning.
- **How to test**: Keep the source policy singular, but require every title, table, and conclusion to say “current-policy-state conditional.” Treat update 101 separately as the shared-warmup state; later anchors are endogenous current32 states. A raw branch is raw delivery with filtered-history moments, not a raw-AdamW history.
- **Relevant evidence**: The design already notes this limit; making it part of the reporting schema prevents later broadening.

### 5. CPU-float64 delta CE is a sufficiently defined proxy for immediate native-training response
- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: Float64 evaluation reduces subtraction cancellation and defines a reproducible high-precision measurement function.
- **What changes if wrong**: Casting parameters and inputs and changing device/precision can change logits, ReLU boundary behavior, CE, and gradients. The primary outcome is then the response of a CPU-float64 surrogate to a native float32 AdamW displacement, not the actual GPU training loss change.
- **How to test**: Freeze before implementation the exact module reconstruction, parameter order, input conversion, reduction, CE definition, model mode, and gradient flattening. Also report native-float32 before/after loss as a secondary concordance diagnostic. Disagreement must be retained and must block claims about native loss, not invalidate the explicitly defined float64 functional.
- **Relevant evidence**: `best-practices-check.md` correctly limits replay identity across devices/releases; it does not establish CPU-float64 measurement equivalence.

### 6. Undefined and near-degenerate branches are handled without changing the estimand
- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: Exact zero targets become zero gradients, positive-target/zero-direction branches are retained as undefined, and failed invariants stop review.
- **What changes if wrong**: `nc=nl=0`, one-sided zero, or very small denominators can yield ambiguous 0/0 rules, unstable scaling, or incomplete 2x2 grids. Computing remaining contrasts or available-case means can silently change the anchor set by outcome.
- **How to test**: Freeze a complete truth table for `(nc zero/nonzero, nl zero/nonzero)`, including both-zero behavior and an arithmetic definition of zero versus near-zero. Construct normalized vectors in float64 rather than multiplying by an unstable ratio. When any cell is undefined, mark every dependent direction, norm, interaction, and aggregate contrast undefined with a shared reason and fixed complete-case mask.
- **Relevant evidence**: The current text specifies positive-target/zero-direction handling but not the both-zero case, near-zero rule, or downstream contrast masks.

### 7. The outcome-to-action table is operational enough to prevent post-hoc interpretation
- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: It lists interaction, persistent direction contrasts, heterogeneity, small separation, and audit failure, and says mixed structures may coexist.
- **What changes if wrong**: “Persist,” “little separation,” “reported numerical resolution,” and “if needed” remain undefined. Overlapping rows allow the same result to justify several next actions. In particular, numerical non-resolution is not evidence that a practically large effect is absent.
- **How to test**: Replace qualitative triggers with a precedence rule: validity and leverage first; then report the complete 2x2 interactions; then classify only exact observed sign/order concordance across the 12 named anchors, without population language. Define numerical resolution solely from prospectively measured replay/auditor error, or remove the “little separation” action. State which specific result would stop reciprocal training, raw-state comparison, or further local studies.
- **Relevant evidence**: The prior parent decision rejected arbitrary stability/equivalence thresholds; the revised table must not recreate them implicitly.

### 8. The fresh-draw bundle and independent audit have distinct evidential roles
- **Category**: Independence
- **Confidence**: Medium
- **Currently assumed because**: Bundle 71901 is labeled sensitivity and unpooled, while an auditor independently reconstructs arithmetic from raw artifacts.
- **What changes if wrong**: The sensitivity run may be narrated as replication, or the numerical audit as confirmation of the scientific effect. Neither is justified: one extra joint state/probe draw gives no stable sign estimate, and arithmetic agreement only validates implementation/reporting.
- **How to test**: Use fixed labels everywhere: “three-bundle descriptive primary,” “one-bundle fresh-draw sensitivity,” and “independent numerical audit.” The auditor must cover both executions, preserve undefined masks, and make no effect-level acceptance decision.
- **Relevant evidence**: This separation is already stated in the revised design and should become a schema invariant rather than prose only.

## Necessary pre-implementation corrections

1. Recast the headline estimand as finite, within-anchor 2x2 delivered-gradient responses on current32 states; remove any wording that promises general separation of direction and magnitude.
2. Freeze per-anchor factor-leverage fields and complete-case rules. Every aggregate must carry the same defined-anchor mask and realized angle/norm separation.
3. Resolve probe dependence: prefer the full auxiliary pool if measured cost fits; otherwise add a second fixed probe or explicitly make the exact 256-example probe part of the estimand. Do not interpret cross-bundle means as state-only variation.
4. Specify the CPU-float64 functional completely and add native-float32 loss as a secondary concordance check. Freeze independent cross-device tolerances before producer code exists.
5. Add a four-case zero/near-zero truth table and propagate undefined status to all dependent contrasts and summaries. No available-case substitution is allowed.
6. Rewrite outcome-to-action rules with validity/leverage precedence and concrete stop decisions. “Little separation” must be tied only to numerical error or removed; it cannot stand in for an equivalence margin.
7. Preserve the audit/sensitivity distinction in artifact schemas and final headings. Neither supplies population confirmation.

## Acceptable design choices after correction

- Four fixed anchor times are adequate for a temporal profile, provided times are not pooled as replicates and update 101 is distinguished from endogenous later states.
- A single current32 source policy is efficient and honest for the limited estimand; adding source policies now would change the question and multiply training burden.
- The five branches are sufficient for the conditional grid plus raw reference; no clean source trajectory, official test set, scalar training arm, or long-run branch is needed.

This is an independent design review only. It does not approve implementation, a pilot, or launch.
