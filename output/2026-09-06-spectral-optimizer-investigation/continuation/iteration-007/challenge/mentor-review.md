# Mentor Review

## Overall Assessment

The proposal is careful, auditable, and honest about adaptive MNIST reuse, whole-policy interpretation, and adverse outcomes. Its main weakness is scientific prioritization: another 20-cell bundle extension would describe instability more precisely, but it is unlikely to change the next decision, whereas the unresolved mechanism question can be attacked more directly and cheaply with prospectively captured common-state one-step counterfactuals.

## What a Senior Researcher Would Do Differently

I would require an explicit decision table before spending more training compute: for each possible pattern of the three mandatory contrasts—current minus AdamW, lagged minus current, and restored minus current—state what subsequent experiment would change. The parent correctly rejects the proposed sign-class and “near-zero” thresholds. Once those thresholds are removed, the present contract has no operational criterion beyond “report the numbers and stop.” That is scientifically legitimate description, but weak grounds for prioritizing 16 primary plus four audit cells.

The existing evidence already supports the decision relevant to lagging: it has no dependable benefit in this recipe. Clean results are uniformly adverse across three primary bundles plus the separate audit bundle; noisy results reverse across bundles; and norm restoration does not isolate magnitude because AdamW displacement remains unmatched. Four more primary bundles cannot turn this into a general claim, and the separate audit bundle cannot be pooled. Favorable new values would motivate a new confirmation; adverse or heterogeneous values would stop the line. In every case, the mechanism question remains next.

I would therefore sequence a sparse common-state mechanism study first. Generate a small number of prospectively specified trajectories, save complete anchors at scientifically motivated stages, and branch one AdamW step from each identical state using raw, current, lagged, restored-lagged, and reciprocal-norm inputs. This directly measures whether direction and pre-Adam magnitude produce reproducible differences in actual AdamW displacement and immediate loss response. It should precede another whole-policy seed extension, not follow it.

## What Hasn't Been Examined Yet

The bundled randomization changes initialization, split, corruption realization, and minibatch order simultaneously. Calling the observed spread “bundle variability” is correct, but it cannot reveal which source drives the reversal or whether interactions dominate. More bundles estimate the combined generator’s variability; they do not explain it.

The primary accuracy contrast also combines optimizer behavior with arm-specific checkpoint selection and exposure. This is a faithful whole-policy estimand if the question is “which training-and-selection policy performs better?” It is not a clean measurement of the filter’s mechanism. The sharp variation in selected steps, including zero post-warmup exposure in earlier cells, makes that distinction load-bearing.

The repeated official MNIST test set is acknowledged appropriately, but the proposed audit bundle is a workflow replication, not an untouched scientific confirmation. Its separation from the primary summary is correct. It should not acquire extra epistemic weight merely because it is called an audit.

Most importantly, prior runs did not retain complete optimizer/observer/RNG state. Saved model parameters and pre-Adam gradient bundles are insufficient for exact counterfactual AdamW steps. If another training run occurs without full anchors, the project risks paying again later merely to recover the state needed for the already-identified mechanism experiment.

## Simpler Alternatives

Use one or a few fixed noisy trajectories rather than 20 new full-policy cells. At several predeclared anchors—at minimum immediately after warmup and at early, middle, and late post-warmup stages—save:

- model parameters;
- AdamW step count, first moments, and second moments;
- complete observer state, including the state needed to reproduce previous and self-inclusive current bases;
- exact next minibatch indices and labels;
- RNG and data-plan identities;
- fixed clean and corrupted auxiliary evaluation probes.

From exact clones, compute the same raw minibatch gradient, construct each candidate delivered gradient, take exactly one AdamW step, and measure full and decay-subtracted displacement, pairwise displacement angles, directional derivatives, and immediate loss changes on the training batch and fixed clean/noisy probes. Reciprocal norm is inexpensive in a one-step branch even though another reciprocal whole-training arm is not justified.

This design cannot establish a long-run mediator, but it answers the immediate question the current evidence leaves open: whether delivery direction, pre-Adam magnitude, or AdamW’s stateful transformation creates a reproducible same-state separation. If it changes idiosyncratically across anchors, that itself is a useful state-dependence result and argues against more elaborate direction-based training studies.

Full common-state anchor capture should be mandatory before any new training, not deferred. Sparse capture in this small network is a modest storage and audit burden compared with rerunning trajectories. The schema and replay checks must be frozen prospectively; incomplete reconstructed states should not be accepted as substitutes.

## Construct Validity / Information Value

The proposed bundle study is not predetermined by construction. The signed whole-policy contrasts are genuinely unknown, the policies are real implementations rather than surface proxies, adverse cells are retained, and the parent has removed arbitrary sign-class and zero-effect claims. There is therefore no strawman-construct failure.

Its information value is nevertheless low relative to the motivating objective. It measures conditional performance variability under one heavily reused benchmark and a bundled source of randomness, not why the spectral optimizer suppresses memorization, why current versus AdamW reverses, or why lagged delivery changes actual AdamW displacement. Since every plausible outcome still leaves same-state mechanism work as the next scientific step, the proposed ordering is inefficient.

The current primary estimand answers a legitimate policy question, but not the mechanism question. That distinction should be explicit rather than allowing more bundle measurements to stand in for understanding the optimizer.

## Key Recommendations

1. Defer the 16-plus-four bundle extension until a predeclared decision table shows how its possible descriptive outcomes would change subsequent work.
2. Require sparse, complete common-state anchor capture before any new training and run the cheaper one-step AdamW counterfactual mechanism study first.
3. Keep any later bundle study strictly descriptive: all three contrasts, all signed cells, arm-specific exposure, unpooled audit results, and no sign-class, zero-effect, equivalence, or independent-confirmation claims.

## Verdict

MAJOR_REVISIONS

The proposed experiment is construct-valid but poorly ordered for the stated objective. Its controls and reporting rules are sound, yet another bundle-variability study is unlikely to change the next scientific decision. Common-state capture and a bounded one-step mechanism design should be made the prerequisite. This review grants no launch approval.
