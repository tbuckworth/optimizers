# Prospective design: four-view observation, single-view delivery

10 September 2026. **Proposed recipe, not an acquired result or launch authorization.** This note uses only the audited scalar summaries and reports linked below. No model, gradient archive, training data or checkpoint was opened, and no experiment was run. The research workflow informs the separation of evidence, assumptions, controls and failure conditions; it adds no approval gate or repeated mean-restoration study.

## Why this route remains worth testing

The [fixed-state report](results.md) and [independent scalar audit](audit.json) support a conditional directional effect, not a size-only account. From unaugmented warmups, translated training gradients help translated heldout CE more in the native direction at either matched data-step norm in all three seeds. The opposite sign holds on original-image readouts. Augmented warmups have adverse mean directional contrasts in all four input/readout cells. These are six parents but only three seed units, measured at one early state; they do not identify the full training mechanism.

The recorded action retains 76.93–80.59% of between-image translated-view-mean variation versus 34.43–54.21% of within-image view variation across the six parents. It already retains 79.05–81.35% of translated mean-gradient energy. Thus “the observer discards the entire useful mean” is not the motivating claim. A plausible remaining question is whether changing the observations that construct the moving span improves useful learning. Preferential geometric retention alone neither predicts that improvement nor establishes semantic selectivity.

## One precisely specified recipe

For each optimizer update, sample the same batch of `B=64` example occurrences as the single-view control. At the current, unchanged parameters, independently translate each occurrence four times using the existing zero-filled integer `dx,dy ∈ {-2,…,2}` distribution. Assigned labels stay attached to source examples. Let

```text
g_t^(r) = (1/B) Σ_b ∇θ CE(fθ(T_tbr x_tb), y_tb),   r=1,…,4
bar_g_t = (g_t^(1) + g_t^(2) + g_t^(3) + g_t^(4))/4.
```

All four gradients use the same parameters; there is no optimizer update between views. Average gradients/loss gradients, not logits. Retain `g^(1)` separately.

The candidate, **O4-D1**, advances the unchanged stable rank-32 centered observer **once** using `bar_g_t`, then delivers `h_t = P_t g_t^(1)` to ordinary ambient-coordinate AdamW. Here `P_t` is the post-observation native action, not an additional mean-restoring operator. The first 100 updates remain unfiltered `g^(1)` delivery while the observer already receives `bar_g`; filtering begins on update 101. Keep Adam learning rate, betas, epsilon, weight decay, observer decay/rank and all numerical conventions unchanged. No gradient renormalization, observer-input gain multiplier, moment projection/reset, clipping change or covariance retuning is part of this recipe.

This requires separating observation from application: an implementation must not call an ingesting filter a second time on `g^(1)`. Adam and the observer each advance once per optimizer update. Four sequential backward passes need no per-example gradient tensor and can release each graph immediately; two running flat buffers suffice beyond the usual states.

## Steelman and its assumptions

The [unified hypotheses, §3.5](../../research/spectral_optimizer_unified_hypotheses_2026-09-10.md) give the fixed-model identity, with independent example occurrences and conditionally independent transformations:

```text
a_i = E_T[g(i,T)],  W_i = Cov_T[g(i,T)]
Cov(bar_g_m) = (1/B) { Cov_i(a_i) + E_i[W_i]/m }.
```

At `m=4`, averaging reduces conditional view variation by four in this population calculation while preserving the expected gradient and between-example term. A useful span could become easier to track if transformation-stable variation identifies useful directions and view-sensitive variation competes with them. The independently checked finite-grid total=between+within identity is compatible with this rationale, but its four-view between term still contains sampling variation; it is not an estimate of a pure population signal.

The actual observer is centered, exponentially weighted, truncated and moving with the learned model. Its update need not equal this population covariance. O4-D1 tests the whole observation policy, not a causal claim that one covariance component mediates endpoint performance.

## Different from the completed mean-complement route

[Unified §2.4](../../research/spectral_optimizer_unified_hypotheses_2026-09-10.md) describes `Pg + (I−P)mu`: changing **delivery** to restore a temporal EMA outside the span while retaining the original observations. O4-D1 instead changes **observations** through simultaneous view averaging and keeps hard current-gradient delivery. It adds no outside-mean term and no temporal response kernel beyond the incumbent observer/Adam histories. The two meanings of “mean” are not interchangeable.

[I13](../2026-09-06-spectral-optimizer-investigation/continuation/iteration-013/results.md) established genuine soft/redraw adaptation recovery but near-raw fixed-realization fitting after unrestricted mean restoration. [I15](../2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/results.md) also established the constructive counterpoint: mean delivery with projected SGDm history allowed fixed-label progress and avoided the unrestricted mean arm's late collapse, although clean underfitting remained. Those are useful completed interventions, not failed ideas to erase or experiments to repeat here. They do not tell us that O4-D1 is better: they used different histories, source states and, for I15, a different base optimizer.

## Smallest useful learning comparison

The minimal clean-label panel is four arms × three fresh paired seed bundles: **12 trajectories**, all classes eligible from update 1, balanced 5,000 training/5,000 heldout examples, the incumbent MLP and 4,000-update settings. All arms use translation; this is an observer-recipe comparison, not a repeat of augmentation-on/off.

| Arm | Observer input, once per update | Adam input after warmup | Batch-gradient evaluations/update |
|---|---|---|---:|
| Raw-D1 | None | `g^(1)` | 1 |
| O1-D1, incumbent | `g^(1)` | `P_t^(1) g^(1)` | 1 |
| O4-D1, candidate | `bar_g` | `P_t^(4) g^(1)` | 4 |
| Raw-D4, averaging control | None | `bar_g` | 4 |

Pair initialization, example IDs, occurrences and the designated first transform; extra views use separate deterministic streams. Within each bundle, Raw-D1/O1-D1/O4-D1 must have identical model and Adam states through warmup because they deliver the same `g^(1)`. The two observers intentionally differ. Raw-D4 averages from update 1, so its warmup state need not match. After filtering begins, comparisons are full-policy trajectory effects, not fixed-model observation interventions.

O4-D1 versus O1-D1 answers the main question at fixed delivery convention. Raw-D4 versus Raw-D1 measures the simpler benefit of averaging the optimizer input. O4-D1 versus Raw-D4 asks whether spending the four views on the observer is competitive with spending them directly on Adam. This four-arm panel is not a complete observer-by-delivery factorial: it cannot estimate whether averaging delivery also helps a filtered optimizer. Do not claim that interaction or silently add more variants.

The clean panel alone cannot establish retained wrong-label protection. If the intended claim is the joint learning/protection tradeoff, the smallest corresponding panel is the same four arms under both clean and fixed 80%-guaranteed-wrong training targets: **24 trajectories total**, three paired seed bundles, clean heldout labels. Exactly 400 examples per true class receive fixed wrong labels, shared across views. This is a scope extension specified here, not a selected or automatically queued run. Earlier clean/noisy trajectories are context, not paired controls for new seeds.

## Readouts, cost and informative failure

Use fixed endpoint original-image heldout CE and accuracy together, absolute progress from warmup, and the entire incumbent 22-point evaluation schedule. A separately fixed translated heldout panel is a secondary transfer readout, not a substitute chosen after seeing signs. For wrong-label training, additionally retain clean-label training metrics and separate true-label/assigned-wrong-label metrics on the corrupted subset. Less wrong-label fitting without useful heldout progress is not a rescue. Report all seed values; views and checkpoints are not extra replicates. No best-checkpoint, learning-rate or view-count search belongs in this first comparison.

Count four batch-gradient evaluations for each O4-D1/Raw-D4 update, versus one for either single-view arm. The proposed clean panel costs 120,000 batch-gradient evaluations; the two-target panel costs 240,000, before evaluations. Neither count is a wall-time estimate. Raw-D4 is matched on view/backward count, not exactly elapsed time: the spectral observer adds work. Report optimizer updates, underlying-example occurrences, transformed-example evaluations and measured time separately. Do not waste extra backward passes in the single-view controls merely to burn equal compute. Any later acquisition would need an explicit byte inventory and the existing bounded local resource profile; this note reserves no GPU and spends nothing.

Important failure modes and interpretation boundaries:

- **Transformation-stable error:** fixed wrong labels survive every view. Averaging can strengthen their between-example gradient structure rather than remove it. View consistency is not correctness.
- **Useful view dependence:** gradients needed to learn translation robustness may be precisely what averaging suppresses. The current geometry already prefers between-image variation, so making that preference stronger may add no value or worsen adaptation.
- **Scale and self-inclusion:** the candidate observes a lower-variance/lower-norm statistic without rescaling it. Its delivery view contributes only one quarter of the current observation, changing current-gradient self-inclusion. Improvement could reflect these finite-observer effects; the panel does not isolate them from better orientation.
- **Adam and objective mismatch:** hard projection of `g^(1)` need not help either heldout objective after Adam. Original and translated readouts already have different local signs. Matching observation counts does not match future Adam states or realized update norms.
- **Uncompetitive use of compute:** beating O1-D1 while losing to Raw-D4 is evidence for a recipe improvement, but not a compelling advantage over ordinary multi-view optimization. A smaller filter deficit caused by raw deterioration is not absolute progress. No reliable clean improvement, or recovery coupled to renewed wrong-label harm, leaves the proposed stronger recipe unsupported in that scope.