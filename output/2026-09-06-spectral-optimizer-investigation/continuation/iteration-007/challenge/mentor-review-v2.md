# Mentor Review

## Overall Assessment

This is a well-focused replacement for the bundle-extension study. Its strongest feature is the complete direction-by-norm factorial comparison under fixed AdamW and observer history; its main weakness is that the primary loss outcome uses a single 256-example auxiliary probe even though one-step differences may be very small and probe-specific.

## What a Senior Researcher Would Do Differently

Use the full 5,000-example auxiliary pool, or several prospectively fixed disjoint probes, for the primary clean-loss measurement. Float64 reduces numerical cancellation but does not reduce finite-sample idiosyncrasy. Because every branch is evaluated on the same examples, full-pool evaluation should remain inexpensive relative to source training and would make “little separation” substantially more interpretable.

Add a zero-input AdamW reference at every anchor. With nonzero historical first and second moments, assigning a zero gradient can still produce a data-dependent-looking displacement from moment carryover, in addition to weight decay. The proposed `delta_data` removes only nominal decoupled decay. Pairwise branch contrasts remain valid without this reference, but a zero-input branch would show how much of each absolute displacement and loss change is inherited optimizer motion rather than a response to the new gradient. It can be an explanatory reference rather than a sixth treatment arm.

Freeze tolerances and the exact definition of “reported numerical resolution” before implementation. The outcome-to-action rule appropriately avoids an equivalence claim, but resolution should combine serialization/replay error, float64 evaluation error, and observed probe sampling variation rather than being inferred after seeing small effects.

## What Hasn't Been Examined Yet

All anchors come from current32 trajectories. This is not a defect in the stated estimand: the document repeatedly and correctly limits conclusions to current-policy states, and the current branch provides a strong source-policy replay control. It does mean the study cannot explain whether lagged or restored policies enter regions where the same local branch comparison changes. A later whole-policy mechanistic claim would require anchors from lagged/restored histories or a source-policy factorial design.

The raw branch is also raw input under filtered-history AdamW moments, not an AdamW-training baseline. This is stated correctly and should remain prominent in results.

The four anchor times cover useful phases, but they are four calendar points rather than states selected by observer events. That is preferable to outcome-based selection, although it leaves repair-trigger-local behavior untested. Do not reinterpret time differences as repair effects.

## Simpler Alternatives

The proposed study is already close to the simplest faithful design. The four projected branches form the complete two-direction by two-input-norm table:

- current direction/current norm;
- previous direction/previous norm;
- previous direction/current norm;
- current direction/previous norm.

This is more informative than another reciprocal whole-training arm because it fixes parameters, moments, observer state, gradient, and probe. Raw provides a useful external reference. Adding only a zero-input diagnostic completes the immediate optimizer-response picture without creating another trajectory.

## Construct Validity / Information Value

The result is not known merely from AdamW arithmetic. Given an anchor, branch displacements are deterministic computations, but their ordering is not statable beforehand because AdamW transforms each coordinate through historical moments and the network’s finite loss response depends on parameter-space geometry and the chosen probes. Measuring first-order predictions alongside finite-loss residuals usefully distinguishes local alignment from nonlinear effects.

The construct is faithful to the limited question: it measures same-state conditional responses, not long-run training performance or semantic denoising. The source-current replay gate directly validates the most important conditioning assumption. Freezing `c` and `l` before branching, ingesting the raw gradient exactly once, and restoring identical optimizer state prevent the endogenous-history confound that invalidated causal interpretation of iteration 006.

The principal information risk is sensitivity of tiny one-step CE changes to a single 256-example probe. Strengthening the primary probe and adding a zero-input reference would make null-looking or moment-dominated outcomes substantially more useful.

## Key Recommendations

1. Evaluate primary auxiliary-clean loss on the full reserved auxiliary pool, or on multiple fixed disjoint probes.
2. Add a zero-gradient AdamW reference to quantify moment carryover plus decay.
3. Keep every conclusion explicitly conditional on current32-generated states; require additional source policies only before making a whole-policy mechanism claim.

## Verdict

MINOR_REVISIONS

The design is construct-valid and materially more informative than another bundle-variability study. The requested changes improve resolution and interpretation without altering its central estimand. This review provides no launch approval.
