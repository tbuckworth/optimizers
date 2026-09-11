# Mentor Review

Returned independently by external_results_audit; saved by the parent.

## Overall Assessment

This is a useful, construct-valid follow-up: it tests learning outcomes that iteration 005’s retention measurements cannot determine. Its strongest features are paired policies, a shared warmup, clean/noisy conditions and both checkpoint selectors; its main risk is interpreting several adaptive whole-policy comparisons as a decomposition of self-inclusion into direction, norm and semantic usefulness.

## What a Senior Researcher Would Do Differently

I would make the estimands explicit before implementation. The first comparison estimates the effect of changing the complete delivery policy from current to previous basis, under a fixed validation-accuracy selection rule. The second compares two policies using the same **own-state current-candidate norm prescription**, not identical realized gradient norms. Their different subsequent trajectories are part of the policy effect, not a methodological defect; they prevent attributing that effect uniquely to direction.

Describe `lagged32_current_norm` as **lagged direction with current-observer magnitude**. It retains current-gradient influence through the norm and therefore is not a fully non-self-inclusive control. The existing caveats largely establish this, but the distinction should appear beside the primary hypotheses, not only in limitations.

Add two inexpensive secondary measurements before freezing:

- Final clean-label and fixed-noisy-label training accuracy, scored from the same logits on the existing training examples. These distinguish suppression of corruption fitting from loss of learning more directly than test accuracy alone. They must never select checkpoints.
- The same-state cosine between current and lagged candidate gradients, with explicit zero-vector nulls. Candidate norms and raw/applied cosines do not identify their mutual angle. This directly measures the directional intervention without storing full gradients.

Keep the four primary condition/contrast groups separate and report all paired magnitudes. Consistent signs across three seeds support a restricted descriptive finding, not superiority beyond this recipe. Preserve the endpoint and CE-selected results even if they contradict the primary comparison.

## What Hasn't Been Examined Yet

Iteration 005 measured its large before/after retention differences at four steps that were all scheduled repair multiples. Iteration 006 records every step, which is valuable: the original observation need not represent ordinary steps equally well. Prespecify a small descriptive breakdown of candidate retention and angle on repair steps versus other active steps. This requires no new run or probe and must not become an outcome-selected primary window.

The existing scalar controls do not establish what happens under actual AdamW update-norm matching. Iteration 004 already showed that substantial gradient attenuation can coexist with update norms close to AdamW’s. Time-varying scaling, unscaled warmup, nonzero epsilon and moment history prevent treating constant-scaling invariance as an explanation by itself. The proposed displacement measurements are appropriate; a different optimizer or update-normalized policy can remain future work.

There is also no reciprocal filtered arm delivering the **current direction with the lagged candidate’s norm**. That omission does not invalidate the stated asymmetric questions. It does mean this is not a complete direction-by-norm factorial decomposition; neither the scalar controls nor norm-restored lagging supplies that missing cell. Do not add it unless the intended claim expands.

Finally, fresh seed bundles do not reset the benchmark’s evidential history. The same dataset and test set have informed several iterations. The all-training-before-test gate prevents within-study selection leakage, but not the broader adaptivity of the research programme. The design already acknowledges this appropriately.

## Simpler Alternatives

The minimal faithful learning study would contain `current32`, `lagged32` and `lagged32_current_norm`: 18 runs across the proposed seeds and label conditions. Adding AdamW gives a useful 24-run study with an absolute reference. The two scalar policies expand this to 36 runs and provide additional control-policy evidence, but neither is required to estimate the two primary contrasts.

Keeping all six is defensible if the bounded pilot supports the resource estimate. If simplification becomes necessary, decide before freezing and remove secondary arms rather than seeds, the clean condition or checkpoint selectors. Do not make that decision from pilot learning outcomes.

Avoid expanding into independent gradient probes or full covariance references. They would answer different questions and are unnecessary here. Likewise, retain the complete all-pairs numerical appendix if convenient, but organize the report around the primary contrasts and a short predefined control hierarchy; fifteen arm pairs per condition should not become fifteen competing stories.

## Construct Validity / Information Value

The headline learning outcome is **not determined by construction**. Lagging may reject batch-specific variation, but it may also discard useful new directions and alter optimization through reduced gradient magnitude and different moment histories. Iteration 005 establishes none of those effects on predictive performance.

The measured construct—clean predictive performance after training with current or lagged delivery—is a faithful instance of the stated learning question. It is not a cheap proxy for that outcome. The norm and collinearity equalities, however, are implementation invariants fixed by the policy definitions; passing them is not scientific evidence that norm restoration improves learning or identifies a mechanism.

This design returns a meaningful verdict on the restricted policy question. It does not return a verdict on whether self-inclusion admits specifically corrupt information, whether lagging is a semantic denoiser, or whether lagging generally improves AdamW. Those stronger claims would require different evidence, not merely favorable results here.

## Key Recommendations

1. Freeze the policy-level estimands and explicitly label norm-restored lagging as lagged direction with current-observer magnitude. Preserve all four primary groups, both selectors and every seed.
2. Add final clean/noisy training accuracy and same-state candidate-angle logging as cheap, non-selecting secondary measurements; prespecify the repair-phase diagnostic breakdown.
3. Keep execution validity separate from scientific success. A zero-lagged/positive-current gate failure is a preserved implementation-domain failure, not an unfavorable learning result; a faithfully executed null comparison remains successful research. Clarify that distinction in the fail-fast table.

## Verdict

MINOR_REVISIONS

The motivating question is sound, its answer is genuinely uncertain, and no essential control is missing for the stated policy comparisons. The recommended changes improve interpretation and information value without requiring additional training arms, semantic probes or a broader sweep.
