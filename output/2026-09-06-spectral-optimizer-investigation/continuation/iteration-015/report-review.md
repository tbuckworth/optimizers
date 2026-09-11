# Independent review of the I15 report

**Reviewer:** Codex

**Date:** 7 September 2026

**Verdict:** Pass; no remaining material numerical or interpretive error found.

## Sources and scope

I reviewed [`results.md`](results.md), SHA-256
`8e244b847ad83b8d7f7a831cf16dfcdb5cdfe9ad367c3ba08e488a4e2a297d18`, and
[`mathematical-interpretation.md`](mathematical-interpretation.md), SHA-256
`9bb6c3497b4d4e0a0ad9d7da133e09cd15f4d5d08e9a024d0e33bd47d2dac317`,
against the pinned [`analysis-001/summary.json`](analysis-001/summary.json),
the passing acquisition audit, and the independent scalar
[`report-audit.json`](analysis-001/report-audit.json), SHA-256
`37018d3694f61476e9a7df094d1473db48623b4a878e5b5d4faadae0685d0933`.
The latter independently matched 19,558 numerical and discrete values with
maximum absolute numerical difference zero. This review did not rerun training,
load tensors, perform forward passes, or replay either audit.

## Substantive checks

The endpoint table retains all 12 registered effects with all three seed values
and the equal-seed mean. Its signs and magnitudes match the summary. In
particular, fixed-target $H$ is adverse for CE in every seed and mixed for
accuracy; clean-target $H$ is favorable but describes less regression rather
than positive learning. All 12 seed-by-target-by-metric values contributing to
$M$ are favorable. $S$ is uniformly adverse across clean-target endpoint cells
and uniformly favorable across fixed-target endpoint cells. The report therefore
preserves the target reversal rather than pooling it into a single mechanism
score.

The validation-selected table retains all 24 registered family effects in eight
compact rows, again with every seed and mean. The accompanying interpretation is
correct: selected $M$ remains favorable in every displayed seed, whereas fixed
$S$ is favorable for accuracy and adverse for CE under both selectors. The
selection language is appropriately calibrated. Selection on the frozen
validation split followed by evaluation on the disjoint auxiliary split is a
valid checkpoint-selection comparison; the report limits it as coarse offline
selection without incorrectly alleging auxiliary-selection leakage. It also
keeps endpoint and selected findings separate.

The complete-curve, absolute-progress, realization-fitting, confidence, and
geometry headlines agree with the pinned summary. The distinction between
relative preservation and positive adaptation is maintained. Claims about the
EMA mean are limited to useful information in this three-seed MNIST/SGDm recipe;
neither document equates that information with a semantically clean direction.
The large fixed-target interaction is described as a policy interaction, not a
causal mediation fraction or a universal momentum mechanism.

During review, one terminology ambiguity was corrected. The saved
`removed_old_buffer` field is the geometric old-buffer action complement
$b-A_tb$ for every policy, but it is actually removed only by projected-history
policies. The final report now calls it the action-complement diagnostic and
states explicitly that native-history arms retained it. This makes the amplitude
and signed-dot discussion faithful to the implemented intervention.

The numerical-fidelity discussion is also properly bounded. The reported Gram,
idempotence, recurrence, data-step, and decay residuals match the audited
summary. They support local float32 conformance to the specified recurrence,
but the final text explicitly notes that small local residuals do not bound the
sensitivity or robustness of an accumulated nonlinear trajectory. The
mathematical companion makes the same distinction and does not treat the moving
native action as a fixed exact orthoprojector.

## Remaining limitations

The conclusions remain conditional on three seeds, one MNIST MLP, one calibrated
SGDm rate, one rank, one corruption process, and one duration. Amplitude,
orientation, temporal smoothing, and endogenous basis evolution remain
entangled. The factorial intervention supports a causal comparison between the
specified history policies on these paired branches, but it does not identify
the semantic content of the complement, a mediation fraction, or transfer to
other tasks and operating points. Both reviewed documents state these limits.

Subject to those declared boundaries, the high-level conclusion is supported:
carried outside-action history is not necessary for the fixed-label current-arm
benefit in this recipe, while explicit mean restoration supplies useful
adaptation information whose effect depends strongly on how SGDm history is
transported.
