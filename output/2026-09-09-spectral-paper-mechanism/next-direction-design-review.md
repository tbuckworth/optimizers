# Prospective review: raw-direction discriminator

9 September 2026. **PASS for implementation next, with one zero-case
clarification before the executable protocol is frozen.** No scientific reason
requires rerunning completed native, orthogonal or norm-matched branches.
This is a design review, not executable-source approval or launch validation.

## The causal distinction is sound

At a common state, the two policies have the same incoming norm target and
differ only in delivered direction. After branching, both evaluate the same
functional law `a(S,g)=||V(Vᵀg)||` on their own states. Therefore the endpoint
contrast tests the total effect of replacing the direction rule within this
state-dependent controller. Subsequent differences in scale, estimator and
Adam history are part of that policy effect; they are not separately controlled
mediators. The design states this correctly.

This does not identify a direct effect of direction at an identical numerical
scale sequence, prove that all learned geometry is unnecessary, or test a
generic scalar schedule. The raw branch's shadow V still affects its scale.
The wording “conditional on a learned scale law” appropriately limits the
question. Preserve that wording in any eventual headline.

## The arm is informative, not a predetermined control

The existing retained-norm observations describe only states visited by the
old norm-matched policy. They neither determine the new trajectory nor make
the raw and projected directions equal. Even at a common state the geometric
identity concerns incoming vectors, not the Adam update or later useful
learning. Accordingly, the raw-policy outcome is genuinely unknown.

A raw deficit would support a benefit of this direction-restricted policy
including its later feedback. A raw gain would show direction restriction is
not necessary for those post-fork endpoints under this replacement. Neither
outcome identifies the role of restriction before step 1500. Small or mixed
contrasts cannot establish equivalence. The design retains these boundaries
and useful held-out readouts rather than making gradient similarity the goal.

## Archived references and implementation conditions

Using the archived norm-matched run is a legitimate reuse of the paired
starting states and frozen recipe, with a reproducibility qualification rather
than a new concurrent control. Matching source behavior, complete input state,
environment and the first-step tensor provenance matters. Known CUDA
sensitivity means small endpoint differences must remain unresolved instead
of being attributed confidently to the direction change. The design does not
claim that one shared first-step comparison establishes deterministic equality
of a later 1,000-step trajectory. No extra rerun is needed to preserve this
honest, limited comparison.

Before implementation is frozen, clarify **“two zeros are uninformative”**:
when the norm target and raw input are both zero, deliver an explicit zero
gradient and perform the ordinary AdamW step. Do not skip it or substitute
`grad=None`: carried moments and decay can still move parameters. Undefined
directional diagnostics should be recorded as such, not converted to favorable
cosines. A zero norm target with nonzero raw input likewise defines a zero
delivered action. This is a specification clarification, not an additional
scientific arm or a request for user approval.

The proposed source review, synthetic fixtures, integrity checks and bounded
storage/runtime validation remain prerequisites to launch. They should verify
that the saved first-step “projected” tensor is the norm-matched projected
action defined in the design, not an ambiguously named unscaled q. Preserving
all five seeds and both fixed endpoints, with no outcome-driven extension,
makes the proposed one-arm acquisition a coherent next step.
