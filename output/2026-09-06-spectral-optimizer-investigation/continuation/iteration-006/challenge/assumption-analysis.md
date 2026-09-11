# Assumption Analysis

6 September 2026. Independent, prospective leaf review; no iteration-006
implementation, dataset access or experiment was performed. No peer review was
read. The assumption-challenger instructions set the structure; the pre-mortem
checklist helped turn the assumptions into concrete failure tests.

## Summary

Five load-bearing assumptions are examined below, including two high-impact
implementation/measurement issues that should be closed before source freeze.
**No redesign of the six-arm scientific comparison is required:** it can answer
the stated whole-policy question, but not identify a unique denoising or
norm-independent causal mechanism. Parent clarifications received during this
review resolve the intended arithmetic and rank-labeling choices; their tests
and final prospective specification remain necessary.

The plan compares six endogenous AdamW policies on three paired seed bundles,
separately under clean and fixed noisy labels. Iteration 005 motivates changing
observation/delivery order; iteration 004 motivates retaining both checkpoint
selectors and scalar controls. Neither earlier result predicts that lagging
will help, and a faithfully measured adverse or null result remains useful.

## Critical Assumptions (Low confidence, High impact)

### 1. Finite positive candidate norms are sufficient for valid float32 norm restoration

- **Category:** Theoretical / Methodological.
- **Confidence:** Low for this implication; it is false in general. The
  frequency of troublesome cases in this MNIST recipe is unknown.
- **Currently assumed because:** The design specifies float64 norms, float32
  delivery and an exact-zero rule, but its initial version does not specify
  multiplication precision or distinguish representable from unrepresentable
  norm targets.
- **What changes if wrong:** A mathematically finite rescaling can overflow
  during conversion of the scale to float32, or the delivered vector can lose
  its entire norm through underflow. Near-zero lagged projections can also
  magnify projection-rounding error in the delivered direction. Silently adding
  an epsilon or clipping the scale would change the policy being tested.
- **How to test:** Freeze the parent's proposed recipe: compute norms and scale
  in float64, multiply `direction.double() * scale`, then cast once to the raw
  gradient dtype. Read back the actual delivered gradient. Require finite
  values, a positive delivered norm whenever the target norm is positive, and
  relative norm error at most `1e-6`; an absolute floor must not excuse complete
  underflow. Include representable extreme-ratio cases that must pass and
  quantization/underflow cases that must fail with preserved context. For
  example, a lagged vector of size approximately `1e-40` restored to norm one
  needs a finite float64 scale exceeding float32's range, although the intended
  output can be representable. Conversely, spreading a smallest-subnormal target
  norm over many coordinates can make every delivered coordinate round to zero.
- **Early warning / risk:** Huge scale, tiny positive lag norm, or positive
  target with zero/incorrect delivered norm. Occurrence likelihood is unknown;
  severity is high because an affected arm cannot be silently substituted.
- **Relevant evidence:** The initial [design](../design-intent.md) already treats
  positive-current/zero-lag norm as a fatal undefined-direction case. The
  implementation check (artifact not distributed in this public snapshot) requires actual-gradient
  gates. The float64-multiply clarification received from the parent during
  review addresses premature overflow, but cannot make every float32 target
  representable. No new numerical experiment was needed for these examples.

### 2. Every policy with a width-32 observer also has a rank-32 delivery operator

## Moderate Assumptions (Medium confidence or Medium impact)

### 3. Accuracy-selected comparisons necessarily measure substantial post-warmup learning

- **Category:** Methodological.
- **Confidence:** Medium; selected steps for fresh seeds are unknown.
- **Currently assumed because:** The headline question concerns delivery after
  step 100, but the primary selector may choose step 0 or the shared step 100.
  Different arms may also select very different amounts of continued training.
- **What changes if wrong:** A small primary difference can reflect selection
  of an identical pre-intervention checkpoint, not similar active trajectories.
  More generally, the primary estimand is the policy **with its checkpoint
  selection rule**, not the effect at a common training step. This does not
  invalidate the chosen primary outcome.
- **How to test:** Preserve the selector as proposed. Report selected steps and
  `max(selected_step - 100, 0)` beside paired outcomes, flag pre-intervention
  selections, and retain the already mandatory final-step comparison. Do not
  exclude warmup-selected arms or change the selector after seeing them.
- **Early warning / risk:** Selected steps 0/100 or widely separated selected
  steps. Likelihood is unknown; interpretive severity is medium.
- **Relevant evidence:** In [iteration 004](../../iteration-004/results.md),
  hard32-minus-AdamW accuracy changed from +18.81 percentage points at the
  endpoint to -2.38 under accuracy selection. Its saved selected steps vary
  considerably across arms. No fresh-seed selection behavior follows from that
  earlier observation.

### 4. The implementation changes only delivery phase, and the measured phase contrast is interpreted at the right level

- **Category:** Methodological / Independence / Theoretical.
- **Confidence:** Medium until canonical regression and state tests pass.
- **Currently assumed because:** The proposed implementation reuses validated
  helpers but adds previous-basis storage, two candidates and new policies.
- **What changes if wrong:** Aliasing, observing twice, or an off-by-one at
  activation/repair could make the treatment something other than the specified
  phase change. Even when correct, a previous basis is not an oracle for
  noise-free signal: it contains earlier visits to the same fixed corrupted
  examples and depends on the policy's previous trajectory.
- **How to test:** Match the current arm to the unchanged canonical computation;
  check exactly one raw-gradient observation per step; exercise steps 100/101
  and 199/200/201; mutate the live basis after cloning and verify the previous
  candidate remains fixed. Compare full warmup model, optimizer and RNG states
  within each `(seed, noise condition)`, including NumPy-global state. Clean and
  noisy conditions should not be required to share post-gradient warmup states.
  The parent's check already requests the substantive aliasing, RNG and
  canonical-current tests; no additional training arm is needed.
- **Early warning / risk:** Wrong observation count, failed current regression,
  previous-basis changes after ingestion, or warmup disagreement within a cell.
  Likelihood is medium before tests; engineering severity is high. Treating
  lagging as semantic noise exclusion would separately be an interpretive error.
- **Relevant evidence:** [Iteration 005](../../iteration-005/results.md) measures
  the complete transition on common AdamW streams; every sampled step also
  schedules repair. Its approximately .84-versus-.41 retention contrast is not
  an isolated outer-product effect and not a filtered-policy learning result.
  Record the proposed every-step contrasts on each new policy's own stream,
  but do not subtract scalar-arm endpoint contrasts to claim an identified
  directional mediator. The design already correctly disclaims that inference.

## Background Assumptions (High confidence, Low impact)

- **Category:** Data/Resource / Baseline / Scope / Scaling / Independence.
- **Confidence:** High for a fixed-recipe policy comparison, not for transfer
  to other models, datasets or optimizer tuning regimes.
- **Currently assumed because:** Earlier audited runs establish the local
  data/model workflow; the study explicitly excludes SOTA and population claims.
- **What changes if wrong:** A failure of storage or runtime gates stops the
  experiment, while a lack of external transfer limits generalization rather
  than invalidating this within-recipe comparison. Clean/noisy pairing does not
  create six independent seed replications, and fresh seeds do not create a new
  official test set.
- **How to test:** Retain the existing hash/mount/occupancy/time gates and use the
  pilot only for resource and invariance evidence. The noisy 220-step pilot does
  not certify late clean-gradient magnitudes; synthetic tests in assumption 1
  must cover the relevant numerical extremes without accuracy-driven choices.
  A larger benchmark or tuned baseline is not a prerequisite for this question.
- **Relevant evidence:** [Iteration 004](../../iteration-004/results.md) completed
  twelve runs and checkpoint evaluations in 159.55 elapsed seconds. This
  supports feasibility, not a guarantee for the new policies or a justification
  for exceeding the proposed 900-second cap. The run state (artifact not distributed in this public snapshot)
  preserves the small-study and reused-data scope.

## Assumption Dependency Map

Numerically valid delivered gradients (1), correctly classified controls (2)
and faithful observer timing (4) are prerequisites for interpreting any learning
contrast. Given those gates, checkpoint selection (3) determines which learning
comparison is primary; benchmark scope (5) bounds where that comparison applies.
None of these conditions turns own-state norm restoration into cross-arm norm
matching or a pure causal mediation experiment.

## Recommendations

- **Fix before freeze:** encode the float64 scaling/cast recipe and strict
  representability tests; correct the blanket rank claim and define baseline
  candidate nulls. The parent's clarifications are suitable resolutions.
- **Verify before pilot/full approval:** canonical current-arm regression,
  once-only observation, nonaliasing previous bases, actual delivered-gradient
  gates and complete within-cell warmup/RNG equality. Preserve failures rather
  than replacing a policy or relaxing a tolerance.
- **Add to reporting, not the arm list:** selected-step exposure and explicit
  pre-intervention selections. Preserve the existing endpoint and CE-selected
  outcomes regardless of primary signs.
- **Accept as bounded limitations:** endogenous trajectories, unisolated
  observer-transition components, reused test data and three seed bundles.
  They do not require new probes, new datasets or additional arms for the
  stated question. This is a conditional design pass, not launch approval.

## Reviewed evidence provenance

Hashes identify the versions read, before any subsequent design amendment:

| Input | SHA256 |
|---|---|
| iteration-006/design-intent.md | `4ba43f2bcc220f7867e791cb7765840d407c0cd52ddf5850774717b6f70bd331` |
| iteration-006/best-practices-check.md | `feca4c695499d882cbfcbf551529ec36e6c07ed768e3bac1d379e93a4d855c03` |
| iteration-004/results.md | `c2bedd41e74c48b5e7eb6d7ae86bba6efc2fb3999322fa54c9cd63638037f795` |
| iteration-005/results.md | `e3d18d9942a1f3fc66f65be6a4b5be25173d128601929c8d0ab3121314fe2550` |
| RUN/state.md | `c26996a4080e21e2a26073866f58522b80b3c3446e736bbc06548d67579be4d3` |

Numerical historical claims above cite the completed audited reports; this
review does not independently reaggregate their raw outcomes or certify an
iteration-006 implementation that does not yet exist.

### Resolution recheck before handoff

The parent's revised design, SHA256
`7d6084173c22e47a59df594c871df3d2c3c169898afdc5d13bc021d4b2234352`,
now corrects the rank statement and explicitly specifies float64 multiplication,
one float32 cast, `1e-6` relative norm and normalized-direction gates, positive
target/positive delivery, and preserved representability failures. Its new
selection-language qualification also addresses assumption 3. These are
adequate prospective resolutions: **no remaining assumption-level blocker to
implementation**, subject to the stated synthetic/invariance tests. This
recheck reads only the parent-authored design amendment, not other reviewers'
assessments, and supplies no pilot or full-run approval.
