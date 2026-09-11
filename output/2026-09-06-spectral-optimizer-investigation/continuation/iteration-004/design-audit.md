# Prospective design audit — iteration 004

**Conceptual verdict: suitable for the two stated small-recipe comparisons.**
This is a design review, not a code audit, pilot result, or launch approval.
Reviewed the complete [design intent](design-intent.md) and [protocol](protocol.md)
before any iteration-004 outcome. The parent retains the source, implementation,
pilot and execution gates. No source edits, runs, commits or subagents were used
for this audit.

Reviewed SHA-256 values:

- Protocol: `4a0800927f1d18b4d4b01aadb2657b7f1c9ada1310f245a24dad6092c44f387f`
- Design intent: `a1e9b7e4d6c1f9cda80b7798af1e1b3a4d7882d47f06c334aa365eefed25704c`

## What the checkpoint design resolves

The protocol fixes the candidate grid at steps 0,100,...,2000, independently
selects strict minimum CE and strict maximum accuracy, preserves the earliest
exact tie, and uses neither an epsilon tie group nor another metric as a
tie-break. All selectors, final and warmup-stop checkpoints receive both test
metrics only after all twelve training/selection runs finish. These rules
prevent test-based checkpoint selection within this experiment. They do not
guarantee that validation selection improves test performance or that a gain
in accuracy is also a gain in CE.

The common step-100 checkpoint is useful: it tests whether continued training
under each policy adds or removes performance beyond the identical warmup
state. Requiring equal parameters **and AdamW moments/counters**, RNG/training
state and raw/applied gradients avoids treating parameter equality alone as
equality of the next-step optimizer state. Hard32 and scalar32 also share their
observer state at that point. Width128 has different estimator state by design.
The common checkpoint is a fixed comparator, not an outcome-selected stopping
rule. Its inclusion in both selector grids bounds selected validation criteria,
not selected test performance.

## What the scalar control does and does not match

At a fixed state with nonzero raw gradient `g` and an exact orthogonal projector
`P`, define `h=Pg` and `alpha=||h||/||g||`. Then `0<=alpha<=1`, and

\[
\|\alpha g\|=\|Pg\|,\qquad
g^\top(\alpha g)=\alpha\|g\|^2,\qquad
g^\top Pg=\alpha^2\|g\|^2.
\]

Thus the scalar candidate has the same norm as the projected candidate but
preserves raw-gradient direction. It does not match the raw-gradient inner
product or, even for SGD, first-order loss change. The numerical protocol
allows alpha up to 1.005 without clamping; therefore strict attenuation on
every floating-point step must not be asserted. The actual commitment is
matching the computed candidate norm within the stated tolerance.

Each observer uses its own trajectory's raw current gradient and is
self-inclusive. The scalar observer is not measurement-only: it changes
training through alpha, although it does not directly rotate the gradient.
Hard-versus-scalar is a comparison of complete adaptive training policies.
It is not matching another arm's realized gradient norms, projector history,
or actual AdamW displacement, and does not isolate direction as a unique causal
mediator of an eventual generalization difference.

For Adam, a constant positive rescaling applied from optimizer initialization
would scale the first moment by alpha and the second moment by alpha squared;
the factors cancel in the normalized update when epsilon is zero. With nonzero
epsilon this changes its effective size. Here alpha varies, and scaling starts
after 100 unscaled steps with existing moments, so that cancellation argument
does not make the control redundant. Conversely, equal pre-Adam gradient norms
do not imply equal AdamW update norms or equal effective learning rates.
The planned actual total and nominal decay-subtracted displacement records are
therefore important; subtracting current nominal decay does not erase its
historical effects on the trajectory.

The zero rules are explicit: identity/warmup alpha is one; under active scalar
control, zero raw or projected norm yields alpha zero and a zero gradient tensor.
It must not skip the optimizer step or pass `None`. Existing moments and weight
decay may still move the parameters. Undefined cosines remain null. Finite,
norm-matching and direction gates are specified. The separate implementation
review should verify these gates inspect the gradient actually delivered to
AdamW, not only the newly constructed candidate tensor.

## Statistical and interpretation checklist

The researcher-review guidance informed the separation between direct planned
comparisons and broader interpretation. The final prospective protocol adopts
the recommendations sent to the author on tie handling, optimizer-state
matching, zero-gradient behavior, endogenous scalar control and reused-test
limits. No broad sweep is recommended or required for this bounded question.

## Final prospective alignment check

Re-read the complete current protocol and newly supplied
[analysis specification](analysis-plan.md) before launch. This addendum preserves
the original reviewed hashes above rather than replacing the review history.
Current reviewed SHA-256 values:

- Protocol: `40fa3dc42612176d955d944d3c052af82d09f46eac58ba893c789a47554fbe53`
- Analysis plan: `03ee1d5479936a2493a0e548aa73bd9d05a243aa3d54fb4fb5489a3fbcc94a47`

The protocol's readability changes do not alter the reviewed recipe, seeds,
zero rules, norm tolerances, selectors, common warmup checks, outcomes or
interpretation. The separately reported implementation hardening is not an
outcome-based recipe amendment; this conceptual check does not certify test
execution or launch readiness.

The previously pending aggregation specification is now resolved. It names the
same two accuracy-selector co-primary contrasts and fixes all six pairwise arm
comparisons, all four checkpoints and both test metrics. Final and selected
test differences from the common warmup checkpoint are explicitly within-run
comparisons, not extra trained arms. Accuracy units and seed-first descriptive
statistics are unambiguous. Strict selector rederivation, all twelve completed
runs before test loading, and 48 saved-state test evaluations are required.

Secondary geometry uses fixed all/early/late windows, means of per-step values,
all-step positive-dot denominators, and nominal decay-subtracted displacement
as the primary update diagnostic. Undefined values retain exact null-step masks;
paired group statistics require all three available pairs with matching masks.
In particular, undefined active-hard-arm alpha is not silently set to zero.
The plan explicitly rejects scalar orthoprojector leakage and norm/learning-
rate/mediator identification beyond the own-trajectory quantities measured.

**Alignment verdict remains pass, with no additional conceptual blocker.**
The source/test, runtime-invariance pilot, frozen-byte/commit and explicit
parent-GO gates remain separate and must still be satisfied. No experiment,
test suite, source edit, commit or subagent was used for this alignment check.
