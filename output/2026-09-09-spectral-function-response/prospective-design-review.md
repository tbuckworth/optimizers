# Prospective finite function-response design review

Codex · 9 September 2026.

Reviewed against `research/grokking_raw_direction_2026-09-09.md` at SHA-256
`4f4af8287a8cb1deff151136cde25651e6d8334f617e0a14305ac426cea610d0`
and the independent raw-direction interpretation review at SHA-256
`ae4850fee88a469ab4a83b1eb25112ed1e9d949ff0615c23aeb40960211e018e`.
This is design review only. I did not open experimental tensors or arrays, run
inference, replay an update, or launch an experiment.

## Verdict

**Proceed as a bounded local discriminator.** The three scientific actions can
separate the immediate finite response to removing the off-span component from
the response to increasing retained-span amplitude. They cannot establish that
either local effect mediates the 1,000-step endpoint difference. A zero-input
Adam step is useful as a secondary carried-history drift reference, but is not
needed for the primary pairwise identification and does not isolate an additive
momentum cause.

## Exact common-state interventions

For each fixed seed 100–104, admit the saved original step-1,500 model, Adam
state, raw gradient `g`, post-observer retained basis `Q`, and native action norm
`a`. Let `P=QQᵀ`, require nonzero `g`, `Pg`, and `a`, and define

<pre>
u = a g / ‖g‖                 raw
t = P u = a Pg / ‖g‖          truncated raw
v = a Pg / ‖Pg‖               projected
ρ = ‖Pg‖ / ‖g‖, so t = ρv
</pre>

Raw versus truncated holds the retained incoming component fixed and removes
only `(I−P)u`. Truncated versus projected holds direction fixed and increases
retained amplitude. This is the minimal action chain needed to distinguish the
two local interventions.

Restore the identical model and complete Adam state independently before each
new zero, truncated, or projected step. Deliver the saved or derived action
directly to AdamW and execute exactly one optimizer step. Do not recompute the
training gradient, update the observer again, or continue a trajectory. Reuse
the receipt-bound saved raw post-step state and its archived pre/raw logits; do
not replay raw. The resulting cross-invocation CUDA sensitivity must remain an
explicit limitation, especially for small local contrasts.

The zero-input step means an AdamW step with a delivered gradient tensor of
zero, not no parameter update. Old moments, bias-correction time and decoupled
weight decay still move parameters. Its response quantifies this combined
carried-history drift. Subtracting it from an action response gives a
conditional action-versus-zero contrast, but does not decompose the nonlinear
interaction between the new action, old moments and denominator.

## Function-space quantities

Evaluate the same full uniform modular grid, in the same fixed order and
environment, before the step and after each response state. For logits
`f(a,b,c)`, center every input over output class `c` before energy calculations.
This removes softmax-irrelevant all-class shifts.

For any centered finite logit response `δf`, define, coordinatewise in output
class,

<pre>
Rδf(a,b,c) = p⁻¹ Σ_x δf(x,(a+b−x) mod p,c).
</pre>

Under the uniform full `p²` input measure, `R` is the orthogonal projector onto
responses constant within each modular-sum class. Store consistently normalized
squared energies for `δf`, `Rδf`, and `(I−R)δf`, and verify the Pythagorean
identity within a prospective numerical tolerance. Define zero-denominator
ratio semantics in advance rather than from observed magnitudes.

The primary response objects should be the pairwise post-logit contrasts
`f_t−f_u`, `f_v−f_t`, and `f_v−f_u`, not only each arm's post-minus-pre response.
Decompose these pairwise contrasts with `R` as well. Individual post-minus-pre
energies include the shared carried-history and decay response and should be
labeled total one-step responses. Action-minus-zero contrasts are useful
secondary diagnostics.

`Rδf` means sum-consistent response, not correct or useful response.
`(I−R)δf` means within-sum variation, not an identified memorized exception.
Energy alone cannot decide whether suppression or amplification is helpful.

## Utility and finite outcomes

Fix the baseline pre-step CE gradient with respect to logits separately on the
training and held-out sets. For a finite logit response `δf`, report

<pre>
U(δf) = −〈∇_f CE(f_before), δf〉,
</pre>

with positive values defined as favorable. This is first-order CE utility
evaluated along an observed finite output-space chord. It is not a parameter
Jacobian-vector product and is not the actual finite loss change. Because the
same baseline gradient is used, utility contrasts telescope exactly.

Also report actual finite CE, correct-class margin and accuracy before and after
each response state, for train and held-out sets, with all five seed values and
predeclared paired summaries. Use explicit favorable sign conventions and no
outcome-selected thresholds. Accuracy is descriptive and may be insensitive to
a single step.

For every scalar outcome `Y` evaluated on the same counterfactual states,

<pre>
(Y_t−Y_u) + (Y_v−Y_t) = Y_v−Y_u
</pre>

exactly. This telescoping check is required. It does not make suppression and
amplification independent causal components under nonlinear Adam and network
responses, and squared energies of separately realized responses need not add.

If finite loss or margin is evaluated after retaining only `Rδf` or only its
residual, label it a post-hoc logit-space counterfactual rather than a separately
realized Adam response. The clean minimal analysis instead uses componentwise
signed baseline utility and complete-action finite outcomes.

## Interpretation rule

Consistent favorable raw-to-truncated finite outcomes would support helpful
local off-span suppression conditional on the inherited state. Consistent
favorable truncated-to-projected outcomes would support helpful local retained
amplification. Both may hold, one may hold, or neither may be visible in one
step. A mixed or null local result does not refute a cumulative feedback
mechanism. No result here establishes trajectory mediation, semantic feature
selection, memorization cleanup, fresh validation, safety transfer, or a new
optimizer recommendation.
