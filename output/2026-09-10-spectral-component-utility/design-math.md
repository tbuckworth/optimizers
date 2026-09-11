# Does signed component utility add a new mechanism question?

Codex — Spectral Optimizer Investigation · 10 September 2026

**Assessment: conditional go for one small common-parent diagnostic, not for
component attribution or another efficacy study.** The new information would
be which evaluation-objective changes accompany the actual update's useful
or harmful clean response. Separate Adam steps on the three components, more
norm-control arms, or comparisons of component energies alone would not answer
that question. Main retains the final scope/resource decision. This diagnostic
is not required before reporting the existing paper.

This independent assessment read the current
root state (artifact not distributed in this public snapshot),
[next decision](../2026-09-10-spectral-strong-augmentation/next-decision.md),
[knowledge index](../../knowledge/index.md) and
[schema](../../knowledge/schema.md), the
[objective derivation](../../research/spectral_augmentation_loss_geometry_2026-09-10.md),
[strong results](../2026-09-10-spectral-strong-augmentation/results.md) and
[completed clean-state diagnostic](../2026-09-10-spectral-augmentation-state/results.md).
It did not read other agents' design outputs, raw archives, checkpoints or
models, and performed no scientific calculation or experiment.

## 1. What is already known, and what would be new

The strong study establishes genuine native learning after warmup, not merely
preservation: unaugmented mean reporting accuracy rises from 36.70% to 79.75%.
But raw translation reaches 84.97%, versus 66.82% for the combined policy;
the combination's primary accuracy and CE costs recur in all three seeds.
Nor does it further suppress wrong-assignment fitting relative to the single
interventions. These are practical findings, not a component mechanism.

The earlier six-clean-state diagnostic already distinguishes direction from
actual Adam data-step size. It also measures between-image/within-view gradient
energy and identifies both useful and adverse native directions. Repeating
those measurements on another checkpoint would add relatively little.

It did **not** measure the fixed-label realization objective F, the exact
view-consistency objective C, or how their changes trade against softened-target
learning S under a common actual displacement. Those signed quantities can
distinguish, locally, “less consistency learning,” “less mean-logit learning,”
and “restriction of realization fitting while useful learning survives.”
The latter is the constructive possibility worth preserving. None follows
from the existing energy-retention numbers.

## 2. Fix the evaluation objectives before defining actions

For a fixed training evaluation panel I, fixed assigned targets \(t_i\), true
classes \(y_i\), and all 25 equally weighted translations, define

\[
\bar z_i=\frac1{25}\sum_T f_\theta(Tx_i),\quad
A(z)=\log\sum_k e^{z_k},\quad q_i=.1e_{y_i}+.9u,
\quad \epsilon_i=t_i-q_i.
\]

The strong recipe's replacement law, not its realized wrong fraction, fixes
\(q_i\). With \(\operatorname{CE}(z,q)=A(z)-q^Tz\), set

\[
\begin{aligned}
S_I&=|I|^{-1}\sum_i\operatorname{CE}(\bar z_i,q_i),\\
F_I&=-|I|^{-1}\sum_i\epsilon_i^T\bar z_i,\\
C_I&=|I|^{-1}\sum_i[25^{-1}\sum_T A(z_{iT})-A(\bar z_i)].
\end{aligned}
\]

Then \(L_I^{\rm aug}=S_I+F_I+C_I\) pointwise at every evaluated parameter
vector. The gradient identity \(g_I=s_I+f_I+c_I\) is exact too, where
\(s_I=\nabla S_I\), etc. Differentiate the moving mean logits; do not introduce
a stop-gradient teacher.

S is not clean CE. Its exact split
\(S=.1\,\overline{\operatorname{CE}(\bar z,e_y)}
+.9\,\overline{\operatorname{CE}(\bar z,u)}\)
should be retained in finite scalar readouts so a change in confidence
regularization is not automatically described as lost class learning. F is
linear in logits, not generally in parameters, and may have either sign.
Reducing it means fitting the realization term, not necessarily harming clean
prediction. It includes correctly assigned examples too: their \(\epsilon_i\)
need not vanish. C is nonnegative but can be zero for a consistently wrong or
uninformative predictor.

Define separate true-label clean readout losses on fixed nontraining images:

\[
H_O=\overline{\operatorname{CE}(f_\theta(x),e_y)},\qquad
H_T=\overline{25^{-1}\sum_T\operatorname{CE}(f_\theta(Tx),e_y)}.
\]

Original-image CE and accuracy remain primary competence readouts; translated
ones are secondary. Neither H is generally the loss at mean logits. A lower
S or C does not substitute for an actual favorable H change.

## 3. Actual inherited-state Adam estimand

For each parent \(\sigma=(\theta,m,v,t,O)\), fix a one-view action batch and
its original assigned targets. Compute its raw gradient \(g_*\) once.
Raw delivery is \(h_R=g_*\). Native delivery \(h_N\) follows the existing
policy: clone the saved observer, observe this current gradient once, and use
its resulting action. It is not an independently chosen fixed projector.

Both actions start from identical model, Adam moments, counters and settings.
In ideal arithmetic, with componentwise squares and divisions,

\[
m_a^+=\beta_1m+(1-\beta_1)h_a,\quad
v_a^+=\beta_2v+(1-\beta_2)h_a^2,
\]
\[
\theta_a^+=(1-\eta\lambda)\theta-
\eta\frac{m_a^+/(1-\beta_1^{t+1})}
{\sqrt{v_a^+/(1-\beta_2^{t+1})}+\varepsilon_{\rm Adam}},
\qquad \Delta_a=\theta_a^+-\theta.
\]

The actual serialized FP32 endpoints, not presumed equality to this ideal
formula, define the measured displacement. Counters advance exactly once;
no resets or coercions to a common clock across different parents. Native's
current-gradient self-inclusion is part of the contrasted policy.

For each \(J\in\{S_I,F_I,C_I,L_I^{\rm aug},H_O,H_T\}\), measure

\[
U_J(a)=-\nabla J(\theta)^T\Delta_a,
\qquad E_J(a)=J(\theta)-J(\theta_a^+).
\]

Positive means decrease of that named objective. Call it **useful learning**
only when the clean readout supports that interpretation. Report both linear
and finite changes, their residual \(E_J-U_J\), and any opposite signs.
No universal nonnegative remainder follows because the network objectives
need not be convex in parameters.

The primary within-parent contrast is

\[
D_J=E_J(N)-E_J(R),\quad
D_J^{\rm lin}=-\nabla J(\theta)^T(\Delta_N-\Delta_R).
\]

Exact accounting gives
\(E_L=E_S+E_F+E_C\) and \(D_L=D_S+D_F+D_C\), with the analogous linear
identities. This does **not** decompose clean-readout improvement \(D_H\)
into three causes; H is a different objective. Do not turn signed summands
or near-zero ratios into a “fraction mediated.”

The action batch and evaluation panel may differ. Consequently \(g_*\) need
not equal \(s_I+f_I+c_I\), even in exact arithmetic. The equality is for
the evaluation objective, not an invented decomposition of the actual sampled
one-view gradient. A difference between them includes both image and view
sampling, not automatically pure augmentation noise.

### Decay and inherited history

Use the actual rounded decay-only parameter vector \(\theta_D\) to separate
\(\Delta_a=(\theta_D-\theta)+(\theta_a^+-\theta_D)\), with differences
computed consistently from stored endpoints. Linear component accounting is
additive at the same parent gradient. Finite decay/data accounting instead
uses \(J(\theta)-J(\theta_D)\) and \(J(\theta_D)-J(\theta_a^+)\).
Decay-only is not Adam with a zero gradient: inherited momentum can still move
parameters in the latter. This clarification needs no extra zero-gradient
Adam arm or additional norm-control roster.

## 4. Accounting versus intervention attribution

Evaluating S, F and C along the full update answers what it changes. It does
not answer which input component generated that movement. Adam is nonlinear:
squaring the summed gradient introduces cross terms, and its denominator and
old moments are shared. Generally
\(\operatorname{Adam}(s+f+c)\neq
\operatorname{Adam}(s)+\operatorname{Adam}(f)+\operatorname{Adam}(c)\).
Those separate steps would also repeat inherited momentum/decay effects.

Removing F or C from a delivered gradient would be a different intervention,
possibly changing both the current observer and Adam denominator. It cannot
be reconstructed by labeling full-update dot products as contributions.
No such intervention is needed or selected here.

Native versus bypass at one common parent is a conditional current-delivery
intervention. Comparing native-trained and raw-trained final models, or
unaugmented and translated warmups, is not that intervention: weights,
moments and observer histories differ. Comparing warmup versus final contrasts
describes state dependence, not when or why the endpoint gap emerged. The
source states are reused and the question was motivated after outcomes.

## 5. Sampling, roster and the finite-view issue

Main's source-only inventory reports 12 native parent states: three seeds ×
none/translation × update 100/final 56,304, with no intermediate checkpoints.
Use the whole fixed roster, with two prospectively fixed one-view action batches
per parent if its resource plan permits. Average those action draws within
parent first, then list all three seed values separately for each stage and
training condition. Do not treat 12 parents or 24 actions as independent seeds,
pool away state reversals, or select the largest existing performance gap.
Images/views must be fixed from plan-only rules before this measurement;
clean labels provide oracle diagnostic information, not a deployable policy.

Exact enumeration of the 25-view support removes within-image Monte Carlo
component bias for the finite panel. It does not remove image-panel sampling
error or make this a population result. If enumeration is replaced by m iid
views, the finite-view decomposition remains pointwise exact, but

\[
\mathbb E S_m\geq S_\infty,\qquad
\mathbb E F_m=F_\infty,\qquad
\mathbb E C_m\leq C_\infty.
\]

The two Jensen biases cancel in the total expected loss; the separate gradient
biases have no analogous guaranteed sign. At m=1, C vanishes by construction.
An average of ordinary per-view CE values is not a substitute for reconstructing
per-image mean logits. Fix transformations and their weights at before/after
evaluation so a finite difference does not include resampling noise.

C is label-independent at the **same** parameters, images and view law. Its
values across differently trained parents can depend on labels through the
learned state. Calling such between-parent differences label-independent
regularization effects would be incorrect.

## 6. Constructive hypothesis, falsifier and decision boundary

The strongest constructive possibility is conditional complementary
restriction: native delivery reduces realization fitting while preserving
positive S/C adjustment and actual clean progress. A same-parent pattern
\(E_H(N)>0\), \(D_H\geq0\), and smaller positive F-decrease
would support that local tradeoff. F's sign alone does not establish protection.
It would not overturn the measured adverse combined-policy endpoint.

The sharper explanatory hypothesis for the combination cost is that native
restricts **useful** consistency adjustment: raw has \(E_C(R)>0\) and positive
clean progress, while native has \(D_C<0\) together with \(D_H<0\). S and F
must be reported jointly to show whether restriction is selective or broad.
This pattern supports a local association, not consistency-mediated causation.

A falsifying outcome for that local consistency account is \(D_C\geq0\)
across the registered translated-parent cells despite adverse \(D_H\): the
clean cost then occurs without reduced local consistency improvement. Likewise,
if neither action decreases C, “lost useful C learning at these states” is
unsupported. Predominant lost S progress, extra F fitting, state reversals or
linear/finite disagreement each changes the explanation; tiny changes or mixed
seeds may simply leave it unresolved. None disproves every earlier-stage or
long-run consistency hypothesis.

Proceed only if the common inherited states, exact objective readouts and
independently checkable actual displacements fit one fixed bounded protocol.
The substantial post-observer rank200 bases must be included in archive
accounting, not hidden behind a vector-only estimate. Main must verify the new
snapshot adapter and resource feasibility; this review does not certify them.
If only endpoint scalar C values, standalone retention or unmatched-state
comparisons are feasible, **no-go**: those would not add enough beyond existing
evidence. Either decision leaves the completed result and paper reportable.
