# Mathematical follow-up: ranked causal hypotheses and identifiable tests

7 September 2026. **Theory and evidence challenge only. Every intervention
below is PROPOSED, not run. This note supplies no authority for a new I7 attempt;
the consumed one-shot remains immutable.**

## Bottom line

The leading hypothesis is not that temporal PCA discovers intrinsically clean
directions. It is that hard filtering imposes a learned, task-dependent
bottleneck on the gradient information entering AdamW. Under this provisional
account, that bottleneck suppresses late fitting in the noisy-MNIST regimes
where memorization uses directions the policy excludes, while suppressing
useful learning on clean MNIST, sparse parity, LoRA and other adverse tasks.
Adam's moments and coordinatewise normalization then turn this input restriction
into a different—and substantially less transparent—restriction on parameter
and function change. Existing evidence supports restricted learning broadly;
it does not yet establish this specific directional mediation.

The ranking below concerns causal importance for the observed anti-memorization
pattern, not mathematical reality. The mechanisms can coexist.

| Rank | Hypothesis | Current assessment |
|---:|---|---|
| 1 | Task-dependent learned directional restriction | Most plausible primary mechanism; supported as a broad capacity/trajectory account, not as semantic denoising |
| 2 | Adam history, adaptive scaling and realized-step attenuation | Plausible co-mechanism and major unresolved confound; existing scalar controls do not match it |
| 3 | The basis tracks optimizer-induced gradient drift/curvature as much as or more than fresh-noise structure | Plausible explanation of why learned orientation matters without implying denoising |
| 4 | Recursive truncation inertia changes which directions become available | Established estimator effect and plausible modifier; weak evidence that it drives task outcomes |
| 5 | Self-inclusion of the current gradient is the source of the beneficial effect | Strong as a measurement-bias/optimization effect, weak as the primary anti-memorization explanation |

## 1. What the filter and AdamW jointly implement

Let the complete pre-step state be

\[
F_t=(\theta_t,\text{model buffers},a_{t-1},b_{t-1},n_t,
     m_{t-1},V_{t-1},S_{t-1},\text{LR/decay/scaler/RNG state}),
\]

where `a,b` are Adam's first- and second-moment states and `m,V,S` are the
observer state. For the realized raw batch gradient `g`, let `P^-` be the native
operator before observation and `P^+(g)` the native operator after the observer
processes `g`. In ideal arithmetic these are orthoprojectors; the production
action is the saved-column map `V V^T` and should be evaluated natively.

For any delivered gradient `h`, define the exact cloned one-step AdamW map

\[
\delta_F(h)=T_F(h)-\theta_t.
\]

This is the right primitive for a complete-state intervention. It holds the
parameters, batch, moments, step counter, decay and schedule fixed, changing
only the nominated input. It does **not** estimate what a separately trained
policy would do after its histories diverge.

Ignoring decoupled decay for display, Adam's data displacement has coordinates

\[
q_i(h)=\frac{\beta_2 b_i+(1-\beta_2)h_i^2}{1-\beta_2^{n_t}},\qquad
r_i(h)=\frac{\beta_1 a_i+(1-\beta_1)h_i}{1-\beta_1^{n_t}},
\]
\[
\delta_i^{\rm data}(h)=-\eta\frac{r_i(h)}{\sqrt{q_i(h)}+\epsilon}.
\]

Thus Adam is not a fixed metric applied to `h`: its diagonal metric depends on
the current input. For coordinates with `q_i(h)>0`, the local derivative is

\[
\frac{\partial\delta_i^{\rm data}}{\partial h_i}
=-\eta\left[
\frac{(1-\beta_1)/(1-\beta_1^{n_t})}{\sqrt{q_i}+\epsilon}
-\frac{r_i(1-\beta_2)h_i}
{(1-\beta_2^{n_t})\sqrt{q_i}(\sqrt{q_i}+\epsilon)^2}
\right].
\]

It can have either sign because of the historical numerator. Even with a
frozen denominator `D`, the displacement is an affine sum of historical motion
and `D h`; `D P g` need not lie in `range(P)`. With rotating projectors and zero
initial momentum,

\[
a_t=(1-\beta_1)\sum_{s\le t}\beta_1^{t-s}P_sg_s,
\]

which lies in the sum, or linear span, of historical subspaces, not generally
the current one.
Elementwise squaring in the second moment and weight decay add further escape.
The observed 60--77% outside-subspace squared displacement is therefore expected
to be possible, while not itself explaining performance
([iteration 003](../continuation/iteration-003/results.md)).

### Fixed-metric clean-descent criterion

For a predictable orthoprojector `P`, conditional corrupted-objective mean
`m=mu+b`, and a fixed positive diagonal preconditioner `D`, a data step
`delta=-eta D P g` has expected first-order clean loss change

\[
E[\mu^T\delta\mid F]=-\eta\mu^T D P(\mu+b).
\]

Neither `P` nor `D` being positive semidefinite makes this expression favorable:
`DP` is nonsymmetric unless `D` commutes with `P`, and `b` may point against the
clean gradient. The current Adam map is harder still because `D=D(h)` and a
historical numerator are present. This is why gradient retention, gradient MSE,
first-order descent and finite loss must remain separate endpoints.

## 2. Complete-state factorial and endpoint definitions

The following proposed design is the common core of all five tests. Capture a
predeclared panel of complete states before, during and after the onset of
memorization. At each state, use common realized training batches and common
independent clean and fixed-corruption probe bundles. Clone the full state for
every one-step fork; do not advance the source state.

Let

\[
u_r=g/\lVert g\rVert,\quad
u_p=P^+(g)g/\lVert P^+(g)g\rVert,\quad
q_r=\lVert g\rVert,\quad q_p=\lVert P^+(g)g\rVert.
\]

Subject to predeclared zero-norm rules, the `2 x 2` input factorial is

\[
h_{rr}=q_ru_r=g,\quad h_{pr}=q_ru_p,\quad
h_{rp}=q_pu_r,\quad h_{pp}=q_pu_p=P^+(g)g.
\]

It identifies input-direction effects at fixed input norm, input-norm effects
at fixed direction, and their interaction, all conditional on the same AdamW
history. It does not automatically match realized update norm. Add a separate
raw-direction control `h_*=q_*u_r`, where `q_*>=0` is the smallest predeclared
root satisfying

\[
\lVert\delta_F^{\rm data}(q_*u_r)\rVert
=\lVert\delta_F^{\rm data}(h_{pp})\rVert.
\]

Because the adaptive map need not be monotone, the protocol must report no
solution or multiple solutions rather than silently changing the target.
Moreover, changing `q` can rotate Adam's coordinatewise data displacement, so
`h_*` matches its final norm but does not provide a pure actual-update-direction
contrast. A cleaner, explicitly artificial diagnostic first computes the native
raw and filtered Adam data displacements, directly rescales the raw displacement
to the filtered displacement's norm (and reciprocally the filtered displacement
to the raw norm), then adds the identical cloned decay displacement. These are
one-step displacement controls, not ordinary Adam policies and not automatically
the existing or failed I7 arms. A zero-input fork `h=0` is an essential negative
control: it measures movement from historical momentum and decay with no current
gradient information. A predeclared random direction of norm `q_p`, independent
of the current batch and labels, tests whether learned orientation matters
beyond a generic same-sized perturbation.

Every “no material difference” falsifier below requires a prospectively fixed
practical-equivalence margin and measurement precision adequate to resolve it.
Failure to reject a zero difference is not a falsifier.

For arm `j`, report these different estimands separately, aggregating state and
batch contrasts seed-first:

1. **Conditional gradient MSE**
   `G_j=E[||h_j(g)-mu||^2|F]`. The closed-form predictable-projector identity
   applies to `P^-g`, not to self-inclusive `P^+(g)g` or norm-restored policies.
   Those require direct repeated-batch estimation at a frozen state.
2. **First-order clean descent**
   `D_j=-E[mu^T delta_F(h_j)|F]`. Positive `D_j` is locally favorable. This is
   about the clean objective gradient, not the current corrupted batch.
3. **Finite clean and corrupted loss changes**
   `Q_j^z=E[L_z(theta+delta_F(h_j))-L_z(theta)|F]`, for independent `z=clean`
   and `z=fixed-corruption` probes. These include curvature and the shared decay
   step. Same-batch training loss is reported separately because it is selected
   by the gradient that produced the update.
4. **Function change**, such as a fixed-probe mean squared logit displacement
   and class-margin change. Equal parameter-step norms are not equal function
   changes.
5. **Generalization** is not identified by any one-step quantity. It requires
   a separately authorized, prospectively selected multi-step policy comparison
   and fresh held-out evaluation. One-step clean validation loss is only a local
   predictive-performance proxy.

The exact direction effect at low input norm, for example, is

\[
\tau_{\rm dir}(Q;q_p)=E[Q(h_{pp})-Q(h_{rp})],
\]

and the direction-by-norm interaction is

\[
E[(Q(h_{pp})-Q(h_{rp}))-(Q(h_{pr})-Q(h_{rr}))].
\]

These are identifiable conditional effects of the one-step delivery choice.
They are not long-run mediation effects of having trained with that choice.

## 3. Ranked hypotheses

### H1 — Task-dependent directional restriction is the primary mechanism

**Claim.** The learned projection restricts which gradient information enters
AdamW. In noisy MNIST this blocks much of the late route to fitting fixed wrong
labels while retaining enough directions for earlier digit features. In other
tasks the same bottleneck removes useful directions. “Anti-memorization” is
therefore a task/phase-dependent capacity effect, not evidence that the leading
covariance space is intrinsically clean.

**Support.** Across old and current code, hard projected arms repeatedly finish
noisy-label training with much higher clean accuracy and corrupted-training
accuracy near 16--18%, while AdamW reaches roughly 43--45% corrupted-training
accuracy and loses clean accuracy. In iteration 006, current, lagged and restored
lagged projected arms all retain the endpoint pattern, whereas both scalar arms
remain AdamW-like. Iteration 004's scalar control also matches its own projected
gradient norm without reproducing endpoint preservation. Learned bases beat the
historical fixed-random control. This is empirical support for a directional
restriction, not proof of semantic selection.

**Exact proposed estimand.** On the complete-state panel, estimate the vector

\[
(\tau_{\rm dir}(Q^{clean};q_p),
  \tau_{\rm dir}(Q^{corr};q_p),
  \tau_{\rm dir}(D;q_p),
  \tau_{\rm dir}(\text{function change};q_p))
\]

and the corresponding matched-final-data-step-norm input contrast
`Q(h_pp)-Q(h_*)`, plus the artificial displacement-rescaling contrast. Compare
the learned direction with the preregistered random same-norm direction. The
mechanism predicts that learned direction selectively reduces immediate
corruption fitting relative to clean progress in the late noisy phase, with a
clean-progress cost in clean or earlier phases.

**Falsifier.** The learned-direction contrasts fall within the predeclared
practical-equivalence margins relative to raw-direction and random-direction
controls, with adequate precision, after both input- and artificial displacement-
norm matching across the predeclared late states, while an authorized repeated
scalar/random restriction reproduces the long-run endpoint and corrupted-label
training pattern. That would remove learned direction as the mediator and favor
a scalar or generic capacity account.

### H2 — Adam history and realized-step attenuation are a co-mechanism

**Claim.** Hard projection matters partly because it changes Adam's numerator,
denominator and their histories, producing smaller and differently directed
data steps and changing the balance with decoupled weight decay. This is more
specific than “lower effective learning rate”: the transformation is
coordinatewise, nonlinear and history-dependent.

**Support.** In iteration 004 the scalar arm's mean applied/raw gradient ratio
(.90565) is comparable to hard32's (.89784), but its decay-subtracted step norm
(.054562) remains near AdamW (.055033), whereas hard32 is .042273. It also fails
to reproduce hard filtering's endpoint. In iteration 006, restoring lagged
pre-Adam norm still leaves noisy realized steps at .035467 versus .043200 for
current filtering. The actual step lies mostly outside the advertised subspace,
and fixed-basis mathematics permits projected-gradient Adam to reverse descent
([mathematical audit](mathematical-audit.md)). Base-optimizer failures further
argue that filter and optimizer cannot be analyzed independently.

**Strongest conflict.** The historical learned-versus-random comparison did not
match retained norm or realized displacement, so it cannot adjudicate this
account. More directly, iteration 006 clean Current-32 has a *larger* mean
decay-subtracted step norm than clean AdamW (.040946 versus .036148) while
learning substantially less; different trajectories still prevent a same-state
causal reading, but this strains a universal smaller-step explanation. Scalar
arms establish that modest input-norm attenuation by itself is insufficient,
although they do not match actual displacement. Rare observed ascent reversals
also make the known two-dimensional reversal an unlikely broad explanation.

**Exact proposed estimand.** Decompose, at fixed `F`,

\[
Q(h_{pp})-Q(h_{rr})=
[Q(h_{pp})-Q(h_*)]+[Q(h_*)-Q(h_{rr})].
\]

The algebraic brackets are exact, but the first is not a pure actual-direction
effect because changing `q` may also rotate Adam's displacement. Report it as a
matched-final-norm input intervention. Separately use the artificial reciprocal
data-displacement rescalings above for an exact norm-controlled displacement-
direction diagnostic. Repeat with (i) the actual live denominator, (ii) a
diagnostic frozen denominator from `h_rr`, and (iii) `h=0`. Report moment,
denominator, data-step, decay-step and total-step contributions without calling
the frozen-denominator or displacement-rescaled diagnostics optimizer policies.

**Falsifier.** If both the matched-final-norm raw-input fork and the artificial
norm-rescaled raw displacement fall within predeclared practical-equivalence
margins of the projected fork's clean/corrupted finite-loss and function changes,
with adequate precision, direction adds nothing material at the tested states
and the scalar/history account is favored. Conversely, if the projected-versus-
raw difference remains large after the artificial displacement control and is
unchanged when historical numerator/denominator contributions are removed in
the diagnostic forks, Adam-history mediation is weakened.

### H3 — Centered covariance primarily tracks training drift/curvature, not denoising

**Claim.** Along a moving neural trajectory, the observer can learn directions
in which the mean gradient changes because parameters move. These directions
may encode useful feature acquisition or memorization dynamics, but they are
not the same object as within-state minibatch noise directions.

**Mathematical basis.** Write `g_t=mu(theta_t)+epsilon_t`. At a fixed stationary
state, the code's post-mean innovations have

\[
\operatorname{Cov}(c_t)=\frac{2\beta^2}{1+\beta}\Sigma_{batch},
\]

after transients. But if the mean gradient drifts by an approximately constant
increment `d`, steady lag gives

\[
E[c_t]\approx\frac{\beta}{1-\beta}d,
\qquad d\approx H_t\delta\theta_{t-1}
\]

under a local quadratic approximation. The covariance update then contains a
drift outer product scaled approximately by
`beta^2/(1-beta)^2`, sampling variation, and cross terms. At `beta=.99`, slow
gradient drift can therefore be strongly represented. This is a dynamics-
weighted curvature proxy at most, not a Hessian or Fisher identity.

**Support.** The implementation centers away a constant mean and uses temporal
batch-mean changes. Clean and fixed-corruption residual retentions rise together
when estimator width increases on endogenous trajectories, with essentially no
selectivity-gap improvement in iteration 003. On the common AdamW streams,
clean and residual gradients are strongly opposed (final cosine about -.897),
and improved covariance fidelity gives only a modest, nonuniform selectivity
change ([iteration 005](../continuation/iteration-005/results.md)). These facts
undercut a simple additive-noise PCA story while remaining compatible with
trajectory geometry.

**Strongest conflict.** Training is minibatch-stochastic, and noisy current-
versus-prior retention gaps are much larger than clean gaps, so fresh batch
variation plainly affects the basis. No existing artifact separates within-
state covariance from between-state drift covariance or measures Hessian-vector
alignment. The drift account is therefore provisional, not the demonstrated
replacement for denoising.

**Exact proposed estimand.** At each frozen `theta`, use independent batch draws
to estimate `Sigma_batch(theta)` and its top-rank projector `P_noise`. Using
predeclared neighboring saved states and large common clean probes, estimate the
EMA-centered mean-gradient drift projector `P_drift`; separately compare
`d_t` with `H_t delta_theta` by Hessian-vector products. Measure projector
overlap and matched-rank captured energy of the learned `P^-`, then feed
`P_noise g`, `P_drift g` and `P^-g` through the same complete AdamW state with
the input-norm and actual-step-norm controls above. The target is the joint
estimand `(subspace overlap, D, Q_clean, Q_corr, function change)`, not accuracy.

**Falsifier.** Across the fixed panel, `P^-` aligns with and behaves like
`P_noise`, while showing little alignment with measured drift or `H delta`, and
the drift-derived projector fails to reproduce its one-step action. That would
reject drift/curvature as the dominant source of learned orientation at those
states. The converse would weaken, but not alone falsify, semantic denoising.

### H4 — Recursive truncation inertia is a secondary mechanism

**Claim.** Repeated rank truncation prevents individually weak innovations from
accumulating, so early dominant modes persist longer than they would in the
full finite-history covariance. This can stabilize a restrictive subspace and
delay both useful learning and memorization.

**Support.** The exact axis-switch construction delays a rank-one switch from
69 observations in the full covariance to 459 at decay .99. The 20-seed
synthetic study favors wider estimation in every predeclared cell. On common
neural streams, storage 128 improves final matched-rank capture from .988071 to
.999839 of the exact rank-32 optimum in every seed
([iteration 005](../continuation/iteration-005/results.md)). This establishes
estimator inertia/fidelity effects, not learning causality.

**Strongest conflict.** Wider estimation has an adverse noisy learning seed,
slightly worsens clean accuracy in every iteration-003 seed, and does not
consistently improve measured clean-versus-corruption retention. Even ideal
truncation admits exact counterexamples where wider storage has worse
matched-rank capture. The neural fidelity improvement is only about 1.18 points
relative to the rank-32 optimum, which itself contains about 60.41% of total
trace. Better covariance approximation has not been shown to cause better
learning.

### H5 — Self-inclusion is a modifier, not the primary benefit

**Claim.** Updating the covariance with the current gradient makes the delivered
operator adapt to its own input. This raises in-sample retention and weakens the
restriction for novel high-energy directions. It may help optimization by
preventing total rejection of new useful directions, but it is poor evidence
of persistence or denoising and is unlikely to be the main source of the
anti-memorization benefit.

**Mathematical basis.** For an ideal full estimator let
`C(alpha)=C_0+alpha c c^T` and

\[
F_k(\alpha)=\sum_{i=1}^k\lambda_i(C(\alpha))
=\max_{\operatorname{rank}P=k}\operatorname{tr}(P C(\alpha)).
\]

`F_k` is a supremum of affine functions of `alpha`, hence convex. Away from
eigenvalue ties, `F_k'(alpha)=c^T P_alpha c`; convexity makes this retained
innovation energy nondecreasing with its own rank-one weight. Thus elevated
post-update retention of `c` is partly an in-sample fitting identity, not an
independent prediction result. The code projects raw `g`, has `c != g`, uses a
truncated mixed-precision state and may repair it, so monotonicity is not a
blanket theorem for the native recorded quantity.

**Support.** The common-stream replay finds post-observation raw-gradient energy
retention about .841 versus .411 under the prior basis, with every saved
state/width increment positive. Iteration 006 again finds materially larger
current than prior retention, especially under noisy labels. Self-inclusion is
therefore quantitatively large.

**Strongest conflict.** Directly training with the prior basis does not produce
a dependable gain. Lagging is worse on clean data in all primary pairs and the
fresh bundle; noisy effects are adverse in the primary mean but positive in the
fresh bundle. Norm restoration does not settle the result, and both current and
lagged projected arms still suppress corrupted-label fitting. This is the
strongest evidence against self-inclusion as the unique or dominant beneficial
mechanism.

**Exact proposed estimand.** At a common complete state form `u_+` from
`P^+(g)g` and `u_-` from `P^-g`. Compare their native deliveries, then compare
`q u_+` versus `q u_-` at common `q`, and finally raw-direction controls matched
to each realized AdamW data-step norm. Estimate pairwise differences in `G`,
`D`, `Q_clean`, `Q_corr` and function change. Use independent future batches to
measure out-of-sample retention under `P^+` and `P^-`; the current batch's
retention is explicitly an in-sample diagnostic.

**Falsifier.** Once input norm and artificial displacement norm are controlled,
the current-versus-prior clean/corruption differential falls within a
predeclared practical-equivalence margin at adequate precision on independent
probes, while lagged and current whole-policy endpoint anti-memorization also
falls within a predeclared practical-equivalence margin at adequate precision.
Existing iteration-006 evidence moves in this direction, but the exact same-state
mediator test has not been performed.

## 4. What existing studies identify—and what they do not

The [predictable-projection identity](../continuation/predictable-projection-theory.md)
is exact for fixed `P^-` under its conditional assumptions:

\[
\Delta_{MSE}=\lVert(I-P)\mu\rVert^2-
\lVert(I-P)b\rVert^2-\operatorname{tr}((I-P)\Sigma).
\]

It says when projection improves conditional gradient estimation. It does not
say that the resulting AdamW step descends the clean objective, that finite loss
falls, or that a trained model generalizes. The audited finite-support example
improves gradient MSE while reversing expected clean quadratic loss change.
Self-inclusive and norm-restored actions are nonlinear in the current sample,
so even this MSE identity cannot be transferred to them by notation.

## 5. Decision rule for subsequent interpretation

The next mechanistic evidence, if separately authorized, should be interpreted
in this order:

1. Check exact state cloning and distinguish observer order from delivered
   gradient.
2. Report input direction, input norm, live AdamW displacement, decay and
   function change as separate quantities.
3. Use independent clean and corruption probes for `D` and finite losses;
   retain same-batch results only as selected/in-sample diagnostics.
4. Treat conditional MSE, first-order descent, finite loss and long-run
   generalization as four different questions.
5. Aggregate over predeclared states and independent batch bundles seed-first;
   dependent steps are not replications.
6. Preserve null roots, zero norms, adverse states and cross-bundle sign changes.

On present evidence, the defensible causal summary is: **the filter changes what
AdamW is allowed to learn, not simply how much it learns; which exclusions help
is task- and phase-dependent, and the adaptive optimizer transforms that
restriction enough that gradient-space diagnostics alone cannot identify the
learning mechanism.**

Production optimizer code is unchanged. This note adds theory and proposed
estimands only, consistent with the [follow-up scope](mathematical-followup-plan.md).
