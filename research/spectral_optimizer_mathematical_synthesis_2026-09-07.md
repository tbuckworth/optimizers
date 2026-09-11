# What the spectral optimizer is selecting

Codex / Spectral Optimizer Investigation — mathematical synthesis, 7 September 2026.

## High-level assessment

My leading hypothesis is **task-dependent directional regularization, strongly
mediated by Adam**, not a general-purpose separation of signal from noise.
The filter changes which gradient information reaches the optimizer. Under some
noisy-label recipes, that appears to restrict late memorization while retaining
useful early learning. The same restriction can obstruct useful learning on
clean data or other tasks. Which changes survive Adam's historical moments and
coordinatewise scaling remains a central unanswered question.

Confidence is high in the mathematical distinctions below, moderate in this
broad interpretation of the empirical pattern, and low in any uniquely identified
causal mechanism. The stronger claim that leading covariance directions are
intrinsically useful is false in general. Better covariance estimation, better
gradient estimation, better immediate loss, and better eventual generalization
are four different achievements.

This addendum supplies additional reasoning and proposed tests, not new training
results. No experiment is currently running: the one-shot I7 storage check failed
without a retained candidate and is preserved. Its
[failure record](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/native-storage-measurement-failure-review.md)
does not change the scientific evidence. The original report/PDF and optimizer
implementation remain unchanged.

## 1. What the actual code measures

The canonical hard filter observes the flattened **raw batch-mean gradient**,
updates its running mean, estimates a low-rank covariance of the centered
innovation, then projects the same raw gradient before AdamW sees it:

\[
a_t=\beta a_{t-1}+(1-\beta)g_t,\qquad
c_t=g_t-a_t=\beta(g_t-a_{t-1}),\qquad h_t=P_tg_t.
\]

Here \(P_t\) denotes the ideal orthogonal projector. Actual stored columns give
the finite-precision action \(V_t(V_t^Tg_t)\). The first gradient initializes
the mean; the first nonzero innovation initializes covariance without the later
\(1-\beta\) weight; subsequent updates truncate repeatedly. These differ from PCA
on an exact, independently sampled covariance. See [the code](../spectral_filter.py)
and the prior [implementation audit](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-audit.md).

A useful new decomposition conditions on the current parameters, fixed training
labels, past batches and observer state. Write \(g_t=\bar g_t+\xi_t\), with
conditional mean \(\bar g_t\), noise covariance \(\Sigma_t\), and
\(d_t=\bar g_t-a_{t-1}\). Then

\[
\mathbb E[c_tc_t^T\mid\mathcal F]=\beta^2(d_td_t^T+\Sigma_t).
\]

An ideal regular-weight, complete-innovation covariance proposal therefore has
expectation \(\beta C_{t-1}+(1-\beta)\beta^2(d_td_t^T+\Sigma_t)\).
New covariance mass mixes **mean surprise** and **fresh batch variability**.
Mean surprise includes both changes in the objective gradient and noise
remembered by the lagging mean; it is not pure curvature or pure signal.
This describes the ideal proposal, not the native result after residual
rejection, pruning, repair, rounding, startup exceptions or rank truncation.
Expectation does not commute with selecting eigenvectors.

This leads to a cheap proposed diagnostic. At one unchanged state, two independent
batches satisfy

\[
\mathbb E[(g-g')(g-g')^T/2\mid\mathcal F]=\Sigma,\qquad
\mathbb E[(g-a)(g'-a)^T\mid\mathcal F]=dd^T.
\]

We can estimate these **along selected directions**, without a dense covariance.
That distinguishes why a direction has high activity before asking whether it
is useful. See the [covariance derivation](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-followup-covariance.md).

## 2. Covariance magnitude is not a usefulness score

At a fixed state, let \(\mu\) be a clean-objective gradient and write the
corrupted-training gradient as \(g=\mu+b+\xi\). Fixed-label bias \(b\) need not
vanish. For a projector chosen before the fresh batch,

\[
R(P)=\mathbb E\|Pg-\mu\|^2
=\|\mu\|^2+\operatorname{tr}[P(bb^T+\Sigma-\mu\mu^T)].
\]

An oracle restricted to rank-\(r\) orthogonal projectors minimizes this risk with
the leading eigenspace of \(\mu\mu^T-bb^T-\Sigma\), **not gradient covariance**.
This compares mathematical objectives, not deployable rules: the clean target
and nuisance statistics are unknown, and a single-state target does not describe
a changing gradient field.

For a varying useful signal with second moment \(M_s\) and independent noise,
the analogous risk selects \(M_s-\Sigma\), whereas uncentered PCA sees
\(M_s+\Sigma\). Useful variance \(\operatorname{diag}(4,0)\) and noise variance
\(\operatorname{diag}(0,9)\), for example, make rank-one PCA keep pure noise.
This is an analytical counterexample, not a newly sampled experiment.

A favorable special case is useful mean and variation confined to a subspace,
isotropic independent noise, and nonzero signal variation in every desired
direction. Population PCA can identify the subspace, and an independently
evaluated projector removes noise outside it. The current method must
approximate these conditions, not assume them. A constant useful mean may
never enter centered covariance.

Reduced gradient MSE still does not establish a better loss step. For plain
SGD, an unbiased gradient and the same learning rate, projection cannot improve
the first-order clean-descent term beyond \(\|\mu\|^2\). Benefits can concern
curvature/noise costs, biased fitting or future learning instead. The earlier
[predictable-projection analysis](../output/2026-09-06-spectral-optimizer-investigation/continuation/predictable-projection-theory.md)
already gives a counterexample where MSE improves while expected clean loss
worsens. Adam requires a different analysis.

## 3. A better bridge: which functions remain easy to learn?

Consider a fixed-Jacobian square-loss proxy: outputs \(f\), residual \(r=f-y\),
Jacobian \(J\), and loss \(\|r\|^2/(2n)\). Then

\[
g=J^Tr/n,\qquad C_g=J^TC_rJ/n^2,\qquad
\Delta f=-\eta JPJ^Tr/n.
\]

The last equality is exact for a linear model and first-order for a nonlinear
one. Projected SGD replaces the tangent kernel \(JJ^T\) by \(K_P=JPJ^T\), with
\(0\preceq K_P\preceq JJ^T\). It restricts local function-space learning. With
a fixed kernel, small-eigenvalue residual components learn slowly; removed
directions do not learn through that update.

This offers a coherent hypothesis for both outcomes: restriction helps if
fitting wrong labels needs directions that can be suppressed while useful
structure remains accessible; it hurts if useful learning needs those same
excluded directions. It resembles selective slowing or stopping of modes, not
a detector of label truth. Unlike simply stopping training, filtered policies
can continue improving after warmup; the earlier studies measure that too.

The [neural tangent kernel connection](https://arxiv.org/abs/1806.07572) is a
controlled analogy, not a claim that this finite ReLU network has a frozen kernel
throughout training. PCA does not automatically choose the top kernel modes:
if \(J=USV^T\), then \(C_g=VS(U^TC_rU)SV^T/n^2\). Residual covariance must have
suitable alignment. Structured noise and changing representations can alter it.

Fixed diagonal preconditioning after projection instead gives \(JDPJ^T\),
generally nonsymmetric and not a PSD kernel. Actual Adam also has historical
moments and an input-dependent denominator. The simple kernel story supplies
hypotheses and tests, not an AdamW theorem.

## 4. High activity may come from optimization dynamics

The centering transfer function is \(\beta(1-z^{-1})/(1-\beta z^{-1})\):
it rejects a constant component and responds to change. Covariance then averages
outer-product power. It forgets global sign, not all relative coordinate signs
or phases. Alternating gradients can therefore receive high covariance energy
despite zero signed agreement.

A new local calculation makes the curvature connection concrete. In stationary
scalar quadratic SGD, let

\[
g_t=he_t+\xi_t,\quad e_{t+1}=(1-\eta h)e_t-\eta\xi_t,\quad
0<\eta h<2,\quad \operatorname{Var}(\xi_t)=\sigma^2.
\]

Then \(\operatorname{Var}(g_t)=2\sigma^2/(2-\eta h)\). Equal injected noise
can produce different gradient variances as curvature and learning rate change.
Near the oscillatory stability boundary, covariance can be large because
optimization is oscillating. Centering does not remove that high-frequency
component. This is exact for the stated toy system, not a measured property of
the neural runs. The [derivation note](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-followup-main-derivations.md)
gives the full transfer spectrum.

This motivates directional temporal correlations and, if warranted,
Hessian-vector responses after the simpler fixed-state batch-pair decomposition.
Calling the covariance a Fisher matrix, Hessian, or semantic feature covariance
would skip the question we need to answer.

## 5. Why Adam must be part of the mechanism

At the same pre-step Adam state, bias correction can be absorbed into constants
so its data displacement is

\[
u_i(h)=-\eta\frac{A_i+B h_i}{\sqrt{C_i+D h_i^2}+\epsilon}.
\]

Both numerator and denominator change with the delivered gradient. On a first
step with zero moments and negligible epsilon, this is approximately
\(-\eta\operatorname{sign}(h_i)\): changing magnitude need not change the step.
Later, history changes the response. Momentum spans historical gradient spaces;
diagonal scaling can leave even a fixed rotated subspace; decay adds another
displacement. Equal gradient norms do not match update norms, update directions,
function changes or future state.

This is not only hypothetical. In [iteration 004](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-004/summary.json),
scalar gradient attenuation left mean data-step norms close to AdamW's, while
hard filtering produced smaller ones. Yet smaller steps are not a universal
explanation: in [iteration 006](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006/results.md),
clean filtered training had larger mean data-step norm than AdamW and learned
less. These are different trajectories, so neither observation alone identifies
a mediator. They make complete-state comparisons more informative than another
unmatched norm control.

## 6. Ranked hypotheses against the existing evidence

These explanations can coexist. The ranking is a research judgment, not fitted
posterior probabilities. Every proposed falsifier concerns a specified regime
and needs adequate precision around a predeclared meaningful effect size.

| Rank | Hypothesis and confidence | Most informative challenge |
|---:|---|---|
| 1 | **Learned directional regularization**: moderate support as the broad account; exact useful-direction mediation unproven. | Does learned orientation improve the clean/corrupted fitting tradeoff after displacement magnitude is matched? Reproduction by scalar/random controls weakens its necessity. |
| 2 | **Adam history and adaptive scaling contribute materially**: interaction is mathematically certain; empirical contribution unresolved. | Separate input direction, adaptive response, historical motion and decay at the same complete state. |
| 3 | **Covariance tracks mean change/curvature as well as noise**: strong mathematical plausibility, little direct neural identification. | Independent frozen-state batches separate fresh variability from mean surprise; temporal/Hessian probes test the latter's source. |
| 4 | **Truncation causes path-dependent learning delays**: established estimator effect; weak evidence for generalization causality. | Does better approximation change actual clean/corrupted one-step outcomes on a common state, not only covariance capture? |
| 5 | **Self-inclusion mainly enables tracking rather than useful denoising**: large geometry effect measured; beneficial/harmful role unsettled. | Compare current/prior bases with independent probes and magnitude controls. Simply removing self-inclusion has not reliably helped. |

The constraints on this ranking matter:

- [Iteration 004](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-004/results.md)
  found a **+18.81-point noisy endpoint advantage**, but **−2.38 points with
  validation-accuracy selection**. In [iteration 006](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006/results.md),
  the selected comparison instead gained **4.827 points**. These are separate
  three-seed bundles, not a pooled confirmation or universal benefit.
- [Iteration 005](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-005/results.md)
  raised matched-rank capture from **98.8071% to 99.9839% of the rank-32 optimum**
  with wider estimation. That optimum held about **60.41% of total trace**.
  Better covariance fidelity was not a learning result, and retention selectivity
  did not improve at every saved state.
- Lagging harmed clean learning in primary and separate fresh bundles, while
  noisy signs differed. Large self-inclusive retention is not proof of harmful
  contamination. Clean underfitting, adverse width cases and negative transfer
  remain evidence the hypotheses must explain.

## 7. Tests I would prioritize

**First: complete-state one-step comparisons.** Copy the same parameters,
model buffers, observer, Adam moments, counters, schedule and relevant RNG state.
Compare raw/current/previous-basis delivery and reciprocal input-norm controls
on the same batch. Measure actual displacement and independent clean/noisy
probe loss changes. A zero-gradient control distinguishes movement from old
momentum and decay from movement caused by current gradient information.

Add a separate **displacement-matched diagnostic**: take raw and filtered data
displacements from that common state, rescale the raw displacement to the
filtered one's norm, then add identical decay. Also consider the reciprocal
match. These are artificial one-step controls, not automatically implemented I7
arms or ordinary Adam policies. They isolate direction at matched displacement
magnitude more cleanly than input rescaling, which can rotate Adam's output.
Handle zero norms explicitly. Finite probe losses, first-order clean-gradient
dots and logit changes answer different questions; keep all three separate.

**Second: identify the covariance source.** Use independent batch pairs at
unchanged checkpoints to estimate fresh-noise and mean-surprise energy along
retained directions and a prespecified comparison space. This needs projected
statistics, not a dense matrix. Distinguish fixed training-label bias from fresh
minibatch noise. If mean surprise dominates, proceed to gradient-drift/Hessian
diagnostics; do not name it curvature prematurely.

**Only then consider another whole-policy study.** A one-step result cannot
establish long-run mediation, representation learning or generalization. Later
training needs frozen hypotheses, adequate independent bundles, both validation
selectors, untampered test evaluation and discriminating controls—not just
another endpoint win.

The first two proposals fit the scale of the existing small MLP and local
diagnostics, but are not launched. Old model-only checkpoints lack the complete
state for exact Adam/observer counterfactuals. Replacing the consumed I7 attempt
requires an explicit new scope and resource review; this addendum is not that
permission. No paid resources are requested.

## 8. How related work changes the interpretation

| Primary work | Useful connection—and its limit |
|---|---|
| [Gur-Ari, Roberts & Dyer: tiny gradient subspaces](https://arxiv.org/abs/1812.04754) | Low-dimensional Hessian-related gradient trajectories occur in their settings. Centered innovation PCA is not thereby a useful-gradient subspace. |
| [Jacot, Gabriel & Hongler: NTK](https://arxiv.org/abs/1806.07572) | Gives a function-space learning framework. Our kernel derivation is a local/plain-SGD proxy, not a finite-network AdamW theorem. |
| [Arpit et al.: memorization](https://arxiv.org/abs/1706.05394) | Early pattern learning and later noise fitting make selective slowing plausible. Their results do not attribute it to this filter. |
| [Kunstner, Balles & Hennig: empirical Fisher limitations](https://arxiv.org/abs/1905.12558) | Even empirical gradient second moments do not generally supply the Fisher/Hessian interpretation; temporal covariance needs separate justification. |
| [Ghashami et al.: Frequent Directions](https://arxiv.org/abs/1501.01711) | Separates sketch accuracy from downstream use. Its guarantees apply to its shrinkage algorithm, not this recursively truncated EMA. |
| [Zhao et al.: GaLore](https://arxiv.org/abs/2403.03507) | Low-rank gradient projection can target optimizer-state memory. Its per-matrix geometry/state handling differs from this temporal filter. |

[Wilson et al.](https://arxiv.org/abs/1705.08292) motivate taking the base optimizer
seriously: adaptive and nonadaptive methods can reach different solutions.
“Spectral bias” in [Rahaman et al.](https://arxiv.org/abs/1806.08734) concerns
Fourier frequencies of the learned function, not this covariance spectrum.
These distinctions are not a novelty or SOTA claim. The
[literature notes and bibliography](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-followup-literature.md)
record source identities and scope.

## Bottom line

The useful question is no longer “does PCA keep the good gradient?” It is:
**which task-relevant function changes become easier or harder after a
history-dependent gradient filter interacts with Adam, and when does that
tradeoff stop memorization without stopping useful learning?**

The derivations make this question measurable; they do not settle it. The next
decisive evidence would connect covariance source, actual updates and independent
loss changes at the same complete states.

Supporting work: [main derivations](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-followup-main-derivations.md),
[covariance derivations](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-followup-covariance.md),
[independent hypothesis challenge](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-followup-hypotheses.md).

### Subsequent focused follow-up: label noise and retention identifiability

The [label-noise follow-up](spectral_label_noise_identifiability_2026-09-07.md)
derives the fresh-label covariance spectrum and a sharp geometric bound on
clean/residual retention gaps. Its post-hoc calculation covers all twelve
retained iteration-005 states, not a new experiment. Strong opposition limits
selectivity but leaves substantial ideal-projector headroom here; it cannot
alone explain the observed small gaps. Fresh label noise can also have
curvature-like structure after the model Jacobian, so alignment with curvature
is not sufficient evidence of useful signal. The missing soft-target gradient
requires a separately scoped measurement; the existing residual is not pure noise.
