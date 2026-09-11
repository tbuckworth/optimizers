# Primary-literature addendum: temporal gradient covariance

**Evidence cut: 2026-09-07.** This is a bounded primary-source comparison, not evidence that the current filter is novel, convergent, or superior to AdamW. The local source of truth for the estimator and counterexamples is [mathematical-audit.md](mathematical-audit.md). I read the primary arXiv records/available full text linked below. An attempted Semantic Scholar API search for gradient covariance, Fisher, noisy labels, and low-rank projection returned HTTP 429 for each query; it supplied no evidence. arXiv's API returned 30 records for `Frequent Directions`; the original paper was selected.

## Closest sources: claims and boundary

## Frozen-Jacobian/kernel connection

This is **algebra under a stated frozen-Jacobian square-loss proxy**, not an empirical claim about the finite network. Let residuals be `r`, outputs have Jacobian `J`, and use loss `||r||²/(2n)` so `g=J^T r/n`. Then

\[
C_g=J^T C_r J/n^2.
\]

For plain projected SGD with an orthogonal parameter projector `P`, output dynamics use the effective kernel

\[
K_P=JPJ^T \preceq JJ^T,
\]

because `I-P` is positive semidefinite. PCA of `C_g` aligns with leading Jacobian/NTK modes only if residual covariance is sufficiently isotropic or otherwise aligned. It need not do so for structured label noise, drift, or the implementation's temporal centering. With a fixed positive diagonal preconditioner `D`, `JDPJ^T` is generally nonsymmetric, so it is not an ordinary PSD kernel; Adam's evolving diagonal state makes the proxy still less literal.

**Proposed discriminator (not run):** at frozen checkpoints, estimate independent clean and corrupted residual covariances and compare their overlaps with `JV_t` (or Jacobian-vector products), alongside raw-gradient and actual-AdamW-update overlap. Precisely estimated lack of excess alignment over rank-matched random bases while the endpoint difference remains would weaken the NTK-mode explanation at those states.

All suggested null-result discriminators below require a prespecified meaningful
effect threshold and adequate precision. Failure to reject a difference is not
evidence of equivalence or a general falsification of a mechanism.

## Reusable connections and falsifiers

### Streaming sketch versus the actual recursive estimator

FD supplies a theorem for its own shrinkage algorithm, not for

\[
C_t\approx {\cal T}_r\{\beta C_{t-1}+(1-\beta)(g_t-m_t)(g_t-m_t)^T\},\qquad m_t=\beta m_{t-1}+(1-\beta)g_t.
\]

The local audit's rank-one counterexample shows repeated truncation can discard
an emerging direction even after it dominates untruncated EMA covariance.
The estimator-fidelity comparison is **already completed** in the
[common-stream replay](../continuation/iteration-005/results.md): wider storage
improved matched-rank capture but supplied no new learning outcome. Do not rerun
it. The remaining **proposed, unrun utility test** asks whether those estimator
differences change independent clean/corrupted one-step outcomes at the same
complete state. A precisely negligible utility difference despite substantial
estimator improvement would weaken truncation as a local learning mediator;
it would not negate the already established estimator effect.

### Tiny gradient subspace versus useful innovation subspace

Locally, `g_(t+1)-g_t` can contain curvature times optimizer motion, batch noise, and model drift. Centering also omits a strictly constant mean direction from covariance accumulation: the audit's `g_t=(1,xi_t)` retains varying nuisance and discards the constant useful component. **Proposed discriminator (not run):** compare independent clean-gradient, innovation, raw-gradient, and Hessian-subspace retained energy. **Falsifier of useful-subspace selection:** clean retention is no better than rank-matched random directions while a gain persists after actual-step matching.

### Noise-learning endpoint versus selective mechanism

The noisy-label papers support testing late memorization, not a particular subspace mechanism. The remaining alternatives are lower effective update, altered AdamW/decay balance, or selective directional change. **Proposed discriminator (not run):** log clean/corrupted probe energy in `V_t`, decay-subtracted actual update, and an actual-step-matched scalar control. **Falsifier of selective suppression:** clean and corrupted probe effects are indistinguishable and the scalar control reproduces the endpoint.

### Projection is not the AdamW update

Wilson et al. gives external motivation; the local audit gives direct algebra: diagonal preconditioning generally does not commute with a rotated `P`, while momentum across changing bases and decoupled decay add off-subspace terms. **Proposed discriminator (not run):** report `||(I-P_t) Delta-theta_t||/||Delta-theta_t||`, preconditioned-step norm, and clean-loss directional change; compare matched SGD and an explicitly update-projected policy as distinct algorithms. Reproducing the effect under matched SGD with negligible leakage would weaken the claim that Adam interaction is necessary, not rule out its contribution in the AdamW runs.

## Bottom line

The literature supports separations, not a universal coherence story: streaming-compression quality differs from learning utility; low-dimensional gradients differ from centered temporal covariance; Fisher differs from curvature; noise memorization is an outcome, not a PCA mechanism; and projected gradients differ from adaptive updates. These motivate independent-probe, estimation-width, Jacobian/residual, and actual-update measurements. They do not resolve contradictory seed bundles or authorize another experiment.
