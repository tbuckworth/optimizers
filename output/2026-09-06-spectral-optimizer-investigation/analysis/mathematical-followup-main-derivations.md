# Additional derivations for the conceptual follow-up

Codex, 7 September 2026. Theory only; no new numerical experiment. These
derivations supplement, rather than replace, the earlier mathematical audit.

## 1. A local function-space interpretation

For a fixed differentiable model linearization with output vector f in R^n,
Jacobian J in R^(n by p), residual r=f-y and square loss ||r||²/(2n), the
parameter gradient is g=J^T r/n. For a fixed orthogonal parameter projector P,
projected SGD gives

    delta_theta = -eta P J^T r/n,
    delta_f = -eta J P J^T r/n.

The second equality is exact for a linear model and first-order for a nonlinear
model. The effective tangent kernel is K_P=J P J^T. Since 0 <= P <= I,
0 <= K_P <= J J^T in PSD order and rank(K_P) <= rank(P). This establishes a
restriction of local function-space learning under these assumptions, not a
theorem that the current AdamW implementation has a low-dimensional trajectory.

For fixed J and time-varying residuals, temporal central covariance transforms
as C_g=J^T C_r J/n² (before truncation and startup effects). With J=U S V^T,

    C_g = V S (U^T C_r U) S V^T/n².

Thus covariance PCA agrees with the top right singular vectors of J only under
additional conditions, for example isotropic residual covariance on its left
singular space. It does not automatically select the leading tangent-kernel or
curvature modes. Residual variation, corrupted labels, current representations
and trajectory history all affect the answer. The constant residual mean is
not itself represented by C_r.

This gives a conditional account of anti-memorization: if useful target
components are learned through the retained function directions while fitting
idiosyncratic labels requires other directions, restriction can preserve early
competence. If useful components need those other directions, the same
restriction underfits. With fixed kernel eigenvalue kappa, a mode's residual
under linear gradient descent is multiplied by (1-eta*kappa/n)^t; weak or
removed modes learn slowly or not at all. Assume 0 <= eta*kappa/n < 1 for the
simple monotone interpretation. This is an analogy to learning-rate-dependent
regularization, not an assertion that the neural optimizer remains in this
fixed-kernel regime or that low kernel eigenvalues identify label noise.

With a fixed positive diagonal preconditioner D after projection, the analogous
matrix is J D P J^T, generally nonsymmetric and not PSD. The symmetric form
J D^(1/2) P D^(1/2) J^T would be PSD, but describes a different update rule.
Actual Adam additionally has input-dependent D and historical momentum.

## 2. The covariance objective is different from gradient-estimation risk

Condition on a fixed dataset, parameters, history and clean target gradient mu.
Write the next independent batch gradient g=mu+b+xi, E[xi]=0, Cov(xi)=Sigma,
where b is fixed corruption bias, not assumed zero. For a predictable rank-r
orthogonal projector P,

    R(P) = E||P g-mu||²
         = ||mu||² + tr[P (b b^T + Sigma - mu mu^T)].

This follows from P²=P and expansion of ||P(mu+b)-mu||². Under exactly rank r,
the oracle minimizer keeps the leading eigenvectors of

    A = mu mu^T - b b^T - Sigma,

not the leading eigenvectors of the measured gradient covariance. If rank is
at most r, only positive-eigenvalue modes need be kept (zero modes are neutral).
For a fixed known mu, A has at most one strictly positive eigenvalue. This is
not a practical rank recommendation: it illustrates the difference between a
single-state oracle target and a changing unknown gradient field. A constant
estimator equal to known mu would bypass this restricted estimator family.

For a varying useful signal s independent of zero-mean noise, the analogous
risk for estimating each s selects leading eigenvectors of E[s s^T]-Sigma.
Uncentered PCA instead uses E[s s^T]+Sigma; centered PCA removes the signal
mean as well. Signal-to-noise ordering and variance ordering need not agree.

A favorable special case is g=U a+xi, with all useful mean/variation in span(U),
isotropic independent noise Cov(xi)=sigma² I, and positive signal variance in
each desired U direction. Population centered covariance identifies U when its
signal eigenvalues stand above the isotropic noise floor. A held-out projector
onto U then retains the useful signal and only r*sigma² of p*sigma² noise power.
The code is not given U or an independent population covariance; current-sample
inclusion, anisotropic noise, finite history, truncation and drifting parameters
all weaken this special-case analogy.

Neither risk identity proves clean-loss improvement. At a quadratic clean
objective with gradient mu and Hessian H, an SGD step has exact expected change

    -eta mu^T P(mu+b)
    + eta²/2 * [(mu+b)^T P H P(mu+b) + tr(P H P Sigma)].

For a smooth nonquadratic loss, add the Taylor remainder. For b=0, projection
cannot improve the first-order benefit over raw SGD at the same eta, since
mu^T P mu <= ||mu||². Benefits can concern the second-order noise/curvature cost,
biased targets, future states or generalization instead. This agrees with the
existing predictable-projection note and does not turn MSE into a loss metric.

## 3. A stationary quadratic exposes a curvature confound

Consider one mode of plain SGD near a quadratic optimum:

    g_t = h e_t + xi_t,
    e_(t+1) = a e_t - eta xi_t,  a=1-eta*h,

where h>0, xi_t are independent with mean zero and variance sigma², and
0<eta*h<2. In stationarity e_t is independent of the current xi_t, giving

    Var(e) = eta² sigma²/(1-a²)
           = eta sigma²/[h(2-eta*h)],
    Var(g) = h² Var(e)+sigma² = 2 sigma²/(2-eta*h).

Thus identical injected noise can generate very different gradient variances
depending on curvature and step size; variance diverges near the stability
boundary. This is a stationary toy calculation for unfiltered SGD, not a fit
to the neural trajectories or a theorem about Adam.

Let L denote one-step delay, not loss. The stationary transfer function is

    g = [(1-L)/(1-a L)] xi.

Combining with the actual centering filter's steady-state transfer function
beta*(1-L)/(1-beta L) gives innovation power spectral density

    S_c(omega) = sigma² beta² |1-exp(-i omega)|^4
                / (|1-beta exp(-i omega)|² |1-a exp(-i omega)|²).

This is normalized so white xi has constant density sigma² and variance is
the integral divided by 2*pi. It shows why high covariance energy can indicate
optimization-driven oscillation rather than statistically reliable structure.
Centering does not remove the near-stability-boundary high-frequency component.
The practical question is how much of the neural covariance comes from batch
variation versus changing conditional means; this derivation does not answer it.

## 4. The exact one-step Adam map explains why norm controls are insufficient

Hold the pre-step first/second moment states, step count and hyperparameters
fixed, and let h be the delivered gradient. Absorb bias correction into

    A_i = beta1*m_previous_i/(1-beta1^t),
    B   = (1-beta1)/(1-beta1^t),
    C_i = beta2*v_previous_i/(1-beta2^t),
    D   = (1-beta2)/(1-beta2^t).

For ordinary Adam without AMSGrad, the data displacement is

    u_i(h) = -eta*(A_i+B*h_i)/(sqrt(C_i+D*h_i²)+epsilon).

AdamW adds its separate -eta*weight_decay*theta_i term. At positive
s_i=sqrt(C_i+D*h_i²), the exact partial derivative is

    du_i/dh_i = -eta*[B/(s_i+epsilon)
       -(A_i+B*h_i)*D*h_i/(s_i*(s_i+epsilon)²)].

With epsilon=0 this reduces to

    -eta*(B*C_i-D*A_i*h_i)/(C_i+D*h_i²)^(3/2).

The map is not a common linear preconditioner across counterfactual gradients.
On the first step from zero moments and with epsilon=0, u_i=-eta*sign(h_i)
for nonzero h_i: amplitude changes within the same sign pattern disappear.
Later, history and denominator changes matter and local response can even
change sign. This does not assert that such cases are common in the neural runs.

Therefore a useful causal comparison must copy the same complete pre-step state,
apply each candidate h, and compare actual displacements and independent-probe
losses. Equal pre-Adam gradient norms do not supply that control. A zero delivered
gradient still allows historical momentum and weight decay to move parameters.
Single-step comparisons answer a local counterfactual question, not the eventual
outcome of retraining with different state histories.

## 5. Small covariance regret need not mean similar useful action

Let P be the exact leading-r covariance projector, Q another rank-r orthogonal
projector and gamma=lambda_r-lambda_(r+1)>0. Covariance capture regret obeys

    R = tr[(P-Q)C] >= gamma*(r-tr(PQ))
      = gamma/2 * ||P-Q||_F².

This follows by expanding Q's diagonal overlaps in C's eigenbasis, pairing
missed mass above the boundary with retained mass below it. Only a substantial
eigengap makes small regret force a nearby subspace. Moreover useful-gradient
energy depends on mu^T P mu, not tr(PC); covariance fidelity alone does not
control its semantics. The measured width32 capture near 99% of the rank32
optimum is consequently evidence about a covariance objective, not a theorem
that its difference from the wider sketch is irrelevant to learning.

All proposed diagnostics and interventions remain unexecuted. Preserve the
consumed I7 attempt; mathematical analysis confers no replacement launch authority.
