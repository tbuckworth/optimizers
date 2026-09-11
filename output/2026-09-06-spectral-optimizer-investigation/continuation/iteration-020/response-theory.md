# I20 response and finite-initialization theory

8 September 2026. **Prospective algebra for a post-hoc saved-stream
discriminator; no new stochastic or neural evidence.** This note reviews a
candidate response family and covariance initialization. It does not read or
regenerate scientific arrays, select an I20 winner, or change the unfavorable
I19 native-risk result.

## Definitions

Write `Q_t=I-P_t`, where each `P_t` is an orthoprojector that may be any causal,
data-dependent function of the saved observations. The I19
constant-preserving (CP) response is

\[
d_t^{\rm CP}=\rho P_t d_{t-1}^{\rm CP}+(1-\rho)P_tg_t+Q_t\mu_t,
\qquad
\mu_t=\beta\mu_{t-1}+(1-\beta)g_t.
\]

The proposed two-sided smoother is

\[
r_t=B_t r_{t-1}+(I-B_t)g_t,
\qquad B_t=\rho P_t+\gamma Q_t.
\]

Every output is initialized to `g_1`. The response grid
`rho in {.9,rho_*}` and `gamma in {.99,.9}`, with

\[
\rho_*=(2.1-\sqrt{.41})/2=(21-\sqrt{41})/20,
\]

is small and interpretable. In particular, the `.9/.9` cell is an exact
direction-blind control rather than another adaptive spatial method.

## Constant preservation and contraction

For a constant input `g_t=c`, the common initialization gives `mu_t=c` and,
by direct induction, both `d_t^{CP}=c` and `r_t=c` for every projector
sequence. No predictability, commutation of successive projectors, or fixed
rank is needed for this pathwise statement.

For the two-sided smoother, conditional on a realized projector sequence, an
error from the constant satisfies

\[
e_t=B_te_{t-1},\qquad
\lVert e_t\rVert_2^2=\rho^2\lVert P_te_{t-1}\rVert_2^2
 +\gamma^2\lVert Q_te_{t-1}\rVert_2^2.
\]

Thus it is a Euclidean contraction whenever
`max(|rho|,|gamma|)<1`, even when the projectors rotate. This is a conditional
response-state claim: if changing a response state were also allowed to change
the estimator and hence `P_t`, the shared-projector comparison would no longer
apply. CP also contracts deviations at rate at most `|rho|` on a constant
stream when its mean is initialized at the constant. With an independently
misinitialized mean, CP is a coupled forced system,
`mu_t-c=beta(mu_(t-1)-c)` and
`d_t-c=rho P_t(d_(t-1)-c)+Q_t(mu_t-c)`, not a one-state contraction.

When `rho=gamma=.9`,

\[
B_t=.9(P_t+Q_t)=.9I,
\]

so the smoother is exactly the isotropic `.9` EMA at every step, independent
of the direction estimator. With the same `g_1` initialization its complete
output array, not merely its stationary risk, must equal that control. This is
the strongest cheap implementation check in the response factorial.

## Fixed and moving projectors

If `P_t=P` is fixed and `gamma=beta`, then CP and the two-sided smoother are
identical. Indeed, CP gives `Qd_t^{CP}=Qmu_t`; starting from `d_1=mu_1=g_1`,
induction yields

\[
d_t^{\rm CP}=(\rho P+\beta Q)d_{t-1}^{\rm CP}
 +[(1-\rho)P+(1-\beta)Q]g_t.
\]

The equivalence generally fails for a moving projector. Let
`delta_t=r_t-d_t^{CP}` and take `gamma=beta`. Exact subtraction gives

\[
\boxed{\delta_t=B_t\delta_{t-1}
 +\beta Q_t(d_{t-1}^{\rm CP}-\mu_{t-1}).}
\]

The extra term is a response-transport term. For fixed `P`, its projected
factor vanishes because `Qd_{t-1}^{CP}=Qmu_{t-1}`. For moving `P_t`, the old CP
state and old mean can differ in a direction newly assigned to the current
complement. Consequently a CP-versus-smoother contrast at `gamma=beta` tests
the handling of moving actions, not just a notational rewrite or a change in
steady fixed-direction decay.

## Why shortening complement memory changes angle sensitivity

For a fixed, data-independent rank-one projector with squared useful-axis
alignment `a`, the stationary random-walk risk of the two-sided smoother is

\[
J(a;\rho,\gamma)=Q_1[a v(\rho)+(1-a)v(\gamma)]
 +[aR_1+(1-a)R_2]n(\rho)
 +[(1-a)R_1+aR_2]n(\gamma),
\]

where `v(q)=q^2/(1-q^2)` and `n(q)=(1-q)/(1+q)`. It remains affine in `a`.
For the I19 strong cell `Q_1=.1`, `R=(1,4)`:

| rho | gamma | useful risk | nuisance risk | alignment to beat common-LTI bound |
|---:|---:|---:|---:|---:|
| .9 | .99 | .499048 | 5.140677 | >.965567 |
| rho_* | .99 | .290257 | 5.554846 | >.929982 |
| .9 | .9 | .689474 | .689474 | impossible |
| rho_* | .9 | .480683 | 1.103642 | >.713963 |

The bound here is the previously reviewed `.658872` optimum for the admissible
deterministic shared unit-gain causal LTI class. Shortening complement memory
from `.99` to `.9` makes a misrouted changing component much less costly, so
the fixed-angle requirement drops sharply for `rho_*`. The price is worse
denoising in the correctly identified nuisance complement: useful-oracle risk
rises from `.290257` to `.480683`. When both decays are `.9`, alignment has no
effect and the stationary risk is the isotropic-EMA value `.689474`, which is
above the common-LTI optimum.

These numbers are sensitivity diagnostics only. The native I19 alignment, or
the alignment of any moving saved estimator, must never be inserted into this
fixed, data-independent formula. Such a projector is correlated with current
observations and response history. The table neither predicts native risk nor
identifies direction error or complement lag as a causal mediation fraction.

## Finite EW covariance initialization

I19's saved full moment is not the usual zero-initialized weighted
accumulator. With `z_t=g_t-mu_t`, it stays zero until the first nonzero
residual, assigns that first outer product unit weight,
`M_t=z_tz_t^T`, and only thereafter applies

\[
M_t=\lambda M_{t-1}+(1-\lambda)z_tz_t^T
\quad (\lambda=.99).
\]

This legacy convention is a useful immutable reproduction arm, but its first
nonzero sample has much greater early weight than in the standard recursion

\[
M_t^{(\lambda)}=\lambda M_{t-1}^{(\lambda)}
 +(1-\lambda)z_tz_t^T,\qquad M_0^{(\lambda)}=0.
\]

After `t` global updates,

\[
M_t^{(\lambda)}=(1-\lambda)\sum_{j=1}^t
 \lambda^{t-j}z_jz_j^T,
\qquad w_t=1-\lambda^t.
\]

The leading eigenvector is unchanged by division by positive `w_t`, but an
absolute or relative availability rule is not scale invariant in the same
way. Apply the gap threshold to
`M_t^{(lambda)}/w_t` (equivalently, rescale the threshold algebraically), and
record that global `t` includes the zero first residual caused by
`mu_1=g_1`. Do not silently restart the weight clock at the first nonzero
residual. Ties or absent directions and their response fallback must be frozen
explicitly.

For example, the I19-style relative rule with tolerance `tau` becomes

\[
\operatorname{gap}(M_t)>\tau\max\{w_t,
 |\lambda_{\max}(M_t)|\},
\]

which is exactly the rule obtained by applying
`gap > tau max(1,abs(lambda_max))` to the normalized matrix.

The estimator roster should therefore retain the original saved native action
and the original saved full direction wherever it was available, and add
**both** standard-weighted `.99` and standard-weighted `.999` sequences using
the same saved residuals, zero initialization, normalization, eigengap rule,
and a common missing-direction fallback. The standard `.99` arm is necessary:
comparing standard `.999` only with original saved full `.99` changes memory
and initialization at once. Any claimed longer-memory effect must be the
paired standard `.999` minus standard `.99` contrast. Original saved full
versus standard `.99` isolates the finite initialization change more directly,
subject to identical eigendecomposition and availability conventions.

There is an additional I19-specific fallback hazard. The saved native action
was `I` when its basis was absent, whereas the saved full-moment action array
was zero when its eigengap was unavailable. A direct response comparison
between those arrays therefore includes estimator absence and different
fallback routing. Reconstructing the full-legacy direction with the same
frozen identity fallback as new estimators removes that particular mismatch,
but is no longer byte-for-byte replay of the saved `full_action` array. Report
absent counts and the convention explicitly; do not label a native-versus-full
contrast as rank truncation alone.

## Steelman and interpretation boundary

The best steelman is a crossed saved-stream analysis:

- hold each direction sequence fixed and compare all four `(rho,gamma)`
  responses, including the exact `.9` EMA identity;
- hold the response fixed and compare original saved native, original saved
  full, standard `.99`, standard `.999`, and fixed useful/nuisance oracles;
- retain all original scalar and model-based controls, initialize every output
  at `g_1`, and report whole horizon first with startup, transition and late
  windows separately.

If standard `.999` improves direction metrics over same-initialization `.99`,
while `.9` complement memory improves risk for the *same* moving direction,
the factorial would support two distinct engineering bottlenecks: finite
direction estimation and brittle directional response. The fixed useful
oracle would show available response headroom, while the fixed nuisance oracle
would expose the adverse direction case. That is the strongest interpretation;
it is still not an additive causal decomposition or proof of mediation.

Main confounds remain finite startup, longer-memory lag after a true direction
change, current-observation self-inclusion, action/response interaction,
rank-truncation versus full-moment recurrence differences, missing-basis
fallback, and the unequal opportunity created by a larger post-outcome policy
grid. Reusing the 32 I19 seeds gives paired counterfactual evidence, not fresh
replication. Any apparent remedy is exploratory and requires a frozen fresh
bundle, including a direction-change adverse case, before an efficacy claim.

The accompanying exact checker uses only deterministic rational fixtures. It
verifies constant preservation, contraction, fixed-projector equivalence, the
moving-projector difference identity, the isotropic `.9` identity, standard
versus legacy moment initialization, weight-mass normalization, and the
defining polynomial for `rho_*`.
