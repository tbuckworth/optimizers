# What the implemented centered EMA covariance can and cannot mean

7 September 2026. This is a theoretical note about the current implementation,
not a new experiment, an optimizer change, or a novelty claim. The derivations
are elementary consequences of conditional second moments and linear systems.
They extend rather than replace the earlier
[mathematical audit](mathematical-audit.md), the
[ideal truncation analysis](../continuation/ideal-truncation-error-theory.md),
and the [predictable-projection analysis](../continuation/predictable-projection-theory.md).

## 1. The exact one-step estimand separates fresh noise from mean surprise

Write `q=1-beta`. Immediately before batch `t`, condition on a history
`F_(t-1)` containing the current parameters, running mean, low-rank observer,
optimizer state, and all previous batches. Assume only that the new batch has

\[
g_t=\bar g_t+\xi_t,\qquad
E[\xi_t\mid F_{t-1}]=0,\qquad
E[\xi_t\xi_t^T\mid F_{t-1}]=\Sigma_t.
\]

Here `bar g_t` is the conditional mean gradient of the current full finite
training objective under the specified next-batch sampling scheme. It need not
be the clean gradient. The code updates the mean before centering:

\[
a_t=\beta a_{t-1}+qg_t,\qquad
c_t=g_t-a_t=\beta(g_t-a_{t-1}).
\]

Define the predictable mean surprise `d_t=bar g_t-a_(t-1)`. In ideal arithmetic,
an already initialized, regular-weight observer that incorporates the complete
innovation would propose

\[
B_t=\beta C_{t-1}+q c_tc_t^T.
\]

Therefore the following conditional identity is exact for that full-innovation
proposal:

\[
E[B_t\mid F_{t-1}]
=\beta C_{t-1}+q\beta^2(d_td_t^T+\Sigma_t). \tag{1}
\]

This is the cleanest interpretation of the actual update. Its expected new
mass is the sum of:

1. squared **mean surprise** relative to the observer's lagging mean; and
2. covariance of the **fresh conditional batch noise**.

The underlying conditional innovation moment
`E[c_t c_t^T | F_(t-1)]=beta^2(d_t d_t^T+Sigma_t)` is exact under the stated
probability model. The realized outer product also contains the zero-mean cross terms
`beta^2(d_t xi_t^T+xi_t d_t^T)`. At finite memory these can materially rotate a
leading subspace even though they vanish in (1). This identity deliberately
excludes startup, native residual rejection, eigenvalue pruning, representation
repair, and floating-point roundoff. Those operations can make the literal
stable-path proposal differ from `B_t`; residual rejection can even introduce
signed approximation error. After spectral truncation,
`E[T_k(B_t)]` is generally not `T_k(E[B_t])`, so (1) is neither a literal
production-state recurrence nor an unbiasedness claim for the retained
eigenspace.

The term `d_t` is not synonymous with parameter drift. Exactly,

\[
d_t=(\bar g_t-\bar g_{t-1})+(\bar g_{t-1}-a_{t-1}). \tag{2}
\]

The first term is drift of the conditional full-training-objective mean; locally it is
approximately `H_(t-1) Delta theta_(t-1)`. The second is observer lag and
contains remembered noise from earlier batches. It is predictable at step `t`
but random across training histories. Thus a single observed `c_t c_t^T`
cannot identify curvature-driven drift and sampling noise.

There is a direct non-training diagnostic. At a frozen checkpoint and observer
state, two independent batches satisfy

\[
E[(g_t-g'_t)(g_t-g'_t)^T/2\mid F_{t-1}]=\Sigma_t.
\]

Several independent probe batches can estimate both `Sigma_t` and `d_t`, making
the two terms in (1) separately measurable. If independent-example assumptions
are adequate, the fresh-noise term should scale approximately as inverse batch
size while the fixed-state mean-surprise term should not. Failure of that
scaling would point to augmentation dependence, sampling without replacement,
or an invalid frozen-state approximation rather than semantic covariance.

### Implementation coefficients and startup

These coefficients match [`spectral_filter.py`](../../../spectral_filter.py):
both stable and legacy paths update `a_t` with `q`, form the post-update
innovation `c_t`, decay old represented variances by `beta`, and give later
outer products weight `q`. Since `c_t=beta(g_t-a_(t-1))`, the regular
observation coefficient on the pre-update residual is `q beta^2`, not `q`.

The first gradient sets the mean and has zero innovation. When the first
nonzero innovation later arrives while `V is None`, the code initializes
`C=c_tc_t^T` rather than `q c_tc_t^T`. This one observation is consequently
overweighted by `1/q` relative to later innovations (100 at the default
`beta=.99`). Any interpretation of early eigenvalues as stationary covariance
must exclude or model this transient.

## 2. The filter is a high-pass detector followed by a low-pass power memory

After transients, with lag operator `L`,

\[
c_t=\frac{\beta(1-L)}{1-\beta L}g_t
=\beta\sum_{j\ge 0}\beta^j(g_{t-j}-g_{t-j-1}). \tag{3}
\]

The audit already records this high-pass transfer function. A useful further
consequence is that the complete covariance observer resembles an
**envelope/power detector**: it high-pass filters the signed gradient, takes an
outer product (removing a sample's global sign), and then low-pass averages that
matrix-valued power with the same decay. Relative coordinate signs and phases
remain in cross-covariances, and smoothing does not erase every temporal phase.
The scalar-direction case rewards persistent *energy of change*, not persistent
pointing.

For `g_t=A cos(omega t)u` in one fixed direction and discrete frequency
`0<omega<pi`, the time-averaged regular covariance contribution is proportional
to

\[
\frac{A^2}{2}\left|\frac{\beta(1-e^{-i\omega})}
{1-\beta e^{-i\omega}}\right|^2 uu^T. \tag{4}
\]

A constant coefficient contributes zero after transients, while alternation
near `omega=pi` is passed almost at full amplitude and becomes positive power.
At the excluded Nyquist endpoint `omega=pi`, `cos^2(omega t)=1`, so the factor
`A^2/2` in (4) is instead `A^2`.
For `beta=.99`, the high-pass knee is approximately
`omega=(1-beta)/sqrt(beta)=.01005` radians per step, corresponding to a period
near 625 steps. The subsequent covariance EMA has an e-folding time near 100
steps. These are soft frequency and envelope scales, not a 100-step hard window
or a rank bound.

The common EMA effective-sample-size formula `(1+beta)/(1-beta)` applies to
independent outer-product observations. Innovations generated by (3) are
generally correlated, especially along an endogenous optimization trajectory;
199 should not be treated as the effective number of independent covariance
samples without an autocorrelation calculation.

### A local quadratic connection: covariance can measure stability margin

Consider one Hessian mode of a quadratic trained by plain SGD:

\[
g_t=h e_t+\xi_t,\qquad
e_{t+1}=(1-\eta h)e_t-\eta\xi_t,
\]

where `xi_t` is iid with variance `sigma^2` and
`a=1-eta h` satisfies `|a|<1`. In stationarity,

\[
g_t=\frac{1-L}{1-aL}\xi_t,\qquad
\operatorname{Var}(g_t)=\frac{2\sigma^2}{2-\eta h}.
\]

Combining this optimizer feedback with (3) gives the innovation spectrum

\[
S_c(\omega)=
\frac{\beta^2|1-e^{-i\omega}|^4}
{|1-\beta e^{-i\omega}|^2|1-ae^{-i\omega}|^2}\sigma^2. \tag{5}
\]

As `eta h` approaches 2 from below, `a` approaches -1 and the optimizer develops
a large alternating component near `omega=pi`; (5) amplifies precisely that
region. Thus a leading centered-covariance mode can identify a stiff direction
near the optimizer's oscillatory stability boundary even when its exogenous
batch-noise variance is no larger and its clean utility is no better. This is a
theorem for the stated scalar SGD model, not for AdamW. Its falsifiable neural
analogue is that leading covariance modes should sometimes have strongly
negative lag-one gradient autocorrelation and large Hessian-update response,
and their energy should move systematically with learning rate. Absence of
these associations would weaken the stability-margin explanation.

## 3. Rank truncation changes memory into an admission-and-survival process

Without truncation, after the first accepted nonzero innovation `c_j`, the code
would represent

\[
C_t=\beta^{t-j}c_jc_j^T
+q\sum_{s=j+1}^t\beta^{t-s}c_sc_s^T. \tag{6}
\]

At a full rank cap, this simple history ceases to hold. An exact special case
shows the relevant mechanism. Suppose the stored basis is orthonormal with
eigenvalues `lambda_1 >= ... >= lambda_k`, and a new `c_t` is orthogonal to its
span. Before configured tolerances, its candidate eigenvalue is `q||c_t||^2`,
while the weakest old candidate is `beta lambda_k`. The new direction is
guaranteed to be admitted when

\[
\mathcal L_t:=\frac{q\|c_t\|^2}{\beta\lambda_k}>1. \tag{7}
\]

It is rejected when `mathcal L_t<1`; at exact equality the outcome depends on
the eigensolver's tie basis and truncation ordering. If it loses, its
contribution is discarded completely and cannot combine with
the next observation unless storage is available then. Repeated subthreshold
evidence is therefore forgotten between competitions. If the incumbent gets no
new energy, decay eventually lowers its barrier, but the previously discarded
mass is not recovered. The initialization exception makes the earliest
incumbent barrier especially large: a regular orthogonal candidate initially
needs roughly `sqrt(beta/q)` times the incumbent innovation norm to displace it,
about 9.95 times at `.99`.

Equation (7) is exact only for an orthogonal candidate. With an in-span
component, the small eigensystem rotates and there is no scalar admission rule.
It nevertheless suggests useful diagnostics: record perpendicular innovation
energy, the pre-update smallest stored eigenvalue, basis turnover, and runs of
similar rejected residuals. A genuine admission-barrier mechanism predicts a
sharp rise in turnover near leverage one, delayed entry of repeated
subthreshold directions, and relief from wider estimation storage specifically
for those directions. No such relationship would weaken this mechanism.

This also explains why wider recursive storage has no monotonic semantic or
even matched-rank capture guarantee. A wider sketch remembers more incumbents
and can raise the barrier against a genuinely emerging direction; the exact
three-axis construction in the
[ideal truncation note](../continuation/ideal-truncation-error-theory.md) proves
that possibility. Stored eigenvalues are survivor statistics of this process,
not an unbiased estimate of the unknown tail spectrum.

### Current-sample inclusion behaves like a high-leverage shock gate

The current code updates the basis and then projects the same raw gradient.
In the orthogonal case above, an innovation just below (7) is removed by the old
and current rank-`k` spaces, whereas one just above (7) installs its own
direction and can preserve much of itself. Large novel shocks are therefore
*more* able to survive current-basis filtering than small novel components.
This is the opposite of an outlier-rejection theorem.

A falsifiable prediction is that current-minus-lagged retained energy should
increase with innovation leverage and basis turnover, especially on large-norm
transients. Existing common-stream measurements establish a large average
self-inclusion gap, but do not yet test this leverage curve or label the shocks
as useful versus nuisance. Existing lagged learning interventions are adverse
on clean data and sign-inconsistent on noisy data, so “self-inclusion is harmful
contamination” is already too simple a causal account.

## 4. High covariance energy is neither SNR nor clean-descent utility

At a fixed state, let `s` be the clean-objective gradient and let the corrupted
training gradient be `g=m+xi`, with conditional noise covariance `Sigma`.
Consider a predictable rank-one projector `P=uu^T` followed by plain SGD with
step size `eta`. For an exactly quadratic clean objective with Hessian `H`,

\[
E[\Delta L_{clean}\mid F]
=-\eta s_u m_u
+\frac{\eta^2}{2}h_u(m_u^2+\sigma_u^2), \tag{8}
\]

where `s_u=u^Ts`, `m_u=u^Tm`, `h_u=u^THu`, and
`sigma_u^2=u^T Sigma u`. This is exact for the stated quadratic and only a local
approximation otherwise.

In the unbiased case `m=s`, with `h_u>0` and `0<eta h_u<2`, direction `u` has
positive expected one-step utility exactly when

\[
\frac{s_u^2}{\sigma_u^2}>
\frac{\eta h_u}{2-\eta h_u}, \tag{9}
\]

with the evident conventions at zero noise. The relevant quantity is therefore
a curvature- and step-size-dependent signal-to-noise threshold, not covariance
energy. With corruption bias, even large `|m_u|` is harmful when `s_um_u<=0`
under a PSD Hessian. Equation (1) shows that the learned temporal eigenvalue can
instead be large because of `d_u^2`, fresh `sigma_u^2`, finite-history cross
terms, or optimizer-driven oscillation. None orders directions by (8).

There is an equally sharp estimation statement. From the predictable-projector
MSE identity, a forced rank-`r` orthoprojector minimizing error to a fixed clean
target `mu`, with `m=mu+b`, selects the top eigenspace of

\[
K=\mu\mu^T-bb^T-\Sigma, \tag{10}
\]

not the top eigenspace of a covariance matrix. If rank is optional, only
positive-eigenvalue directions of `K` should be retained; forcing a negative
one worsens MSE. For a random zero-mean useful signal `s` independent of noise,
with covariance `M_s`, the analogous oracle is the positive/top eigenspace of
`M_s-Sigma`, whereas ordinary PCA sees `M_s+Sigma`.

A two-dimensional counterexample is immediate: let
`M_s=diag(4,0)` and `Sigma=diag(0,9)`. Top-total-variance rank-one PCA chooses
the pure-noise second coordinate and has signal-estimation risk 13, worse than
the raw risk 9; the utility oracle `M_s-Sigma=diag(4,-9)` chooses the first
coordinate and has zero risk. This is a proof by construction, not evidence
about the repository's tasks. The actual current-basis policy is also
sample-dependent, so (8)--(10) are diagnostic predictable-policy benchmarks,
not performance theorems for self-inclusive AdamW.

## 5. Ranked provisional explanations and discriminating observations

The ordering below reflects mathematical plausibility plus the existing mixed
evidence, not established causal frequencies.

1. **Activity/stability restriction.** The basis tracks sustained high-pass
   power generated jointly by optimizer motion and batch variation. Restricting
   learning to those active directions can regularize late training, but the
   directions are not intrinsically clean. Test with frozen-state decomposition
   (1), directional lag autocorrelations, Hessian-vector products, and the
   clean-utility score (8). It is weakened if leading modes consistently track
   clean utility while showing no relation to drift, curvature, or stability
   margin.

2. **Task- and phase-specific utility alignment.** Noise-robustness gains arise
   when active directions happen to have better clean-versus-corruption utility
   than the discarded complement. Estimate `s`, `m`, and `Sigma` on independent
   clean/corrupted probes at matched checkpoints, then compare (8) or a
   preconditioned analogue inside versus outside the basis. It is weakened if
   utility selectivity is absent across the phase where the learning curves
   separate, unless a clearly measured long-horizon mechanism replaces it.

3. **Finite-rank hysteresis.** Which activity survives is materially controlled
   by admission barriers and early overweighting, explaining slow adaptation
   and nonmonotone width effects. Passive replay can test (7) without changing a
   training trajectory. It is weakened if basis entries and exits are poorly
   predicted by residual leverage and no repeated rejected directions are
   observed.

4. **Self-inclusion mainly improves rapid tracking, not independent denoising.**
   Current observation inclusion preserves high-leverage drift and shocks. Plot
   the current-minus-lagged retention gap against leverage, batch-loss spikes,
   clean utility, and nuisance utility. The account is weakened if the gap is
   unrelated to turnover/leverage or is selectively largest only for
   independently verified clean directions.

The most informative next measurement is therefore not another top-eigenvalue
curve. It is a matched-checkpoint table decomposing each candidate direction
into mean surprise, fresh batch noise, temporal autocorrelation/stability,
clean descent utility, and admission leverage. That table would distinguish
what the covariance is large *because of* from whether retaining it helps.
