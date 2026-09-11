# I18 — Can the native observer learn the constructive tracking direction?

Prospective protocol, 8 September 2026. No scientific stream has been generated
at creation. Continues I17's [proposed discriminator](../iteration-017/next-native-tracking-design.md),
not a replay or an additional neural arm. User authorizes autonomous research;
no new approval gate. Source review, synthetic checks and resource verification
precede first execution. Paid spend/reservation remains $0 of $100.

## Question and scope

Can the unchanged stable rank-one observer realize the positive population
tracking example? Separate direction learning from the choice of response
state. This is exogenous, two-dimensional signal tracking, not optimization,
generalization, semantic denoising or a production-default recommendation.
Preserve adverse cells and positive oracle headroom even if native routing fails.

## Fixed membership and stream

- 32 independent noise seeds: integers 18000 through 18031 inclusive.
- Each seed generates exactly one CPU float64 Gaussian array of shape (4000,2),
  using a private torch.Generator seeded with that integer. Multiply its columns
  by (1,2), giving noise covariance diag(1,4). Reuse that same array across all
  three drifts and both rotations. No exploratory scientific draws or seed search.
- Drift magnitudes a1 in {0,.01,.03}; s_t=(a1*(t-1),0), g_t=s_t+epsilon_t,
  one-based t=1,...,4000. Initial signal is zero, not supplied to any estimator.
- Rotations: identity and Q=[[cos(pi/4),-sin(pi/4)],
  [sin(pi/4),cos(pi/4)]]. Rotate the complete signal/noise construction and oracle
  directions. Rotation is paired numerical/equivariance sensitivity, NOT another
  independent seed or extra favorable outcome to select.
- 192 observer trajectories, 768000 observations. Every policy within a
  trajectory receives exactly the same raw gradient, updated mean and action.
- Report fixed windows t=1:100 (startup), 101:1000 (transition), and 1001:4000
  (primary late tracking). All windows retained; no best-step/window selection.

## Native observer and moment diagnostic

Use the committed SpectralGradientFilter in spectral_filter.py, unchanged:
rank=1, decay=.99, warmup=0, filter_strength=1, adaptive='none', normalize='none',
weighting='hard', stable_update=True, relative_eig_tol=1e-8,
absolute_eig_floor=0, stabilize_every=100. A dummy model has one two-element
float64 CPU parameter and an SGD base optimizer solely to satisfy the wrapper
API. Never call optimizer.step(), forward(), or backward(). Assign raw g as
the parameter gradient and call filter_grad() exactly once per observation.
Native first mean is g_1; later means update BEFORE centering. Float64 is an
explicit idealized numerical setting, not the earlier neural float32 setting.

Record the actual post-ingest hard action A=V V^T, or I when V is absent,
matching the native fallback. Use native parameter.grad as the delivered raw
projection; verify it agrees with A*g. Do not replace A with an ideal/rounded
projector, change its sign, freeze it, or repair it beyond native code.

Track a full untruncated 2x2 residual second moment separately, using the same
z=g-mu. Match native initialization: zero until the first nonzero z, then z*z^T
(unscaled); subsequent steps .99*M+.01*z*z^T. This diagnostic never delivers
updates. Retain its matrix and leading-action alignment, plus eigenvalue gap;
when the gap is <=1e-10*max(1,abs(lambda_max)), report the direction unavailable
instead of selecting a favorable tie. Initial/native absent-basis observations
are flagged separately from learned rank-one alignment.

## Response rules and controls

rho=.9, beta=.99. At t=1 all estimators deliver g_1, with states initialized to
make this explicit. No estimator knows the true signal or realized noise.
The optimal shared EMA and fixed-direction oracles below know generating
parameters/directions and are clearly labelled oracle controls.

Mean-preserving delivery: h=native_projection+(mu-A*mu).
For t>=2:

```
native_i17: b = rho*A*b_previous + h
            d = (I-rho*A)*b                 # retain b, NOT d
native_cp:  d = rho*A*d_previous + (I-rho*A)*h
```

Initialize b_1=solve(I-rho*A_1,g_1), d_cp,1=g_1. This is a new common-stream
initialization, not a claim to reproduce I17's inherited neural state. The CP
rule preserves a constant g=mu from matched initialization for any A when native
delivery equals A*g; contraction from an initial error additionally needs a norm
bound (orthogonal projectors suffice). Because A is the actual native action,
retain and quantify finite idempotence/orthogonality errors and response
residuals. For fixed exact P it reduces to fast EMA on P and slow EMA outside P.

Uniform EMA controls q=.9,.99,.999, all initialized at g_1, plus raw g.
For nonzero drift add the analytically optimal shared-decay EMA. Its lag
L=q/(1-q) is the unique nonnegative root of
2*a1^2*L - 10/(2*L+1)^2 = 0; solve by deterministic bisection (100 iterations,
bracket [0,10000]), with no outcome access. For zero drift omit this policy:
the stationary-risk infimum occurs as q approaches 1 and is not attained for
q<1. q=.999 remains only a fixed finite control, not an invented optimum.

I17 normalized scalar mixtures k in {0,.5,.9,1}:

```
h_k = k*g+(1-k)*mu
b_k = rho*k*b_previous+h_k
d_k = (1-rho*k)*b_k
```

Initialize b_k,1=g_1/(1-rho*k). k=0 equals the .99 EMA; k=1 equals the .9 EMA.
Keep those duplicate labels for explicit algebra checks, not extra independent
evidence. These mixtures are not the full class of isotropic temporal filters.

Fixed useful-direction oracle: P=Q*diag(1,0)*Q^T, route fast q=.9 on P and slow
q=.99 on I-P. Fixed nuisance oracle uses the complementary P. Implement them
directly as P*EMA_.9+(I-P)*EMA_.99, to provide a separate recurrence control.
No additional EMA grid, drift, horizon or data-dependent tuning is admitted.

## Predictions and decision discipline

Primary reporting uses identity rotation, late window, paired independent seeds.
Rotation repeats must be shown separately, not pooled to double n.

1. Strong-drift constructive prediction (.03): native average squared alignment
   with the changing direction exceeds .5 and native_cp has lower mean MSE than
   every prespecified uniform EMA (including the analytic oracle) and each
   scalar mixture. Report all pairwise effects, not just the most favorable one.
   This deliberately demanding conjunction may fail even if some benefits exist.
2. Learnability limitation: useful-direction oracle wins, population and/or full
   moment favors that direction, but the native rank-one action does not. This
   distinguishes missing directional headroom from finite/truncated estimation.
3. Response-state limitation: native_cp beats native_i17 on the identical stream
   and action. This identifies a response-rule contrast here, not the fraction
   of I17's neural gap caused by gain or any semantic mechanism.
4. Necessary adverse boundary: zero and weak .01 drift allow nuisance variation
   to dominate the population spectrum. Report any native tracking harm without
   using it to cancel strong-drift positives. The weak-drift useful oracle can
   still outperform uniform smoothing despite the wrong population direction.
   In the zero-drift cell, e1 alignment is only generating-axis alignment; there
   is no changing direction and the word useful is not an empirical designation.

Population predictions and fixed-route asymptotic risks are computed from the
previous exact theory, separately from empirical results. Finite horizon,
initialization, action/data dependence and truncation remain empirical issues.
Do not apply a fixed-projector risk formula to a data-adaptive moving action.

## Outputs, audit and uncertainty

Retain per-observation g,s,mu,A, native delivery, full moment, all policy outputs,
native basis/rank status and diagnostic residuals in non-pickle numeric arrays.
Archive fixed metadata, source hashes, exact policy order and seeds, software
versions, elapsed/resource records and a completion inventory. Each stream is
written once to an exclusive path; no resume, replacement seed or overwritten
partial output. Failed/missing streams remain explicit; a hard numerical error
stops the attempt rather than silently replacing it.

Independent analysis reads saved arrays, not rerunning native observers or RNGs.
Check membership/hashes/shapes/finiteness, raw signal construction and rotation,
mean/order/action delivery, both response recurrences, scalar/EMA/oracle controls,
full-moment recurrence, duplicate-control parity and rotation sensitivity.
Calculate per-seed/window MSE, mean error vector and temporal error dispersion
(MSE=sum squared mean error+dispersion), alignment and orthogonality/idempotence.
Dispersion is descriptive window error variance, NOT a causal noise variance or
an independent-sample standard error over time. Report paired MSE differences,
seed signs and mean +/- standard error across 32 seeds; time steps are not
independent replicates. No multiplicity-adjusted significance claim is planned.

Synthetic tests may use hand-written deterministic streams and non-scientific
test seeds outside 18000:18031. They are correctness fixtures, not outcomes to
tune the scientific membership. Freeze all acquisition code/protocol before the
first scientific draw. Audit/report source is separately reviewed before use.

## Bounded resources and no replay

CPU only, one numerical thread/one-core quota; CUDA hidden; no cloud or GPU work,
downloads, neural checkpoints, dataset loads or production changes. Hard unit
caps: 4GiB RAM, zero additional swap, 1800 seconds runtime plus 5-second stop
grace, Restart=no. Cooperative acquisition cap 1600 seconds; array output cap
1GiB, total root cap 2GiB and require >=10GiB free at admission. Persist under
an exclusive mktemp directory on /tmp/spectral-experiment-artifacts/ and record the actual
mount/space before launch. No scientific smoke or replay is required for these
tiny CPU tensors. If the bound stops the attempt, retain partial coverage and
do not automatically restart it or broaden membership.

This is a prospective claim/resource boundary, not overall goal completion.
Existing I1-I17 handles and delivered reports/emails remain consumed/unchanged.
