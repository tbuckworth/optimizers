# Next discriminator: anisotropic response at matched stationary gain

8 September 2026. Prospective mathematical design after accepted I16 results.
**No I17 implementation, smoke, acquisition or outcome exists.** Original
autonomous authority, resource limits and no-restart rules remain unchanged.

## Reflection

I16 supplies an isotropic competitor with genuine learning progress, but its
family changed stationary gradient gain tenfold and effective gradient-level
shrinkage by the inverse factor. Different update energies leave uniquely
spatial and uniquely temporal explanations unresolved. The [results](results.md)
retain the one positive spectral primary mean and all metric disagreements.

The constructive hypothesis remains: different directions may benefit from
different response speeds. A changing useful component might need fast
tracking while an unhelpful stochastic component benefits from averaging.
Covariance could route these responses helpfully, unhelpfully or not at all.
[I8's favorable construction](../iteration-008/results.md) demonstrates an
available spatial mechanism, not the answer on this neural panel.

## Separate gain from temporal shape

Retain I16's recurrence, with rho=.9 and beta=.99:

\[
\mu_t=\beta\mu_{t-1}+(1-\beta)g_t,\qquad
b_t=\rho k b_{t-1}+kg_t+(1-k)\mu_t.
\]

Instead of `-eta*b_t`, deliver data update `-eta*(1-rho*k)*b_t`. Keep the
manual shrinkage factor `1-eta*lambda` common, eta=.03 and lambda=.01.
Do not silently let decay follow a changed optimizer learning rate. The
zero-state transfer is

\[
\bar G_k(L)=(1-\rho k)
\frac{k+(1-k)(1-\beta)/(1-\beta L)}{1-\rho kL}.
\]

Every k has DC gain1 and constant-state condition `g+lambda*theta=0`.
Their temporal shapes and finite-horizon movements still differ. These are
conditional identities for a fixed external gradient stream, not literal
descriptions of endogenous neural trajectories with inherited h100 history.

For an ideal fixed orthogonal P, retain mean/projected history

\[
b_t=\rho Pb_{t-1}+Pg_t+(I-P)\mu_t,
\]

but deliver `-eta*(I-rho*P)*b_t`. The selected span has normalized momentum
response `(1-rho)/(1-rho*L)`; its complement has EMA response
`(1-beta)/(1-beta*L)`. Both have unit DC gain and stationary penalty lambda.
The ideal difference is now temporal response by direction: mean lags9 and99.

For a fixed symmetric non-idempotent action A, `I-rho*A` also cancels the
constant-state gain whenever it is invertible. The two-response interpretation
specifically requires orthogonal P. The actual action moves and has finite
precision: log its algebra and action defects, without silently repairing or
replacing the native operator. Matching conditional gain does not match the
actual update norm at every point.

## Mean lag does not order noise suppression

For an exogenous scalar white-noise gradient with variance sigma-squared, a
fixed linear kernel with coefficients psi_n has output variance
`sigma^2*sum(psi_n^2)`. Temporally correlated neural gradients need not obey
that white-noise model. With r=rho*k, the exact normalized coefficients are

\[
\psi_n=(1-r)(a r^n+c\beta^n),\quad
a=k-\frac{(1-k)(1-\beta)r}{\beta-r},\quad
c=\frac{(1-k)(1-\beta)\beta}{\beta-r}.
\]

Their squared sum is

\[
(1-r)^2\left[\frac{a^2}{1-r^2}+\frac{c^2}{1-\beta^2}
+\frac{2ac}{1-r\beta}\right].
\]

| k | DC gain | Mean lag, updates | White-noise variance / input variance |
|---|---:|---:|---:|
|0|1|99|.00502513|
|.5|1|50.31818|.09949010|
|.9|1|14.16316|.08602942|
|1|1|9|.05263158|

The intermediate policies have longer mean lags **and higher** noise variance
than normalized raw momentum. A sharp current component plus a small long tail
allows this; k is not a single smoothing slider. These own derivations were
checked against10,000-term sums at all four k values: mass, first moment,
squared sum and nonnegativity, relative/absolute tolerance1e-12 (26177a).
The reproducible [stdlib check](check_normalized_response.py) retains those
identities and the lag/variance counterexample without any training or data.
This is an algebra check, not a measured neural noise model or learning run.

## Ranked next steps

### 1. Matched-gain scalar and spectral continuation

**Do:** Reuse the same six complete I14 SGDm h100 parents. Reuse I16's six
k0 curves unchanged, since normalization leaves that policy identical. Run
normalized k=.5,.9,1 and normalized mean/projected spectral under both targets
and all three seeds200/201/202. Use existing plan rows100:2000 and the unchanged
observer:24 new1900-update branches,45,600 updates, no source replay.

**Tests:** Eight primary spectral-minus-joint-selected-scalar contrasts:
two targets, both validation selectors, both auxiliary metrics. Freeze the
four-k/six-horizon rule and earliest-horizon/lower-k ties before admission.
Retain every fixed-policy curve, h100 progress, realization residual,
confidence, data/total energy and path geometry. No survivor means or retries.

**Needs:** Local RTX3090, sequential worker, proposed6GiB host/zero swap,
4GiB Torch GPU and3GiB artifact caps. I16 used about512s scalar-update time
for34,200 updates, suggesting order10–15minutes here, not an admission
guarantee. A frozen synthetic smoke and conservative forecast must fit a
30-minute scientific runtime cap. Review code, complete-state seams and
independent analysis before acquisition. Expected paid spend$0.

**Kills the idea if:** The strongest anisotropic-response prediction for this
panel fails if normalized spectral loses both fixed-target metrics to the
normalized scalar envelope under both selectors, especially without a clean
progress gain. This does not rule out other tasks. A consistent spectral gain
with actual progress would strengthen temporal routing, not prove semantic
direction discovery. Partial/metric-dependent outcomes must stay separate.

The leading global-response account predicts a strong scalar competitor. The
best opposing hypothesis is that normalizing spectral's high-gain directions
restores a useful balance absent in I16's comparator. Neither outcome is known
from the algebra. No sign prediction is retrofitted to an I17 outcome.

### 2. Fresh-panel confirmation if a useful policy survives

**Do:** Freeze a candidate and comparably tuned scalar competitor, then use a
new seed panel or task. Do not call I14–I16 fresh independent confirmation.

**Tests:** Whether the practical ranking survives adaptation to this panel,
retaining metric/selector distinctions.

**Needs:** A separate prespecified protocol and estimate, not admitted now.

**Kills the idea if:** Consistent reversal limits the proposed practical
advantage, without erasing the original conditional result.

## Not worth pursuing now

- Repeating I14–I16 or extending horizons after observing outcomes.
- A decay-only factorial is weaker than step1: it matches the stationary
  penalty while leaving a tenfold gradient-gain difference.
- Treating small algebra residuals, low movement or covariance retention as
  semantic denoising, causal mediation or long-run robustness.
- Changing production optimizer defaults on this small adaptive panel.
