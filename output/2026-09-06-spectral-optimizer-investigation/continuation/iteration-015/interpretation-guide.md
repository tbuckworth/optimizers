# I15 interpretation guide

Written during acquisition without inspecting an I15 outcome. This guide fixes
how to read the registered contrasts; it adds no estimand or decision rule.

## The local intervention is identifiable; its semantic value is not

At one common state, first idealize the post-ingest action as a fixed
orthoprojector $P$ and write $Q=I-P$. Current delivery is the same in both
history cells, $h=Pg$. Their buffers and decay-adjusted data steps are

\[
b^N=\rho b+h,\qquad b^P=\rho Pb+h,
\qquad d^N=-\eta b^N,\quad d^P=-\eta b^P.
\]

Let $r$ be the common manual-decay displacement. In ideal real arithmetic,
projecting old history changes the parameter step by

\[
d^P-d^N=+\eta\rho Qb.
\]

For locally smooth clean-CE utility $U=-L_{\rm clean}$,

\[
U(\theta+r+d^N)-U(\theta+r+d^P)
  =-\eta\rho\,\nabla U(\theta+r)^\top Qb
   +O(\lVert d^N\rVert^2+\lVert d^P\rVert^2).
\]

Thus native history is locally useful when its removed component points in a
clean-utility-improving update direction. In the implementation, replace $Qb$
by the measured $b-A_tb$. The implemented one-step identity holds up to the
recorded finite-precision displacement and recurrence defects, while
“orthogonal” requires the reported action errors to be small. This differential
argument applies to smooth CE, not discontinuous accuracy.

The saved raw-minibatch dot does not determine this sign. A fixed corrupted
batch gradient is a sample from the fixed-label objective, not the gradient of
auxiliary clean population utility. It contains softened-objective structure,
realization-specific residual, sampling variation and state dependence.
Even its alignment with the delivered data step measures local optimization of
that batch, not clean generalization.

## Reading $H$, $M$ and $S$

$H>0$ says native carried history beats projected history for current delivery
at the registered endpoint. Because both branches begin at the same state and
use the same future batches, $H$ identifies the conditional total policy effect
of changing the history rule: it includes every endogenous downstream change
in gradients, actions and parameters. Together with a large measured removal
and weaker absolute progress after projection, it supports carried history as
useful at this SGDm operating point. It does not identify a direction-only or
semantic mediation effect. The separate current/native-versus-raw comparison
is not part of $H$ and cannot explain its sign.

$M>0$ says the post-ingest EMA complement helps when old outside history is
removed. Positive absolute clean progress is the strongest useful-learning
version. Under fixed labels, concurrent lower fixed loss, greater confidence
or more negative $R_\zeta$ shows realization fitting alongside any clean gain,
not denoising.

$S>0$ says the marginal mean effect is larger with projected than native
history. The strongest substitution pattern is jointly $H>0$, $M>0$, $S>0$,
plus positive absolute clean progress: removing carried history hurts, and an
explicit temporally smoothed component restores some learning. Even this is
not a mediation fraction. Under an ideal fixed action, native mean history has
outside-buffer DC gain $1/(1-.9)=10$, whereas projected history has gain one.
The cells therefore differ in amplitude and frequency response as well as
orientation. Moving $A_t$, different future gradients and nonlinear loss
curvature further destroy an additive decomposition.

Other combinations remain informative. Here “approximately zero” is only a
descriptive summary of retained seed values, not an equivalence claim or a
predeclared threshold:

- $H>0$, $M\approx0$: carried history may help, but an unaccumulated mean is an
  insufficient replacement; this does not test the tenfold native-mean route.
- $H\leq0$, $M>0$: old outside history is unnecessary or harmful here, while
  the newly delivered mean component can still support learning.
- $S\approx0$: saturation, unequal amplitudes or trajectory divergence can hide
  overlap; it does not establish independent mechanisms.
- $S>0$ with adverse cell levels or no absolute progress is an interaction
  without a positive useful-learning result.
- $S<0$ can reflect native accumulation or synergy, not semantic superiority of
  carried history.

Clean and fixed targets must be read separately. Fixed-label gradients combine
the softened corruption objective with the one realized label residual, and
both the EMA and covariance action are learned from that mixture. Accuracy can
remain good while clean CE/calibration worsens because uniform-replacement
softening preserves class ordering more readily than clean probabilities.

## Bounded falsifiers