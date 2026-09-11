# Independent mathematical review: augmentation, fixed wrong labels and filtering

Codex — Spectral Optimizer Investigation · 10 September 2026

**Verdict:** the proposed cross-entropy and KL identities are exact under the
stated assumptions. They give a constructive consistency-regularization
account of augmentation, but not a theorem that augmentation removes fixed
wrong labels or that covariance filtering preserves its useful effect. This
note is first-principles mathematics, not a scientific acquisition, numerical
audit, novelty claim or explanation certified by the running experiment.

Context read: [knowledge index](../../knowledge/index.md),
[schema](../../knowledge/schema.md), and the earlier
[wrong-label augmentation note](../2026-09-10-spectral-wrong-label-augmentation/mathematical-note.md).
No scientific arrays, models, checkpoints or live job handles were opened.
The derivations below require no experiment or numerical calculation.

## 1. Exact objective identity and the direction of KL

Fix an input \(x\), parameters \(\theta\), and an assigned label \(\tilde y\).
Let the finite augmentation distribution have probabilities \(w_t\geq0\),
\(\sum_t w_t=1\), independent of \(\theta\). Define

\[
z_t=f_\theta(T_t x),\quad A(z)=\log\sum_k e^{z_k},\quad
\bar z=\sum_t w_t z_t,\quad p_t=\operatorname{softmax}(z_t),\quad
p_\star=\operatorname{softmax}(\bar z).
\]

For a target distribution \(a\), write
\(\operatorname{CE}(z,a)=A(z)-a^Tz\). In particular, the assigned hard target
is \(a=e_{\tilde y}\). Linearity of the target term gives

\[
\boxed{\mathbb E_T\operatorname{CE}(z_T,a)
=\operatorname{CE}(\bar z,a)+R},\qquad
R=\mathbb E_T A(z_T)-A(\bar z)\geq0.
\]

The inequality is Jensen's inequality. Moreover,

\[
\operatorname{KL}(p_\star\Vert p_t)
=p_\star^T(\bar z-z_t)+A(z_t)-A(\bar z).
\]

Taking the weighted average removes the linear term, hence

\[
\boxed{R=\mathbb E_T\operatorname{KL}(p_\star\Vert p_T)}.
\]

The orientation matters: the logit-mean prediction is the **first** KL
argument. This is not generally the reversed orientation and is not the
Jensen–Shannon divergence. Also, \(p_\star\neq\mathbb E_T p_T\) generally:
it is the normalized weighted geometric mean of the view probabilities.
At finite logits, \(R=0\) exactly when all positive-weight views have the same
probabilities; their logits may still differ by view-dependent constants
times the all-ones vector. No invariance of the internal representation is
implied.

The finite-support identity is exact, not a small-translation approximation.
Differentiation requires differentiable logits at the point considered;
piecewise differentiable networks satisfy the formulas away from their
nondifferentiable boundaries. A continuous augmentation law needs additional
conditions permitting differentiation under the expectation. Parameter-
dependent sampling probabilities would add derivative terms.

## 2. Exact gradient and fixed-assignment decomposition

Let \(J_t=\partial z_t/\partial\theta\) have shape classes × parameters and
\(\bar J=\mathbb E_TJ_T\). Then

\[
\begin{aligned}
g&=\mathbb E_T J_T^T(p_T-a),\\
r:=\nabla R
&=\mathbb E_TJ_T^Tp_T-\bar J^Tp_\star
=\mathbb E_TJ_T^T(p_T-p_\star),\\
g&=\bar J^T(p_\star-a)+r.
\end{aligned}
\]

The derivative of the moving center \(\bar z\) is included; it is not a
stop-gradient teacher. If differentiating the KL expression directly, both
arguments move with \(\theta\).

Now write each fixed assigned target as

\[
e_{\tilde y_i}=q_i+\epsilon_i,\qquad
q_i=\mathbb E[e_{\tilde Y_i}\mid x_i,y_i],\qquad
\mathbf1^T\epsilon_i=0.
\]

Here \(q_i\) is the corruption-law soft target, independent of \(\theta\);
\(\epsilon_i\) is the once-drawn assignment residual. Exactly,

\[
\boxed{L_i^{\rm aug}
=\underbrace{\operatorname{CE}(\bar z_i,q_i)}_{\text{soft-target term}}
-\underbrace{\epsilon_i^T\bar z_i}_{\text{fixed realization}}
+\underbrace{R_i}_{\text{view consistency}}},
\]

\[
\boxed{\nabla L_i^{\rm aug}
=\bar J_i^T(p_{\star,i}-q_i)
-\bar J_i^T\epsilon_i+r_i.}
\]

“Linear force” means linear in logits: its parameter gradient
\(-\bar J_i^T\epsilon_i\) generally changes as the representation changes.
For a dataset, average these identities over examples. At an externally fixed
\(\theta\), expectation over the assignment mechanism removes the residual
term, leaving the soft-target-plus-consistency objective. This does **not**
justify setting that term to zero at \(\theta(\tilde Y)\) learned from the
same assignments: parameters and Jacobians then depend on those assignments.
Exact-count corruption also couples examples, even when each has the stated
marginal soft target.

For symmetric guaranteed-wrong corruption with probability \(\rho\),

\[
q=\left(1-\rho-\frac\rho{K-1}\right)e_y
+\frac\rho{K-1}\mathbf1.
\]

Its correct-class argmax is retained only for \(\rho<(K-1)/K\), with a tie
at equality. Thus the earlier \(\rho=.8,K=10\) recipe has a truth-favoring
gap \(1/9\). In contrast, uniform replacement with selection probability
\(\alpha\) has \(q=(1-\alpha)e_y+(\alpha/K)\mathbf1\): \(\alpha=.9\)
gives a gap \(.1\) and expected actual-wrong fraction \(.81\). These are
different corruption conventions, not interchangeable “90% wrong” labels.

## 3. What this constructively explains—and what it cannot

There is a genuine favorable mechanism available. Augmentation introduces a
nonnegative cost for prediction disagreement across views. If transformations
preserve the true task and useful patterns share across examples, this can
make view-specific realization fitting less attractive while retaining a
truth-favoring population signal. It changes both the consistency penalty
and the classifier \(\bar z\) on which targets act; it is not simply the
unaugmented loss plus an unrelated penalty. Reducing \(R\) need not reduce
the label residual, and reducing total augmented loss need not reduce \(R\).

Crucially, repeated views retain **the same** \(\epsilon_i\). They are not
fresh independent label votes. A classifier that assigns the same wrong
prediction to every view can have \(R_i=0\) while fitting the wrong label
perfectly. If disjoint example orbits can be assigned arbitrary constant
predictions, consistency alone cannot rule out memorization. Conversely,
overlapping or statistically shared task-preserving views can constrain that
freedom. Which situation the model and data approximate is empirical.

At the same fixed parameters and view set, \(R_i\) and \(r_i\) do not contain
the assigned label. Along differently labeled training trajectories, their
values can differ because \(\theta\) differs. The label-independent formula
therefore does not certify label-independent learned behavior, true-label
recovery, improved held-out performance or an optimizer-specific advantage.

## 4. Covariance is a different object

With \(\delta_T=z_T-\bar z\), an exact integral remainder is

\[
R=\mathbb E_T\int_0^1(1-s)\,
\delta_T^T H_A(\bar z+s\delta_T)\delta_T\,ds,
\quad H_A(z)=\operatorname{diag}(p)-pp^T.
\]

For small logit deviations this gives the local approximation

\[
R\approx\tfrac12\operatorname{tr}
\left[H_A(\bar z)\operatorname{Cov}_T(z_T)\right].
\]

This is curvature-weighted **logit** variation, not parameter-gradient
covariance. For example, at fixed \(i,\theta\), put
\(c_T=J_T^T(p_T-q_i)\) and \(d_T=-J_T^T\epsilon_i\). Then

\[
\operatorname{Cov}_T(g_T)
=\operatorname{Cov}_T(c_T)+\operatorname{Cov}_T(d_T)
+\operatorname{Cov}_T(c_T,d_T)
+\operatorname{Cov}_T(d_T,c_T).
\]

The fixed wrong label can therefore contribute view-dependent gradient
variation through \(J_T\), including cross terms. No eigendirection is
automatically a component of \(\nabla R\), semantic invariance, useful
learning or memorization. The existing temporal observer additionally mixes
minibatch sampling with parameter drift and finite-memory estimation. Even
perfect identification of \(r\) would not make its scalar retention a
signed-utility certificate.

## 5. Exact two-dimensional counterexample: 99% retention, reversed improvement

First, for an orthogonal projector \(P\), gradient descent on the **total**
objective delivers \(-\eta Pg\). Its first-order decrease of \(R\) is
\(\eta r^TPg\), not \(\eta\|Pr\|^2\). The latter applies to projecting
\(r\) alone. Take

\[
r=(1,1/10)^T,\quad g=(-1,20)^T,\quad P=\operatorname{diag}(1,0).
\]

Then

\[
\frac{\|Pr\|^2}{\|r\|^2}=\frac{100}{101}>.99,\qquad
r^Tg=1,\qquad r^TPg=-1.
\]

Thus ordinary total-gradient descent decreases \(R\) to first order, while
projected total-gradient descent increases it, despite retaining over 99% of
its gradient energy. Both updates decrease the total objective locally:
\(g^Tg=401\), \(g^TPg=1\).

This is realizable by the exact augmented binary CE above, not just arbitrary
vectors. Let \(\theta\in\mathbb R^2\), take two equally probable views with
logits \((m+d,0)\) and \((m-d,0)\), assign the first class, and define

\[
m=4\theta_1-\frac{199}{5}\theta_2,\qquad
d=\log3+4\theta_1+\frac25\theta_2.
\]

Writing \(A_2(a)=\log(1+e^a)\), the loss and Jensen term are

\[
L=\tfrac12[A_2(m+d)+A_2(m-d)]-m,
\quad R=\tfrac12[A_2(m+d)+A_2(m-d)]-A_2(m).
\]

At \(\theta=0\), \(\sigma(d)=3/4\), \(\sigma(-d)=1/4\), so
\(R_m=0\), \(R_d=1/4\). Therefore
\(\nabla R=(1,1/10)\), while the mean-logit CE gradient is
\((-2,199/10)\), giving total \(g=(-1,20)\), exactly as required.
Smoothness makes the opposite signs actual finite changes for sufficiently
small positive step sizes. This is a theoretical construction, not an
optimizer run or a claim that these vectors occur in MNIST.

A useful sufficient bound, with \(\rho_R=\|Pr\|^2/\|r\|^2\), is

\[
|r^TPg-r^Tg|
\leq\|r\|\sqrt{1-\rho_R}\,\|(I-P)g\|.
\]

Preservation follows if the unprojected positive utility exceeds this bound,
not from high retention alone. At exactly 100% retention, \(Pr=r\) does
preserve \(r^TPg=r^Tg\) for this ideal orthogonal projection. Positive scalar
norm matching does not change its sign; Adam history, coordinate scaling and
weight decay require signed **actual-displacement** accounting rather than
this plain-gradient formula.

## Implication for the current account

The strongest defensible connection is: augmentation supplies an exact
view-consistency term while fixed wrong assignments still act on the
view-mean classifier. Spectral filtering can help or hinder either part
depending on the joint delivered direction, not merely which standalone
gradient has greater covariance or retention. This provides a sharper
interpretation of existing useful and adverse outcomes, without converting
them into a demonstrated causal decomposition or changing the live protocol.
