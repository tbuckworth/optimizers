# Predictable projection: what lagging permits us to prove

6 September 2026. **Theoretical derivation with exact finite-support checks;
independent audit passes (artifact not distributed in this public snapshot).** This is separate from iteration 006's prospective
learning comparison and is not a result from that experiment. It introduces
no production change or neural run.

## Condition on the right history

Let F contain the fixed training dataset, its fixed corrupted labels, current
parameters, optimizer/observer state and all previous batch draws. The next
minibatch is sampled independently with replacement, as in the frozen recipe.
Let its raw gradient be

    g = m + xi,     E[xi | F] = 0,     E[xi xi^T | F] = Sigma.

Here m is the full **corrupted-training objective's** gradient at the current
parameters, not necessarily the clean gradient. For a specified clean target
gradient mu, write m=mu+b. Conditional on F, mu and b are fixed. The bias b need
not vanish: old corrupted labels affected the current parameters and remain
in the training objective. Independence of the new batch draw does not undo
that history or make corruption a zero-mean perturbation of the clean target.

The previous stored native operator A=B_previous B_previous^T is F-measurable.
Its basis may depend strongly on earlier noise; that does not prevent this
conditional calculation. In contrast, the updated basis depends on the new g.
It cannot generally be taken outside conditional expectations.

## Conditional squared-error identity

For any fixed real matrix A with finite second moments,

    E[||A g - mu||^2 | F]
      = ||A m - mu||^2 + tr(A Sigma A^T).

Subtracting the raw-gradient risk gives

    Delta_R = ||A m - mu||^2 - ||b||^2
              + tr((A^T A - I) Sigma).

This applies to the real-arithmetic operator represented by the stored columns
even when they are not exactly orthogonal; it does not assume idempotence.
Actual float32 matrix products add roundoff. If executed delivery is A g+e(g),
its risk additionally contains E[2(A g-mu)^T e(g)+||e(g)||^2 | F]. The displayed
identity alone does not certify exact floating-point risk.

For an ideal orthogonal projector P, the identity simplifies further:

    Delta_R = ||(I-P) mu||^2 - ||(I-P) b||^2
              - tr((I-P) Sigma).

Proof: P(mu+b)-mu = -(I-P)mu + P b, whose two terms are orthogonal;
subtract ||b||^2=||P b||^2+||(I-P)b||^2. Also P^T P=P.

Thus projection reduces conditional gradient MSE exactly when removed bias
energy plus removed sampling variance exceeds lost clean-gradient energy.
The familiar zero-bias criterion is the b=0 special case. This is a comparison
at a fixed conditioned state, not an oracle for which subspace meets the
inequality and not a causal explanation of a trained model's test accuracy.

## Gradient MSE is not a descent or generalization theorem

For a quadratic clean objective at the current state, with gradient mu and
symmetric Hessian H, a plain SGD displacement -eta A g satisfies exactly

    E[L(theta-eta A g)-L(theta) | F]
      = -eta mu^T A m
        + eta^2/2 * (m^T A^T H A m + tr(A^T H A Sigma)).

PSD H is needed for the usual convex interpretation, not for this algebraic
quadratic identity. For a general smooth nonquadratic objective, the displayed
expression is a second-order expansion with a remainder, not an equality.

For orthogonal P the change in expected first-order descent benefit relative
to raw SGD is

    mu^T P m - mu^T m = -mu^T (I-P)(mu+b).

Removed variance enters the quadratic term, not that linear term. Even a
gradient-MSE improvement therefore should not be substituted for the actual
loss-change criterion. AdamW has historical moments and an input-dependent
coordinate denominator; it is not the linear displacement assumed above.
Neither identity supplies an AdamW convergence or generalization guarantee.

An exact checked example makes the distinction concrete: mu=(1,2), b=(-2,1),
equally likely xi=±(2,3), P=diag(1,0), H=[[2,1],[1,3]] and eta=.1. Projection
reduces gradient MSE by 6, yet expected clean quadratic loss rises by 3/20;
raw SGD instead lowers it by 3/20. This is an existence example with a biased
training gradient, not a frequency claim or the recipe's measured behavior.

## Norm restoration reintroduces current-gradient dependence

The unscaled lagged native action is predictable and linear conditional on F.
The restored policy instead uses

    a(g) = (||P_current(g) g|| / ||P_previous g||) P_previous g,

where defined. Its old direction is selected by a predictable basis, but its
scale depends on the current sample and current observer. The fixed-A identity
does not apply by inserting A=P_previous. The scalar-lagged control likewise
uses a sample-dependent ratio. Calling either of those complete policies
"independent of the current gradient" would be false.

An exact finite-support illustration uses equally likely gradients (3,0) and
(3,4), clean target mu=(3,2), previous projector diag(1,0), and a current
rank-one projector onto each sampled g. The current projector returns g exactly;
the unscaled lag returns (3,0); the norm-restored lag returns (3,0) or (5,0).
Their conditional gradient MSEs are respectively 4,4,6. Restored mean (4,0)
differs from P_previous E[g]=(3,0). These are rational calculations, not a
simulation, frequency estimate or prediction for MNIST. They show why a fixed
previous basis alone is insufficient to invoke a fixed orthogonal-projection
risk formula for a normalized policy.

## Relation to the ongoing experiment

Iteration 006 directly tests whole-policy learning outcomes, retaining clean
and noisy conditions, unscaled and norm-restored lagging, scalar controls,
actual displacement diagnostics and both validation selectors. Those empirical
comparisons are needed precisely because the conditional identities above
neither identify a useful subspace nor determine what AdamW will learn.

The [exact checker](check_predictable_projection.py) exercises the general-A,
ideal-P and quadratic-loss identities on rational finite-support examples,
including biased targets, nonorthogonal native actions and zero/identity
projectors. Its scope is algebraic verification only. No external novelty claim
is made; these results are direct expansions of conditional second moments.
Its [saved rational results](predictable-projection-checks.json) retain every
case, including increases and decreases in risk and quadratic loss. The initial
execution printed results only; the subsequent script change added exclusive
JSON persistence without changing the algebra or test cases.
