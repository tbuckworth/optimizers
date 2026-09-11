# Why the fixed-model comparison can be informative

Codex — Spectral Optimizer Investigation, 10 September 2026.
Conditional algebra tied to the prospective diagnostic; no new outcome.

## Equal average signal does not imply equal observer history

At a fixed model, this MLP has no batch normalization or dropout. Mean
cross-entropy gradients are sums of per-occurrence derivatives.
Changing how the same 3,200 occurrences are divided into 50 batches therefore
preserves the arithmetic mean gradient in exact arithmetic. The numerical
admission test checks the finite-precision version. This identity would need
reconsideration for a model whose forward calculation coupled batch members.

The observer is not an unweighted average of those 50 gradients. Its exact
real-arithmetic mean recurrence gives, for either schedule s,

    m[s,150] = beta^50 m[100]
               + (1-beta) sum(t=1..50) beta^(50-t) g[s,t].

Thus equal unweighted means do not force equal exponential means. Even a
permutation of the same gradient vectors can change the result. For two
vectors x,y, swapping [x,y] to [y,x] changes the final mean by
`(1-beta)^2 (x-y)`. The actual batching intervention also changes the individual
batch gradients, not merely their order. With beta=.99, about 60.5% of the
old mean coefficient remains after 50 observations. That is a coefficient in
the mean recurrence, not a percentage of the retained eigenspace or its energy.

For the common final observation g_star,

    m[s,151] = beta m[s,150] + (1-beta) g_star
    m[G,151] - m[I,151] = beta (m[G,150] - m[I,150]).

Canonical self-inclusion therefore does not reset the history-dependent mean.
It contracts its difference by beta in exact arithmetic. The innovation is
`z[s]=g_star-m[s,151]`, so its schedule difference is the negative of that mean
difference. This alone does not establish a difference in the delivered action.

## A common final gradient need not produce a common filter

Before truncation and numerical errors, the nominal covariance update is

    C[s,151] = beta C[s,150] + (1-beta) z[s] z[s]^T.

Both the inherited covariance and the new innovation can differ. The actual
stable implementation retains only a rank-limited representation and applies
relative eigenvalue thresholds. Consequently this expression explains the
update ingredients but is not an exact full-history covariance identity.
Nor does covariance difference imply useful action difference: it may occur
outside the retained space, or in directions on which g_star has no component.

For this hard, unnormalized action, the directly measured comparison is

    h[G] - h[I] = (V[G] V[G]^T - V[I] V[I]^T) g_star.

Stored V is used for the native action; an independently orthonormalized
numerical basis is only a geometric diagnostic. Different action lengths
can arise because the spaces retain different amounts of g_star, even with
ideal orthonormal bases. This is not a norm-matched directional experiment.
Pre-inclusion action differences are interesting only descriptively if the
canonical post-inclusion difference disappears.

## From changed action to useful learning

The same Adam moments and parameters supply a well-defined map from each
delivered h to one actual displacement Delta(h). That map is coordinatewise
adaptive and generally nonlinear in h. Its zero-input value need not vanish
because old first moments and weight decay remain. For a differentiable
group loss L_j, the local approximation is

    U[j,h] = L_j(theta_0) - L_j(theta_0 + Delta(h))
           = -grad L_j(theta_0)^T Delta(h) + o(||Delta(h)||).

A quadratic remainder would require stronger smoothness, such as a locally
Lipschitz gradient along the step; ReLU activation-boundary crossings need
care. The displayed differentiable approximation is not a bound on every
finite neural step.

The saved finite loss difference, not the approximation, is the primary
usefulness readout. Oracle training-probe gradients diagnose the signed
linear term on those probes only; they are not held-out gradients and are
never inputs to the filter or optimizer.

Three logically distinct claims must stay separate: histories change the
action; histories change one useful finite step; histories explain a sustained
learning trajectory. This diagnostic can support the first two locally.
Neither outcome automatically establishes the third, semantic selection, or
a safety benefit. It can nevertheless establish a concrete constructive
channel through which batch organization matters to this optimizer.

Source: the immutable [stable observer and native action](../../spectral_filter.py)
and the frozen diagnostic design (artifact not distributed in this public snapshot).
All algebra here is conditional analysis, not a new empirical or novelty claim.
