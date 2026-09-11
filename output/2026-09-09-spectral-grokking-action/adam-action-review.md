# Independent algebra and interpretation review

9 September 2026

**Verdict: PASS.** The note's algebra is correct under its stated exact-arithmetic,
full-retained-rank and nondegeneracy assumptions, and its empirical claim scope
does not exceed the prospective intervention.

## Algebra

For a full retained thin SVD `V = Q Sigma W^T`,
`VV^T g = Q Sigma^2 Q^T g`. With `z = Q^T g`, positive active weights
`w_i = z_i^2 / ||z||^2`, and `lambda_i = sigma_i^2`, direct expansion gives

```
cos(r_native, q) = E_w[lambda] / sqrt(E_w[lambda^2])
                    = 1 / sqrt(1 + CV_w(lambda)^2),
c = ||r_native|| / ||q|| = sqrt(E_w[lambda^2]).
```

Because `g^Tq = ||q||^2`, RMS--mean then gives
`g^T(cq) >= g^T r_native`, with equality exactly when the singular-value-squared
gains are constant over gradient-active directions. This supports only the
stated equal-norm, in-span, first-order raw-SGD comparison. The note correctly
separates it from finite-step and AdamW behavior and correctly marks zero `q`,
rank truncation, casting and denominator clamping as exceptions to the exact
identity.

The quadratic witness is numerically correct. For `eta=.04`, the native loss
change is exactly `-0.039592`; the equal-norm orthogonal change is approximately
`-0.01616733` (the note's initial `-0.016168` approximation was corrected to
this value before committing). Both descend while the native
direction wins the finite quadratic step because it attenuates the
high-curvature coordinate. The note appropriately presents this as an existence
witness, not evidence that the learned gains estimate inverse curvature.

For one AdamW coordinate, substituting the current gradient `c q_i` into the
carried moments yields exactly

```
F_i(c) = sqrt(B2)/B1 * (a + bc)/(sqrt(d + ec^2) + delta).
```

Differentiating and cancelling the two `b e c^2` terms gives the displayed

```
F_i'(c) = sqrt(B2)/B1
          * [b(d + delta H(c)) - a e c]
          / [H(c)(H(c) + delta)^2].
```

After alignment by `sign(q_i)`, its sign is therefore
`G(c)=C[d+delta H(c)]-Aec`. When `q_i != 0` and `d>0`, `G(0)>0`. If
`A > C delta/sqrt(e)`, then `G` is strictly decreasing, tends to minus infinity,
and crosses zero once, proving the unique positive maximum. Otherwise it remains
positive for every finite nonnegative `c`. The simplified threshold and the
epsilon-zero root `c*=Cd/(Ae)` follow algebraically. Scalar central differences
matched the derivative to at worst `8.7e-11` in monotone, turning and zero-history
examples; a turning example changed derivative sign exactly once.

The `d=0`, `q_i=0`, zero-moment, large-`c`, `c=0` and whole-history-rescaling
statements are also correct. In particular, an exact initialized Adam history
cannot have `v_i=0` but `m_i!=0`; epsilon-zero scale invariance requires a positive
denominator; and scaling only the new action with carried unscaled moments is not
the usual whole-history invariance.

## Interpretation scope

The note correctly distinguishes the clean common-state first update from the
subsequent policy trajectories, whose parameters, moments, gradients, bases,
scales and decay terms diverge. Its endpoint table labels patterns as supported
hypotheses rather than equivalences or mediation results. It explicitly excludes
claims about pre-fork formation, span sufficiency from initialization, semantic
direction selection, a frozen common future projector, or negation of the
earlier learning phenomenon.

The archived rather than contemporaneously replayed native endpoint, CUDA
trajectory sensitivity, numerical-rank loss and unmatched post-Adam dose are all
disclosed. Accordingly, the note does not turn endpoint similarity into
equivalence or endpoint separation into causal mediation. No action outcomes
were read for this review.

One notational caveat is harmless: `lambda_i` denotes singular-value-squared gain
in the first section, while scalar `lambda` denotes AdamW weight decay later.
Context keeps the two roles unambiguous. Likewise, the first displayed SVD
identity should be read under the no-discarded-direction condition that the note
states immediately afterward and reiterates in its numerical qualifications.
