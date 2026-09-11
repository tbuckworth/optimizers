# Shared structure, rarity, and what batch covariance actually measures

Codex — Spectral Optimizer Investigation, 9 September 2026.

**Status: conditional mathematical analysis, not a neural result.** These
standard covariance identities sharpen the user's motivating hypothesis and
the next experiment design; no novelty claim is made for the identities.
They concern groups of **examples**, not the parameter-coordinate clusters
in the recent graph pilot.

## 1. A constructive version of the original intuition

At fixed model parameters, suppose a per-example gradient contains a shared
component that is present with probability q:

    G = a + I b,        I ~ Bernoulli(q).

Here b is one parameter-space direction with possibly many nonzero coordinates.
For a mean of B independent draws,

    E[g] = a + q b,
    Cov(g) = q(1-q) b bᵀ / B.

The b component has only one potentially nonzero covariance eigenvalue

    λ_b = q(1-q) ||b||² / B.

This gives a precise favorable case for the user's intuition: a shared feature
can create a large rank-one contribution from coordinate co-movement. Many
independent, mutually orthogonal example-specific directions with smaller
individual eigenvalues can be excluded by keeping the leading space. Neither
this construction nor the word "shared" says whether b is a correct rule, an
incorrect shortcut, or a feature relevant to some other task.

There are three important qualifications even before neural dynamics:

- Frequency, squared amplitude and coordinate support interact. A rare strong
  direction can outrank a common weak one. Large coordinate support increases
  ||b||² only if the amplitudes are otherwise comparable; reparameterization
  can change that comparison.
- A feature present with exactly the same gradient contribution in every
  example has q=1 and contributes **zero centered covariance** in this model.
  Perfectly persistent agreement and large centered variance are different.
- For q>0 and b≠0, relative to its own squared mean contribution, the variance
  is `(1-q)/(B q)`. Rarity can mean large *relative* batch fluctuation, not an
  intrinsically low-variance or incoherent feature. Absolute eigenvalue rank
  still depends on q, amplitude and competing directions.

With several independent Bernoulli features and orthogonal b_j, the eigenvalues
are `q_j(1-q_j)||b_j||²/B`. This supplies both a favorable regime and a reversal
without changing the filter: useful features win only when these weighted
energies and the retained rank favor them. Correlated feature presence adds
cross terms; it is not this diagonal model.

## 2. The general group decomposition

Let S index mutually exclusive example groups, with probabilities p_s,
conditional means μ_s and conditional covariance matrices Σ_s at a fixed model.
Write μ = Σ_s p_s μ_s. For B independent mixture draws, total covariance gives

    Cov(g_iid) = [W + H] / B,
    W = Σ_s p_s Σ_s,
    H = Σ_s p_s (μ_s - μ)(μ_s - μ)ᵀ.

To see this, expand `G-μ = (G-μ_S) + (μ_S-μ)`; the conditional expectation of
the first term is zero, so the two cross terms vanish. Averaging independent
draws divides the covariance by B.

Now take exactly n_s = B p_s independent samples within each group, assuming
these counts are integers, and average all B gradients with equal weights.
The expected gradient remains μ, but

    Cov(g_stratified) = W / B,
    Cov(g_iid) - Cov(g_stratified) = H / B  ⪰ 0.

Thus changing group-count randomness can remove a whole covariance component
without changing the expected objective gradient. Sampling here is with
replacement; finite-population/no-replacement corrections are separate.

The two-group deterministic case has W=0: an exactly proportioned batch has
constant gradient and no population centered covariance, while iid group
counts expose a rank-one direction proportional to `μ_1-μ_0`. The iid leading
direction is the **difference between groups**, not necessarily either group's
mean gradient and not their common useful component.

This does not say stratification is better or worse. It says the filter's
selection statistic is partly a property of the sampler. A plain optimizer is
also affected by sampling variance, so a naive training comparison of the two
samplers would not isolate a spectral mechanism.

## 3. Connection to the actual streaming estimator

The [current stable code](../../spectral_filter.py) forms

    m_t = β m_(t-1) + (1-β) g_t,
    c_t = g_t - m_t = β [g_t - m_(t-1)].

For iid gradients at an unchanged model, in the stationary limit and with
0 ≤ β < 1, the lagging mean is independent of the new draw and has covariance
`(1-β)Σ_g/(1+β)`. Therefore

    E[c_t c_tᵀ] = 2β² Σ_g / (1+β).

For β>0 this is a scalar rescaling of the population batch covariance, so the
same group-composition directions remain in its expectation. At β=0 the
innovation is identically zero and no leading eigenspace is defined by it.

This is **not** an exact description of the running native basis: model motion,
time correlations, mean surprise, initialization, finite memory and recursive
truncation matter. Selecting an eigenspace of a random moving estimate is not
the same as selecting the eigenspace of its expectation. The first nonzero
innovation also receives a different startup weight in the code. The existing
[conditional innovation analysis](../../research/spectral_optimizer_mathematical_synthesis_2026-09-07.md)
and neural measurements retain those limits. Grokking's full-batch setting has
no fresh group-count variation, so a successful grokking effect must involve
other sources of temporal change.

## 4. What should be measured next?

Do not assign "useful" and "harmful" by eigenvalue or frequency. Construct
independent ground-truth evaluation: ordinary rule accuracy, performance on
rare correct cases, and behavior after a spurious feature is removed/reversed.
Give the native optimizer neither those group labels nor a supervised oracle.
Retain a favorable random-corruption condition and clean competence checks.

A small diagnostic can then separate the above explanation from a slogan:
at fixed registered checkpoints estimate group means and within-group
variability, and report directional quantities `vᵀHv` and `vᵀWv` on the
**already chosen** space. Do not materialize a P×P covariance. Also report
signed clean/group loss utility of actual Adam parameter steps; projected
gradient energy alone is not learning or safety.

An eventual sampler intervention would need stronger controls. Simply changing
training batches changes both filtering and the base optimizer. An observer-only
comparison can hold the delivered training batch fixed while feeding separate,
matched-expectation iid versus stratified probe batches to the covariance
observer at the same model. That adds gradient computation and changes the
observer policy; it must be called an intervention, not native reproduction.
Exact model matching applies only at a common checkpoint or in an offline
diagnostic: once different observer actions drive training, their models
diverge. Continued branches have matched sampling policies, not identical
future model states or realized probe gradients.
It is a later discriminator, not an additional arm silently added now.

## Bottom line

The original idea has a real constructive mechanism: shared gradient events
can concentrate covariance and permit selective continuation. But "common",
"coherent", "large eigenvalue" and "useful" are not synonyms. Testing how the
native filter behaves when those properties come apart is scientifically
informative; running the algebraically predetermined two-group example alone
would not establish neural usefulness or a safety result.
