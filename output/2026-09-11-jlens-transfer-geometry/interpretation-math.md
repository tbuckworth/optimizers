# A score direction is not always the pattern associated with that score

Codex — Spectral Optimizer Investigation · 11 September 2026

The useful mathematical distinction is between a **measurement rule** and
the **variation in activations associated with its measurement**. A fixed
direction u supplies a measurement z = uᵀ(h−c), with a fixed reference c
(the original fit mean in this study). Let μ = E[h] on the population now
being considered, and m = E[z]. With covariance C, linear regression gives:

```text
d = Cov(h,z) / Var(z) = Cu / (uᵀCu)
E_linear[h | z] = μ + d(z − m)
m = uᵀ(μ − c)
```

When c = μ, the score is centered and the expression becomes μ + dz. Keeping
the original c on new data instead changes the intercept, not d or pairwise
score gaps. This is a least-squares projection, not a claim that the true
conditional expectation is linear. For any fixed linear readout L, the corresponding
coefficient is Ld. For a covariance eigenvector with positive variance,
d = u/(uᵀu), so the direction itself also describes the associated pattern
(up to normalization). For arbitrary u, those two objects can differ.

There is relevant established work here, not a novelty claim. Haufe and
colleagues distinguish extraction filters from activation patterns in linear
neuroimaging models; their transformation is A = Σx W Σs⁻¹. The one-score
case has precisely the covariance form above. Their [author poster](https://f1000research-files.f1000.com/posters/docs/263125503)
states the relation and summarizes the [2014 NeuroImage paper](https://pubmed.ncbi.nlm.nih.gov/24239590/).
The poster and indexed abstract were inspected; the publisher's full paper
returned an access error, so no full-paper review is claimed. Applying this
distinction to J-Lens is our interpretation, not evidence supplied by that
neuroimaging study about language-model semantics.

## Why data changes can matter without changing the model

Keep the model, u and L fixed. A different text distribution can change C
to C′. The score rule remains the same; its associated pattern becomes:

```text
d′ = C′u / (uᵀC′u)
readout regression coefficient = Ld′
```

This supplies a reason not to identify a readable signed-PC list with a
universal concept label. It also supplies a steelman: on the population where
it was fitted, a PCA direction really can be a compact summary of shared
variation, rather than the properties of one arbitrarily selected example.
Neither statement says whether that summary is useful on a particular pair.

## What the geometry test can and cannot establish

The input-space diagnostic compares u with C′u. With q = uᵀu and positive
variance, write d′ = u/q + e. Then uᵀe = 0, and:

```text
cos = (uᵀC′u) / (||u|| ||C′u||)
||e|| / ||u/q|| = sqrt(1/cos² − 1)
```

A small cosine means the data-associated activation pattern differs from
the scoring direction. It does not prove that their token readouts differ.
For example, u=(1,0), C′=[[1,1],[1,2]] gives d′=(1,1). The readout L=(1,0)
annihilates the difference; L=(0,1) exposes it. In general the observable
difference is Le, not e alone. Numerical fit alignment is also in-sample;
small, low-rank fresh covariance estimates need not identify population C′.

Finally, J-Lens's normalized vocabulary output is not the fixed linear
surrogate Lh itself. The [earlier source-linked note](../../research/jlens_covariance_interpretation_2026-09-10.md)
derives the ranked-token connection and its finite-precision and residual-
variance qualifications. This diagnostic cannot certify a causal Jacobian,
semantic truth, clusters, optimizer behavior or safety benefit. It can check
whether the clean eigenvector simplification survives in the saved panels.
