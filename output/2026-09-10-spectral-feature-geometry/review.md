# Feature-geometry supplement: mathematical review

Codex — Spectral Optimizer Investigation · 10 September 2026

**Disposition: conditional identities and examples checked. No new scientific
acquisition, claimed novelty or established neural mediation.**

An independent leaf derived the
[note](../../research/spectral_feature_geometry_2026-09-10.md); main read it
in full and checked its assumptions, vectorization, eigenvalue multiplicities,
mean subtraction, examples and relation to current code. The separately
inspected [primary sources](source-review.md) constrain the interpretation.
This is not a fresh experimental replication or an audit of the live study.

## Exact checks

At uniform predictions, the fresh uniform-label error has conditional mean
zero and second moment Π/K. Column-major vectorization therefore gives the
stated Kronecker covariance, including nonzero feature means. Class-contrast
multiplicity and a strict feature gap give the stated block-rank projector;
the vec identity then yields right-feature truncation. Cross terms vanish
in the nonuniform-prediction split by conditional mean zero, not by assuming
unconditional independence. The mixed-label formula subtracts its nonzero
mean outer product. The finite-population factor uses covariance normalized
by1/N, as stated.

In the binary example, the true-label error equals −s c/√2, so the clean
gradient is −(a,bns)⊗c/√2. Both clean and uniform-label second moments equal
diag(a²,b²)⊗ccᵀ/2; subtracting the mixed mean gives the two stated centered
eigenvalues. All-signal retention is a property of this fixed projector at
W=0, not of a learned temporal estimate or an Adam displacement.

## Main finite-enumeration arithmetic

One single-thread NumPy check used only four fabricated feature vectors
`[(2,1),(1,−2),(−1,.5),(3,.25)]`, three equally likely classes, and every
label enumerated exactly. The nonuniform-prediction check used
`W=[(.2,−.1),(−.15,.3),(.05,.1)]`; the mixture check used true labels
`[0,1,2,0]` and replacement probability.7. There was no random sampling,
file input/output, scientific array, model loading, Torch or GPU use.

Maximum absolute errors:

| Identity | Error |
|---|---:|
| Uniform-label mean | 2.082e−17 |
| Uniform-label covariance | 1.111e−16 |
| Nonuniform-prediction covariance | 1.666e−16 |
| Mixed-label mean | 5.552e−17 |
| Mixed-label covariance | 1.111e−16 |
| Vectorized projector/right-feature action | 1.111e−16 |

All were below1e−12. Exhaustive binary sign/label enumeration additionally
gave nonzero eigenvalues(.5,1.98), mean-direction retention1 up to floating
point, and variance trace2.48→1.98 in the favorable case; (.495,2), retention0,
and trace2.495→2 in the adverse case. Exchanging binary class names does not
change these covariance or retention statements.

## Scope maintained

This connects the user's shared-feature intuition to an exact familiar
special case and supplies both a genuine favorable mechanism and its boundary.
High feature energy is not an invariant semantic ranking. Structured geometry
under fully random labels cannot supply the missing true class mapping.
Current full-network blocks, learned representations, fixed corruption,
temporal centering/truncation and downstream Adam are explicitly distinct.

## Readable artifact