# Independent algebra and interpretation review

Codex · 9 September 2026.

Reviewed source: `direction-interpretation.md`, SHA-256
`818c3443fc01ac78869cc636cebc02802d3dc791a515e848aa17cd9764f0e326`.
This was a prospective, source-only review. I did not inspect results, load tensors
or arrays, run numerical analysis, or mutate experiment code or data.

## Verdict

**Pass.** I found no remaining material algebraic error or interpretation
overclaim. Two precision issues identified during review were corrected in the
reviewed source before this verdict: the distinction between the full stored
operator used for the scale and the threshold-retained projector, and the need
for a separate native/baseline contrast before calling a raw-policy result an
improvement.

## Algebra checks

- Because `P=QQᵀ` is an orthogonal projector and `q=Pg`,
  `gᵀq=gᵀPg=‖Pg‖²`. The stated cosine is therefore
  `‖q‖/‖g‖=ρ`, and the equal-norm distance identity is
  `‖d_raw−d_proj‖²=2a²(1−ρ)`. These identities do not require the
  retained range of `P` to equal the exact range of `VVᵀ`.
- `A=VVᵀ` is symmetric, so `‖Ag‖²=gᵀA²g`; hence the stated formula
  `α²=(gᵀA²g)/(gᵀg)` is exact. It asserts no unwarranted relation
  between `A²` and `P`.
- The first-order SGD ordering follows from Cauchy–Schwarz: raw maximizes
  `gᵀd` over Euclidean vectors of norm `a`, while normalized `Pg` maximizes it
  over norm-`a` vectors in `range(P)`. This assumes a positive step size and is
  only a directional-derivative statement; the note explicitly denies a finite
  step guarantee.
- In the quadratic example,
  `d_raw=(1/√2,1/√2)` and `d_proj=(1,0)`. Thus the linear terms are
  `−√2η` and `−η`, while the half-quadratic terms are
  `25.25η²` and `0.5η²`. At `η=0.05` these are approximately
  `−0.0075857` and `−0.04875`, respectively. The example is also
  compatible with the action definitions, for example by taking
  `A=P=diag(1,0)`, which gives `a=1` for the stated gradient.
- In the frozen-preconditioner example,
  `gᵀD d_proj=100` and
  `gᵀD d_raw=(100+1)/√2≈71.42`. This is a valid counterexample to
  transferring the Euclidean first-order ordering to a fixed positive diagonal
  preconditioner. The note correctly does not identify this frozen map with the
  actual history-dependent Adam update.

## Interpretation checks

The source now correctly says that `a=‖VVᵀg‖` uses the full stored
operator while `P` describes only its threshold-retained numerical span. If a
nonzero singular direction is discarded, the common target norm may include a
contribution not represented in `range(P)`; equal action norm and the
common-state direction comparison remain exact, but exact gain-versus-span
separation does not.

The endpoint claims are appropriately conditional. A projected advantage is
evidence for the total post-fork directional-restriction policy relative to the
specified raw alternative, not semantic feature selection or a uniquely
identified mechanism. A raw advantage weakens necessity of retained-span
restriction for those raw-policy endpoints, but does not establish improvement
over native or baseline without that separate contrast. The cautions about
trajectory feedback, inherited state, mixed outcomes, fixed endpoints, and
transfer are sufficient.

Optional wording hardening only: “maximal immediate descent” could be rendered
“maximal first-order decrease” to make its directional-derivative meaning fully
explicit. The surrounding heading and finite-step caveat already make the
current wording unambiguous, so this is not a scientific defect or blocker.
