# Iteration 002: first Adam step can reverse projected-gradient descent

The prospectively fixed example succeeded using the current repository's hard
projection method and CPU float64 PyTorch Adam. Projected SGD reduced the
quadratic loss, but first-step Adam fed the same projected gradient increased
it. The positive original-gradient inner product establishes a direction
reversal, not merely overshoot from an otherwise descending direction.

For `L(theta)=||theta||²/2`, initialize `theta=g=(-2,1.1)` and fix
`u=(1,2)/sqrt(5)`, `P=u u^T`. The current
`SpectralGradientFilter._project_gradient` returned
`h=Pg=(.040000000000000036,.08000000000000007)`.
Since `g^T h=||h||²=.008`, `-h` is a strict descent direction. Adam uses
LR .1, betas (.9,.999), epsilon 1e-8, zero weight decay, and freshly initialized
moments. Every arm executes exactly one optimizer step.

| Intervention | Actual parameter displacement, rounded | gᵀ displacement | Exact loss change, rounded | Displacement outside range(P) |
|---|---|---:|---:|---:|
| Unfiltered Adam | (+.100000, −.100000) | −.309999998 | −.299999998 | .134164078 |
| Projected-gradient SGD | (−.004000, −.008000) | −.000800000 | −.000760000 | 0 |
| Projected-gradient Adam | (−.099999975, −.099999988) | +.089999964 | +.099999960 | .044721343 |
| Project filtered Adam's displacement back through P | (−.059999990, −.119999980) | −.011999998 | −.003000001 | 0 |

The final row is a deliberately different intervention. It does not describe
the production optimizer. The unfiltered row's subspace leakage is expected;
that arm has no subspace restriction.

First Adam's bias-corrected update is
`delta=-eta*h/(abs(h)+epsilon)`. Here it changes the projected direction's
coordinate ratio from 1:2 to almost 1:1, moving the step outside the retained
line. That creates a component whose inner product with the discarded part of
the original gradient outweighs descent within the retained subspace.
Consequently `g^T delta>0`. For this quadratic, the additional curvature term
`||delta||²/2` is also positive. Under this fixed first-step state, reducing the
positive LR scales the ascent direction but does not change its sign. This
last statement is algebraic, not a learning-rate sweep.

The prospective derivation in [theory.md](theory.md) explains the general
boundary: for a fixed symmetric positive-definite D and orthogonal projector
P, `g^T D P g>=0` for all g if and only if D and P commute. Adam's D depends on
Pg, so this fixed-matrix theorem is not itself a universal characterization of
Adam. Nevertheless, along gradients with the same Pg, its first-step D is
fixed, allowing the construction used here. Reprojecting the filtered
displacement gives `-eta*P*D*P*g` and restores weak first-step directional
descent, without guaranteeing finite-step decrease at arbitrary LR or
later-step descent with historical momentum.

## Validation, provenance and limits

The [protocol](protocol.md), theory and script were written and their hashes
recorded before execution. No candidate, tolerance, LR, vector or arm was
changed after observing results. The example ran on 6 September 2026,
10:23:05–10:23:06 UTC, Python 3.12.3, PyTorch 2.11.0+cu128, explicitly on CPU
with one thread. All 19 arithmetic checks passed, with maximum absolute error
`4.440892098500626e-16` against the frozen `1e-12` threshold. These include the
hand-derived projection, first-Adam displacement formula, orthogonal-projector
identities and exact quadratic loss-change identity in every arm.

[result.json](result.json) retains all unrounded vectors, optimizer inputs,
losses, directional changes, leakage norms, validation errors and source
hashes. Its SHA-256 is
`a489f3676ba78e85e31e9dbc0ab8573f088e5904a89090838aa3d4559e910a2a`.
The executed [script](run_diagnostic.py) SHA-256 is
`a78694749a52a4f622f10b81b0ec3b4dfa4c4c7ec0a407c8d82a28726ab84e80`.
The current filter source SHA-256 is
`9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`,
at repository HEAD `de3b95df3702d100a7482a199ffc0483e89bb7ae`.
Post-execution checks verified that the protocol, theory, script and filter
files still matched every hash stored in the result.

The independent audit (artifact not distributed in this public snapshot) separately reproduced all four displacement
vectors with PyTorch, recomputed the 19 checks and confirmed the signs with
50-digit Decimal arithmetic. It also identifies a provenance limit: source
hashes in the result are assembled after execution, and the files had no
pre-execution commit. Their earlier freeze is supported by the session/tool
chronology, not independently established by the result's hashes alone.

Post-execution wording erratum, raised by the parent reviewer: the frozen
theory says "Positive definiteness of D_h and P". D_h is positive definite,
but the rank-one P is **positive semidefinite**, not positive definite. The
block proof correctly treats P as an orthogonal projector; the wording error
does not change the derivation or numerical example. The frozen theory was
not silently edited.

This is an existence result for an initialized, fixed basis and the first Adam
step. It does not test how a temporal covariance estimator learns this basis,
how often such geometry appears in neural training, whether an earlier
experiment encountered it, or whether Adam generally fails to converge.
Current `_project_gradient` was executed directly; `filter_grad` and its
covariance-update/warmup behavior were intentionally outside scope. Execution
involved no neural run, GPU workload, API call or production edit. A subsequent
prospective measurement could log `g^T delta` and actual subspace leakage in a
real training trajectory; this diagnostic supplies no frequency estimate.
