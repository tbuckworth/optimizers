# Iteration 002 protocol: hard gradient projection before first Adam step

Prospective protocol written 6 September 2026 before executing this example.
The parent explicitly authorized this bounded CPU diagnostic and supplied the
candidate below. No parameter search, candidate tuning, production edits,
training run, GPU use, or paid API call is planned. The initial delivered report
and PDF remain unchanged. This iteration will not be committed before review.

## Question and frozen example

Can an orthogonal projected gradient that is a descent direction for the
original loss become an ascent step after Adam's coordinate scaling?

Use CPU float64, the quadratic `L(theta)=0.5*sum(theta**2)`, initial parameters
and raw gradient `theta=g=(-2,1.1)`, unit vector `u=(1,2)/sqrt(5)`, and fixed
orthogonal projector `P=u u^T`. Learning rate is 0.1. Adam uses fresh zero
moments, betas (0.9,0.999), epsilon 1e-8 and weight decay zero. There are no
other optimizer steps, clipping, warmup, covariance updates or basis learning.
The current `SpectralGradientFilter._project_gradient` provides the hard
projection with `normalize='none'`, `weighting='hard'`, `filter_strength=1`,
and an explicitly installed fixed rank-one basis. Calling this private method
isolates projection and does not claim to exercise the learned-basis pipeline.

Run the following four arms on identical initial parameters:

1. Unfiltered first Adam step using g.
2. SGD using h=Pg.
3. First Adam step using h=Pg.
4. Project the parameter displacement produced by arm 3 back through P before
   applying it. This is an explicitly different intervention, not current
   optimizer behavior. Its Adam state is still computed using h.

The fixed predicted projection is h=(.04,.08). Its raw-gradient inner product
is g^T h=.008>0. Predicted SGD loss change is -.00076. Ignoring epsilon only
for this verbal approximation, the filtered Adam displacement is (-.1,-.1),
g^T displacement=.09>0, and its quadratic loss change is .10>0. The unfiltered
Adam and reprojected-displacement arms are predicted to reduce the loss. The
exact finite-epsilon formulas and assumptions are in `theory.md`.

## Observables and validation

Record input/projection vectors, actual parameter displacement, raw-gradient
directional change, exact before/after loss, displacement norm, and the norm of
the displacement outside range(P). Also record analytical first-Adam
displacements `-lr*h/(abs(h)+epsilon)` and analogous unfiltered values, plus
the maximum absolute discrepancy from PyTorch. Check the projected gradient
against the hand-derived (.04,.08) and P g; check symmetry/idempotence of P;
check `g^T Pg=||Pg||^2`; and check the quadratic change identity
`L(theta+delta)-L(theta)=g^T delta+0.5||delta||^2` in every arm.

All arithmetic validation tolerances are absolute 1e-12. The primary success
condition requires filtered SGD `g^T delta<0` and loss change<0, but filtered
Adam `g^T delta>0` and loss change>0. These signs are conclusions to test, not
grounds for changing the example or silently dropping a result. A sign
failure is retained and reported as a failed candidate. An arithmetic
validation failure marks the execution invalid and requires a documented
prospective amendment before rerunning. No broad convergence or neural-model
frequency inference is licensed by either outcome.

The script requires `--run`, checks that a result artifact does not already
exist, and emits a single JSON record to stdout; the agent preserves that
record using apply_patch. Record protocol, theory, script and current filter
source SHA-256, Git HEAD/status, UTC start/end, Python/Torch versions, all raw
vectors and validation outcomes. Record the saved result SHA-256 separately.
