# Independent review: stochastic tracking steelman

Status: PASS. Reviewed 8 September 2026 by the independent mathematical-review
leaf agent (`i17_analysis`), separate from the note/checker author. This is a
review of conditional mathematics and deterministic checks, not an experiment,
native-observer audit or empirical confirmation.

## Reviewed source pins

- stochastic-tracking-steelman.md (artifact not distributed in this public snapshot):
  `abbb672d4aa602262c91507661aedb2584c4fdffa80a35142e9037e4e179ca34`.
- [check_stochastic_tracking.py](check_stochastic_tracking.py):
  `b0f57ad646a23e86b06c11949e1eb469731b54629506e87674e2dc2e181b564a`.

Both files were read completely. The checker was then run with CUDA hidden and
numerical thread limits set to one: **50 checks PASS**, terminal `928e06`.
It imports only the standard library and generates no random stream or array.

## Mathematical findings

1. The tail representation gives process-error coefficient `-c_(k+1)` and
   observation-error coefficient `c_j-c_(j+1)`. Independence and finite second
   moments yield the stated trace risk; Gaussianity is unnecessary.
2. Independently expanding the square-completion certificate reproduces that
   risk because `Q_sum+2R_sum=R_sum(q+1/q)`. Square-summable tails justify all
   cross sums. Equality uniquely requires `c_j=q^j`, so the ordinary EMA is
   optimal over the stated admissible common causal LTI class, including signed
   kernels—not just over a grid of EMA decays.
3. The exact optimum is `(sqrt(201)-1)/20`, approximately .658872344. The
   directional witness has exact risk `18869/37810`, approximately .499047871.
   The checker's negative Riccati-polynomial value at the positive witness
   verifies the strict inequality without relying on rounded display values.
4. The scalar error recurrence and finite-initialization formula are correct.
   The post-mean residual covariance is
   `beta² Q/(1-beta²) + 2 beta² R/(1+beta)`; its measurement term correctly
   retains current-observation/updated-mean dependence. The selection threshold
   is exactly .06; process variance .1 favors the changing axis and .01 does not.

## Scope and qualifications

The final note includes the necessary mean-square admissibility condition;
unit gain alone is insufficient for a random-walk convolution. A finite first
absolute kernel moment is a sufficient condition covering EMA/DEMA, while the
certificate also holds on the stated square-summable increment-error domain.
The stationary result is not substituted for finite startup. Data-dependent,
time-varying, nonlinear, noncausal and coordinate-specific comparators remain
outside the common deterministic causal-LTI claim.

The fixed directional route is an oracle witness, not a theorem about native
finite-memory direction acquisition. Population leading alignment, finite
random estimates, truncation and response transport remain distinct. No neural
convergence, optimizer-default change or alteration of I18 evidence follows.
No load-bearing mathematical defect remains in these pinned sources.
