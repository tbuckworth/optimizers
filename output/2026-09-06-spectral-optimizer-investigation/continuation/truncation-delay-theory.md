# Continuation: exact delay from repeated rank-one truncation

Written after the initial report was delivered on 6 September 2026. This is a
new theoretical diagnostic, not a revision of the delivered PDF and not a
neural-performance result.

## Setup and prediction

Let the initial covariance state be `C_0 = kappa e1 e1^T`, with `kappa > 0`.
Every subsequent centered innovation is `c_t = a e2`, with `a > 0`. Directions
are orthonormal. For decay `0 < beta < 1`, exact EMA covariance is

$$
C_t^{\mathrm{exact}}=
\kappa\beta^t e_1e_1^\top+a^2(1-\beta^t)e_2e_2^\top.
$$

The first strict top-direction switch occurs when

$$
\beta^t<\frac{a^2}{\kappa+a^2}.
$$

The recursively truncated rank-one estimator discards every new contribution
`q = (1-beta) a^2` until that *single contribution* exceeds the decayed
incumbent eigenvalue. Before switching its state is
`kappa beta^t e1 e1^T`; it switches when

$$
\beta^t<\frac{(1-\beta)a^2}{\kappa}.
$$

For a threshold `r > 0`, the first positive integer satisfying `beta^t < r`
is `max(1, floor(log(r)/log(beta)) + 1)`. Strict comparison avoids making a
claim about a particular eigensolver's choice at equal eigenvalues. None of
the numerical cases below is at an exact tie.

With unit innovation amplitude, the prospective predictions are:

| Decay | Initial eigenvalue | Exact-estimation switch | Recursive rank-one switch |
|---|---:|---:|---:|
| .9 | 1 | 7 | 22 |
| .9 | 10 | 23 | 44 |
| .99 | 1 | 69 | 459 |
| .99 | 100 | 460 | 917 |
| .999 | 1 | 693 | 6905 |
| .999 | 1000 | 6906 | 13809 |

The large initial values are explicit initial-state conditions. They are
illustrations of incumbent scale, not claims that a normally trained model
has exactly that eigenvalue or that the startup exception alone caused it.

## What the calculation identifies

Wider rank-two estimation, followed by a rank-one delivered projection, keeps
the accumulating new evidence and switches with the exact reference. Its
delivered dimension is unchanged. Recursive truncation can therefore create
a large adaptation delay independently of delivered dimension or numerical
orthogonality error. It does not permanently reject every emerging direction:
the incumbent eventually decays enough for a single innovation to enter.

At unit initial eigenvalue and unit amplitude, ignoring integer rounding,
the delay ratio is `log(1/(1-beta))/log(2)`. It grows as decay approaches one.
This is a statement about this deliberately orthogonal stream, not a universal
ratio for minibatch training. Nonorthogonal observations may rotate the
incumbent continuously, and sufficiently large individual observations can
cause a much earlier switch.

## Current-code check

The companion script initializes two current stable filters to the same
explicit covariance and mean state. One stores rank one, the other rank two;
both are evaluated using only their leading direction. At each step it supplies
`g_t = m_(t-1) + c_t / beta`, so post-update centering yields the prescribed
innovation exactly. This construction intentionally controls innovations;
it is not a stationary raw-gradient process or a model training trajectory.

Numerical eigenvalue cutoffs are disabled for this idealized check, float64 is
used, and no parameters are trained. Successful agreement verifies this
mechanism in the current implementation, not its practical importance in
MNIST, parity, Numerai or EM. The separately frozen stochastic-stream protocol
tests sensitivity to realistic perturbations before any neural inference.
