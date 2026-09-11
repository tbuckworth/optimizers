# I16 mathematical interpretation

I16 completed all 18 new scalar branches without a numerical failure. The
independent audit passed 5,304,819 checks, 431 file hashes and 30 complete-state
digests. The numerical statements below come from the pinned
[summary](analysis-001/summary.json); the accepted raw collection is indexed by
collection.json (artifact not distributed in this public snapshot). This is three-seed evidence
for one inherited MNIST/SGDm setting, not a universal account of spectral
filtering.

The subsequent [independent report corroboration](analysis-001/report-audit.json)
agrees exactly on22,814 derived numeric/discrete values. This verifies the
arithmetic, not a causal mechanism or a broader replication claim.

## The registered scalar test

For utility `U = -CE` or accuracy, each primary contrast was spectral
mean/projected history minus the best validation-selected scalar response. The
scalar family chose jointly over four `k` values and six horizons, whereas the
spectral arm chose over six horizons only. The eight registered seed values and
means were:

| Target | Selector | Auxiliary utility | Seeds 200, 201, 202 | Mean |
|---|---|---|---:|---:|
| clean | min validation CE | -CE | -.13152, -.14115, -.12420 | -.13229 |
| clean | min validation CE | accuracy | -.0324, -.0290, -.0346 | -.0320 |
| clean | max validation accuracy | -CE | -.13191, -.15797, -.11822 | -.13603 |
| clean | max validation accuracy | accuracy | -.0332, -.0316, -.0346 | -.03313 |
| fixed | min validation CE | -CE | -.11515, -.09587, -.04709 | -.08604 |
| fixed | min validation CE | accuracy | +.0466, +.0070, -.0040 | +.01653 |
| fixed | max validation accuracy | -CE | -.00844, -.02424, +.02313 | -.00318 |
| fixed | max validation accuracy | accuracy | -.0462, -.0422, -.0606 | -.04967 |

Thus seven of eight mean signs match the prospective scalar prediction. The
sole positive mean, fixed-target accuracy after minimum-CE selection, is mixed
across seeds. After maximum-accuracy selection, the scalar choice was `k=0`
in every fixed-target seed and spectral accuracy was lower by 4.97 percentage
points on average. The opposing prediction—a fixed-target spectral advantage
across both metrics, both selectors and multiple seeds—did not occur. The
validation-to-disjoint-auxiliary comparisons are legitimate within-study
selection; they are nevertheless coarse checkpoint selection, and the scalar
panel had more candidates. They are not official-test or fresh-task
confirmation, and the broader sequence of hypotheses was adaptive.

The result is stronger than a selection-only comparison. At h2000, `k=0`
beat spectral mean/projected on CE and accuracy for both targets in every seed:
all 12 seed-by-target-by-metric differences favored `k=0`. Its mean advantage
was .00773 CE and .447 accuracy points under clean training, and .03231 CE and
8.57 points under fixed corruption. Spatial selection was therefore not needed
to beat the spectral policy at this operating point. This does not establish
mechanistic equivalence or make a global claim that spatial selection is never
useful.

## Temporal response, gain and regularization

The scalar recurrence is

\[
\mu_t=.99\mu_{t-1}+.01g_t,\qquad
b_t=.9k b_{t-1}+kg_t+(1-k)\mu_t.
\]

Under a fixed exogenous stream, its DC gain is `1/(1-.9k)`, giving 1,
1.818, 5.263 and 10 for `k=0,.5,.9,1`; its normalized mean lags are 99,
50.32, 14.16 and 9 updates. With the common manual decay coefficient .01, a
constant-state calculation gives the effective gradient-level regularization

\[
\lambda_{\mathrm{eff}}(k)=.01(1-.9k),
\]

namely .01, .0055, .0019 and .001. An ideal fixed spectral projector instead
has gain/lag 10/9 in its selected span and 1/99 outside it, with corresponding
penalty `.01(I-.9P)`. Consequently, I16 compares inseparable changes in memory,
bandwidth, gain, finite-horizon dose and effective shrinkage. Its outcome cannot
be attributed specifically to smoothing or to the absence of spatial selection.
The actual action also moves with the neural trajectory, so the fixed-action
formula is a comparison principle rather than a potential for the measured run.

Nor was `k=0` merely stalled. From h100 to h2000 its auxiliary clean CE fell
by .0768 and accuracy rose 3.35 points under clean training; under fixed
corruption CE fell by .1377 and accuracy rose 18.75 points. Spectral progressed
too—by .0691/2.91 points clean and .1054/10.17 points fixed—preserving I15's
constructive result that mean/projected history supports real learning. But the
doses were radically different: full-window data-step squared energy/path were
.0168/5.42 for clean `k=0` versus 4.04/85.77 spectral, and .00649/3.40 versus
1.94/60.12 fixed. Better `k=0` utility with far less movement rules out a
pure-stall description, but it is not an amplitude-matched causal comparison.

## Realization fitting and what remains

Increasing `k` produced progressively stronger fixed-realization fitting.
At h2000, `R_zeta = L_fixed - L_soft` was -.0717, -.1504, -.6301
and -1.1662 for `k=0,.5,.9,1`; confidence rose from .1563 to .3742. Spectral
was near `k=0` at -.0600 and .1574, yet `k=0` learned fixed-target auxiliary
accuracy better. Reduced realization fitting is therefore compatible with the
result, but it neither uniquely explains utility nor warrants a semantic
denoising claim. Spectral's full-window data-step action-complement fractions
were .00260 clean and .00141 fixed, versus .620 and .530 for `k=0`, confirming
that the spatial intervention operated without showing that its admitted
directions were useful. The largest recorded recurrence residual/bound ratio
was below .000826, with ideal-step defect `4.27e-7` and decay-realization error `3.10e-7`.
These are local implementation checks, not bounds on accumulated nonlinear
utility sensitivity or causal evidence.

This negative discriminator does not erase I8's constructive toy success, where
useful variation made learned spatial selection outperform global stopping, nor
I15's neural evidence for genuine progress after mean restoration. It supplies
a competitive isotropic witness at this operating point, not evidence that
the two policies work through an identical mechanism. A precise next
mathematical discriminator, if studied separately,
is a decay-matched scalar factorial with

\[
\lambda(k)=.001/(1-.9k),
\]

which holds the conditional fixed-point coefficient at .001 while retaining
the distinct temporal kernels. Persistence of the low-`k` advantage would
challenge a fixed-point-shrinkage-only account, leaving temporal response,
gain and finite-horizon dose as alternatives. Disappearance would support a
regularization interaction, not establish regularization as sufficient:
changing decay also changes transient trajectories. That prospective contrast
would not identify semantic subspaces or a unique mediation mechanism, and
this note authorizes no new acquisition.

The later [next-design note](next-gain-matched-design.md) strengthens this
proposal by matching both stationary gain and effective shrinkage and adding
a normalized spectral candidate. It also shows why mean lag alone is not a
reliable ordering of noise attenuation. This is prospective mathematics, not
an I17 outcome.
