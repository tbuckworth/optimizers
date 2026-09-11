# Author-side interpretation of the completed common-gradient replay

6 September 2026. This is a bounded **author-side interpretation**, not an
independent reference reconstruction, training replay or numerical certification.
I read the completed [summary](summary.json) and all twelve raw scalar snapshot
records, and performed descriptive arithmetic on those stored scalars only.
The independent reference/raw-aggregation audits are separate work.

Summary SHA256: `81ebeacbd84a9936529128fdc8f27be0b1b93721793933cad9fd021dc57e461e`.
Execution manifest SHA256: `b8368fe5ee628dee648a2167e909a0380c6abc2b3fcc7d11976a707dd86e3478`.
The manifest reports all three replays and twelve references complete with
all gates passed; no new accuracy or checkpoint selection was computed.

## 1. Positive primary result, with a specific denominator

At the predeclared final state, width128 improves matched-rank span capture in
all three reused seed streams. Seed differences in the fraction of optimal
rank-32 reference energy are +.0111822954, +.0122733654 and +.0118488209;
mean +.0117681606, descriptive sample SD .0005499891. Thus mean capture rises
from **98.8071% to 99.9839% of the optimal rank-32 energy**.

Every earlier recorded prefix also favors width128 on span capture. These
secondary means describe dependent repeated states, not additional replicates:

| Step | Width32 capture | Width128 capture | Paired gain |
|---|---:|---:|---:|
| 200 | .994168 | .999971 | .005803 |
| 500 | .989362 | .999885 | .010523 |
| 1000 | .988213 | .999860 | .011646 |
| 2000, primary | .988071 | .999839 | .011768 |

The normalized squared rank-32 projector distance also favors width128 in all
twelve states; final means are .0962223 versus .00070914. This explains why
near-optimal energy need not mean nearly identical spaces: some subspace error
can occupy relatively weak directions. All reference projectors are reported
valid; observed boundary gaps relative to the leading eigenvalue range from
.00041054 to .00200821, above the prespecified gate. There are no excluded
rank/gap-null states supporting these averages.

## 2. Full-covariance error is a different, unequal-capacity comparison

Final mean relative Frobenius covariance error falls from .237671 at width32
to .060126 at width128, with improvement in every recorded state. This compares
covariances represented by up to 32 versus 128 modes. It does not hold covariance
storage capacity fixed, and its size should not be substituted for the primary
rank-32 effect. The matched-rank energy and projector results supply the separate
evidence that the wider estimate also improves its *leading 32-dimensional span*.

The finite-history target uses the same rounded innovations and canonical
startup weight. It is not a population covariance or a semantic oracle.
Production discrepancies can include residual rejection, pruning and finite
precision as well as recursive rank truncation. The separate
[ideal-truncation theory](../ideal-truncation-error-theory.md) gives explicit
counterexamples to universal width monotonicity; the present positive pattern
is an observation on these streams, not a general theorem.

## 3. Native useful/corruption retention: smaller and heterogeneous effects

Final-state means of the stored squared-gradient-norm retentions are:

| Probe/contrast | Width32 | Width128 | Exact-reference projector |
|---|---:|---:|---:|
| Training clean | .421548 | .430220 | .428649 |
| Fixed-corruption residual | .357167 | .357517 | .355711 |
| Noisy training gradient | .368333 | .369375 | .369542 |
| Disjoint clean | .381265 | .389311 | .387010 |
| Clean minus residual | .064382 | .072703 | .072938 |

The final clean-minus-residual gap improves in all three seeds, by .009405,
.011749 and .003810 (mean .008321). Nevertheless, wider estimation does **not**
uniformly retain more clean gradient or remove more corruption residual:

| Seed | Change in clean retention | Change in residual retention | Gap change |
|---|---:|---:|---:|
| 3 | -.013386 | -.022791 | +.009405 |
| 4 | +.050832 | +.039083 | +.011749 |
| 5 | -.011431 | -.015241 | +.003810 |

The small positive mean clean change is driven by seed 4; average residual
retention is almost unchanged. The gap improves through different component
changes across seeds. Seed 5's final gap remains small even for the near-optimal
wide estimate (.003746; reference .003864).

Do not generalize the final positive gap contrast to every prefix. It is
negative at step500/seed4 and step1000/seeds3,5; the step1000 mean is -.001009.
Its complete four-state seed means are positive (.003729, .002455, .007955;
mean .004713), but these remain secondary and must not replace the final primary
covariance-capture endpoint. Earlier adverse contrasts remain visible in
500/seed4 (artifact not distributed in this public snapshot),
1000/seed3 (artifact not distributed in this public snapshot) and
1000/seed5 (artifact not distributed in this public snapshot).

The wide native retentions closely follow the reference projector. A
**post-outcome descriptive check** of stored values finds width128 closer to
reference clean and residual retention in all twelve states; maximum absolute
wide-reference discrepancies are .003792 and .005378 respectively. Closeness
is not universal for every other probe: it holds in 10/12 noisy-gradient and
11/12 auxiliary-clean comparisons. Reference fidelity explains what operator
is being approximated, not whether that operator is scientifically preferable.

The clean gradient and corruption residual are strongly anticorrelated (raw
cosines range approximately -.913 to -.828). Since noisy=clean+residual, their
signed cross term is essential. Separate retention ratios cannot be treated
as an additive accounting of the projected noisy gradient or as counts of
semantic/memorization directions. The residual uses fixed training noisy labels
at a state learned from those labels; the independent batch draw does not remove
that conditioning. Disjoint clean probes measure a different, held-out gradient.

## 4. Strong self-inclusive response, without a consistent width effect

For the same current raw g_t, replacing the previous basis by the just-updated
basis increases retained squared norm at all twelve states for each observer.
Increments range .274–.570 for width32 and .289–.501 for width128. Final means
are .381691 and .396299: current-gradient retentions rise from approximately
.425 to .807/.821 after observation.

The reference covariance also includes g_t by design. There is no lagged exact-
reference solve in this experiment. Comparing current-gradient retention with
independent-probe retention additionally compares different draws and batch
sizes (64 versus 256); that entire difference cannot be assigned to self-inclusion.

## Bottom line

This replay establishes a useful measurement bridge: on the same observed
AdamW states and innovations, widening estimation markedly improves fidelity
to the finite-history leading rank-32 space. It also gives a modest final
clean-versus-residual retention-gap improvement, with mixed component and
earlier-prefix behavior. It does not establish that this fidelity caused the
previous studies' accuracy differences, that a reference projector would learn
better, that width128 is universally better, or that the covariance identifies
semantic signal. The observers did not alter training, and these are three
reused randomized streams—not new policy outcomes or independent test data.
