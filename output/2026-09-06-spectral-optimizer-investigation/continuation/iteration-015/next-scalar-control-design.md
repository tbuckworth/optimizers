# Prospective scalar temporal-control design

## Scalar family

From each of the same six I14 SGDm h100 parents, continue the same saved batch
rows 101–2000 and labels. Ingest each raw gradient once into the unchanged
canonical observer and use its post-ingest mean
$\mu_t=.99\mu_{t-1}+.01g_t$, but ignore the learned basis in delivery. For a
fixed scalar $k$, define

\[
b_t=.9k b_{t-1}+kg_t+(1-k)\mu_t,
\qquad \theta_{t+1}=(1-.03\times.01)\theta_t-.03b_t.
\]

Use the fixed grid $k\in\{0,.5,.9\}$. This is exactly **18 new branches**:
three values × clean/fixed targets × seeds 200/201/202, each for 1,900 updates.
Reuse, do not rerun, the six I14 raw-SGDm curves as $k=1$: their observer still
ingests once, while the delivered recurrence is exactly
$b_t=.9b_{t-1}+g_t$. Reuse I15 mean/projected as the spectral comparator.

For a stationary constant gradient, $\mu=g$ and the scalar DC gain is

\[
G_k(1)=\frac{1}{1-.9k},
\]

giving gains $1$, $1.818\ldots$, $5.263\ldots$, and $10$ for
$k=0,.5,.9,1$. More generally, with lag operator $L$,

\[
G_k(L)=\frac{k+(1-k)(1-.99)/(1-.99L)}{1-.9kL}.
\]

Under an ideal fixed spectral action, mean/projected has gain 10 inside its
selected space and gain 1 outside. The scalar family applies one intermediate
response to every coordinate. This is the intended contrast.

## Frozen comparison

Retain every seed, $k$, target and scheduled horizon
h100/250/500/1000/1500/2000. Primary comparison should deliberately favor the
scalar baseline: within each seed and target, jointly select $(k,h)$ from all
four scalar values and six horizons by (a) minimum validation clean CE and (b)
maximum validation clean accuracy, with earliest-horizon then smaller-$k$ ties.
At the separately selected point, report both auxiliary clean CE and accuracy.
Compare these with I15 mean/projected selected by the same validation metric
over its six horizons. This yields eight signed spectral-minus-best-scalar
utility contrasts: two targets × two selectors × two auxiliary metrics, with
all three paired seed values and their mean. Missing contributions make the
contrast unavailable; never survivor-average.

“Best scalar” means best on the specified validation selector, not an oracle
upper bound on auxiliary performance. The larger scalar search gives it more
validation opportunities, but finite validation noise can also hurt auxiliary
performance. Retain each fixed-k comparison so that search effects stay visible.

Mandatory secondary results are the unselected endpoint and full curve for
each fixed $k$, mean/projected and raw; absolute progress from the common h100
state; validation and auxiliary CE/accuracy/confidence; and, for fixed labels,
training clean/fixed/soft CE and accuracy, $R_\zeta$, confidence and true-label
probability. Report cumulative data-step energy/path length and raw-gradient
dots descriptively so a low-$k$ win is not silently called equal learning at a
much smaller optimization dose. Do not tune learning rate, extend duration,
rescale steps, add another $k$, or select on auxiliary outcomes.

## Interpretation

If mean/projected beats the validation-selected scalar envelope across both
clean metrics with positive absolute progress, the strongest valid conclusion
is that this rank-32 anisotropic response adds value beyond these global
gain/bandwidth controls. It would support spatial selectivity, not prove that
the selected space is semantically clean or that anisotropy uniquely mediates
I15. If a scalar matches it, learned spatial selection is unnecessary at this
operating point; isotropic temporal smoothing or gain control becomes the
simpler account. Mixed CE/accuracy or clean/fixed results remain mixed rather
than a composite win.

$k$ is not a pure “slowness” knob. It jointly changes instantaneous raw/EMA
mixing, momentum memory, DC gain, frequency response, step amplitude and the
entire endogenous trajectory. At finite h2000, low $k$ may underfit merely
because its effective response is smaller; curves and absolute progress expose
but do not eliminate this ambiguity. Conversely, the spectral arm's inside
gain 10 and outside gain 1 are ideal fixed-action descriptions, not literal
claims for its moving finite-precision action.

Finally, EMA is not a clean-label oracle. With fixed corruption it averages
conditional minibatch variation, but a persistent realization-specific force
in the finite training objective remains in its long-run mean. Changes in
$R_\zeta$, fixed accuracy and confidence must therefore accompany clean
utility. Scalar success would show that spatial filtering was not necessary;
it would not establish denoising.

Any implementation must bind the accepted I14/I15 summaries and source
artifacts, restore each h100 model/SGDm/observer/RNG state independently, verify
the neutral h100 evaluation and complete-state digest before all updates, use
the identical saved plans, and prove synthetic first-step $k=1$ parity with the
unchanged raw-SGDm code without repeating a real I14 update. The same local
resource ceilings and failure/no-retry discipline
apply. No model, split, seed, target or cloud resource is added by this design.
