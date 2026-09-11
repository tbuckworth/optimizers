# Independent batch-composition theory review

Date: 2026-09-09
Verdict: **PASS — no algebraic defect found.**

For `G = a + I b`, `I ~ Bernoulli(q)`, the mean and batch-mean covariance are
exactly

`E[g] = a + qb`, `Cov(g) = q(1-q)bb^T/B`.

The sole nonzero eigenvalue is `q(1-q)||b||^2/B`. The stated relative variance
`(1-q)/(Bq)` follows after dividing by the squared mean contribution `q^2
||b||^2`. The note correctly handles the important reversals: centered variance
vanishes at `q=1`, peaks at intermediate frequency for fixed amplitude, and can
be dominated by a rare but strong direction. For independent Bernoulli
indicators and mutually orthogonal `b_j`, the listed eigenvalues follow; without
either assumption, cross-covariances or nonorthogonal mixing prevent that
diagonal reading.

The mixture decomposition is also exact. The law of total covariance gives
`Cov(G)=W+H`. Fixing counts to `n_s=Bp_s` while independently sampling within
each group yields

`Cov(g_stratified)=B^-2 sum_s n_s Sigma_s=W/B`,

so iid-minus-stratified covariance is `H/B`, which is positive semidefinite.
The two-group deterministic special case therefore exposes only the between-
group mean difference under iid count variation. The assumptions of integer
counts, with-replacement conditional draws, equal weighting, and a fixed model
are all stated.

For the code's update-first EMA,

`c_t = beta(g_t-m_{t-1})`.

At a fixed model under stationary iid gradients, `m_{t-1}` is independent of
`g_t` and has covariance `(1-beta)Sigma_g/(1+beta)`. Hence

`E[c_t c_t^T] = 2 beta^2 Sigma_g/(1+beta)`

for the zero-mean stationary innovation. At `beta=0`, `c_t=0`. The note
correctly limits this identity: the native estimator is nonstationary,
self-inclusive, recursively truncated, and specially initialized. In
particular, the code initializes the first mean to the first gradient, while
the first later nonzero basis vector is stored without the stationary
`sqrt(1-beta)` weighting, so the startup qualification is real.

The favorable/adverse interpretation is appropriately symmetric. Sampler
composition can make a shared event spectrally visible, but eigenvalue size
does not identify correctness, rarity, or utility. Conversely, stratification
removes between-group covariance without proving an improvement, and it also
changes noise seen by the base optimizer.

One implementation-design precision should be preserved if the proposed probe
intervention is developed: two observer policies can share an exact model only
at a common checkpoint or during non-updating diagnostic replay. Once their
different bases alter delivered actions, their model states diverge. A causal
training comparison therefore needs common-state one-step branches or explicit
paired trajectories; “matched expectation” alone does not make realized probe
gradients equal. This is not a defect in the identities or current native code,
and the note already labels the proposal a later non-native intervention.

No simulation, saved-data analysis, or source modification was performed.
