# Iteration 001: estimation rank and adaptation to a covariance switch

Prospective protocol written 2026-09-06, after initial report delivery. **Status: prepared; experiment must not execute before the parent sends GO.** The initial report, PDF, evidence files and production optimizer remain unchanged. This iteration studies a synthetic covariance mechanism, not neural generalization.

## Question and primary prediction

At fixed delivered gradient-projection rank one, does retaining more directions in the covariance estimator reduce delay when a new orthogonal direction becomes dominant? Repeated rank-one truncation can discard weak new evidence before it accumulates. Wider estimation should reduce that specific approximation error. It need not improve estimation of the instantaneous population target in every noisy finite sample; exact finite EMA itself has sampling error and memory lag.

The primary outcome is integrated post-switch error relative to the new dominant axis. Secondary outcomes separate agreement with the finite-sample exact EMA from agreement with the population expected EMA. Null and adverse results remain in the report, and all seeds/configurations are retained.

## Frozen design

Dimension is 8. Seeds are the integers 0 through 19. Every estimated or reference covariance delivers exactly one leading eigenvector after the stream starts; all measured post-switch states have a nonzero basis. There is no optimizer, parameter update, training loss, dataset, GPU or API call.

| Factor | Levels |
|---|---|
| Estimation rank | 1, 2, 4, 8 |
| Exact finite-sample reference | Full dense EMA, no rank truncation or eigenvalue cutoff |
| EMA/mean decay | .9, .99 |
| Isotropic background variance ν | 0, .1 |
| Initialization | `current_overweight`, `regular_weight` |
| Delivered projection rank | 1 throughout |
| Raw-stream seeds | 0, 1, …, 19, paired across all method/initialization conditions |

For decay .9, collect **50 pre-switch and 50 post-switch** observations. For decay .99, collect **500 pre-switch and 500 post-switch** observations. These explicit counts avoid floating-point rounding in a nominal five-memory-window formula. They are approximately five EMA e-folding times on each side of the switch. The primary metric uses the entire post-switch window, with no selected burn-in removal. Early/late descriptive windows are the first/last 10 post-switch observations for .9 and first/last 100 for .99; their boundaries will not be retuned after observing results.

Let e₁ and e₂ be the first and second coordinate axes. Raw gradients are independent, zero-mean Gaussian observations with

\[
\Sigma_{\rm pre}=e_1e_1^T+\nu I_8,\qquad
\Sigma_{\rm post}=e_2e_2^T+\nu I_8.
\]

Generate each observation as a scalar standard normal times its phase's signal axis plus `sqrt(ν)` times an independent 8-vector of standard normals. Thus dominant variance is 1 for the clean condition and 1.1 for the background condition; the other variances are respectively 0 or .1. Background changes finite-sample orientation, not the known optimal rank-one population axis.

Random streams use NumPy's `default_rng(SeedSequence([20260906, seed, phase]))`, where phase 0 supplies pre-switch draws of shape `(n_pre,9)`, phase 1 supplies post-switch draws `(n_post,9)`, and phase 2 supplies independent isotropic probes `(n_post,256,8)`. The first normal column gives signal and the remaining eight give background. Each condition regenerates these named streams, so decay .9 receives the first 50 observations from the same phase sequence used for .99; ν=0 and .1 share underlying signal/background normals. All estimation ranks and initialization modes receive exactly the same raw observations and probes within their paired cell. No seed is replaced.

## Mean centering and covariance recurrence

Use the canonical order: `m₁=g₁`, `c₁=0`, and for `t≥2`,

\[
m_t=\beta m_{t-1}+(1-\beta)g_t,\qquad c_t=g_t-m_t.
\]

The first nonzero innovation occurs at t=2 with probability one. For `current_overweight`, initialize `C₂=c₂c₂ᵀ`, matching the current class. For `regular_weight`, initialize `C₂=(1−β)c₂c₂ᵀ`. This is an explicitly labeled factorial, not a silent production fix. Subsequently each rank-k estimate computes

\[
C_t^{(k)}=T_k[\beta C_{t-1}^{(k)}+(1-\beta)c_tc_t^T].
\]

The dense mechanistic model symmetrizes before `numpy.linalg.eigh`, removes eigenvalues at or below `1e−8 λ_max` (canonical relative tolerance, absolute floor zero), and keeps the largest k. The exact finite-sample EMA uses the same raw innovations and initialization but **never truncates or floors its covariance**. Rank eight is therefore a nearly exact numerical control, not defined by fiat to equal the exact reference. Its discrepancy is measured.

The delivered projector is always `P_t=v_tv_tᵀ` for the largest estimated eigenvector, even for estimation ranks 2/4/8. The current canonical hard method ordinarily delivers its entire retained basis; this experiment deliberately separates estimation rank from delivered rank to test the proposed mechanism. It does not relabel default production behavior as a rank-one-delivery method.

## Two distinct population references

The instantaneous post-switch population target is `P*=e₂e₂ᵀ`. This defines adaptation success and the primary error.

Separately compute the **expectation of the full, mean-centered finite EMA**, not an uncentered EMA of Σ. Let `Q_t=E[m_tm_tᵀ]`. Since observations are independent and zero-mean,

\[
Q_1=\Sigma_1,\quad Q_t=\beta^2Q_{t-1}+(1-\beta)^2\Sigma_t,
\]
\[
R_t=E[c_tc_t^T]=\beta^2(\Sigma_t+Q_{t-1}).
\]

Set `E[C₁]=0`, initialize `E[C₂]` as either `R₂` or `(1−β)R₂`, then propagate `E[C_t]=βE[C_(t−1)]+(1−β)R_t`. This recurrence includes the random initial mean and the first-innovation overweight exactly. Its leading projector is the population-EMA reference. It differs from the expectation of a sampled leading projector and must not be described as the latter.

For a near-degenerate reference eigenvalue, defined as within `max(1e−14, 1e−10 λ_max)` of the largest, reference-agreement scores use the full tied top eigenspace to avoid an arbitrary eigenvector penalty. Delivery still uses one eigenvector returned by the eigensolver. The axis target is never degenerate. Record reference top-eigenspace dimension and eigengap so tied cases remain visible.

## Metrics fixed before execution

For each method, seed, decay, background and initialization, record at every post-switch observation:

1. Axis overlap `o_t=(v_tᵀe₂)²` and axis error `1−o_t`. The **primary integrated post-switch error** is `Σ_t(1−o_t)`, with its horizon-normalized version `mean_t(1−o_t)` also reported. Low is better.
2. Agreement `v_tᵀP_top,finite,t v_t` with the exact finite EMA's leading eigenspace, and agreement with the population EMA's top eigenspace. These separate low-rank approximation error from statistical/memory error. They are not interchangeable with axis error.
3. Independent-probe mean squared action error. Draw 256 independent `z∼N(0,I₈)` per post-switch observation from the named probe stream, and average `||(P_t−P*)z||²`. Also compare to the finite and population **rank-one reference projectors**. Probe draws never influence estimation. For nondegenerate references, the analytic population probe error is exactly `2[1−(v_tᵀv_ref)²]`; record it alongside Monte Carlo estimates. At tied references the sampled rank-one action discrepancy remains explicitly tied-choice-dependent; eigenspace overlap is the interpretation metric.
4. Leading vectors/eigenvalues, retained rank, relative covariance Frobenius error versus the exact finite EMA, and the exact/population leading gaps. Retain raw gradients for independent replay.

Sustained adaptation requires **axis overlap at least .9 for 5 consecutive post-switch observations** at β=.9, and **20 consecutive observations** at β=.99. If the first qualifying window starts at the one-based post-switch step s, adaptation time is s. If none qualifies, store `adaptation_step=null`, `censored=true`, and censoring time `n_post−window+1` (46 or 481). Never substitute zero, drop a censored seed, or average only successful seeds. For a bounded descriptive comparison, `restricted_adaptation_step=min(T,censoring_time)` uses the censoring time for failed seeds and is labeled restricted rather than an uncensored mean delay. Report censor counts and the paired restricted-time differences versus rank one and the exact finite reference.

Primary comparisons pair each wider rank against rank one within the same seed/decay/background/initialization. Report all 20 individual differences, their mean, median, sample standard deviation, and counts of positive/negative/near-zero differences (near zero defined as absolute difference ≤1e−12 for horizon-normalized error). Also report aggregate comparisons to exact finite EMA. No significance gate, confidence-interval selection, best-seed selection or pooled cross-factor success claim is planned. There are 640 estimated condition traces and 160 exact finite reference traces; these are paired trajectories, not 800 independent seeds.

## Implementation validation and stop conditions

The standalone script uses CPU NumPy float64 for dense 8×8 covariance updates. After generating the frozen streams, validate **seed 0**, both decays, both backgrounds, both initialization modes and all four estimation ranks: 32 trace comparisons against the current stable `SpectralGradientFilter` in CPU float64. Compare every represented covariance state and post-switch delivered leading-projector geometry. Set `relative_eig_tol=1e−8`, `absolute_eig_floor=0`, `stabilize_every=100`; manually increment its observation count as `filter_grad()` would. For the corrected initialization condition only, rescale its singular values by `sqrt(1−β)` immediately after the first basis initializes, within the standalone harness. Do not edit production code.

The recurrence-validation gate requires maximum covariance Frobenius error divided by `max(||C_dense||_F,1e−12)` below `1e−6` in all 32 traces. Leading-projector Frobenius error must be below `1e−4` whenever the dense leading relative gap exceeds `1e−5`; tied/near-tied comparisons are counted and disclosed separately. A failed gate retains all generated artifacts but marks the run invalid for mechanistic conclusions; investigate and prospectively amend the protocol before rerunning, retaining the failed run. Do not silently adjust tolerances or discard mismatching traces.

Additional invariants: all stored covariances/metrics finite; delivered vectors unit to 1e−10; covariance symmetry to 1e−10 relative; no significant negative eigenvalue below `−1e−10 max(trace(C),1)`; and independent-probe analytic axis error equal to `2(1−overlap)` by construction. The held-out Monte Carlo error is descriptive and is not required to match its expectation within an outcome-selected tolerance.

The script exits unless explicitly invoked with `--run`. It refuses to overwrite an existing results directory. Execution writes protocol/script/source hashes, Python/NumPy/Torch versions, timestamps, every unrounded per-seed metric, compressed float64 trace arrays, validation results, and paired summaries under this iteration's `results/`. Anticipated runtime is a few CPU minutes; no resource expansion is authorized. This protocol and script are prepared before GO, and their hashes are recorded at execution.

## Interpretation limits and prospective falsifiers

The prediction is weakened if wider estimation does not reduce finite-reference error or axis adaptation error, or if low-rank shrinkage improves the noisy variant relative to exact EMA. Improvement only under the overweighted initialization would localize the result to initialization history rather than a general rank effect. Similar behavior under both modes would support recurrent truncation as a contributor beyond startup. Agreement with exact EMA but continued long axis lag would implicate memory/sampling behavior instead of truncation.

A favorable result establishes a controllable mechanism in this synthetic stream. It does not demonstrate that sparse parity failures, MNIST noise benefits, Numerai performance, or emergent-misalignment outcomes are caused by the same mechanism. Neural measurement would remain a subsequent, separately scoped experiment.
