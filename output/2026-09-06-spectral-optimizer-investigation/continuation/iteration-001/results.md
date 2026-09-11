# Iteration 001 results: wider estimation reduces axis-switch delay

The frozen CPU experiment supports the proposed truncation mechanism. **Estimating two covariance directions while evaluating a rank-one projector reduced post-switch error in every seed in each of the eight prespecified decay/background/initialization cells.** Widening further approximated the exact finite EMA more closely. The benefit persisted with regular observation weighting at initialization, so it was not explained solely by the canonical first-innovation overweight. No optimizer updates were performed.

This is evidence about a synthetic changing covariance, not about neural generalization, parity circuits, noisy-label learning, finance or alignment. The clean variant is deliberately constructed so the full history occupies at most two axes; rank two recovering the exact covariance in that variant is a mechanistic control rather than an unexpected discovery.

## Execution and validation

The [prospective protocol](protocol.md) was approved before GO, and the [script](run_experiment.py) was statically checked before execution. The CPU-only run started at **2026-09-06 10:13:11 UTC** and completed at **10:13:31 UTC**, taking **20.18 wall-clock seconds**. It produced all 800 planned condition records and all 32 canonical implementation comparisons. No failed configurations or seeds were removed, and no configuration or tolerance was changed after observing outcomes.

Validation (artifact not distributed in this public snapshot) passed in all 32 traces: maximum relative covariance Frobenius discrepancy between the dense mechanistic model and the canonical stable class was **3.23×10⁻¹⁴**; maximum leading-projector discrepancy was **4.71×10⁻¹³**. There were no near-tied post-switch validation steps. These are far inside the prospective 10⁻⁶ covariance and 10⁻⁴ projector gates. Every condition met the sustained-adaptation criterion within its horizon; censor counts are zero throughout.

The execution manifest records protocol SHA256 `89f258c2741a49a7fb713ae4124754ff5dd69ad6aa259b628d21f01b753d9f95` and script SHA256 `6442e376138b0fc4d7dce90740864d9737e29a357ad81b4a2a5299ff01890e4c`, along with source hash, versions and timestamps. See execution.json (artifact not distributed in this public snapshot).

## Primary outcome: integrated error toward the new axis

Entries below are the mean across 20 paired seeds of the horizon-normalized integrated error, `mean_t[1−(v_tᵀe₂)²]`. Smaller is better. Every condition evaluates a rank-one projector. The clean stream has no background; the noisy stream adds isotropic variance .1. Rank-eight and the exact finite EMA agree to numerical precision in these runs; they remain separately stored and evaluated.

| Decay | Background variance | Initialization | Estimate rank 1 | Rank 2 | Rank 4 | Rank 8 | Exact finite EMA |
|---|---:|---|---:|---:|---:|---:|---:|
| .9 | 0 | Current overweight | .228846 | .126063 | .126063 | .126063 | .126063 |
| .9 | 0 | Regular weight | .227679 | .125714 | .125714 | .125714 | .125714 |
| .9 | .1 | Current overweight | .250700 | .164432 | .160812 | .158075 | .158075 |
| .9 | .1 | Regular weight | .251129 | .162725 | .159946 | .157548 | .157548 |
| .99 | 0 | Current overweight | .396488 | .128360 | .128360 | .128360 | .128360 |
| .99 | 0 | Regular weight | .393877 | .127149 | .127149 | .127149 | .127149 |
| .99 | .1 | Current overweight | .380352 | .149559 | .139989 | .132904 | .132904 |
| .99 | .1 | Regular weight | .377232 | .151254 | .138703 | .131472 | .131472 |

The unnormalized integrated sums use 50 post-switch observations at .9 and 500 at .99; all unrounded sums and averages are retained in [per-seed-results.json](results/per-seed-results.json). Because the horizon scales with memory, cross-decay comparisons concern the specified windows, not equal amounts of data or computation.

Within each of the eight cells, rank two beats rank one in all 20 seeds on the primary metric. The paired mean reduction ranges from .08627 to .26813. This is not 160 independent seeds: the same 20 named random streams are reused across factors. Means, medians, sample standard deviations and every signed seed difference appear in [paired-summary.json](results/paired-summary.json); no significance threshold was applied.

## Adaptation time: memory lag versus truncation lag

Adaptation is the first one-based post-switch step beginning a run of overlap at least .9 for five consecutive observations at decay .9, or 20 observations at decay .99. These are the prospective definitions, rather than thresholds selected to maximize separation. All runs adapted, so restricted and observed adaptation times coincide here.

| Decay | Background | Initialization | Rank 1 mean step | Rank 2 | Rank 4 | Rank 8 / exact finite |
|---|---:|---|---:|---:|---:|---:|
| .9 | 0 | Current overweight | 14.90 | 7.80 | 7.80 | 7.80 |
| .9 | 0 | Regular weight | 14.75 | 7.75 | 7.75 | 7.75 |
| .9 | .1 | Current overweight | 19.40 | 12.95 | 12.75 | 12.00 |
| .9 | .1 | Regular weight | 19.35 | 12.80 | 12.55 | 12.00 |
| .99 | 0 | Current overweight | 222.15 | 65.35 | 65.35 | 65.35 |
| .99 | 0 | Regular weight | 220.45 | 64.65 | 64.65 | 64.65 |
| .99 | .1 | Current overweight | 236.90 | 78.75 | 74.10 | 69.65 |
| .99 | .1 | Regular weight | 234.30 | 80.20 | 73.20 | 69.10 |

The analytic **expected centered EMA**, including running-mean variance and initialization, changes its leading axis at step 7 for decay .9 and step 70 for .99 in all background/initialization cells. Those are population-EMA times, not the time of the instantaneous covariance switch, which occurs before the first post-switch observation. Finite exact EMA differs through sampling and mean-estimation variability. For example, its mean sustained time in the clean .99/current cell is 65.35 rather than exactly 70. Averaging leading projectors is not the same operation as diagonalizing expected covariance.

Rank one's additional delay relative to the finite exact reference is substantial: 156.8 mean observations for the .99 clean/current cell and 167.25 for the .99 background/current cell. Removing the startup overweight leaves comparable gaps of 155.8 and 165.2. The experiment therefore identifies recurrent truncation as a source of delay after a long pre-switch period, beyond the direct startup-weight effect.

## Exact-reference agreement, independent probes and the adverse seed

Wider estimation sharply reduces discrepancy from the exact finite leading eigenspace. In the .99 background/current cell, mean finite-eigenspace error falls from .243378 at rank one to .011893 at rank two, .003905 at rank four, and numerical zero at rank eight. This isolates a covariance-approximation effect rather than merely claiming that a larger estimator is closer to the chosen new axis.

The held-out probe metric evaluates projector action on 256 fresh independent isotropic Gaussian vectors at every post-switch observation. Those probes do not feed the estimator. Its analytic expectation is twice the axis error, and the recorded Monte Carlo values show the same large effect; raw probe sample covariances remain available for replay. This does not provide another independent experiment—it is an independent evaluation of the same fitted projectors.

There is one important exception to a universal “exact is always better” story. For seed 3, decay .9, background .1 and regular initialization, rank eight/exact has **.0004899554676301016 higher** horizon-normalized axis error than rank one. Rank two still improves over rank one in that seed. The finite exact estimator is not an oracle for instantaneous population orientation; finite-sample noise and the timing of transitions can occasionally favor shrinkage. The adverse result remains in the paired arrays and should not be rounded to equality or excluded.

Initialization changes end-point summaries modestly in this design, which uses approximately five EMA memory windows before switching. That supports an effect beyond startup in this setting. It does not demonstrate that initialization never matters: a shorter pre-switch phase or a different stream could expose much stronger startup dependence.

## What this updates

The synthetic mechanism is now supported by both controlled algebraic examples and a prospectively specified stochastic stream: **restricting estimation state can prevent a new direction's evidence from accumulating, even when the evaluated projector rank is held fixed.** Retaining extra estimation directions reduces this error here. The empirical gap grows with longer memory in the tested .9/.99 contrast.

## Artifacts and replay

- Independent audit (artifact not distributed in this public snapshot): all primary metrics, paired summaries,
  adaptation windows, raw streams, exact reference and expected covariance
  reaggregated without importing the experiment's analysis functions.
- [Protocol](protocol.md) and [standalone script](run_experiment.py).
- [All 800 unrounded condition records](results/per-seed-results.json) and [all paired comparisons](results/paired-summary.json).
- Population references (artifact not distributed in this public snapshot), 32 full validation traces (artifact not distributed in this public snapshot), and execution manifest (artifact not distributed in this public snapshot).
- Compressed float64 traces (artifact not distributed in this public snapshot), approximately 26 MiB: raw gradients, independent probe covariances, leading vectors/eigenvalues, ranks, all post-switch metrics, and population covariance trajectories. Keys contain the decay/background/seed/initialization and metric name; method order is `rank1, rank2, rank4, rank8, exact_finite`. This raw archive remains local and is excluded from Git; its exact size and SHA256 are tracked in raw-artifact-manifest.json (artifact not distributed in this public snapshot).

The script deliberately refuses to overwrite the existing results directory. Independent replication should run a copied iteration directory or use a separately reviewed output-path amendment, retaining this original run and its hashes.
