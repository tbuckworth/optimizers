# I16 results: an isotropic temporal control beats the tested spectral SGDm candidate

7 September 2026. This report interprets the fixed I16 study defined in the
[protocol](protocol.md) and prospective predictions (artifact not distributed in this public snapshot). The
primary numerical authority is the independently derived
[summary](analysis-001/summary.json), whose [audit](analysis-001/audit.json)
passed. All 18 new confirmation branches completed through update 2000; there
were no numerical failures. Six I14 raw trajectories and six I15
mean/projected-history trajectories were hash-bound references, not reruns.

## Executive result

The strongest opposing anisotropic prediction failed; seven of eight leading
scalar mean-sign predictions held. A validation-selected scalar
temporal-control family outperformed the I15 spatially selective
mean/projected-history comparator in seven of eight three-seed primary means.
The exception was fixed-target auxiliary accuracy under the minimum-validation-
CE selector: spectral minus scalar was **+1.653 percentage points**, with mixed
seed effects of **+4.66, +0.70, and -0.40 points**. This is not the predicted
uniform spectral advantage.

The clearest result used the maximum-validation-accuracy selector. It selected
the most strongly smoothed scalar, `k=0`, in every fixed-target seed. Mean
auxiliary accuracy was **64.73% for scalar versus 59.77% for spectral**, a
spectral-minus-scalar effect of **-4.967 points**, adverse to spectral in all
three seeds. At the fixed h2000 endpoint, `k=0` also beat spectral in all three
seeds on both clean CE and clean accuracy. Its mean advantages were 0.03231 CE
and 8.573 points. On clean-target training, both selectors chose raw SGDm
(`k=1`) in every seed, and spectral was worse on both auxiliary metrics.

## Design and evidence integrity

I16 continued the same six SGDm h100 states used by I14 and I15: seeds
200/201/202 by clean/fixed training target. The new branches used
`k=0,.5,.9`; I14 raw SGDm supplied `k=1`. At every new step the unchanged
rank-32 observer ingested the raw gradient. The delivered gradient and retained
old momentum were

\[
h_t=k g_t+(1-k)\mu_t,\qquad
\widetilde b_{t-1}=k b_{t-1},\qquad
b_t=.9\widetilde b_{t-1}+h_t,
\]

where \(\mu_t\) is the post-ingest EMA mean. Thus `k=0` uses the post-mean and
erases old momentum, while `k=1` is literal raw SGDm. The comparison is a
temporal-response family, not a pure learning-rate or norm-matched control.

For each seed and target, the scalar procedure selected jointly over four
values of `k` and six recorded horizons using either minimum validation clean
CE or maximum validation clean accuracy. Ties were fixed prospectively by
earliest horizon and then lower `k`. Spectral was selected separately over its
six horizons. Auxiliary data never selected a checkpoint. Primary effects are
\(U(\text{spectral})-U(\text{scalar})\), with \(U=-\)auxiliary clean CE for CE
and \(U=\)auxiliary clean accuracy for accuracy; positive always favors
spectral.

Acquisition was frozen at commit
`f8c99aa82d8de106148cde90e6814069d692d953`; analysis was frozen at
`b7d17f1dae58329413caf1df0069a408a97d3fad`. The audit recorded 5,304,819
checks, zero errors or warnings, 431 file-hash operations, 1,314,171,445 bytes
streamed through those operations, and 30 complete-state tree digests. These
are verification counters, not counts of unique physical files. The root held
417,762,251 logical regular-file bytes. The only runtime-tree entry was an
empty `torchinductor_titus/` directory. The lossless
JSON collection (artifact not distributed in this public snapshot), SHA-256
`cc112411fd23c43f0c87d876491fc8a91ecd8e018fd58488668a037420a0687a`,
contains all 40 original JSON records: 211,944,593 source bytes compressed to
23,363,545 bytes.

A separately implemented standard-library [report corroboration](analysis-001/report-audit.json)
subsequently reconstructed all primary/selected/endpoint comparisons, all
30 trajectories and their180 scheduled points, all960 metric aggregates, and
all104 path/component rows directly from the lossless archives. All22,814
numeric/discrete values agreed exactly (maximum absolute difference0).
This adds independent arithmetic verification, not experimental replication.

## The eight primary comparisons

Accuracy entries are percentage points. Each cell lists seeds 200, 201, 202,
then their arithmetic mean. No survivor averaging was needed, and these three
seeds do not justify population uncertainty claims.

| Target / selector | Metric | Seed effects | Mean |
|---|---:|---:|---:|
| clean / minimum CE | CE | -0.13152, -0.14115, -0.12420 | **-0.13229** |
| clean / minimum CE | accuracy (pp) | -3.24, -2.90, -3.46 | **-3.200** |
| clean / maximum accuracy | CE | -0.13191, -0.15797, -0.11822 | **-0.13603** |
| clean / maximum accuracy | accuracy (pp) | -3.32, -3.16, -3.46 | **-3.313** |
| fixed / minimum CE | CE | -0.11515, -0.09587, -0.04709 | **-0.08604** |
| fixed / minimum CE | accuracy (pp) | +4.66, +0.70, -0.40 | **+1.653** |
| fixed / maximum accuracy | CE | -0.00844, -0.02424, +0.02313 | **-0.00318** |
| fixed / maximum accuracy | accuracy (pp) | -4.62, -4.22, -6.06 | **-4.967** |

Seven means favor scalar. Six of those seven also have the same sign in every
seed. The fixed maximum-accuracy CE effect is near zero and mixed, so it should
not be presented as a robust CE advantage. Conversely, the lone positive mean
is also mixed and does not establish a robust spectral advantage.

## What validation selected

Each outcome cell below is auxiliary `CE / accuracy`; selection used only the
named validation metric.

| Target | Seed / selector | Scalar choice and outcome | Spectral choice and outcome |
|---|---|---|---|
| clean | 200 / min CE | `k1,h1500`: .23799 / 93.14% | `h1500`: .36951 / 89.90% |
| clean | 201 / min CE | `k1,h2000`: .19074 / 93.68% | `h2000`: .33189 / 90.78% |
| clean | 202 / min CE | `k1,h2000`: .23316 / 93.54% | `h2000`: .35736 / 90.08% |
| clean | 200 / max acc | `k1,h2000`: .23760 / 93.22% | `h1500`: .36951 / 89.90% |
| clean | 201 / max acc | `k1,h2000`: .19074 / 93.68% | `h1500`: .34871 / 90.52% |
| clean | 202 / max acc | `k1,h2000`: .23316 / 93.54% | `h1500`: .35139 / 90.08% |
| fixed | 200 / min CE | `k1,h500`: 1.85378 / 52.20% | `h2000`: 1.96892 / 56.86% |
| fixed | 201 / min CE | `k.5,h2000`: 1.86706 / 58.80% | `h2000`: 1.96293 / 59.50% |
| fixed | 202 / min CE | `k.5,h1500`: 1.90185 / 57.94% | `h1500`: 1.94894 / 57.54% |
| fixed | 200 / max acc | `k0,h1000`: 1.99101 / 66.88% | `h1000`: 1.99945 / 62.26% |
| fixed | 201 / max acc | `k0,h2000`: 1.93870 / 63.72% | `h2000`: 1.96293 / 59.50% |
| fixed | 202 / max acc | `k0,h1500`: 1.97207 / 63.60% | `h1500`: 1.94894 / 57.54% |

The fixed-target selectors answer different questions. Minimum CE chose a
heterogeneous scalar procedure and obtained mean CE 1.8742 versus 1.9603 for
spectral, but spectral retained higher mean accuracy, 57.97% versus 56.31%.
Maximum accuracy uniformly chose `k=0`, obtaining the large accuracy advantage
while the mean CE difference was small (1.9673 scalar, 1.9704 spectral), not
an established equivalence.

## Full fixed-policy curves and endpoints

The following sequences are equal-seed auxiliary means at horizons
`100/250/500/1000/1500/2000`. They retain weak and nonmonotone arms rather than
showing only selected checkpoints.

| Target | Policy | Clean CE sequence | Clean accuracy sequence (%) |
|---|---|---|---|
| clean | `k0` | .4219/.3879/.3698/.3616/.3534/.3451 | 87.40/88.56/89.31/90.07/90.49/90.75 |
| clean | `k.5` | .4219/.3672/.3484/.3250/.3110/.2983 | 87.40/89.21/90.09/90.81/91.29/91.62 |
| clean | `k.9` | .4219/.3459/.3130/.2703/.2462/.2398 | 87.40/89.93/90.88/91.93/92.58/92.95 |
| clean | `k1` | .4219/.3344/.2838/.2490/.2229/.2205 | 87.40/90.27/91.49/92.49/93.27/93.48 |
| clean | spectral | .4219/.4036/.4088/.3754/.3565/.3528 | 87.40/88.19/87.93/89.40/90.17/90.31 |
| fixed | `k0` | 2.0902/2.0767/2.0228/1.9929/1.9731/1.9525 | 43.46/50.13/58.52/64.01/64.11/62.21 |
| fixed | `k.5` | 2.0902/2.0322/1.9781/1.9318/1.9062/1.8864 | 43.46/56.27/60.38/61.77/59.05/56.82 |
| fixed | `k.9` | 2.0902/1.9731/1.9240/1.9186/1.9399/2.0187 | 43.46/56.28/54.44/43.27/36.83/30.69 |
| fixed | `k1` | 2.0902/1.9395/1.8968/1.9484/2.0563/2.2294 | 43.46/51.59/49.68/33.97/27.89/23.96 |
| fixed | spectral | 2.0902/2.0530/2.0387/1.9985/1.9857/1.9848 | 43.46/49.50/50.87/56.05/56.73/53.63 |

Clean-target performance improves roughly with `k`: raw ends best, while
`k=0` and spectral remain underfit. Fixed-target behavior is nonmonotone.
`k=.9` and raw initially improve, then collapse as training continues; `k=.5`
and especially `k=0` retain much more clean utility. This horizon dependence
is why endpoint and validation-selected comparisons must remain distinct.

The complete h2000 spectral-minus-scalar effects are below. CE is utility
(positive means spectral has lower CE); accuracy is again percentage points.
Triples are seeds 200/201/202. These fixed-policy comparisons do not use a
selector and should not be substituted for the primary validation-selected
estimands.

| Target / scalar | CE seed triple | CE mean | Accuracy seed triple (pp) | Accuracy mean (pp) |
|---|---:|---:|---:|---:|
| clean / `k0` | -.00383/-.00140/-.01796 | -.00773 | -.10/-.64/-.60 | -.447 |
| clean / `k.5` | -.04721/-.05511/-.06130 | -.05454 | -.90/-1.50/-1.54 | -1.313 |
| clean / `k.9` | -.10691/-.12051/-.11174 | -.11305 | -2.46/-2.52/-2.96 | -2.647 |
| clean / `k1` | -.13167/-.14115/-.12420 | -.13234 | -3.16/-2.90/-3.46 | -3.173 |
| fixed / `k0` | -.02597/-.02424/-.04673 | -.03231 | -6.62/-4.22/-14.88 | -8.573 |
| fixed / `k.5` | -.09999/-.09587/-.09937 | -.09841 | -.74/+.70/-9.52 | -3.187 |
| fixed / `k.9` | +.03528/+.02695/+.03954 | +.03392 | +27.26/+27.24/+14.34 | +22.947 |
| fixed / `k1` | +.31637/+.17842/+.23907 | +.24462 | +35.08/+33.06/+20.88 | +29.673 |

Thus spectral loses to `k0` on both endpoint metrics in all three seeds and
loses mean CE and accuracy to `k=.5` for both targets. Against fixed-target
`k=.9` and raw, however, it wins strongly and uniformly. The scalar response
family spans both sides of the spectral result: weak response underfits clean
training but protects clean utility under corruption; high response learns the
clean target quickly but eventually fits the fixed realization. Spectral
protects against the latter failure, but is not uniquely able to do so.

## Absolute progress, realization fitting, and confidence

All policies share the h100 state. On fixed targets, `k0` progresses from
43.46% to 62.21% auxiliary accuracy and reduces CE by 0.13770. Spectral reaches
53.63% and reduces CE by 0.10539. Raw instead loses 19.50 accuracy points and
increases CE by 0.13923. This is improved learning by the low-response arms,
not merely preservation of h100, although their later peaks and endpoints
differ.

The training realization residual \(R_\zeta=\mathrm{fixedCE}-\mathrm{softCE}\)
provides a complementary boundary. Its common fixed-target h100 mean was about
-0.02727. At h2000 it was -0.07167 for `k0`, -0.15044 for `k=.5`, -0.63010
for `k=.9`, -1.16616 for raw, and -0.06001 for spectral. Thus `k0` nearly
matches spectral's protection but actually fits the fixed realization slightly
more; the strong scalar accuracy is not evidence of perfect corruption
rejection. Fixed-target confidence similarly ends at .1563 for `k0` and .1574
for spectral, versus .3742 for raw. On clean targets, `k0` and spectral remain
low-confidence/underfit (.8405 and .8425) relative to raw (.9659), consistent
with a global response constraint rather than target-aware selectivity.

## Optimization dose and geometry

The scalar result cannot be separated from dose. Over steps 101–2000, mean
fixed-target data-step squared energy/path length were only **0.00649/3.395**
for `k0`, versus **1.94469/60.124** for spectral and **12.61483/148.121** for
raw. Clean-target values were **0.01679/5.420**, **4.03752/85.769**, and
**2.86885/70.478**, respectively. The late 1001–2000 window retained 44.2% of
fixed `k0` energy, 54.1% of fixed spectral energy, and 75.6% of fixed raw
energy. Integrated raw-gradient/data-step dots were negative in every listed
arm; for fixed `k0`, spectral, and raw they were -0.273, -13.478, and -85.413.
These quantities show very different paths, not matched interventions.

Current-action-complement fractions also differ: for fixed targets they were
52.96% (`k0`), 21.25% (`k=.5`), 23.72% (`k=.9`), 30.83% (raw), and 0.141%
(spectral). The near-zero spectral fraction verifies spatial confinement of
its actual step; it is not a mediation fraction or proof that the confined
subspace is semantically clean. New scalar algebra residuals stayed far below
their recorded homogeneous bounds (maximum residual/bound ratio below .00083),
with empirical ideal-step defects below 4.27e-7. Small local defects do not
bound accumulated nonlinear trajectory sensitivity.

## Evidential strength and unresolved alternatives

Confidence is high in the conditional arithmetic and trajectory ordering: the
acquisition is complete, primary signs are preserved seed by seed, and the
independent audit binds reused and new artifacts. Confidence is moderate that
global temporal/dose regularization is the main explanation for this SGDm
panel. Confidence is low that any single scalar property—bandwidth, DC gain,
momentum memory, or path length—has been identified causally. These quantities
co-vary with `k`, and each branch's parameters alter its later gradients and
learned basis.

There are only three confirmation seeds and one model, SGDm rate, batch size,
rank, observer decay, and corruption construction. The fixed condition inherits
90% label replacement—81% incorrect in expectation—not an exhaustive model of
label noise. No confidence interval, sign pattern, or three-seed mean converts
these trajectories into a population theorem. Steps were not norm-restored or
matched across arms; the large dose differences are part of the treatment. A
stronger spatial-specificity claim would require a control matching the
relevant temporal transfer and realized dose without learned directions, or a
task where such controls fail while spectral succeeds. I16 supplies neither.

## Interpretation and boundaries

I16 weakens a spatial-necessity account for the I15 SGDm result. The simplest
supported account is now task-dependent temporal regularization: reducing
high-frequency/current-gradient response and carried momentum can improve
fixed-label clean utility, while the best response strength depends on target
and metric. This is compatible with I15's constructive finding that
mean/projected history learned useful information; I16 shows that its strongest
fixed-label accuracy was not specific to learned spatial selection.

Two asymmetries prevent a universal ranking. First, the I15 spectral comparator
was chosen after inspecting I15 outcomes on this same panel, whereas the I16
scalar grid was prospective. Second, within I16 scalar received 24 `(k,h)`
validation candidates versus six horizons for spectral. Validation selection
is legitimate here, but the study is neither an equal whole-program tuning
budget nor a fresh-benchmark replication. The same three seeds, dataset,
architecture, corruption realizations, and SGDm operating point remain.