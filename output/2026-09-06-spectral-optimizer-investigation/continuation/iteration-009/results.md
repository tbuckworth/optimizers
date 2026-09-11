# I9: preserved learning trajectories, but no late one-step denoising advantage

7 September2026. Prospective three-seed neural experiment; source/protocol
commit `4a71576`. Completed once and audited. This is new MNIST-training-split
evidence, not an official-test benchmark, a retry of I7, or another run of I8.

## High-level finding

The filter again strongly reduces late fitting of corrupted labels and preserves
clean classification accuracy. But the predeclared, same-state comparison finds
its delivered direction **worse for immediate clean cross-entropy at matched
data-step size**, in all three seed averages. Best sampled validation accuracy
is effectively tied in mean, while best validation cross-entropy favors AdamW
in every seed. Endpoint preservation and instantaneous clean-loss improvement
are different claims; this experiment supports the former, not the latter.

The strongest surviving interpretation is a restriction on what the network
continues fitting over time. A more specific, provisional possibility is that
it favors the softened population objective of symmetric label corruption over
fitting the particular corrupted-label realization. That can preserve class
rankings without recovering clean-label probabilities. The mathematics below
makes this possibility concrete; I9 does not establish it as the causal account.

## What ran

Fresh seed bundles100,101,102, each with paired raw AdamW and actual current32
filter trajectories: 2,000 steps, batch64, 5,000 fixed-corrupted training images,
5,000 disjoint clean-validation images and5,000 disjoint auxiliary images.
Architecture784–64–ReLU–10; unchanged canonical stable rank32 filter. Actual
incorrect-label fractions were .8158,.8074,.8106; nominal .9 replacement permits
replacement by the original digit. No official-test image was read.

Four complete model/AdamW/observer/RNG anchors per source, after steps100,500,
1500,2000. Frozen old-basis diagnostics use32 independent batch pairs. The
separate update batch observes the filter once before native projection.
Four real AdamW forks plus two artificial reciprocal data-displacement controls
share the same starting parameters, moments and counter. Independent auxiliary
gradient and finite-loss samples are separate. See the frozen [protocol](protocol.md).

## Primary result: native direction is locally adverse

Primary outcome is auxiliary clean cross-entropy change, averaged over the two
late current32-source anchors within each seed, then across seeds. Negative
contrasts would favor native. Common AdamW decay is retained in all finite-loss
evaluations; only the data-displacement norms are matched, not total-step norms.

| Seed | Native direction at raw data norm minus raw | Native actual minus raw direction at native data norm |
|---|---:|---:|
|100|+0.000393428|+0.000356646|
|101|+0.000162455|+0.000151381|
|102|+0.000298882|+0.000288378|
|Mean|+0.000284922|+0.000265468|

Five of six late anchor contrasts are adverse for both controls. Seed101 at
step1500 is favorable (about−0.0000357 for each); it is preserved, not discarded.
These are small local effects in cross-entropy units. Three seed bundles are
not six or24 independent optimizer replications, and no significance or
equivalence claim is made. All144 intervention cells are defined.

At these same current-source late states, actual raw/native updates have mean
auxiliary clean-loss changes +.000592813/+.000821662: **both increase clean CE**
on average. Meanwhile both reduce auxiliary soft-q loss, by about.0000905 and
.0000885. Independent gradients, finite loss, corrupted labels and soft targets
must therefore not be conflated. All other anchors, raw-source states, losses,
input-norm controls and signed utilities remain in the
[complete summary](analysis-001/summary.json) and original probe JSON (artifact not distributed in this public snapshot).

## Learning curves: endpoint preservation is not best-stop superiority

Means across the three paired seeds; peaks are descriptive selections on the
same validation data and sampled only every100 steps. They are not unseen-test
evaluations or claims about an unsampled continuous optimum.

| Measurement | Raw AdamW | Current32 |
|---|---:|---:|
|Endpoint clean-validation accuracy|30.04%|46.77%|
|Best sampled validation accuracy|52.39%|52.39%|
|Endpoint validation CE|2.08652|2.03423|
|Best sampled validation CE|1.85505|1.97564|
|Endpoint fixed-corrupted-label training accuracy|45.22%|16.63%|
|Endpoint fixed-corrupted-label training CE|1.65510|2.25876|

Endpoint accuracy differences are +10.90,+12.38,+26.90 percentage points.
Best-accuracy differences are +.18,−1.16,+1.00 points, mean+.00667 points; this
near-zero mean is not statistical equivalence. Raw best CE is better in all
three seeds. These results preserve, rather than resolve, historical I4/I6
selected-checkpoint sign changes. The new split/RNG namespace and validation-only
measurement also make this a distinct comparison, not a pooled continuation.

Full21-point curves, including training-clean performance, are directly retained
in all-curves.json (artifact not distributed in this public snapshot). Their temporal pattern supports
anti-memorization. It does not by itself identify which aspect of the filter
causes it or show that useful learning always exceeds an early-stopped baseline.

## Covariance source: mostly independent batch variation

For the previous mean a and frozen previous action P, independent pairs estimate
F=.5||g−g'||² and S=(g−a)·(g'−a), with innovation I=F+S. Means estimate conditional
batch covariance trace and squared mean surprise. Conditional here means the
fixed corrupted dataset and the already-learned state, not fresh label redraws.

Across seeds at current32 steps1500/2000, mean within-seed F/I is .9496/.9482;
after P it is .9130/.9161. P retains .4869/.4907 of fresh covariance energy and
.5069/.5097 of total innovation. Thus these retained innovations are not
surprise-dominated. Raw-source late F/I is .9293/.8973, with only .3504/.3519 of
fresh energy retained. The full seed/anchor results are in the summary.

Surprise estimates remain signed: current32 seed102 at step2000 has negative
estimated surprise, giving F/I>1. This is finite-pair estimation noise, not
negative population squared surprise. Ratios involving estimated surprise can
be unstable or exceed1 and must not be read as literal variance-retention bounds.
The basis and current conditional covariance both depend on training history;
this finding does not exclude optimizer/trajectory influence on their directions.

## Retained variation is not automatically clean utility

Previous-basis retention at late current32 states, seed-first means:

| Gradient/component | Step1500 | Step2000 |
|---|---:|---:|
|Training clean|.533|.570|
|Auxiliary clean|.525|.576|
|Training soft-q|.831|.826|
|Fixed-noisy training|.726|.723|
|Soft-q minus clean|.542|.571|
|Fixed-noisy minus soft-q|.470|.527|

The complete signed seven-component Gram matrix is retained, including both
residual components. Their energy fractions are not additive explanations:
cross terms and signs matter. Strong soft-target retention coexists with the
adverse primary clean-loss comparison. Higher clean retention at current versus
raw states compares different learned states, not a matched-state causal effect.

Native actual data-step energy outside the current gradient basis is about
57.9%/61.6% at the late anchors. Its norm is about96.5%/93.6% of the same-state
raw candidate's norm. Adam's delivered displacement is not confined to the
gradient subspace. Reciprocal controls match these actual data norms to maximum
relative error7.37e-8, so the primary result is not a disguised input-norm match.

## Conceptual refinement: classify well without recovering clean probabilities

**Theory, followed by a provisional interpretation.** For clean population
class probabilities r(x), uniform label replacement with probability rho gives

```
q(x) = (1-rho) r(x) + rho/K.
argmax q(x) = argmax r(x)        for rho<1, preserving ties.
```

With sufficient capacity and population-risk minimization, noisy cross-entropy
targets q, not r. At rho=.9,K=10 the class signal is compressed tenfold, but
its ordering survives. For a deterministic true class, q has mass.19 on it and
.09 elsewhere: a correct argmax can coexist with much worse clean negative
log likelihood than a confident correct predictor. The population clean-CE
excess at q is E_x KL(r(x)||q(x)); it vanishes only where q=r almost surely.
This is not a theorem about
finite-sample training, calibration of the I9 network, or an attained optimum.

This scaling/translation and classification distinction is established related
work, not a novelty claim: see §3.3 of
[Lukasik et al., ICML2020](https://proceedings.mlr.press/v119/lukasik20a/lukasik20a.pdf).
That paper connects label smoothing with noise correction and a regularization
view. [Patrini et al., CVPR2017](https://arxiv.org/abs/1609.03683) instead develop
explicit loss corrections using a noise-transition model. Spectral filtering
does not perform that correction and has no corresponding recovery guarantee.

On I9's fixed inputs define q_i=.1 one_hot(y_clean,i)+.09 and
zeta_i=one_hot(y_fixed,i)−q_i. Algebraically,

```
L_fixed(theta) = L_q(theta) + R_zeta(theta)
R_zeta(theta)  = -mean_i zeta_i^T log p_theta(x_i)
G_fixed       = G_q + G_zeta.
```

At fixed parameters independent of corruption draws, the realization residual
has expectation zero over fresh draws. At parameters learned from those fixed
labels it need not; conditional on the realized dataset it is a fixed function.
I9's q_i uses the observed clean label, not knowledge of the true population r(x).

**Working hypothesis:** learned restrictions may preferentially keep fitting
shared structure in L_q while slowing realization-specific R_zeta fitting.
This offers a genuine favorable account of preserved classification without
requiring projected gradients to approximate G_clean or each step to reduce
clean CE. Soft-q retention and the curve pattern are compatible with it, but
are not proof: generic underconfidence or other regularization can also produce
the accuracy/CE pattern. The one-step primary is adverse, and best-stop superiority is
still not established. It also leaves open whether Adam-state history, basis
rotation or effective capacity is the principal cause of the restriction.

## Next discriminating work

The next new experiment should branch from these complete states, not retrain
the six completed source trajectories. Predeclare short continuation horizons,
paired future batches and both classification and probability-loss metrics.
Compare raw/native behavior on fixed labels with a clearly labeled soft-q
shadow objective that removes the fixed realization term. Soft-q uses unavailable
clean information and is a mechanism comparator, not a deployable baseline.

If the native benefit grows over a branch while fixed-label residual fitting
diverges, that supports a cumulative trajectory restriction. A much smaller
effect after removing the realization term would strengthen the specific
memorization account; persistent harm or unchanged effects would weaken it.
Observer freezes and moment-state interventions can then distinguish retained
geometry from Adam memory. Neither a one-step null nor a favorable toy is a
reason to end the research; neither justifies ignoring this adverse primary.

## Evidence and audit limits

Completion manifest (artifact not distributed in this public snapshot) indexes86 original artifacts.
Compact scalar evidence is committed here. Complete tensor anchors, probe
vectors and saved plans remain under
`/tmp/spectral-experiment-artifacts/spectral-i9-001.2ykC0Z/artifacts`; they are not backed up.
[Independent audit](analysis-001/audit.json):8,697 checks pass, covering file and
complete-tree hashes, norms, signed utilities/Gram entries, projection geometry,
leakage and pair scalar identities. A separate
[reciprocal audit](analysis-001/reciprocal-audit.json) passes243 decay, norm,
reapplication and warmup checks. Independent JSON aggregation reproduces the
primary, covariance and learning-curve numbers. Ten focused CPU tests and a
source-identical synthetic GPU smoke passed before acquisition.

Individual pair gradients were deliberately not stored: their geometry cannot
be independently tensor-reconstructed from the saved scalars alone. Finite
forward losses were implementation-reviewed and their contrasts independently
aggregated, not independently reevaluated on another device. Complete state and
plans permit a future targeted replay without retraining. Three seeds, one small
architecture, one reused dataset and one severe symmetric-noise level limit
generalization. These findings do not change the optimizer defaults.
