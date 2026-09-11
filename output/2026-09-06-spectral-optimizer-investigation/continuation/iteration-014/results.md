# I14 — fixed-label endpoint benefit survives under momentum SGD, while plain SGD is adverse

Codex — Spectral Optimizer Investigation, 7 September 2026.
Three fresh randomized confirmation bundles on small MNIST, following a separate
two-seed clean-baseline calibration. This is a conditional optimizer-mechanism
study, not an optimizer leaderboard, a fresh-dataset benchmark or a production
recommendation.

## Main finding

The native rank-32 current filter does not require AdamW's adaptive second
moment to produce a favorable fixed-corruption endpoint in this recipe.
With SGDm, filtering improves auxiliary clean CE and accuracy at update 2,000
in all three confirmation seeds: mean benefits are **+0.154752 CE** and
**+18.72 percentage points**. AdamW is also favorable in every seed on both
metrics, by **+0.091230 CE** and **+23.56 points**. Thus a coordinatewise
second-moment transform is not necessary for these measured fixed-label
effects.

The time course and stopping controls sharpen the interpretation. Under fixed
labels, SGDm's mean CE contrast remains adverse through update 1,500 and turns
favorable only at 2,000; AdamW CE turns favorable at 1,500. Validation-selected
CE comparisons remain adverse for every optimizer and target. The favorable
SGDm/AdamW fixed endpoints therefore mainly describe late preservation against
raw deterioration, with some absolute filtered progress, rather than a general
advantage over stopping raw training earlier. The clean costs and plain-SGD
failure are not secondary exceptions: they are central boundary conditions.

Evidence strength is **supported within this fixed design**: all 36
confirmation trajectories completed and the endpoint signs repeat across
three paired seed bundles. The bundles reuse one MNIST source dataset and the
rates are independently calibrated but not exhaustively tuned, so this is not
task-level replication or a universal optimizer claim.

## Design, calibration and evidence

The [prospective protocol](protocol.md) froze acquisition at commit `d48f20a`.
Every trajectory uses the same 784–64–ReLU–10 MLP, batch size 64, 2,000 updates,
explicit decoupled weight-decay coefficient `.01`, and the canonical observer:
rank 32, covariance decay `.99`, 100-update warmup, stable update and hard
weighting. Both arms ingest each raw gradient once. Raw restores that gradient;
current32 delivers the native post-ingest projection. Full state and evaluation
digests agree through update 100, so the first treatment delivery is update
101. The h100 filter effect is therefore exactly zero by construction.

Calibration and confirmation use globally disjoint pools from the reused
MNIST training source. The 18 calibration trajectories are seeds 190/191 ×
three bases × three prospective rates, all clean-target raw baselines. All 18
completed and all nine rates exceeded the required 85% validation-accuracy
floor in both seeds. Selection used only mean final validation CE:

| Base | Selected rate | Calibration CE, seeds 190/191 | Accuracy, seeds 190/191 |
|---|---:|---:|---:|
| SGD | .1 | .358011 / .332751 | 90.44% / 91.62% |
| SGDm(.9) | .03 | .269452 / .215649 | 92.62% / 93.82% |
| AdamW | .001 | .283689 / .250379 | 92.42% / 93.04% |

SGD and AdamW select interior grid points. SGDm selects its upper boundary,
where its mean calibration CE is still improving; untested higher rates could
alter its operating point. No confirmation outcome was used to choose or
expand the rates.

Confirmation is exactly 36 fresh trajectories: seeds 200/201/202 × SGD/SGDm/
AdamW × clean/fixed targets × raw/current32 policies. Within a seed, all 12
cells share weights, examples, batches and the saved corruption plan. The
fixed-label plans replace 4,492/4,520/4,490 of 5,000 labels, of which
4,016/4,028/4,032 are actually incorrect. The nominal 90% replacement is
therefore about 81% incorrect labels, as expected under uniform replacement.
All 36 trajectories completed 2,000 updates with no numerical failure. Results
below are equal-weight means over three paired seeds; steps and checkpoints are
not additional replications.

For CE, define benefit as

\[
B_{\rm CE}=L_{\rm aux,raw}-L_{\rm aux,current32};
\]

for accuracy, define

\[
B_{\rm acc}=A_{\rm aux,current32}-A_{\rm aux,raw}.
\]

Positive values favor filtering. The complete registered aggregation and every
seed value are in the [analysis summary](analysis-001/summary.json); the
lossless scalar archive (artifact not distributed in this public snapshot) preserves all 94
original JSON files.

## Twelve registered endpoint effects

The following are the 12 primary effects at update 2,000. Each cell lists the
three paired seed values followed by their arithmetic mean. Accuracy is in
percentage points.

| Base / target | CE benefit: seeds 200, 201, 202; mean | Accuracy: seeds 200, 201, 202; mean |
|---|---:|---:|
| SGD / clean | −1.083056, −1.057449, −.971889; **−1.037465** | −34.76, −26.82, −27.26; **−29.613** |
| SGD / fixed | −.198147, −.206285, −.249395; **−.217942** | −1.74, −13.94, −19.56; **−11.747** |
| SGDm / clean | −.306139, −.334261, −.274506; **−.304969** | −7.36, −7.32, −7.20; **−7.293** |
| SGDm / fixed | +.234272, +.092254, +.137729; **+.154752** | +29.20, +16.58, +10.38; **+18.720** |
| AdamW / clean | −.156225, −.197906, −.182775; **−.178969** | −4.88, −5.38, −5.18; **−5.147** |
| AdamW / fixed | +.102282, +.095503, +.075904; **+.091230** | +29.52, +24.74, +16.42; **+23.560** |

Every sign is repeated in all three seeds. There is no evidence here for a
clean-target benefit: filtering is adverse under each base, and particularly
damaging under plain SGD. Conversely, fixed-target benefit is not universal
across bases: it appears under SGDm and AdamW but reverses under SGD.

The descriptive SGDm-minus-SGD interaction is +.372694 CE and +30.467 points
under fixed labels; AdamW-minus-SGD is +.309172 CE and +35.307 points. SGDm
and AdamW are closer: AdamW-minus-SGDm is −.063522 CE but +4.840 points. These
are not momentum or adaptive-moment causal effects. The bases differ in rate,
the per-update shrinkage factors are respectively `1−.1×.01`,
`1−.03×.01` and `1−.001×.01`, and their native step dynamics differ.

## Absolute learning and the h100 reference

Raw and current arms are identical at h100. Their mean auxiliary outcomes at
that common stopping point are:

| Base | Clean target: CE / accuracy | Fixed target: CE / accuracy |
|---|---:|---:|
| SGD | .646041 / 84.42% | 2.206207 / 35.613% |
| SGDm | .421912 / 87.40% | 2.090186 / 43.460% |
| AdamW | .532745 / 86.173% | 2.058584 / 48.647% |

The next table separates relative effects from actual learning. `Δh0` and
`Δh100` use positive-good units: CE reduction and accuracy gain from the named
checkpoint to h2000.

| Base / target / policy | h2000 CE / accuracy | Δh0 CE / pp | Δh100 CE / pp |
|---|---:|---:|---:|
| SGD / clean / raw | .317569 / 91.560% | +1.990929 / +80.307 | +.328472 / +7.140 |
| SGD / clean / current | 1.355034 / 61.947% | +.953464 / +50.693 | −.708993 / −22.473 |
| SGD / fixed / raw | 1.940575 / 51.367% | +.367923 / +40.113 | +.265632 / +15.753 |
| SGD / fixed / current | 2.158517 / 39.620% | +.149981 / +28.367 | +.047690 / +4.007 |
| SGDm / clean / raw | .220500 / 93.480% | +2.087998 / +82.227 | +.201412 / +6.080 |
| SGDm / clean / current | .525469 / 86.187% | +1.783030 / +74.933 | −.103557 / −1.213 |
| SGDm / fixed / raw | 2.229414 / 23.960% | +.079084 / +12.707 | −.139228 / −19.500 |
| SGDm / fixed / current | 2.074662 / 42.680% | +.233836 / +31.427 | +.015524 / −.780 |
| AdamW / clean / raw | .260194 / 92.540% | +2.048304 / +81.287 | +.272551 / +6.367 |
| AdamW / clean / current | .439163 / 87.393% | +1.869336 / +76.140 | +.093583 / +1.220 |
| AdamW / fixed / raw | 2.073719 / 28.647% | +.234779 / +17.393 | −.015135 / −20.000 |
| AdamW / fixed / current | 1.982489 / 52.207% | +.326009 / +40.953 | +.076094 / +3.560 |

The fixed-target SGDm benefit is predominantly protection: raw loses 19.50
accuracy points after warmup while current loses .78 points. Its filtered CE
improves only .015524 after h100, whereas raw CE worsens .139228. AdamW is a
stronger mixed case: current improves both clean CE and accuracy after h100,
while raw loses on both. Plain SGD refutes a generic “filtering prevents noisy
failure” account in this operating point: both arms improve, and raw improves
substantially more.

Under clean targets, raw improves after h100 for all three bases. AdamW current
also makes absolute progress, but less than raw. SGDm current slightly
deteriorates and SGD current collapses sharply. Filtering can therefore retain
some useful learning under AdamW while imposing a much harder adaptation cost
under nonadaptive SGD.

### Horizon pattern

The fixed-target mean effects are not favorable from treatment onset. At h250,
CE/accuracy benefits are −.102684/−13.68 points for SGD,
−.113786/−2.40 for SGDm and −.084292/−2.96 for AdamW. AdamW accuracy first
turns positive at h500 and CE at h1500. SGDm accuracy turns positive at h1000,
but CE is still −.017995 at h1500 and becomes +.154752 only at h2000. SGD is
adverse at every post-warmup checkpoint on both metrics. All clean-target
effects are adverse at every post-warmup checkpoint and generally worsen with
time. The late crossovers support an endpoint-protection interpretation rather
than immediate clean-direction improvement.

## Validation-selected comparisons remain mostly adverse

Each trajectory was independently selected over the six nonzero checkpoints,
once by minimum validation CE and once by maximum validation accuracy. Both
auxiliary metrics were then evaluated at both selectors. These are supporting
comparisons, not replacements for the fixed endpoint. Mean effects are:

| Base / target | Min-validation-CE selector: CE / pp | Max-validation-accuracy selector: CE / pp |
|---|---:|---:|
| SGD / clean | −.311862 / −9.360 | −.326584 / −7.147 |
| SGD / fixed | −.216273 / −15.233 | −.148797 / −11.100 |
| SGDm / clean | −.189650 / −5.673 | −.189780 / −5.700 |
| SGDm / fixed | −.162773 / −3.527 | −.205765 / −.820 |
| AdamW / clean | −.178789 / −5.027 | −.179052 / −5.060 |
| AdamW / fixed | −.076892 / +3.927 | −.057494 / +1.300 |

Every selected CE contrast is adverse. Mean selected accuracy is favorable
only for AdamW fixed labels, and its CE remains adverse in every seed under
both selectors. The seed detail is less uniform than that mean: SGDm fixed
accuracy is favorable in seed 200 but adverse in seeds 201/202 under each
selector. AdamW fixed accuracy is favorable in all three seeds under the
minimum-CE selector (**+6.70, +1.72, +3.36 points**), while the
maximum-accuracy selector is **+.24, +4.36, −.70 points**. Thus the SGDm fixed
endpoint advantage disappears on average under both validation selectors,
whereas AdamW retains a small, metric-dependent selected accuracy benefit.
This is consistent with raw trajectories having useful earlier checkpoints
before late deterioration, not with filtering dominating raw training at its
own best sampled stop.

The selected horizons also expose how different the paths are. For clean SGD,
the minimum-CE selector chooses raw/current at 2000/250 in all seeds, while the
accuracy selector chooses raw at 1500/2000/2000 and current at 100 in all
three. For fixed SGDm, minimum-CE raw/current horizons are
500/250, 500/2000 and 250/250; fixed AdamW chooses 500/2000,
500/2000 and 500/1500. These are legitimate fixed checkpoint-selection
comparisons: selection uses validation data and evaluation uses the disjoint
auxiliary set. They are nevertheless coarse, offline six-checkpoint selectors,
not online stopping policies, untouched official-test estimates or independent
task replications.

## Fixed-realization fitting and confidence

The fixed-label raw branches do fit their particular corruption realizations,
most strongly under SGDm and AdamW. Here
`R_zeta = fixed training CE − expected-soft training CE`; more negative values
mean more realization-relative fitting, not better clean utility. Confidence
is mean maximum training probability.

| Base / policy | Fixed CE / fixed accuracy at h2000 | R_zeta at h2000 | Confidence at h2000 |
|---|---:|---:|---:|
| SGD / raw | 2.192839 / 21.667% | −.124716 | .172796 |
| SGD / current | 2.289170 / 14.093% | −.010557 | .126376 |
| SGDm / raw | 1.681968 / 42.947% | −1.166165 | .374152 |
| SGDm / current | 2.278414 / 15.127% | −.022100 | .142256 |
| AdamW / raw | 1.639775 / 45.707% | −1.038770 | .328625 |
| AdamW / current | 2.255629 / 17.747% | −.051913 | .158464 |

At h100, raw and current share `R_zeta` of −.012971/−.027273/−.036415
for SGD/SGDm/AdamW. By h2000, filtering keeps realization fitting and
confidence near their warmup scale, while raw SGDm and AdamW substantially fit
the fixed realization. This supports a protection component in their endpoint
effects. It is not semantic denoising: the same restriction damages every
clean-target endpoint, and plain SGD suppresses realization fitting yet still
loses badly on clean auxiliary utility. Suppression of fixed-label fitting is
neither sufficient for a favorable clean result nor evidence that the learned
subspace contains only clean signal.

## Update geometry: projection, momentum and adaptive scaling separate

The prospective optimizer note (artifact not distributed in this public snapshot) distinguishes the
delivered gradient from the actual optimizer step. For SGD, the decay-subtracted
data step is directly proportional to the delivered gradient. SGDm applies an
accumulated first-moment buffer, and AdamW applies first- and second-moment
state coordinatewise; neither downstream transform must preserve the current
span.

The table reports current32 data-step leakage as summed outside-span squared
energy divided by summed data-step squared energy within each branch, then
equal seed averaging. `Full` includes the unfiltered 100-step warmup; `101+`
isolates the active-filter window.

| Base / target | Full trajectory | Steps 101–2000 |
|---|---:|---:|
| SGD / clean | .008318 | 3.72e−12 |
| SGD / fixed | .008109 | 6.10e−12 |
| SGDm / clean | .093223 | .007884 |
| SGDm / fixed | .044636 | .019258 |
| AdamW / clean | .790835 | .775849 |
| AdamW / fixed | .606094 | .591656 |

The active-window SGD step is numerically confined to the current span, as
predicted. SGDm leakage is small but nonzero because its momentum buffer
contains history under previous bases. AdamW steps remain mostly outside the
current span despite receiving projected gradients, consistent with its
coordinatewise adaptive transform. Raw active-window data-step leakage, for
reference, is 13.64%/16.81% for SGD clean/fixed, 23.81%/30.83% for SGDm and
78.92%/68.42% for AdamW; these use each raw branch's own evolving basis.

This geometry makes the necessity result more specific. Favorable fixed-label
SGDm endpoints occur with only about 1–2% outside-span data-step energy, so
AdamW's large diagonal-remapping leakage is not required for that effect.
Momentum history remains a possible modifier. The geometry does not show that
momentum causes the benefit: bases, rates, decay contractions and endogenous
trajectories differ.

The [mathematical interpretation](mathematical-interpretation.md) gives a
second exact reason not to attribute the cross-base split only to leakage. At
a constant ideal fixed point, plain SGD with gradient \(g\) and decoupled
coefficient \(\lambda\) requires \(g=-\lambda\theta\). Unnormalized momentum
has \(b=g/(1-\rho)\) and requires
\(g=-(1-\rho)\lambda\theta\). With \(\rho=.9\), the same nominal coefficient
therefore corresponds to one tenth the gradient-level regularization in this
limiting calculation. The neural trajectories are neither constant nor
deterministic, but the identity and the sharply different raw fixed-label
progress—SGD improves after h100 while SGDm and AdamW deteriorate—provide
alternate explanations for the sign split. “Momentum can coexist with the
benefit” is supported; “momentum caused it” is not.

Nor is filtering a common scalar shrinkage. Over steps101–2000, mean
current-versus-raw data-displacement norms are `.08549/.07391` for clean SGD
but `.03533/.06617` for fixed SGD; `.04771/.03709` for clean SGDm but
`.02797/.07796` for fixed SGDm; and `.04103/.03599` for clean AdamW but
`.04186/.05586` for fixed AdamW. Filtering increases mean data-step magnitude
in all clean cells and decreases it in all fixed cells. These are path-dependent
descriptive measurements, not norm-matched interventions or mediation estimates.

## Interpretation and hypotheses after I14

1. **AdamW's adaptive second moment is not necessary for the tested fixed-label
   endpoint effect — supported.** SGDm is favorable on both primary metrics in
   every seed, and its actual steps have far less current-span leakage than
   AdamW's. This conclusion is conditional on the selected `.03` SGDm recipe,
   rank 32, 2,000 updates, one architecture and one reused dataset.

2. **Optimizer memory or downstream transformation moderates an overly hard
   projection — plausible, not identified.** Plain SGD's exact active-span
   confinement accompanies severe clean underlearning and adverse fixed
   outcomes. SGDm performs much better with a small amount of momentum-induced
   leakage; AdamW performs better still on fixed accuracy with much larger
   leakage. But the cross-base rate and decay differences, SGDm's upper-grid
   selection, and endogenous path changes prevent a causal momentum claim.

3. **The native fixed-label benefit is mainly late protection, with
   optimizer-dependent continued learning — supported.** Raw SGDm and AdamW
   deteriorate sharply after h100 under fixed labels. Current SGDm largely
   preserves accuracy and makes a small CE gain; current AdamW improves both
   metrics. The benefit emerges late and is substantially weakened or reversed
   by validation selection. Plain SGD instead lets raw continue improving more
   than current.

4. **Projection alone is a generally useful directional regularizer —
   contradicted at this operating point.** Plain SGD is adverse in all seeds on
   all four endpoint target/metric cells, while all bases are adverse on clean
   targets. This does not erase earlier single-seed legacy successes, but it
   prevents treating the current stable filter as generic denoising or as an
   optimizer-independent default.

5. **Large AdamW outside-span step leakage is the sole route to benefit —
   disfavored.** SGDm provides a counterexample within this experiment. The
   result does not distinguish buffer memory, basis rotation, rate scale,
   decay, or other trajectory effects.

The historical single-seed cross-optimizer studies remain relevant but used
legacy covariance updates, test-selected rates, and different optimizers. I14
is stronger fresh paired evidence for the current stable implementation; it
does not pool away those earlier mixed results or convert one task into a
general theorem.

## Integrity, audit exception and limits

The original independent CPU audit is **not a blanket pass**. It retains
status `fail` because the artifact root contained one unlisted entry: an empty,
non-symlink `torchinductor_titus` runtime-cache directory. Exactly
1,199,604 of 1,199,605 checks passed, including all 502 declared artifact
hashes, 1,198,339,589 streamed bytes, 162 complete-state tree digests, all
18 calibration and 36 confirmation trajectories, five reconstructed plans,
and all 12 primary estimands. The maximum recorded training residual identity
error is zero and the maximum leakage-energy identity error is
`1.73e−18`. Saved scalar evaluations were not rerun through the model, and
intermediate model checkpoints were hash-checked rather than semantically
replayed.

The separate runtime-directory supplement (artifact not distributed in this public snapshot)
accepts only that exact empty directory after no-follow identity and
before/after emptiness checks. It pins the failed audit and summary, rehashes
all 502 artifacts and binds phase completion and attempt records. It does not
rewrite the audit, rerun its scalar/state semantic checks, or exempt any
nonempty, hidden, nested, symlink or differently named entry. This bounded
disposition supports using the otherwise complete evidence while preserving
the original failure honestly.

The corrected lossless collector archives 94 JSON files: 169,757,860 original
bytes compressed to 21,965,320 bytes. Bulk tensor states remain hash-bound on
the verified large volume but are not part of a Git clone and are not backed
up. No official test set, model-forward audit, per-step vector replay, extra
rate, optimizer, seed or restart was used. Acquisition changed no production
optimizer default, used no paid compute, and does not authorize a universal
recommendation.

## Conclusion