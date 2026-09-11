# I20: better direction memory plus a less brittle response

8 September 2026. **Audited exploratory counterfactual on 32 reused I19 seed
bundles. Not fresh replication, neural training evidence or a default change.**

## Executive finding

There is now a constructive candidate beyond ordinary common smoothing in
the deliberately favorable, strongly changing-signal case. Combining a longer
covariance-estimation memory with a shorter complementary response gives
whole-horizon mean squared error **0.620737**, versus **0.694234** for EMA .9
and **0.663074** for the known-model common Kalman control. The latter margin
is modest: **0.042337 ± 0.021946 SE**, favorable in **23/32 reused seeds**.
This is a promising lead, not confirmed superiority.

The same candidate loses to both controls in **every seed** when signal change
is weak or absent. The practical implication is therefore not “replace the
moving average.” It is that direction estimation and the cost of a wrong
direction can jointly limit spectral performance, and that a matched
direction/response design merits fresh confirmation under favorable and
changing-direction conditions. Better covariance estimation still has no
semantic test for which variation is useful.

All numerical claims below refer to the accepted
[summary](analysis-001/summary.json), whose per-seed rows and registered
contrasts retain the full grid, and its independent
[audit](analysis-001/audit.json). Earlier [I19 results](../iteration-019/results.md)
remain unchanged.

## What was held fixed and what changed

The [protocol](protocol.md) reuses every saved I19 observation, mean, native
action and latent evaluation target: seeds19000–19031, process variances
0/.01/.1, paired rotations0/π4,4000 steps,192 streams. There was no random
redraw, native-observer replay, optimizer step, neural pass or GPU work.
All outputs start exactly at the same first observation.

The direction factor contains native, legacy full moment, zero-initialized
EW .99, zero-initialized EW .999, and separately labelled useful/nuisance
direction oracles. EW .999 versus EW .99 isolates forgetting with matched
initialization and availability rules; comparing only with legacy full .99
would confound memory and its overweighted first residual. Missing full/EW
directions use identity, while native actions are copied exactly. In the
strong cell each non-oracle estimator is unavailable only at the first step
of each seed; none is unavailable in the late window.

Each direction is crossed with two fixed fast decays, .9 and
rho*≈.729844, and three response rules: previous constant-preserving `cp`,
two-sided recurrence with complementary decay .99 (`rec99`), and the same
recurrence with complementary decay .9 (`rec9`). The same settings are used
in every process cell. The rho=.9/rec9 member is **exactly ordinary EMA .9**,
not a new learned-direction success. Useful/nuisance oracles know the
generating direction; common Kalman knows the process/noise model.

Primary results are identity rotation over all4000 steps. Startup1–100,
transition101–1000 and late1001–4000 are secondary. Every summary gives equal
weight to all32 reused seed bundles. There are27,648 new per-seed/window
metrics,864 new means,472 retained parent means,2160 registered contrasts and
270 identity-whole primary contrasts. No pooled winner or extra tuned selector
is introduced, and the grid is outcome-informed rather than a fresh test set.

## Performance: whole horizon first

Mean vector MSE; lower is better. All spectral rows below use rho*.
These are descriptive cross-cell comparisons, not a pooled ranking.

| Policy | No signal change | Weak change | Strong change | Strong late, secondary |
|---|---:|---:|---:|---:|
| Reused EMA .9 | .271223 | .313665 | .694234 | .685154 |
| Reused common Kalman | .009863 | .220254 | .663074 | .658199 |
| Native direction, previous CP | .641107 | 1.079888 | 1.427746 | 1.181915 |
| Native direction, rec99 | .643147 | 1.077966 | 1.757737 | 1.476799 |
| Native direction, shorter complement rec9 | .683563 | .723785 | .678755 | .633349 |
| EW .99 direction, rec9 | .688220 | .729694 | .749578 | .731527 |
| EW .999 direction, rec9 | .686042 | .727748 | **.620737** | **.567253** |
| Fixed useful-direction oracle, rec9 | .374243 | .385582 | .488067 | .482186 |

The whole-horizon candidate/EMA difference is **+.073497 ± .023537 SE**,
favorable in27/32 seeds. Against common Kalman it is **+.042337 ± .021946**,
23/32 favorable. Those are reductions of approximately10.6% and6.4% in
the corresponding mean MSE, not percentages of seeds or confidence levels.
The stronger comparator margin is less than two reported SEs; these are not
multiplicity-adjusted significance results.

Late-window candidate MSE is.567253, with differences **+.117901 ± .024974**
against EMA and **+.090946 ± .023611** against common Kalman, both29/32
favorable. This late gain must not replace the whole-horizon primary. Startup
and transition remain adverse against common Kalman: candidate.964500/.760820
versus common Kalman.730030/.671884. The positive whole average therefore
survives early costs, rather than showing uniform improvement at every stage.

Weak-change candidate MSE.727748 is much worse than.313665/.220254 for the
two common controls; no-change MSE.686042 is much worse than.271223/.009863.
Both controls win all32 seeds in both regimes, over the whole and late
windows. The original non-star CP/.9 no-change advantage over EMA .9 also
remains present (.228737 versus.271223); it is not erased by highlighting
the strong-cell candidate.

## Direction memory: the matched prediction holds

Strong-cell useful squared alignment, averaged within available steps and
then equally across seeds:

| Estimator | Whole mean | Late mean ± SE | Late mean action-change norm |
|---|---:|---:|---:|
| Native | .691128 | .755563 ± .039265 | .008604 |
| Legacy full .99 | .596209 | .615854 ± .019982 | .038700 |
| Standard EW .99 | .596183 | .615855 ± .019982 | .038700 |
| Standard EW .999 | .787525 | **.861536 ± .038076** | **.006808** |

The prospectively stated late EW .999 > EW .99 alignment prediction holds:
the paired difference is+.245680 ± .028949 SE, favorable in30/32 seeds.
At fixed rho*/rec9 response, longer memory
reduces strong whole MSE by **+.128842 ± .014036**, favorable in30/32 seeds;
late reduction is **+.164274 ± .016783**, also30/32. This is a direct
same-initialization, same-input, same-response memory contrast in this toy.
The late legacy/standard .99 similarity also limits an initialization-only
explanation of the stronger .999 late direction result.

This does not establish that covariance fidelity is semantic usefulness.
Under weak change, EW .999 late useful-axis alignment is only.000560, versus
.005082 for EW .99: it stabilizes selection of the nuisance direction.
With no changing signal, the corresponding e1 alignment is merely a reference
axis diagnostic, not accuracy at tracking a genuine useful direction.
Nor is native-versus-full a pure rank-truncation intervention: estimator
recurrences and initialization differ.

## Response design: shorter memory helps; transport alone does not

For the **same saved native direction**, changing CP to rec99 worsens strong
whole MSE from1.427746 to1.757737. The registered effect is
**−.329990 ± .025736**, adverse inall32 seeds; late is also adverse inall32.
Simply carrying both response components forward is not the remedy here.

Shortening rec99's complementary memory to.9 instead reduces native strong
whole MSE to.678755: **+1.078981 ± .111077**, favorable in31/32 seeds.
But that native variant still has a worse whole mean than common Kalman:
effect **−.015682 ± .024213**,18 favorable/14 adverse seeds. Its late mean
is favorable, but remains secondary and mixed (23/32).

Longer covariance memory while retaining CP also does not clear common
Kalman in whole mean: EW .999/rho*/CP is.985065. For EW .999, shortening
rec99's complement reduces whole MSE by **+.593108 ± .087012**,31/32
favorable, to the.620737 candidate. The factorial thus exposes two useful
engineering choices in combination, not a uniquely identified additive
mediation decomposition or proof that either intervention universally helps.
With rho=.9, rec9 outputs match parent EMA .9 with **zero measured numerical
discrepancy** for every estimator in the accepted audit, preventing that
isotropic special case being counted as evidence
that learned directions are necessary.

## How the mathematics fits—and where it stops

The reviewed [response theory](response-theory.md) proves two identities.
For a fixed ideal projector, CP and rec99 are identical under matched
initialization. When the projector moves, their difference has an additional
term projecting the old CP-versus-mean discrepancy into the newly assigned
complement. Both preserve constant input; rec99 is not simply CP rewritten.

The second point is the lag/noise trade-off. With an imperfect fixed,
data-independent direction, complementary decay.99 strongly smooths noise
but makes a changing signal misassigned there expensive to follow. At rho*,
shortening it to.9 reduces the theoretical squared-alignment threshold to
beat the admissible common-LTI stationary bound fromabout.930 toabout.714.
The price is worse ideal-direction risk (.480683 instead of.290257).
This is **conditional fixed-direction mathematics**. Do not substitute the
observed native/EW mean alignment into its risk formula: these actions move
and depend on current observations and response history.

The measured direction-memory contrast and fixed-action response contrasts
support a constructive interpretation: noisy finite-memory direction
estimates and a costly slow complement were both practical limitations in
the favorable toy. They do not identify the causal explanation of neural
anti-memorization, establish optimizer-general superiority, or turn centered
covariance into a useful-information detector. Known anisotropy and direction
oracles remain important sources of the conditional headroom.

## Next test and confidence

Freeze a small confirmation roster on **fresh independent seeds**, retaining
the two common controls, previous CP, native rec9, matched EW .99/.999 rec9,
and labelled directional references. Include the existing weak/no/strong
conditions and a **true-direction switch** with fixed response settings.
The longer covariance memory may then lag and lose its gain; this adverse
case is essential to the steelman, not grounds to reject it without testing.
Specify the moving latent-signal construction and causal oracle before
launch. Do not relabel this I20 grid or paired rotation as confirmation.

Confidence is high in the audited finite-stream arithmetic and conditional
algebra; provisional in the candidate's useful operating regime; insufficient
for fresh-seed, changing-direction or neural efficacy. No production code,
optimizer default or paid-compute allocation has changed.

The paired rotations are numerically consistent: the maximum absolute
rotated-minus-identity mean MSE difference over all432 policy/process/window
pairs is2.47e−14. This checks the paired coordinate construction; it supplies
no additional independent seeds or general neural invariance claim.

## Integrity and artifacts

Reconstruction completed192/192 streams in66.125s; independent audit completed
with42,444 checks over155,125,424 numeric values andzero errors. Original
source freezes, parent hashes, exact live-to-terminal handles, storage and
resource bounds are in launch.md (artifact not distributed in this public snapshot). Both handles are consumed;
neither should be restarted.

- Acquisition freeze:11c428f2347137377563c69a4f629c1e75192198.
- Analysis freeze:e5e4fc6574788b27895694a10e700555fc2c4600.
- Audit SHA:01ab83a51ab4610f374bad2a1fb086513650ca5ea5ec04890254e5bc1e4a982d.
- Summary SHA:d8aee4475c0b18e85f09e2daa63aad84cfd0e484eda0f37fd0c90fb6c3daed69.

The presentation roster (artifact not distributed in this public snapshot) was chosen before outcome
access. Figures are compact subsets; all registered means, contrasts and
negative cases remain in the accepted summary. Paid spend/reservation:0/100USD.
