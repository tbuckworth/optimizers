# I15 results: mean preservation helps, but carried history is not the fixed-label explanation

7 September 2026. This report interprets the frozen I15 SGDm history-transport
study defined in the [protocol](protocol.md) and
prospective predictions (artifact not distributed in this public snapshot). The primary numerical source is the
[independently derived summary](analysis-001/summary.json); its
[audit](analysis-001/audit.json) passed. The new acquisition contained 18/18
complete confirmation branches and no numerical failures. I14's six `raw` and
six `current32` SGDm trajectories were hash-bound references and were not
rerun.

## Headline findings

I15 separates delivered gradient information from the treatment of the old
SGDm momentum buffer. The four factorial cells are delivery
`current`/`mean` by history `native`/`projected`; `raw` is a separate I14
reference.

1. **Carried outside-action history was not necessary for the fixed-label
   benefit in this recipe.** At h2000, the registered history effect H
   under current delivery was slightly negative for clean CE in all three
   fixed-target seeds. Accuracy signs were mixed. On clean-target training,
   H was positive in all seeds, but both current-delivery arms had regressed
   from h100; native history merely regressed less.
2. **Adding the EMA mean complement robustly improved projected-history arms.**
   Every registered M seed value was positive for both target conditions
   and both metrics. The fixed-target `mean_projected_history` arm improved
   from 43.46% mean auxiliary accuracy at h100 to 53.63% at h2000, while its
   clean CE fell from 2.090 to 1.985.
3. **The mean-by-history interaction was target-dependent, not a general
   substitution result.** S was negative in all clean-target cells but
   strongly positive in all fixed-target endpoint cells. The positive fixed
   interaction arose because `mean_native` moved toward the raw arm's
   realization-fitting collapse while `mean_projected_history` did not. Under
   validation selection, fixed-target S remained positive for accuracy but
   became negative for CE under both selectors.
4. **Projection did remove the intended geometric quantity.** Projecting the
   old buffer drove current-delivery data-step action-complement energy from
   roughly 0.8–1.9% to about 2.5e-11 as a fraction of data-step
   energy. For mean delivery it reduced roughly 10% outside fractions to
   0.14–0.26%. This verifies the intervention's geometry; it does not make the
   removed energy a causal mediation fraction.

## Design and evidence integrity

All new branches started from the six I14 SGDm h100 `current32` states: seeds
200/201/202 by clean/fixed target. They used the same `.03` learning rate,
`.9` momentum, `.01` manually decoupled weight decay, data splits, fixed-label
corruptions and batch plans through h2000. At each step the unchanged observer
ingested the raw gradient once. `current_projected_history` delivered the
native current action and replaced only the old momentum buffer by its current
action. `mean_native` delivered the post-ingest mean-preserving direction while
leaving history alone. `mean_projected_history` combined mean preservation with
old-buffer projection. It did not project away the newly delivered mean
complement.

The acquisition was frozen at commit
`2bccbc6a883f1c4c11950965f9801d2c03d4350b`; the analyzer was independently
frozen at `61764c22ecf15318b41473cbc37e0a57ba85b073`. The three I15 plan JSONs
were byte-hash and exact-tree equal to their I14 counterparts. The audit
reported 4,357,895 checks, zero errors, 395 file-hash verification operations,
1,276,734,041 bytes streamed through those operations, and 30 complete-state
tree digests. These are verification operations, not 395 claimed unique files
or a claim that the streamed-byte counter is physical allocation. The I15 root
held 465,965,738 logical regular-file bytes under the 3 GiB cap. Its declared
runtime tree contained only an empty `torchinductor_titus/` directory.

The raw archival collector subsequently completed without modifying the
acquisition: 40 JSON records representing 260,148,592 source bytes were
collected into 30,772,819 bytes. Its collection record is
`raw-results-001/collection.json` (artifact not distributed in this public snapshot), SHA-256
`b50bd33762ac58b77fc0755418934432e503f3a92343724f33cef6cfffa700fa`.
An independent standard-library recomputation from the I14/I15 collected JSON
then matched 19,558 numeric and discrete scalar values exactly (maximum
absolute numerical difference 0), including all primary, selector, trajectory
and displacement-geometry rows. See
[`analysis-001/report-audit.json`](analysis-001/report-audit.json), SHA-256
`37018d3694f61476e9a7df094d1473db48623b4a878e5b5d4faadae0685d0933`.

## Registered h2000 effects

Utility is negative auxiliary clean CE for “CE” and auxiliary clean accuracy
for “accuracy.” Thus positive is favorable in both columns. Accuracy entries
below are percentage points. Each cell gives seeds 200, 201 and 202, then the
arithmetic mean. No uncertainty interval or population-level claim is implied.

| Family | Target | Metric | s200 | s201 | s202 | Mean |
|---|---|---:|---:|---:|---:|---:|
| `H = U(C,N) - U(C,P)` | clean | CE | +0.03283 | +0.02895 | +0.03727 | +0.03302 |
| H | clean | accuracy (pp) | +0.70 | +0.44 | +0.64 | +0.593 |
| H | fixed | CE | -0.00096 | -0.00198 | -0.00359 | -0.00218 |
| H | fixed | accuracy (pp) | -2.28 | +5.72 | +0.22 | +1.220 |
| `M = U(M,P) - U(C,P)` | clean | CE | +0.20730 | +0.22206 | +0.18758 | +0.20565 |
| M | clean | accuracy (pp) | +4.90 | +4.86 | +4.38 | +4.713 |
| M | fixed | CE | +0.08114 | +0.08419 | +0.09775 | +0.08769 |
| M | fixed | accuracy (pp) | +3.60 | +22.20 | +10.72 | +12.173 |
| `S = M - [U(M,N) - U(C,N)]` | clean | CE | -0.09843 | -0.11244 | -0.08312 | -0.09800 |
| S | clean | accuracy (pp) | -2.70 | -2.38 | -2.76 | -2.613 |
| S | fixed | CE | +0.22992 | +0.13368 | +0.21921 | +0.19427 |
| S | fixed | accuracy (pp) | +29.50 | +37.44 | +18.86 | +28.600 |

The clean H signs are consistent across seeds. Yet `current_native` fell
from 87.40% at h100 to 86.19%, and `current_projected_history` fell to 85.59%.
Its positive H is therefore evidence for relative preservation, not for
positive adaptation. Fixed H fails the prospective leading prediction:
the CE direction is adverse in every seed, while the mean accuracy benefit is
driven by mixed signs.

In contrast, M is directionally uniform in all 12 seed-by-target-by-metric
cells. Mean preservation supplies useful information after history projection.
It does not prove that the EMA estimates a semantically clean direction: the
same policy changes amplitude and the endogenous model, basis, gradients and
future optimizer state.

The interaction S is not stable across target conditions. Under clean
training, the mean effect was actually larger with native history: mean-native
versus current-native improved CE by 0.30364 and accuracy by 7.327 pp, compared
with 0.20565 and 4.713 pp under projected history. Under fixed labels, native
mean delivery harmed CE by 0.10658 and accuracy by 16.427 pp, whereas projected
mean delivery improved them by 0.08769 and 12.173 pp. That reversal produces
the large fixed S. It is evidence of a real interaction between mean
delivery, history treatment and target process, not a general estimate of
“fraction mediated by momentum.”

## Complete learning curves and absolute progress

The following cells are mean auxiliary clean CE / accuracy (%) across the
three seeds. Every policy shares the same h100 state within target and seed.

| Clean target | h100 | h250 | h500 | h1000 | h1500 | h2000 |
|---|---:|---:|---:|---:|---:|---:|
| raw | .422/87.40 | .334/90.27 | .284/91.49 | .249/92.49 | .223/93.27 | .221/93.48 |
| current-native | .422/87.40 | .412/87.86 | .442/87.11 | .463/86.87 | .483/87.17 | .525/86.19 |
| current-projected | .422/87.40 | .421/87.71 | .456/86.51 | .480/86.21 | .508/86.79 | .558/85.59 |
| mean-native | .422/87.40 | .341/89.95 | .288/91.44 | .249/92.59 | .226/93.32 | .222/93.51 |
| mean-projected | .422/87.40 | .404/88.19 | .409/87.93 | .375/89.40 | .357/90.17 | .353/90.31 |

| Fixed target | h100 | h250 | h500 | h1000 | h1500 | h2000 |
|---|---:|---:|---:|---:|---:|---:|
| raw | 2.090/43.46 | 1.939/51.59 | 1.897/49.68 | 1.948/33.97 | 2.056/27.89 | 2.229/23.96 |
| current-native | 2.090/43.46 | 2.053/49.19 | 2.068/48.25 | 2.078/49.01 | 2.074/46.86 | 2.075/42.68 |
| current-projected | 2.090/43.46 | 2.075/48.26 | 2.081/45.41 | 2.083/49.69 | 2.083/49.59 | 2.072/41.46 |
| mean-native | 2.090/43.46 | 1.941/54.24 | 1.903/52.44 | 1.931/36.51 | 2.016/30.01 | 2.181/26.25 |
| mean-projected | 2.090/43.46 | 2.053/49.50 | 2.039/50.87 | 1.999/56.05 | 1.986/56.73 | 1.985/53.63 |

On clean training, raw and mean-native both made about +6.1 pp absolute
progress from h100 and ended near 93.5%. Mean-projected made +2.91 pp, a real
recovery from h100 but still about 3.2 pp below raw. Both current-delivery arms
ended below h100. Thus mean preservation helps even without carried outside
history, while native accumulation remains important for matching raw clean
learning at this operating point.

For fixed targets, raw and mean-native improved initially and then collapsed.
Mean-native ended at 26.25%, only 2.29 pp above raw and 17.21 pp below its h100
level. Current-native approximately preserved h100 (-0.78 pp), and
current-projected declined by 2.00 pp. Mean-projected was the only fixed-target
cell with strong positive endpoint progress: +10.17 pp. It also had the best
mean h2000 clean CE among these five policies. This is the strongest practical
I15 result, but it is specific to fixed corruptions, SGDm `.03`, this rank and
this horizon.

## Validation-selected auxiliary comparisons

Selection is legitimate: each policy/seed chose among the six frozen horizons
using its validation split and was then evaluated on the disjoint auxiliary
split. It is nevertheless coarse offline checkpoint selection, not online
early stopping, an official test set, or task replication. Each effect cell
below gives `[s200,s201,s202]; mean`; accuracy is in percentage points.

| Target | Selector | Metric | H | M | S |
|---|---|---|---|---|---|
| clean | min validation CE | CE | [.00611,.01037,.00000]; .00549 | [.08450,.06465,.03941]; .06285 | [-.12539,-.13102,-.12039]; -.12560 |
| clean | min validation CE | acc. pp | [-.06,+.30,.00]; +.08 | [+3.62,+2.20,+1.84]; +2.55 | [-3.62,-2.52,-3.40]; -3.18 |
| clean | max validation accuracy | CE | [.00611,.01037,.00000]; .00549 | [.08450,.04783,.04538]; .05924 | [-.12327,-.14784,-.11441]; -.12851 |
| clean | max validation accuracy | acc. pp | [-.06,+.30,.00]; +.08 | [+3.62,+1.94,+1.84]; +2.47 | [-3.42,-2.78,-3.40]; -3.20 |
| fixed | min validation CE | CE | [.00103,-.00198,.01236]; .00380 | [.08114,.08419,.10647]; .09060 | [-.08383,-.05721,-.04026]; -.06044 |
| fixed | min validation CE | acc. pp | [-.66,+5.72,+2.14]; +2.40 | [+3.60,+22.20,+8.40]; +11.40 | [+5.10,+12.58,+4.56]; +7.41 |
| fixed | max validation accuracy | CE | [-.00637,.02328,-.02793]; -.00368 | [.08067,.16067,.10647]; .11594 | [-.04847,-.03195,-.06050]; -.04698 |
| fixed | max validation accuracy | acc. pp | [+2.96,-.72,+.88]; +1.04 | [+5.84,+13.16,+8.40]; +9.13 | [+5.48,+6.14,+4.84]; +5.49 |

The selected M effect remains positive in every displayed seed. The fixed
selected interaction is metric-dependent: positive for accuracy in all seeds,
negative for CE in all seeds. This is not a contradiction—the selected
checkpoints and metrics differ, and cross-entropy responds to confidence as
well as the argmax—but it rules out summarizing the result as one scalar
“rescue.”

For context, fixed-target mean-projected achieved mean selected auxiliary
accuracy 57.97% under minimum-validation-CE selection and 59.77% under
maximum-validation-accuracy selection, versus raw's 52.49% under both. The
corresponding mean CE values were 1.960 and 1.970 for mean-projected versus
1.884 for raw. Per-seed mean-projected accuracy was `[56.86,59.50,57.54]%`
under the first selector and `[62.26,59.50,57.54]%` under the second; raw was
`[52.20,50.80,54.48]%`. Thus the accuracy advantage is not a one-seed artifact,
while raw retains the better CE.

## Realization fitting and confidence

At the fixed-target endpoint, R-zeta = fixed CE - soft CE, and the
training confidence diagnostics were:

| Policy | fixed CE | soft CE | R-zeta | max probability | true-label probability |
|---|---:|---:|---:|---:|---:|
| raw | 1.682 | 2.848 | -1.166 | .374 | .176 |
| current-native | 2.278 | 2.301 | -.022 | .142 | .129 |
| current-projected | 2.281 | 2.302 | -.021 | .144 | .129 |
| mean-native | 1.674 | 2.787 | -1.113 | .360 | .177 |
| mean-projected | 2.245 | 2.305 | -.060 | .157 | .143 |

Raw and mean-native strongly preferred the realized fixed labels over the soft
expected target and became much more confident; both lost clean auxiliary
accuracy late. Mean-projected retained a much smaller negative R-zeta,
confidence closer to the current-filter arms, and substantially better clean
accuracy. This is consistent with suppressing native momentum amplification of
the delivered mean complement. It does not identify semantic denoising:
R-zeta, confidence and clean utility are associated trajectory outcomes,
and the policy also changes future representations, gradients and bases.

## Geometry, history amplitude and numerical fidelity

The mean current-action-complement fractions below aggregate energy within
each branch and then weight the three seeds equally. Values are percentages,
shown as data displacement / total displacement.

| Target/policy | steps 101–2000 | steps 1001–2000 |
|---|---:|---:|
| clean current-native | .788 / .897 | .675 / .785 |
| clean current-projected | ~0 / .144 | ~0 / .137 |
| clean mean-native | 11.366 / 9.158 | 8.345 / 5.256 |
| clean mean-projected | .260 / .136 | .277 / .103 |
| fixed current-native | 1.926 / 2.070 | 1.505 / 1.637 |
| fixed current-projected | ~0 / .179 | ~0 / .155 |
| fixed mean-native | 9.920 / 8.944 | 9.520 / 8.522 |
| fixed mean-projected | .141 / .196 | .136 / .156 |

For the current-projected cells, the nonzero total fraction is mostly the
separately applied decay; the data displacement itself is in the current action
to numerical precision. Mean-projected retains a small outside component
because the mean complement is deliberately delivered after projecting old
history.

Amplitude was not controlled and did not move uniformly. Over steps 101–2000,
mean-native versus mean-projected old-buffer squared energy was 1.409 versus
2.363 on clean targets, but 5.246 versus 1.137 on fixed targets. The old-buffer
action-complement energy (the saved `removed_old_buffer` diagnostic) was 0.162
versus 0.011 on clean and 0.548 versus 0.008 on fixed. This component was
actually removed only in projected-history policies; native-history arms
retained it.
Applied-delivery energy was 0.298 versus 0.634 on clean and 1.200 versus 0.250
on fixed. These target-dependent reversals matter: projecting history cannot be
described simply as “making steps smaller.” They also fit the prospective
fixed-action warning that native mean history adds another low-pass/DC gain,
while endogenous trajectories need not follow the ideal fixed-(P) ordering.

Signed dots tell the same descriptive story without establishing mediation.
For fixed mean-native, mean raw-gradient/data-displacement alignment was
-0.0335 over all steps and -0.0485 late, compared with -0.00709 and -0.00732
for mean-projected. Old-history-action-complement/data-displacement dots were
-0.0160 and -0.0218 for mean-native but about -5e-5 for
mean-projected. Again, the native arm retained this measured component while
the projected arm removed it. A negative raw-gradient/data-displacement dot is
first-order descent-oriented for that training minibatch. A negative dot with
an old-history component means the displacement opposes that component; it is
not descent for an identified clean objective. All magnitudes combine direction
and amplitude.

The saved implementation errors were small. The largest current-basis Gram
orthogonality error was 1.49e-6; the largest measured action
idempotence defect was 1.75e-6, against a maximum recorded
roundoff reference of 1.16e-4. The largest saved algebra residual
was 5.06e-7, the largest idealized data-step defect
6.22e-7, and the largest manual-versus-nominal decay discrepancy
4.51e-7. These support faithful execution of the specified
float32 recurrence. They do not prove that the learned span is the uniquely
right optimization subspace or turn action residuals into exact causal
orthogonal components. These are local per-step algebra checks; their small
residuals do not bound the sensitivity of an accumulated nonlinear training
trajectory or its final utility to perturbations.

## Conclusions and limits

I15 rejects the simplest carried-history account of I14's fixed-label SGDm
result: deleting old outside-action momentum did not reliably hurt the current
filter. It simultaneously strengthens a different mechanism claim. The EMA
mean contains useful adaptation information even when old history is projected,
and projecting history can prevent the severe fixed-label realization fitting
seen when that mean is natively accumulated. The strongest supported statement
is therefore about a target-dependent interaction between incoming mean
information and optimizer memory—not that leakage is universally useful, that
mean preservation is clean, or that SGDm momentum is the sole cause.

This is a three-seed MNIST MLP study at one calibrated SGDm rate, one rank,
one corruption level and one 2,000-step schedule. I14 references were accepted
and hash-bound rather than semantically replayed. New terminal complete states
were independently tree-digested; intermediate model-only states were file-
hash checked but were not complete optimizer/observer/RNG snapshots. The
validation/auxiliary procedure is sound for the frozen comparison, but it uses
only six offline horizons and no official test set or fresh dataset. No
norm-matched factorial was added, so temporal accumulation, amplitude and
orientation remain entangled. Finally, outside-action energy and signed dots
are mechanism diagnostics, not clean-utility fractions. These qualifications
limit generalization; they do not erase the internally consistent target
reversal or the all-seed mean-preservation effects.
