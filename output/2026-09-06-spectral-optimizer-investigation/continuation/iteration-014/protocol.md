# I14 — current-stable filter effects across base optimizers

Prospective protocol, 7 September 2026. Autonomous continuation of the user's
explicit optimizer-transfer priority. No production change, completed-run
restart, paid job or additional approval gate. The
[I13 candidate](../iteration-013/next-cross-optimizer-design.md) motivates this
fresh-source study; all I9–I13 sources and outcomes remain immutable.

## Question and scope

Does the native stable filter retain a useful neural effect without AdamW's
coordinatewise second-moment transform? Does ordinary momentum change the
effect? This is a conditional necessity discriminator, not an optimizer
leaderboard or an identified mediation fraction. Favorable SGD filtering would
show AdamW is not necessary for that measured effect. An SGD null/adverse result
would not prove necessity: operating point, finite horizon and task remain limits.

Confirmation is exactly **36 fresh trajectories**: seeds200/201/202 × bases
SGD/SGDm/AdamW × raw/current32 × clean/fixed90%-replacement labels. Each runs
2,000 updates with batch64. All12 cells of a seed share fresh initial weights,
examples, batch indices and corruption plan. No Adam-trained parent is reused.
Architecture is the existing784–64–ReLU–10 MLP; images are divided by255,
without augmentation, schedule, clipping, early termination on finite loss,
weight reset, rank tuning or official-test evaluation.

## Separate baseline calibration

Before any real confirmation trajectory, run **18 clean raw-baseline curves**:
seeds190/191 × three bases × three rates, each2,000 updates:

| Base | Candidate learning rates |
|---|---|
| SGD | .03, .1, .3 |
| SGDm(.9) | .003, .01, .03 |
| AdamW | .0003, .001, .003 |

An eligible rate must finish with finite metrics and at least85% clean-validation
accuracy in **each** calibration seed. This conservative competence floor is
below the roughly93% raw-clean accuracy in prior small-MLP work; it rules out a
chance-level baseline as an informative optimizer discriminator. Among eligible
rates select lowest mean final clean-validation CE, smaller rate on an exact
tie. Missing/weak seeds cannot be survivor-averaged; retain all adverse/failed
curves. If no rate is eligible for a base, do not launch confirmation. Do not
expand the grid after observing outcomes. Freeze the selected-rate JSON and
its source/raw hashes before confirmation; derive the selection independently
from every calibration curve during confirmation preflight.

Calibration only evaluates training and validation data. No auxiliary result
selects a learning rate. The observer is passive even during calibration, to
match the raw execution recipe. Every selected rate is then fixed for both
target regimes and both policies of its base. This gives a raw-calibrated
operating point, not separately tuned fairness for each filtered optimizer.
Report selected raw competence and boundary-grid choices; a poor raw baseline
makes necessity interpretations weak, not evidence against every SGD recipe.

The SGD/SGDm grid offset reflects the latter's unnormalized momentum convention;
it is not a theorem that their rates should differ by ten. Initial-step norm
matching is not additionally performed or substituted after seeing outcomes.
Total new real-data exposure is **108,000 updates** (36,000 calibration plus
72,000 confirmation), explicitly separate from prior studies and from a single
synthetic smoke.

## Plans and disjointness

The global NumPy SeedSequence `[20260907,14,0,0]` permutes all60,000 MNIST
training indices. Its first15,000 form the calibration pool; the remaining
45,000 form confirmation. These phase pools are disjoint. Within each seed,
independent streams `[20260907,14,seed,stream]` choose the pool permutation
(stream0), CPU model initialization seed(stream1), .9 replacement mask(stream2),
uniform replacement digits(stream3) and2,000×64 training batch indices(stream4).
The first/next/next5,000 indices are train/validation/auxiliary. Calibration's
auxiliary allocation is unused. Across seeds, examples can recur; do not claim
dataset-level independence. The MNIST source dataset is reused from earlier work.

All five plans are saved before calibration, including outcome-free confirmation
plans. Calibration and confirmation use the same source-bound planning module;
confirmation loads and verifies the saved plans rather than creating replacements.
Record exact replacement and incorrect-label counts. Ninety-percent uniform
replacement implies81% expected incorrect labels, not90%. Clean cells train on
true labels; fixed cells train on the one saved corruption realization.

## Optimizer and observation semantics

All bases use an explicit decoupled coefficient WD=.01. Plain SGD has no momentum.
SGDm uses the PyTorch recurrence `b_t=.9*b_(t-1)+h_t`, first buffer `b_1=h_1`,
dampening0 and Nesterov false. Both use torchSGD with coupled weight_decay0;
after backward and gradient delivery, multiply parameters by`1-lr*.01`, then
call the data/momentum step. AdamW uses native decoupled decay.01, betas(.9,.999),
epsilon1e-8, AMSGrad/maximize/capturable/differentiable false, foreach/fused false.
No parameter-specific exceptions. Gradients must exist for every parameter.

Different calibrated rates imply different per-update decay factors even with
the same coefficient. Thus cross-base interactions do not isolate optimizer
geometry or momentum alone. Within-base raw/filter rate and decay are identical.
Save nominal decay, total displacement and decay-subtracted data displacement
separately. Neither gradient-norm nor total-step comparisons are substituted
for data-step geometry. SGD data steps are expected to lie in the current
retained span up to native basis/parameter-rounding error; momentum and adaptive
scaling need not preserve that property.

Canonical observer: rank32, decay.99, warmup100, stable_update true,
stabilize_every100, relative_eig_tol1e-8, absolute_eig_floor0, hard weighting,
normalize/adaptive none. Both arms call native `filter_grad` once per raw batch
gradient. Raw restores its unchanged gradient before the base step; current32
uses the native post-ingest delivery. Within each base/target/seed pair, full
model/optimizer/observer/RNG state and evaluations must agree through update100;
the first projected delivery is update101.
No second observation, lagged basis, mean32 or leakage variant is included.

## Outcomes, selection and diagnostics

Evaluate h0 and `{100,250,500,1000,1500,2000}`. The fixed h2000 endpoint gives
12 primary filter effects: three bases × two targets × auxiliary-clean CE and
accuracy. Utility is negative CE or accuracy; positive `B=U(current)-U(raw)`
favors filtering. Retain all three paired seed values and their arithmetic
mean separately, without p-values, composites or target pooling. Report
`B(SGDm)-B(SGD)` and `B(AdamW)-B(SGDm)` descriptively, with their rate/decay limits.

Also select each trajectory by minimum validation CE and maximum validation
accuracy over the six nonzero checkpoints, earliest-horizon tie breaking.
Report **both auxiliary CE and accuracy under both selectors**, plus h100-stop
and absolute progress from h0/h100. Do not replace an adverse fixed endpoint
with a favorable selected result. Three seed bundles are the replications;
steps, checkpoints and multiple metrics are not additional trials.

Training records include fixed/clean CE and accuracy, expected-soft CE,
R_zeta=fixed-minus-soft CE and confidence. Retain all validation and auxiliary
clean CE/accuracy/confidence curves. Every step records raw/applied norm,
actual data/total/nominal-decay norm, current-basis data/total leakage, and signed
raw-gradient dot data displacement. Aggregate leakage by summed energies
within trajectory, then equal seeds, not by survivor or per-step fractions.
Per-step tensors are not retained, so these diagnostics permit scalar identity
checks rather than full per-step vector replay or semantic noise attribution.

Retain all scheduled model checkpoints, full optimizer/observer/RNG states at
h0/h100/h2000, hashes/topology/source/plan/environment metadata, and complete
JSON step records. On numerical failure retain the partial curve/steps, typed
failure, final failed state and last successful evaluation's model checkpoint.
Intermediate complete optimizer states are not saved; do not claim otherwise.

## Failure and resource discipline

Only explicit nonfinite training/evaluation arithmetic after valid h0 may seal
an individual failed trajectory and continue other independent cells. It makes
required endpoint/stopping contrasts unavailable; no earlier checkpoint or
survivor mean replaces a failed arm. If no complete calibration rate remains
for a base, confirmation cannot proceed. CUDA/eigensolver, topology, provenance,
serialization, disk, memory or wall-limit failures abort the entire phase.
No finite-loss rejection, clipping, retry, silent code change or automatic
cloud fallback. Preserve partial artifacts; consumed attempts never restart.

If either member of a confirmation pair fails before a valid h100, both members
must have identical startup-failure fingerprints covering complete terminal
state, successful diagnostics/evaluations and failure position. Otherwise abort
the phase as a pre-treatment reproducibility failure. Do not label asymmetric
behavior before filtering starts as a scientific filter effect. Identical
common failures remain missing comparisons, with warmup coverage counts explicit.

One exclusive root under verified`/tmp/spectral-experiment-artifacts` on`/dev/RECONFIGURE_FOR_LOCAL_STORAGE`, shared
**3GiB** artifact ceiling across smoke/calibration/confirmation. Each phase has
an exclusive attempt record and new output child. Reserve1MiB for failure
metadata. Host cgroup6GiB, zero swap, TorchGPU4GiB, one numerical CPU thread,
CPUQuota100%, Restart=no; launch only with16GiB available host and8GiB free GPU.
Leave unrelated GUI/Stremio processes untouched.

One source-identical synthetic GPU smoke covers all12 base/policy/target cells
for110 steps, all warmup pairs and full-state roundtrips, no real MNIST outcomes.
Smoke cap100s cooperative/120s whole unit; calibration1800s/2100s;
confirmation3600s/3900s. These conservative ceilings include per-step state
validation and scalar geometry, not just model updates. Before either real
phase, scale the complete smoke elapsed time by36,000/1,320 or72,000/1,320 and
a1.5 safety factor. Both projections must fit their ceilings; record them in
the manifest. Otherwise stop before real acquisition and
document the estimate, not silently enlarge limits or restart the smoke.
These larger caps account for the declared108,000 updates, rather than copying
I13's18,000-update cap. No paid compute or reservation; cumulative budget remains
$0 spent of the user's$100. Before acquisition freeze all implementation/test/
protocol source hashes in a commit and run CPU synthetic plus independent
scientific/implementation review. These are execution checks, not user gates.
