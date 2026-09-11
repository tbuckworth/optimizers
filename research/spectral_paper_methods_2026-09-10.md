# Methods for the three-contribution paper core

Codex — Spectral Optimizer Investigation · 10 September 2026

This consolidates the **completed** studies behind the
[compact paper core](spectral_paper_core_2026-09-10.md). It is a manuscript
methods supplement, not a new protocol, acquisition or reanalysis. Prospective
protocols retain their historical status wording; their linked completion and
audit reports determine whether work is finished. The broader
[evidence draft](spectral_paper_draft_2026-09-09.md) retains earlier experiments,
including grokking, scalar controls and contradictory optimizer versions.
Section F additionally consolidates the later completed ordinary-augmentation
sequence, including the strong rank-200 bridge; these designs are distinct from
the rare-class continuations below.

## A. Shared neural setup and policy definitions

The selectivity-boundary and batch-composition studies use an MLP with dimensions
784–64–10, one ReLU hidden layer and 50,890 trainable parameters, including
biases. Inputs are MNIST pixels divided by 255. All data come from the pinned
official **training** IDX files: the official test split is not used. Each
seed selects 550 training images from each of the nine non-8 digits and 50
digit-8 images, giving 5,000 training images. A disjoint 500 images per digit
from the same source forms a balanced 5,000-image held-out set.

Each seed has one 100-step clean warmup using only the 4,950 non-8 training
images. AdamW receives raw gradients during warmup, while a spectral observer
records the same gradients. Every continuation within that seed restores the
identical model, Adam, observer and random-generator state. Thus digit 8 is
both rare and newly introduced after a warmup that treated it as a negative
output class. The design does not isolate frequency from novelty or digit
difficulty.

AdamW uses learning rate 0.001, weight decay 0.01, betas (0.9, 0.999), epsilon
10⁻⁸, and disabled foreach/fused paths. Batches have 64 examples. Each branch
continues for 1,900 updates to absolute step 2,000. There is no learning-rate
schedule, clipping, image augmentation or outcome-dependent early stopping.

The stable spectral observer operates globally on the flattened gradient,
with maximum rank 32, decay 0.99, raw rather than normalized covariance,
hard action, numerical repair every 100 observations, relative eigenvalue
tolerance 10⁻⁸ and absolute eigenvalue floor zero. It updates from the current
raw batch gradient **before** applying its action. Initialization, centering,
finite-rank truncation and repair are the canonical implementation, not an
ideal full-history PCA substitution; see Section 2 of the evidence draft and
[spectral_filter.py](../spectral_filter.py).

Let N(O,g) denote the canonical native action after updating observer O with
g. It is V(Vᵀg) when the basis exists, with the canonical identity fallback
otherwise. The three policies supply these gradients to ambient-coordinate
AdamW:

<pre>
raw:       h = g
native32:  h = N(O,g)
norm_raw:  h = ‖N(O,g)‖ g / ‖g‖, for ‖g‖ > 0
</pre>

Every observer follows its **own** policy/model trajectory. The norm control
shares a functional rule with native, not native's realized norm sequence.
Norm/scaling calculations use float64 before casting delivery to float32;
nonzero norm matching has a fixed dtype-scaled numerical guard. Zero delivery
is an explicit zero gradient followed by an ordinary AdamW step, not `None`
or a skipped step. Carried moments and decay can therefore move parameters.
Neither gradient projection nor gradient-norm matching implies an equivalent
constraint on the resulting Adam displacement.

## B. Useful protection and selectivity boundaries

Seeds 202609111, 202609112 and 202609113 cross four label/cue conditions with
the three policies, giving 36 continuations. All continuations within a seed
use identical precomputed sampled ID occurrences, drawn uniformly with
replacement from the 5,000-image training pool. Rare labels remain correct
and rare images unpatched in every condition.

- **Clean:** unchanged inputs and true labels.
- **Diffuse:** select each majority example independently with probability
  0.9 and replace its label uniformly among the nine non-8 labels, including
  its original label. The expected changed-label fraction is therefore 0.8,
  not 0.9. These assigned labels remain fixed across training.
- **Shared:** choose exactly 500 true-nonzero majority examples, assign label
  0, and set their upper-left 3×3 pixels to white.
- **Sham:** retain exactly the same wrong-label assignments and total/per-true-
  digit patch counts as Shared, but reallocate patches approximately
  independently of poison status within each eligible true digit. Allocation
  uses floor counts and fixed largest-remainder completion, followed by
  sampling without replacement. This is conditional finite-count balancing,
  not exact unconditional independence. Full rules and saved contingency
  counts are in the [prospective protocol](../output/2026-09-09-spectral-selectivity-boundary/protocol.md).

Diffuse and Shared are not corruption-severity-matched conditions. Shared
versus Sham changes patch/target association while preserving targets and
patch prevalence; it does not preserve each individual image's pixels.

The primary endpoint is step 2,000. True-label held-out accuracy and
cross-entropy (CE) are reported for rare digit 8, every common digit, the
nine-digit common macro average and the balanced total. Curves and changes
from warmup distinguish preservation against deterioration from acquisition.
Training predictions are scored against both true and assigned labels, with
actually changed-target examples distinguished. Lower wrong-label fit alone
is not counted as useful protection.

Cue evaluation compares patched and unpatched versions of the **same** held-out
images. For the eight nonzero common digits (4,000 examples), define

<pre>
E(policy, cell) = P(prediction=0 | patched) − P(prediction=0 | unpatched)
D = [E(native, Shared) − E(native, Sham)]
  − [E(raw, Shared) − E(raw, Sham)].
</pre>

Negative D is a reduction in this cue-association interaction. It is not
evidence of zero residual susceptibility. The separate digit-8 population,
raw patched target-0 rates, norm-control interaction, patched accuracy and
patched CE remain secondary reported outcomes, not replacements for D.
Population membership is based on true labels, never a policy's predictions.

Completed outcomes and per-seed evidence:
[report](../output/2026-09-09-spectral-selectivity-boundary/results.md),
[checked summary](../output/2026-09-09-spectral-selectivity-boundary/results/checked-summary.json).

## C. Equal exposure, changed batch composition

Fresh seeds 202609121, 202609122 and 202609123 cross Clean/Diffuse labels,
Interleaved/Grouped schedules and the same three policies: 36 continuations.
The architecture, split recipe, warmup, optimizer, observer and endpoint are
unchanged. These are not extra replications of the first study's different
schedule/cue conditions.

Within each of 38 consecutive 50-update blocks, both schedules use exactly
the same 3,200 indexed occurrences, including repetitions and fixed assigned
labels. Rare and common within-group occurrence lists are shared. If R
occurrences are rare, Interleaved distributes them with per-batch counts
floor(R/50) or ceil(R/50); Grouped fills batches with up to 64 rare occurrences
until all R are placed. A shared random ordering of batch positions controls
which positions receive the allocations. Remaining slots contain the common
occurrences, and paired within-batch permutations are applied.

Exact multiset equality, counts and named random streams are persisted and
checked. Neither schedule is the earlier iid schedule. At a fixed model their
block-average loss gradient agrees in exact arithmetic; gradients along their
subsequently diverging trajectories need not agree. The intervention changes
ordinary optimizer history as well as the observer.

For each rare/common accuracy and CE endpoint M, report all paired schedule
effects and the primary interaction

<pre>
I = [M(Grouped, native) − M(Interleaved, native)]
  − [M(Grouped, raw) − M(Interleaved, raw)].
</pre>

Positive favors accuracy; negative favors CE. Report the norm-control
interaction separately. A favorable interaction caused by deterioration of
raw is not called a recovery of rare recognition in native. Fixed selected
training-gradient probes at count-defined events supply secondary local
diagnostics, not additional independent seeds or endpoint mediation tests.

Exact schedule construction and diagnostic membership:
[protocol](../output/2026-09-10-spectral-batch-composition/protocol.md).
Completed results and audit links:
[report](../output/2026-09-10-spectral-batch-composition/results.md).

## D. Frozen-model observer-history intervention

This diagnostic reuses the three step-100 parents of Study C, after its
outcomes were known. It is a local, outcome-informed intervention, not fresh
confirmation of the 2,000-step endpoints. At each frozen model, compute the
first 50 saved batch gradients under each schedule and each of Clean/Diffuse:
600 batch-gradient measurements. Feed each history into a distinct copy of
the inherited observer without changing the model or Adam.

FP64 means of the two FP32 gradient streams must agree within the predeclared
bound 128 ε₃₂ times their average gradient norm, plus 10⁻¹⁰. The common input
g* is the float32 cast of the symmetric average of those FP64 means. Its
bytes are identical in the native arms and raw control. It is a block-mean
assigned-target gradient, not a fresh 64-example minibatch.

After the histories, each observer is at count 150 while Adam remains at
100. Observe g* once more before applying the primary native action, preserving
canonical self-inclusion. Adam's counter is not advanced to match the observer.
From fresh copies of the same model and Adam100, deliver native Interleaved,
native Grouped, raw g*, or explicit zero, and take one AdamW step to Adam101.
The zero step is shared across label cells within seed: 21 physical steps
support 24 logical action/cell/seed entries. No subsequent training follows.

Primary utility is true-label rare/common held-out CE before minus after;
report absolute utilities, the Grouped-minus-Interleaved contrast, and raw/zero
comparisons. Both direction and gain may change. Pre-inclusion geometry alone
does not establish useful canonical delivery, and a favorable relative
contrast need not be an absolute improvement.

Two fixed true-label training probes—32 rare and 54 common examples—supply
oracle diagnostic gradients q, not delivered actions. Their span retention,
signed −qᵀΔθ and finite probe loss describe different stages. The held-out
loss is from a different population, so −qᵀΔθ is not its Taylor decomposition.

Exact state/counter/input construction:
design (artifact not distributed in this public snapshot),
[execution protocol](../output/2026-09-10-spectral-observer-pathway/protocol.md).
Completed measurement and independent saved-data audit:
[report](../output/2026-09-10-spectral-observer-pathway/results.md).

## E. Saved-vector accounting and evidence limits

A later calculation reads the saved input g*, diagnostic q and delivered h
without rerunning a model, observer or optimizer. It computes qᵀg*, qᵀh,
qᵀ(h−g*) and paired grouping differences in FP64. It joins, rather than
recomputes, the accepted actual Adam and held-out-loss scalars. Every seed,
label cell, probe and native/raw/zero control is retained. Compensated-sum
checks and cancellation-aware sign rules are numerical checks, not a new
independent scientific audit or an estimate of causal mediation.
[Protocol](../output/2026-09-10-spectral-observer-signal/protocol.md);
[results](../output/2026-09-10-spectral-observer-signal/results.md).

Across the preceding neural studies, paired seeds are the replication unit. Shared
warmups, the same three reused diagnostic parents, repeated checkpoints,
two label cells and multiple contrasts are not independent replications.
Per-seed values, means, sample SD/SE and signs are descriptive; there is no
hyperparameter selection, equivalence claim, composite victory score or
selected stopping point. The limited recipe breadth and frequency/novelty
entanglement limit generalization even where every seed has the same sign.

The acquired environment is local RTX 3090, PyTorch 2.11.0+cu128 and pinned
source/data/configuration receipts. Reproducibility does not imply bitwise
agreement across platforms or later library versions. Each completed report
binds the source freeze, immutable output root and saved-data checker. This
supplement does not re-certify those artifacts or authorize their replay.

The emerging claim concerns useful **conditional learning behavior and its
local history pathway**, not semantic truth detection, a direct long-tail
memorization theorem, practical optimizer superiority or demonstrated model
alignment. The separate J-Lens pilot is an interpretation side study and is
not pooled with optimizer-training evidence.

## F. Ordinary augmentation and the observation/delivery follow-up

### F.1 Clean and fixed-wrong-label training comparisons

These studies retain the 784–64–10 MLP, FP32 pixels divided by 255, canonical
stable rank-32 observer and AdamW settings above, but use **balanced** 500-per-
class training and disjoint 500-per-class heldout sets. Every class is present
from update 1. There are 4,000 updates of batch 64, with projection beginning
after update 100; no rare-class introduction or inherited common-only warmup.
Each study crosses raw/native32 and none/one-view integer translation, giving
12 trajectories. Clean seeds are 202609141–143; wrong-label seeds are
202609151–153. The studies are not a paired clean×noise factorial.

Split, occurrence and translation plans use distinct PCG64 SeedSequence
streams. Occurrences are sampled with replacement; dx/dy are independently
uniform in −2…2 per occurrence, with zero fill and no interpolation. Translation
is active throughout warmup, not only after the observer starts filtering.
Raw/native branches within an augmentation mode have identical warmup model/
Adam digests and predictions; different augmentation modes do not share that
warmup. In the wrong-label study, exactly 400/500 examples per true class are
selected independently of the other plans and assigned one of the nine other
labels, giving 4,000 actually wrong targets. Those targets remain fixed for
each example under all views and policies.

Original training/heldout logits are retained at 0,100,200,400,…,4000 (22 states).
Primary outcomes are heldout CE and accuracy at 4,000; report all paired
augmentation benefits and the native-minus-raw interaction, with lower CE
favorable. Wrong-label training logits are additionally scored against true
and assigned labels, including the actually corrupted subset; that subset's
false-label fit is not inferred from overall assigned accuracy. Warmup progress
and all curves remain visible; no selected best checkpoint or significance
claim is introduced. Exact source/data/plan pins and independent NumPy
saved-logit checks are in the [clean protocol/report](../output/2026-09-10-spectral-general-augmentation/results.md)
and [wrong-label protocol/report](../output/2026-09-10-spectral-wrong-label-augmentation/results.md).

### F.2 Fixed-clean-parent direction/scale diagnostic

Reuse the six clean native step-100 parents (three seeds × two augmentation
modes) without continued training. A fixed 64-example training panel, its
original view and four random translated versions supply five batch gradients
and per-example geometry. Both heldout objectives use the same256 source IDs,
disjoint from training: one original view and one fixed-translated view.
At fixed parameters the empirical
four-view total=between-image-mean+within-image-view identity uses denominators
64 and 64×4; it is also evaluated after the **recorded pre-update** QQᵀ action,
including its numerical Gram matrix. Finite-view mean variability is not a
population semantic component.

Each actual raw/native AdamW proposal starts from a private identical parent;
native observes only its candidate once (100→101) before projection. With
rounded FP32 decay endpoint θ_D and rounded data differences d_R,d_N, controls
materialize θ_D+d_R‖d_N‖/‖d_R‖ and θ_D+d_N‖d_R‖/‖d_N‖. Ratios use FP64 norms;
actual materialized norms are checked. Zero norms or ratios above 100 yield
explicit unavailable controls, never clipping or replacement. All were available.
The decay-only comparator is not a zero-gradient Adam step. Path fractions 1
and .1 scale the whole displacement including decay; the latter is not another
optimizer update. Both directional contrasts, absolute CE/accuracy changes and
same-objective signed derivatives are retained. Four translated draws are
averaged within each parent first, all-or-null, not treated as extra seeds.
[Exact protocol and tolerances](../output/2026-09-10-spectral-augmentation-state/protocol.md),
[completed results](../output/2026-09-10-spectral-augmentation-state/results.md).

### F.3 Fresh clean four-view observer comparison

Fresh seeds 202609161–163 use the same balanced small-model recipe and 4,000
updates, with translated training throughout. Raw1 delivers g1; native1
observes/delivers the canonical action on g1; observer4 observes the sequential
FP32 mean of four separately differentiated same-state gradients once and
delivers the current projected g1; raw4 delivers that mean directly. Raw1,
native1 and observer4 share exact model/Adam warmup identities. Raw4 uses its
mean during warmup and is a different trajectory. Every arm advances Adam
once per update; each native observer advances once, never once per view.
After warmup, the basis includes the current observation before delivery.

Independent additional-view and fixed-translated-heldout streams extend the
shared split/occurrence/first-view plans. All 22 readouts save original training,
original heldout and secondary fixed-translated heldout logits. Original
heldout endpoint remains primary. Five fixed contrasts are observer4−native1,
observer4−raw4, observer4−raw1, raw4−raw1 and native1−raw1, with CE signs
reversed for positive benefit. Four-view arms use 16,000 batch-gradient calls
per trajectory versus 4,000; total 120,000, with 72,000 extra relative to four
single-view arms. Timings are descriptive rotated sequential measurements,
not matched time-to-accuracy claims. The independent audit checks 58 artifacts,
264 states and 792 panel evaluations; checkpoint bytes are hashed, not unpickled.
[Protocol](../output/2026-09-10-spectral-multiview-clean/protocol.md),
[results](../output/2026-09-10-spectral-multiview-clean/results.md).

These three additions introduce no noisy four-view result or new claim of
independent neural replay. The [augmentation evidence section](spectral_augmentation_sequence_2026-09-10.md)
preserves their useful local/secondary effects and historical evidence beside
the separately completed larger-rank comparison below.

### F.4 Strong rank-200 augmentation bridge with validation-only selection

Fresh seeds 202609171–173 each cross raw AdamW/native stable rank 200 with
none/translation. The ordered FP32 MLP is 784–256–128–10 with two ReLUs,
ordinary biases and 235,146 parameters; CPU initialization uses the seed before
GPU transfer. AdamW remains lr .001, betas (.9,.999), epsilon 1e−8, weight decay
.01, foreach/fused disabled. Canonical stable hard filtering uses rank 200,
decay .99, warmup 100, no normalization/adaptation, full strength, relative
eigenvalue tolerance 1e−8, absolute floor 0 and repair every 100 observations.
Current gradients are observed once before projection after warmup; Adam
advances once with carried moments, no clipping, reset or gain restoration.

Four independent PCG64 SeedSequence([stream,seed]) streams are frozen.
Stream 0 permutes all 60,000 official **training** IDs: first 50,000 train,
next 5,000 clean validation, last 5,000 clean reporting. Splits are random, not
exactly class-balanced; roles are disjoint within seed, but seeds can overlap.
Stream 1 draws a .9 replacement mask and then uniform labels 0…9 for all 50,000
positions; assignments stay fixed across all views. Replacement can equal
truth, so expected actually wrong fraction is .81, realized
80.962/81.016/81.012%. This differs from the earlier exactly 80%-wrong law.
Stream 2 supplies 72 independent full permutations of the training positions.
Stream 3 supplies occurrence-indexed independent shifts (dy,dx) in −2…2.
Unaugmented arms share occurrences but omit shifts; translated arms share
both. Raw pixels are divided by 255, translated with zero fill, then standardized
by (x−.1307)/.3081. Thus padding is standardized black, not standardized zero.
All training, validation and reporting readouts use original images.

Each pass uses 781 batches of 64 plus one of 16, never dropping the tail:
72×50,000 = 3.6M exposures and 56,304 updates per arm. Historical 60×60,000 has
equal exposure but 56,280 updates and different repetition/data roles; this is
a bridge, not exact reproduction. The 74 fixed readouts are 0, 100 and 782×epoch
for epochs 1…72. Raw/native warmup model+Adam digests and predictions match
within augmentation mode. Each saved state includes original logits on all
three roles; complete warmup/final states have the correct two-hidden-layer
snapshot schema, not the older one-hidden-layer restoration format.

Only train and validation metrics are calculated during acquisition. After all
twelve trajectories finish, each arm selects minimum validation CE over its 74
readouts, with earliest exact tie. The frozen selector converts to FP64,
subtracts row maxima, computes log(sum(exp(shifted logits))) minus the target
logit, and uses ordered Python math.fsum divided by count. All twelve choices
are exclusively persisted and readback-verified against validation arrays
before reporting metrics are calculated. The independent audit repeats that
validation-first verification; stored reporting arrays are not cryptographically
blinded. Both selected reporting metrics use the same checkpoint; selection
requires access to clean validation data, not a universally free resource.

Co-primary outcomes are final original reporting accuracy and CE for native+
translation versus raw+translation. Retain every paired seed, all four arms,
native−raw without augmentation, each policy's augmentation effect and their
interaction. Secondary views are own validation-selected reporting, h100→final
learning, means over exactly epoch endpoints 61–72, all curves and original-
training true/assigned fit including the actually wrong subset. No reporting-
based stopping, tuning or significance/equivalence claim is introduced.

Source freeze `1f954b8`; acquisition completed once and the independent NumPy
saved-array audit is PASS (937 artifacts, 888 logit states, 12 selection choices;
maximum scalar discrepancy 1.4552e−11 within fixed 1e−10 tolerances). It hashes
opaque checkpoint bytes without unpickling or neural replay; model/Adam digest
pairing remains a producer assertion, not reconstructed state correctness.
Two logit passes respect its 2 GiB memory cap. A subsequent scalar review
corroborates all registered summaries, not another scientific replication.
The 12 arms total 675,648 gradient evaluations and 43.2M example exposures;
observed timings are descriptive, not optimized or matched-target speed.
The [prospective protocol](../output/2026-09-10-spectral-strong-augmentation/protocol.md),
[completed report](../output/2026-09-10-spectral-strong-augmentation/results.md),
[audit](../output/2026-09-10-spectral-strong-augmentation/audit.json) and
[interpretation review](../output/2026-09-10-spectral-strong-augmentation/interpretation-review.md)
are authoritative. No model/data computation was rerun for this supplement.

### F.5 Common-parent signed component utility

Use the twelve existing native strong-bridge snapshots only: seeds202609171–173,
prior none/translation, steps100/56304. Exact checkpoint/readout/source pins
are in the inventory (artifact not distributed in this public snapshot)
and manifest (artifact not distributed in this public snapshot).
Per seed, two disjoint64-example translated action batches and a disjoint256-
example training evaluation panel I are fixed across its four parents; a128-
example nontraining reporting panel R supplies clean evaluation. Evaluate all25
equally weighted integer translations dy,dx∈{−2,...,2}, including original.
All actions are translated, including hypothetical augmentation onset at
none-trained parents. Complete deterministic selection and reduction rules
remain in the [frozen protocol](../output/2026-09-10-spectral-component-utility/protocol.md).

Both actions privately restore the same weights, Adam moments/counter and
observer. Compute the assigned-label batch gradient once. Raw bypasses the
observer; native observes and filters once; each advances canonical inherited
AdamW once, with no reset, gain change or norm matching. Evaluation-component
gradients are never action inputs. No training continuation or missing-state
reconstruction occurs. Initial original predictions on500 reporting examples
must exactly reproduce the archived parent readout before any new action.

On I, L is direct mean-per-view assigned CE; S is CE of mean logits for
q=.1×true+.9×uniform, F=(q−assigned)·meanlogits, and C is the Jensen gap of
logsumexp across views. Thus L=S+F+C. Differentiate all six baseline objectives
S,F,C,L,H_O,H_T; H_O is original true CE on R and H_T mean-per-view true CE on R.
The moving mean-logit center is differentiated, not detached. Save actual
wrong-subset original/per-view assigned/true CE and accuracy separately; F
is neither a wrong-subset metric nor a pure memorization component.

Full points preserve actual endpoint bytes. Tenth points use FP32 subtraction,
then FP32 multiplication by.1, then FP32 addition to the parent. Corresponding
decay-only points use the same rounding sequence. E_J=J(parent)−J(point);
U_J=−∇J(parent)·(point−parent), with the actual points cast to FP64 before
subtraction/dot accounting. D_J=E_J(native)−E_J(raw). Positive means more
decrease of that named loss. Full/tenth raw/native/decay points give144 readouts.
The tenth point is not a new lower-rate Adam step or scaled measured effect.

Primary is translated-final full-step D_HO and D_C, all three seed values;
first average the two draws within each parent. Retain all8 stage/mode/path
cells, six objectives, actual fit metrics and individual draws. Finite values
of magnitude≤1e−8 nats and linear utilities≤1e−10 are marked numerically small,
not used as sign evidence or statistical equivalence margins. The chosen panels,
parents and hypothesis are outcome-informed reuse, not fresh confirmation.

Source freeze18f63cf; original results SHA256
`3f685f592e0cf3dfd6a761fc86c8abea8af25aa4744cc8b9e1aeae8f53946096`.
One [independent saved-array audit](../output/2026-09-10-spectral-component-utility/audit.json)
passes50,388 checks,144 endpoints and376 artifacts. It reconstructs saved
objectives, dots, Adam/projection/path arithmetic and provenance, not model
inference, autograd or the full streaming-observer history. Independent scalar
interpretation and report checks are not another scientific replication.
[Completed results](../output/2026-09-10-spectral-component-utility/results.md)
and [all-cell interpretation](../output/2026-09-10-spectral-component-utility/interpretation-analysis.md)
are authoritative; no measurement was rerun for these method notes.
