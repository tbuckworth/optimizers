# Strong-regime augmentation bridge — design only

Codex — Spectral Optimizer Investigation · 10 September 2026

**Prospective recommendation, not an acquisition protocol or launch.** No
attempt is created here. Main will settle the exact protocol, source and
resource admission. This design uses the historical stable positive and the
[regime comparison](regime-comparison.md), not preliminary multiview outcomes.

## Practical question and fixed scope

Does the previously successful **stable global rank-200** recipe add useful
late-horizon noisy-label robustness beyond ordinary one-view translation?
Test the strong version directly, without a rank/LR search or another small
local diagnostic. Twelve fresh trajectories: proposed seeds
`202609171/172/173` × raw AdamW/native spectral × none/translation.
All training is noisy; clean labels are reserved for specified evaluation.
No additional clean-training factorial, per-matrix arm, legacy arm, four-view
averaging, new augmentation family, cloud job or automatic follow-up is included.

The strongest straightforward stable reference is global rank200, not the
same-seed-selected per-matrix rank64 setting. Its historical seed42 result was
78.77% final clean-test accuracy versus38.93% for matched AdamW. The matrix
result was81.74%, but rank64 was selected using that seed's test set. The older
three-seed rank200 positive used legacy numerics and Adam, so it is supportive
context, not the implementation to silently substitute here.
[Stable raw reference](../../results/noisy_mnist_hard_curves/final/global_stable_hard_r200_n90_s42.json),
[matched AdamW](../../results/noisy_mnist_hard_curves/final/adamw_n90_s42.json),
[selection/protocol record](../../results/noisy_mnist_hard_curves/summary.json).

## Faithful model and optimizer; explicit data-role change

- Model:784→256→128→10 ReLU MLP,235,146 parameters, full-model training.
- Pixel transformation: uint8→float32 pixel/255, then
  `(pixel−0.1307)/0.3081`, matching the historical model input.
- AdamW: lr.001, betas(.9,.999), eps1e−8, weight_decay.01;
  freeze single-tensor, non-fused deterministic FP32 implementation.
- Native: canonical stable global hard rank200, covariance decay.99,
  warmup100, full filter strength; no adaptive/normalized/soft weighting.
  Retain current stable numerical tolerances and scheduled repair, pinned
  explicitly in the eventual protocol. Observe the current batch gradient
  once before filtering it; Adam advances once. No moment resets or gain match.
- No training LR schedule or optimizer-dependent duration. All four arms use
  identical per-seed initialization, source occurrences and first-view plan.

Use **only the already-present pinned official training IDX**, not downloaded
data. Partition its60,000 source IDs into50,000 training,5,000 clean validation,
and5,000 clean reporting IDs. Propose one global PCG64 permutation per seed
using SeedSequence([0,seed]), sliced in that order. These are random—not
exactly class-balanced—splits; archive every ID and true-label count. Source
IDs are disjoint within a seed and may overlap across seeds. The official
10,000-example test set remains untouched.

This resolves test-selection contamination, but **is not exact historical
reproduction**: that study trained on all60,000 and reported the official
test set. Holdout population, training population and effective repetitions
change. Do not compare absolute percentages across these studies. The raw/no-
augmentation versus native/no-augmentation arms provide a fresh internal
reference for whether the historical protection pattern transfers to this
evaluation discipline. Failure of that reference is informative; it does not
justify quietly changing the split, rank or horizon.

## Corruption and exposure

Preserve the historical corruption **law**, not the small study's exact80%
wrong-label construction. On the50,000 training IDs in saved order, use one
independent stream to draw Bernoulli(.9) corruption indicators, then uniform
integer replacement labels0…9. A replacement may equal the true label.
The expected actual wrong fraction is81%; archive selected-for-replacement
and actually-wrong masks separately and report realized counts. Labels remain
fixed across all epochs and translated views. Validation/reporting labels are
never corrupted. Proposed PCG64 SeedSequence([1,seed]) is a new reproducible
realization, not a claim to reproduce historical RandomState draws.

Match **3.6million training-example exposures per arm**, the historical
60epochs×60,000. With50,000 training examples this is72 complete shuffled
passes, batch64, retaining the final16-example batch:56,304 updates, compared
with the historical56,280. This deliberately matches example exposure almost
exactly in update count, while acknowledging72 versus60 repeats per training
example. Neither “60epochs reproduced” nor “same data” is an accurate label.
Generate each epoch permutation sequentially from stream2 and retain the
actual occurrence order plus batch boundaries; do not sample with replacement.

Translation is exactly one view per occurrence, independent dx/dy integers
−2…2, stream3, shared between raw/native translated arms. Apply zero-filled
translation in **raw [0,1] pixel space before standardization**, so padding
becomes standardized black, not zero in standardized coordinates. Original
arms use those same occurrences without translation. No extra views or
backward passes: each arm has56,304 batch-gradient evaluations;12arms have
675,648 in total. Rotate the four-arm order by seed index and retain timing.

## Evaluation and selection, fixed before reporting-set readouts

Evaluate the initial state, update100, and each complete epoch endpoint:
74states per trajectory. Save logits on all50,000 original training examples,
5,000 original validation examples and5,000 original reporting examples.
No transformed evaluation panel is needed for this minimal bridge.

**Primary:** clean reporting accuracy and mean CE at the fixed final exposure
(update56,304). The primary recipe contrast is native+translation versus
raw+translation. Report all three paired seeds and their arithmetic mean for
both metrics; no significance or optimum claim from three seeds. Also show
all four absolute endpoints, native/no-augmentation versus raw/no-augmentation,
and within-optimizer augmentation effects. The unaugmented reference is not
a post-hoc replacement primary if the augmented comparison is adverse.

**Predeclared stopping comparison:** for each trajectory, choose exactly one
of the74saved checkpoints by minimum **clean validation CE**, with earliest
checkpoint breaking exact ties. Freeze and verify these choices from validation
arrays alone before computing or inspecting any selected reporting metrics.
Then report accuracy and CE on the reporting split at that same checkpoint.
Do not choose another checkpoint for accuracy, use a reporting-set maximum,
or select configurations/seeds after seeing outcomes. This is a meaningful
baseline when clean validation labels are available; their availability is an
explicit resource, not assumed in every proposed deployment.

Retain complete curves. Separately quantify:

1. Useful learning: clean reporting improvement from update100 to the fixed
   endpoint, with all seed signs. Low noisy-label fit alone is not success.
2. Noise fitting: original-train accuracy/CE against assigned labels and true
   labels, plus actually-wrong-subset target-fit and true-fit, from the same
   logits. Whole-train accuracy is not a precise wrong-label memorization test.
3. Late-horizon preservation: predeclare mean clean reporting performance over
   epoch endpoints61–72, alongside endpoints and validation-selected outcomes.
   No retrospectively chosen “stable interval.”
4. Cost: batch-gradient counts, example exposures, synchronized training,
   augmentation/evaluation time, total time, and selected-checkpoint exposure.
   No claim of matched compute simply because update counts match.

A fixed-endpoint advantage without a validation-selected advantage supports
robustness to continued training, not necessarily superior attainable accuracy.
A native gain over its unaugmented arm but loss to augmented raw is useful
recipe improvement, not a baseline win. A win here would not establish semantic
selection, alignment, or transfer beyond this corruption/augmentation regime.

## Prospective artifact and local resource budget

Historical runtimes were132.7s for AdamW and485.2s for stable global rank200;
maximum recorded global basis storage was179.40MiB. These are **historical
measurements, not promises under the new pinned one-thread environment**.
Six raw plus six filtered runs at those timings sum to about62minutes before
additional augmentation and changed evaluation/implementation costs. A rough
planning range is1.5–3hours for acquisition, not a benchmark prediction.
[Historical scalar budget source](../../results/noisy_mnist_hard_curves/summary.json).

Proposed ceiling for main's resource review: local RTX3090 only,1CPU,
16GiB hostRAM/noSwap,8GiB GPU allocation,8GiB artifact cap,3hours cooperative
deadline/3.5hours hard systemd limit,Restart=no. This is a ceiling, not an
instruction to occupy the GPU now. Verify actual free capacity and require
at least16GiB free on the large mounted volume; no displacement of existing
GPU clients, automatic retries, autoloads, downloads or paid compute ($0).
If implementation inventory or timing is incompatible, revise the proposal
explicitly before execution rather than silently shrinking the scientific task.

Store three plans, initial states, and warmup/final states for each arm;
per-step scalar diagnostics; all74readouts; configuration/source receipts.
Uncompressed FP32 logits alone require about2.13GB
(12×74×60,000×10×4bytes). Write one bounded NPZ per checkpoint, approximately
2.4MB, rather than an oversized all-training-history NPZ. Stable filtered
warmup/final snapshots add roughly1.7–1.9GB for six filtered trajectories;
raw states and full occurrence/shift plans add less, with metadata/header
reserve. A conservative8GiB cap is plausible but must be replaced by an exact
source-level byte inventory before launch. Existing128MiB artifact-read caps
must not be applied blindly to the roughly190MB rank200 snapshots: stream-hash
those opaque files without retaining/unpickling them during audit.

One later independent CPU audit should reconstruct splits/corruption/order/
shifts, all scalar metrics, validation-only checkpoint selection and contrasts,
with checkpoint bytes hash-only. Prospective audit cap:1CPU,2GiB RAM/noSwap,
15minutes cooperative/20minutes hard,64MiB report outside the immutable archive.
It is not a training replication. Budgeting this scope is the next design
decision; this document creates no acquisition or approval workflow gate.
