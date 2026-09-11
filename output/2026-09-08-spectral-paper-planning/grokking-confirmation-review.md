# Independent review: current-stable modular-addition confirmation

8 September 2026. Bounded pre-launch review; no scientific experiment was run
and no prospective outcome was inspected.

## Verdict

**PASS for the registered seed-100 operational bundle after the reviewed files
are frozen.** I found no remaining launch-critical defect in the fixed three-arm
comparison, post-update measurement semantics, state preservation, or bounded
launcher. Seed 100 remains part of the registered five-seed comparison; it is
not a discardable pilot. Seeds 101--104 must be retained as the fixed remainder
after the planned resource check, irrespective of seed-100 outcomes.

## Historical task and provenance

- The recovered modular-addition JSON inventory contains only seed 42 in the
  exploratory runs and seeds 42--46 in the matched three-arm runs. A restricted
  filename/content search found no recovered modular-addition use of seeds
  100--104.
- `experiments/grokking_model.py` is unchanged from raw-result commit
  `cc2b164c4b4edfeaa37c24be2467992bfb7a0d63`. Its fixed architecture has exactly
  227,313 trainable parameters. At p=113 and `train_frac=.3`, the deterministic
  split contains 3,830 training and 8,939 held-out ordered pairs.
- The new AdamW settings match the historical source: learning rate .001,
  weight decay 1, betas (.9, .98), full-batch cross-entropy, and no schedule.
  The filtered arms match rank 200, decay .99, warmup 100 and unit-strength hard
  projection. The current legacy and stable arms share the current canonical
  implementation and differ only in `stable_update`.
- I executed an authorized deterministic CPU software fixture against the
  implementation recovered directly with `git show` from `cc2b164`. Across 20
  steps of a fixed small three-class problem, current
  `SpectralGradientFilter(stable_update=False)` was bitwise equal in loss, all
  model parameters, AdamW state, `V`, `S`, `grad_mean`, and `step_count` to the
  historical `WeightCovarianceFilterV2`. This verifies the retained legacy path
  under current software for the exercised fixed mode.

The old runs did not preserve their PyTorch/CUDA environment. Consequently,
the fixture does **not** establish bitwise equality across the historical and
current machines or software versions. The confirmation is a same-task current-
software comparison, not an exact replay of old floating-point execution.

There is also a harmless but important reporting offset in the historical
runner: raw `epoch=0` was evaluated after its first optimizer update, so a raw
historical threshold at epoch 2500 means 2501 completed updates. The prospective
harness uses a genuine pre-update step 0 and evaluates step 2500 after exactly
2500 updates. The historical paired difference of 1,170 is unaffected, but old
absolute values should be called logged epoch labels or converted by adding one.

## Prospective numerical and measurement checks

- Each child process seeds before data/model construction. Data shuffling uses
  its own seed-local generator, so all three arms for a seed receive identical
  ordered splits and model initialization without arm-dependent RNG consumption.
- Training is exactly 6,000 updates. Evaluation is pre-update only at step 0,
  then common and post-update at steps 50, 100, ..., 6000. Both train and held-
  out loss/accuracy are recomputed from the same post-update parameter state.
  No arm stops on threshold attainment.
- The recorded first 90% crossing is correctly interval-censored on the 50-step
  grid. Non-crossings remain right-censored, and sustained attainment is
  reported separately without claiming indefinite stability.
- CUDA synchronization brackets training and evaluation timing. Checkpoint
  capture plus write is timed separately, including the large filter-basis
  device-to-host copy; setup/progress/other time and total child wall time are
  retained. These are shared-machine instrumentation, not a dedicated timing
  benchmark.

## Checkpoint fidelity and operational safety

Every fixed checkpoint preserves model parameters and buffers, full AdamW state,
the filter basis/spectrum/centered mean/projection rank/counters/diagnostics,
resolved filter configuration, CPU and all-device CUDA RNG states, step,
measurements and timing so far, parameter layout, split identity, configuration,
and source identity. Restoration checks the source/configuration/split/layout,
restores optimizer and filter tensors to their required devices, restores RNG,
and requires the filter counter to equal the training step.

Checkpoint writes are exclusive and atomic on the target filesystem. Reads
require a singly-linked regular file, verify SHA-256 before parsing, and use
`torch.load(..., weights_only=True)`. The sequential launcher creates a new
exclusive large-volume root, pins the harness/model/filter/test/protocol source
hashes, stops on a changed source, failure, timeout or STOP marker, and validates
the exact 121-row evaluation grid and ten-checkpoint roster. It streams and
rechecks every checkpoint hash and writes batch completion only after a final
source check. The proposed one-hour child limit, two-hour outer limit, 16 GB
memory ceiling and one-process execution are compatible with the expected
roughly 18 GB total checkpoint footprint across all five seeds.

## Verification performed

The focused unit suite passed all six tests on CPU. It covers the fixed roster
and arm settings; exact next-step continuation for both legacy and stable filter
states; safe, hash-bound, exclusive checkpoint serialization; order-sensitive
split identity; interval/right censoring; and exclusive JSON writes. The
separate historical fixture above adds direct legacy-path equivalence evidence.

No claim follows yet about whether stable filtering preserves the historical
learning result. That remains the prospective outcome, and this review does not
weaken the existing task-dependence, wall-time, or legacy-version caveats.
