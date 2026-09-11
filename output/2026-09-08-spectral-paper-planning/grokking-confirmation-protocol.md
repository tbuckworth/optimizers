# Current-stable modular-addition confirmation

8 September 2026. Prospective within-benchmark confirmation; no results at
registration. Authorized by the user's explicit approval of the emailed paper
plan. This replaces the old I21 tracking next-step, not completed experiments.

## Question and scope

Does the useful historical modular-addition learning phenomenon survive the
current stable covariance update? Does the stable update improve or weaken it
relative to the legacy numerical path? Accuracy alone will not identify why.
This is the entrance to the approved rule-learning mechanism study, not a
training-efficiency campaign or a new clustering experiment.

## Fixed comparison

- Arms: AdamW; AdamW plus `stable_update=False`; AdamW plus
  `stable_update=True`. The last two use the same current canonical source,
  selecting its explicitly retained legacy versus stable path.
- Fresh seeds and paired data splits: 100, 101, 102, 103, 104. Check recovered
  modular-addition run inventory before launch for collisions. Pair model
  initialization and data within seed. Five seeds is a modest confirmation
  sample, not a power-calculated guarantee; historical baseline/filter
  threshold separation is roughly 1,170 steps with little five-seed variation.
- Same model as the historical experiment: `experiments/grokking_model.py`,
  p=113, width=128, four attention heads, MLP width=512, one layer, GELU and
  LayerNorm. This is not assumed identical to Nanda et al.'s circuit setup.
- Full-batch training on 30% of the 113² input pairs, exact modular-addition
  targets; remaining pairs evaluated without selecting hyperparameters.
- AdamW learning rate 0.001, weight decay 1.0, betas (0.9, 0.98); no schedule.
  Filter rank cap 200, decay 0.99, warmup 100, strength 1, hard weighting,
  no normalization or adaptive rank. Stable numerical tolerances retain their
  documented source defaults; all resolved settings saved.
- Fixed 6,000 optimizer updates per arm, with initial evaluation at step 0 and
  common post-update evaluation every 50 steps. Do not stop an arm when it
  crosses the accuracy threshold. If unfinished at 6,000, report right
  censoring; do not choose an outcome-driven extension in this confirmation.
- No additional learning rates, ranks, seeds, benchmark changes or control
  arms silently added. The first seed bundle can run before the remaining four
  to check operational feasibility, but its scientific outcomes remain in the
  registered five-seed comparison. Do not rerun it as a discarded pilot.

## Readouts and interpretation

Primary descriptive effect: paired change in the first recorded step reaching
90% held-out accuracy, stable versus AdamW and stable versus legacy. Each
observed crossing is interval-censored by the 50-step evaluation grid; a
non-crossing is not assigned a fictitious 6,000-step crossing.

Report every seed, final held-out accuracy/loss, and sustained attainment
(first evaluation at or above 90% after which every remaining scheduled
evaluation stays at or above 90%). This endpoint-dependent sustained metric
does not claim indefinitely stable generalization. Include training accuracy
and loss curves, which distinguish memorization from later generalization.
Report paired differences and uncertainty with n=5 made explicit; do not count
checkpoints or examples as independent replicates. No headline p-value is
planned for this small confirmation.

Separately report synchronized cumulative training seconds, evaluation seconds,
checkpoint/I/O overhead and end-to-end wall time. Instrumented, desktop-shared
times are not a dedicated speed benchmark. Actual delivered gradient geometry,
circuit formation and cleanup remain subsequent mechanism questions, not
conclusions from these curves.

## Preservation and operational bounds

New harness only; do not modify canonical optimizer/model or historical runner.
Save resolved configuration, source hashes/commit, exact split identity,
software/hardware metadata, metrics, and complete model/Adam/filter/RNG state.
Fixed checkpoint steps: 0, 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000.
These enable later, explicitly specified checkpoint interventions. No test-
selected “best” checkpoint is substituted for the full trajectory.

Checkpoints must include the filter basis, spectrum, centered mean, counters,
projection rank and configuration, not only model weights. CPU fixture tests
must establish same-state continuation before scientific launch. Safe tensor-
and-primitive loading only (`weights_only=True`); never load unknown pickles.

Use a new exclusive directory under `/tmp/spectral-experiment-artifacts/`, not the root disk
or existing result directory. One RTX3090 process at a time; pin one CPU math
thread initially to avoid small-matrix thread oversubscription. First launch
is bounded to seed100's three arms, a two-hour total hard limit and one-hour
per-arm limit, with no paid cloud/judges. Stop on nonfinite state, failed arm
or resource fault; preserve incomplete artifacts and diagnose rather than
silently restarting. A scientific arm's live or completed handle is consumed.
Remainder of the fixed roster can follow after timing/resource review, without
selecting which seeds or arms to keep on scientific outcomes.

Checkpoint storage at maximum rank is roughly 182 MB of basis per filtered
checkpoint, about 18 GB over the registered filtered runs before metadata and
other state. Available large-volume space was 686 GB at planning. Retain all
raw measurements; copy compact results/manifests into the repository.

## Implementation check

Installed PyTorch is 2.11.0+cu128 on RTX3090. The
[official checkpoint tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
requires optimizer state for resumption; the custom filter needs explicit
additional serialization. The [versioned CUDA documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html)
explains asynchronous execution, so synchronize timing boundaries. The
[reproducibility notes](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
limit reproducibility claims across versions/platforms; record the actual
environment instead of promising historical bitwise identity.

## Continuation authority