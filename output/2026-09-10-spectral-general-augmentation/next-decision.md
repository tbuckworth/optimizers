# Next decision after ordinary augmentation

Codex · 10 September 2026

The 12-trajectory ordinary-translation acquisition and 58-artifact independent
audit are **complete and consumed**. Do not repeat either. This is the clean,
balanced all-classes-from-initialization comparison requested by the user;
the earlier cue/rare/masking work is not a substitute or current priority.
Report all four absolute outcomes and the positive post-warmup native learning,
not only a baseline ranking. See [results](results.md).

## Selected continuation: fixed-state mechanism protocol

Prepare one bounded diagnostic using the **six saved native step-100 snapshots**
(three seeds × original/translated warmup). Their model/Adam states match raw
within condition; their observers are already trained. Do not restart model
training, replay the 4,000-update acquisition, or silently alter these states.

At each fixed model/Adam state, compare original examples with independently
translated views on the same prospectively selected training examples. Separate
the mean gradient from within-example view variation. Measure how the recorded
observer admits each and, importantly, the signed useful-loss utility of the
**actual Adam proposals** for raw/native and a matched-data-step-norm control.
Gradient retention alone cannot identify useful learning or the optimizer step.
Prospectively fix examples/views, objective readouts, norm convention, budgets,
artifacts and numerical tolerances before any new diagnostic acquisition.
Do not update the saved observer/model or call a local derivative a long-run
training effect. Keep original and translated objectives distinct.

This asks whether the main restriction is direction, scale, or neither, while
steel-manning a useful native response. A mean-preserving, scale-preserving or
later-handoff intervention may follow **only as a separate evidence-motivated
choice**. A later handoff can inherit more AdamW competence rather than produce
new learning under filtering; that attribution must be tested. Rank/LR sweeps,
further cue work, broad speedruns, paid judges, harmful-data work, cloud calls,
and optimizer-default changes are not selected.