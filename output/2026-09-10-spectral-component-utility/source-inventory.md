# Saved-state feasibility: strong-regime component utility

Codex — Spectral Optimizer Investigation · 10 September 2026

## What actually exists

The completed producer's `results.json` contains 27 checkpoint receipts:
three initial, twelve warmup and twelve final. Its 888 intermediate/fixed
readout files contain logits, not restorable intermediate models. The
parent inventory (artifact not distributed in this public snapshot) copies the twelve native-state
receipts and original source/data pins from the producer, whose SHA-256 is
`e98125d3bcbd85e402c8269797e6cb48797b9ebbe9d9715897f8244203c69e8a`.
The accepted independent audit is
`6e2677ae3feb41a3da16ea11639a15871577ff8cd469c6afcda0b8c67096ec7c`.

All twelve candidate native paths were checked with filesystem `stat` and match
the recorded sizes. Each of six warmups is 97,843,895 bytes; each of six finals
is 192,843,767 bytes. Total candidate input is 1,744,125,972 bytes. This turn
has not deserialized, rehashed or replayed those checkpoint contents; the
previous accepted audit supplies their historical byte-integrity evidence.
Any acquisition must freshly hash each exact input before deserialization.

The inventory also now binds all six original plan NPZ/JSON receipts and each
parent's matching original warmup/final readout. Parent rows are explicitly
sorted by seed and none/translate; each row lists warmup before final. These
are copied producer receipts, not newly replayed observations. The readout
schema is `{step,train,validation,reporting}`; the equality check uses the
first 500 rows of `reporting` with the matching plan's reporting IDs/labels.
Fresh byte verification and role/shape checks remain acquisition requirements.

| Available state | What it supports | What it does not support |
|---|---|---|
| Native update 100, both augmentation modes, all three seeds | First-filtered-step counterfactual at a common model/Adam/observer parent | Late-regime effect or comparison between warmup modes holding weights fixed |
| Native update 56,304, both modes, all three seeds | Local filter-versus-bypass counterfactual at each actual final native parent | Causal explanation of the earlier endpoint gap between different trained models |
| Raw warmup/final | Model/Adam reference, no learned observer | Native action without specifying how a new observer is obtained |
| Every epoch's logits | Previously audited loss/accuracy curves | Gradients, hidden representations or missing optimizer states |

No intermediate checkpoint can be reconstructed from these logits. No new
training is justified merely to fill that convenience gap. Initial snapshots
can have different raw/native tracker presence because the branch roster rotates;
they are not needed for the selected candidate local question.

## Source-level restoration requirements

Read the full [strong helper](../../experiments/spectral_strong_augmentation_core.py),
the producer's acquisition/save path, the full strong data helper and relevant
I9 clone/RNG/tracker/restore functions. The strong schema is
`spectral_strong_augmentation_snapshot_v1`, with ordered model specification
784–256–128–10 and 235,146 parameters. It saves CPU copies of:

- Model parameters, six parameter gradients and all module modes.
- AdamW state dictionary, including per-parameter step, first and second moments.
- Every tracker attribute except the model/optimizer/parameter aliases.
- Python, NumPy, torch CPU and CUDA RNG state.

The old `neural_core.restore` explicitly requires `i9_neural_snapshot_v1` and
the one-hidden-layer model. It cannot be reused by relabeling the new snapshot.
The strong helper deliberately has **no restore function**. A new strict adapter
is therefore implementation work still required, not a verified existing feature.
It must rebuild the exact ordered layers and ownership aliases, preserve tensor
dtypes/devices (in particular device V/mean versus CPU FP64 S), gradient/mode
state, all tracker attributes and complete Adam state. Validate round-trip
tree identity and canonical counters on fabricated states before any real load.

Restoration must not change the original checkpoint, global RNG state outside
its controlled context, or parent objects used for another candidate. Baseline
original-image predictions should be checked against saved readout logits;
the producer uses chunks of 500, so comparison must preserve those batching
semantics rather than treating altered-GEMM roundoff as a different model.
Byte-identical restoration claims remain unproven until that explicit check.

The older [direction/size helper](../../experiments/spectral_augmentation_state_core.py)
has a hard rank-32 basis cap and the old experiment's roster assumptions. It
cannot silently serve rank200. Its private joint-copy and actual-displacement
conventions are useful precedents; new code must leave the frozen helper intact.

## Current framework checks

Checked the versioned PyTorch **2.11** documentation, matching the source
environment. The moving `/stable` alias currently redirects to 2.14 and is
not the version basis for this inventory.

No installed library, optimizer or canonical acquisition source was changed.

## Resource feasibility, not admission

A twelve-parent, two-action-batch design would have 24 native post-observer
bases. At the rank200 upper bound, these cost
24 × 235146 × 200 × 4 = 4,514,803,200 bytes before metadata. This dominates
output size; a vector-only budget would be wrong. Smaller actual warmup ranks
must not be relied on for the admission bound. Parent checkpoint copies need
not be duplicated in the new archive; pin their existing receipts.

An 8 GiB output cap can accommodate those bases, compact gradients/displacements,
logits and receipts, subject to an exact implementation inventory. Process one
parent/action at a time; no all-parent GPU retention or full P×P covariance.
Proposed local limits are 1 CPU, 8 GiB host/no swap, 4 GiB GPU allocation,
20-minute cooperative/30-minute hard deadline, plus one separately bounded
saved-array audit. Runtime is provisional, not a benchmark or a launch promise.
No paid resource is needed or reserved. The exact roster and scientific
distinction are decided separately; this inventory alone selects no execution.
