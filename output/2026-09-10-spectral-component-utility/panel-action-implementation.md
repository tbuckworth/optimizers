# Panel selection and faithful action implementation

Codex — Spectral Optimizer Investigation · 10 September 2026

Main authored the new import-inert panel and private-action helpers. All tests
use fabricated metadata/tensors/models, never the original plans, checkpoints,
images or a GPU. These modules provide no CLI or experiment launcher.

## Fixed panel API

`select_panels(role_arrays, seed)` in
[`spectral_component_utility_panels.py`](../../experiments/spectral_component_utility_panels.py)
accepts exactly seven role/label arrays extracted from a verified strong plan.
It checks int64 shapes, digit labels and a disjoint exhaustive partition of
source IDs 0…59999. The original plan hash and label/source correspondence
remain the runner's responsibility; no plan is generated or loaded here.

Explicit streams40/41 produce two disjoint 64-example action batches and a
separate 256-example component panel, with fixed int8 dy/dx action shifts.
Original reporting order supplies the first128 evaluation IDs and first500
baseline-check IDs. All selected labels, wrong masks, IDs/positions and the
exact ordered25-view support are returned as independent contiguous arrays.
There is no class balancing, role-size override, extra seed or global RNG draw.
The same seed call therefore supplies every parent stage/mode identically.

Six fabricated tests cover literal separate stream reconstruction, every output
mapping, label-independent selection, role disjointness, complete view order,
empty wrong mask, read-only input/global RNG preservation and malformed inputs.
The first run passed all six in0.047s under a60s timeout.

## Private raw/native action API

`paired_actions(parent, raw_gradient)` in
[`spectral_component_utility_actions.py`](../../experiments/spectral_component_utility_actions.py)
requires the exact six-parameter strong native model at100 or56304, canonical
optimizer/tracker ownership/configuration and a finite FP32 gradient vector.
That gradient is supplied once by the caller; the helper never computes a
different action gradient per policy. Joint model/Adam/tracker deep copies
preserve ownership and inherited state. Raw bypasses the observer; native
observes/projects once. Each optimizer advances once. No private endpoint
becomes a parent for the next branch or batch.

Returned CPU tensor records contain exact before/after parameters and rounded
decay-only endpoints, raw/delivered gradients, before/after Adam moments and
counters, and native post-basis/singular values. The unused raw observer clock
is honestly unchanged. That raw endpoint is not a canonical native snapshot
with synchronized clocks and must not be passed to `snapshot_with_rng`.

Fabricated tests compare exact canonical updates at both prescribed stages,
including moments, gradients and bases; prove parent/input immutability and
repeat-call equality; reject a changed activation despite unchanged parameter
names; and compose restore, one common action gradient, both actions, all six
diagnostic derivatives and full/0.1-path S/F/C finite accounting.

The first four-action-fixture run passed in1.117s. Main's later topology
validation patch introduced an indentation typo, caught immediately by the
suite import before any science. It was fixed without changing scientific
settings or tolerances. The full suite then passed47 tests; the composed
fixture was added afterward. The final acceptance receipt owns final counts,
hashes and independent review, not this historical intermediate count.

## Boundaries

All original scientific sources are unchanged. Actual source-plan checking,
GPU checkpoint restoration/baseline equivalence, the image transform/evaluation
runner, guarded artifact writer, exact live-object inventory and independent
saved-array auditor are still separate work. A CPU fixture pass cannot admit
or certify a real model replay. Resource limits, source freeze and once-only
attempt handling remain mandatory for the selected local diagnostic.
