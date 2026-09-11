# CPU component checkpoint

Codex parent, Spectral Optimizer Investigation, 6 September 2026.
Evidence: dataset-free engineering fixtures only, not a neural result.

## Accepted scope

Parent read the complete implementation and tests, incorporated independent
review, and ran all 38 iteration007 CPU tests successfully in 1.021 seconds.
The source identities and numerical maxima are in
cpu-component-checks.json (artifact not distributed in this public snapshot).

- `response_math.py`: six owned float32 inputs, native signed-zero preservation,
  the fixed norm/direction gates, all 15 delivered-pair geometries and all 15
  scalar contrasts, complete three-bundle masks without available-case means.
- `state_core.py`: exact pre-forward model/AdamW/observer/RNG capture and restore,
  owned tensor/primitive encoding, restricted trusted-local codec, and tiny CPU
  source-next-step replay including actual parameters, moments and observer state.
- Independent numerical fixtures: NumPy Adam equations versus native CPU32,
  explicit NumPy MLP forward/backward versus Torch CPU64, fixed error ceilings,
  dot propagation and common-baseline cancellation. No producer imports.

The numerical tests cover 128 Adam coordinates at four counters and four scales.
Maximum error/envelope ratios are 0.0260208517 for first moments, 0.0390498263
for second moments and 0.0470917987 for parameters. CPU64 CE and gradient-component
disagreements are at most 1.1102230246251565e-16 and 2.914335439641036e-16.
These small fixtures support implementation feasibility, not universal numerical
guarantees, full-size runtime, or scientific conclusions.

## Review findings and fixes

The independent response-code reviewer returned PASS with two minor hardening
items: malformed huge integers raised OverflowError instead of ValueError, and
the RNG-neutrality invariant lacked a regression. Both are fixed and tested.
The reviewer independently checked every coefficient, zero case, missing-cell
mask, storage ownership and absence of RNG/data/CUDA activity. Parent had already
fixed native negative-zero preservation before that review.

Parent state-core review required actual module class/config checks, ordered live
optimizer/observer aliases, exact device bindings, strict boolean/counter/RNG
domains, finite nonnegative second moments and singular values, raw-byte equality
including signed zero, and failed-restore RNG rollback. Regressions cover these
and malformed/unknown fields. The tiny fixture records a real live next step
before restoring the anchor; a replay-generated endpoint is not used as its
source witness. Restored parameters, full moments/counters and all observer
fields match this live source byte-for-byte.

An early exists-check plus replace could overwrite a concurrently created file.
Publication now atomically refuses replacement; tests preserve both existing
and raced target contents. This trusted-local codec publishes through a temporary
same-directory link and removes its own temporary name. It is not the scientific
artifact controller: it does not implement manifest indexing, retained partial
scientific failures, path/hash/size provenance or aggregate accounting. Do not
use it unwrapped for scientific artifacts or treat transient aliases as storage
savings. Scientific publication still needs the full no-alias/cap contract.

## Parent verification

From the repository root, with `CUDA_VISIBLE_DEVICES=''` and
`PYTHONDONTWRITEBYTECODE=1` for numerical tests:

```text
python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007 -p 'test_*.py' -v
38 tests, OK: 13 response + 15 state-core + 10 independent numerical fixtures.

python3 -m unittest discover -s tests -v
17 tests, OK.

python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation -p 'test_goal_reminder.py' -v
6 tests, OK.

python3 scripts/lint_knowledge.py
Knowledge lint passed: 14 pages, 11 indexed content pages.

git diff --check
No whitespace errors.
```

All 61 tests passed. CUDA remained uninitialized in the iteration007 suite.
The reminder's actual dry-run sees the exact active goal and would queue; it did
not send another message. The timer remains enabled/active/waiting with next
trigger 19:18:19 BST and unchanged OnActiveSec/OnUnitActiveSec of two hours.
This received one-off check is not a second schedule. No experiment was restarted.

## Explicit remaining work

The parent accepts TL/Tq and K=32 unchanged for CPU implementation under the
[updated component decision](implementation-decision.md). Projection-screen
and native-loss-concordance fixture coverage remain pending. Full scientific
envelopes, exact measurement fields, artifact/byte controller, producer loss
module, branch integration and independent raw-artifact audit are not implemented.
GPU state/device replay, native capture neutrality, full-size loss runtime and
the 632-MiB planning estimate are unmeasured. No pilot or scientific execution
is approved. No MNIST plans or outcomes have been generated for this iteration.