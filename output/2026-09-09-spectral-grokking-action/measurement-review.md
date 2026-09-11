# Independent action-state measurement source review

9 September 2026

**Verdict: PASS for future source freeze and acquisition after the action batch
has completed.** This review did not inspect the live action checkpoints, run
model inference, restart training, or execute the measurement. It covered only
the new measurement runner and its synthetic contract fixtures, with the
accepted prior summary/manifests/scalar provenance inspected only as metadata.

Three material admission issues present during review were corrected before this
verdict:

- Valid action seeds can carry different repository commit IDs because unrelated
  commits occurred during the live batch. The collector no longer equates each
  seed commit with the batch-start commit. It preserves every commit as
  provenance while requiring the complete action source-hash dictionary to be
  identical.
- Merely recording current readout hashes did not prove reuse of the accepted
  readout. The collector now requires the accepted acquisition source map to
  equal the current eight-file map and every reused native scalar's map, and the
  accepted analysis source map to equal the current seven-file analyzer map.
- CPU execution or a changed software/device environment could otherwise pass.
  The collector is now CUDA-only and exactly binds Python, PyTorch, NumPy,
  platform, CUDA/cuDNN, RTX3090 identity/capability, CPU-thread setting and
  observable backend flags to the accepted prior/action environments.

## Contract findings

- `expected_roster()` is exactly 35 unique states: five native step-1501
  diagnostic states and orthogonal/norm-matched states at 1501, 2000 and 2500
  for every seed 100--104. Batch and per-seed completion markers, order, seven
  checkpoint receipts per seed, on-disk checkpoint names, source hashes,
  environments, parent legacy step-1500 receipts and metrics hashes are all
  checked before any model is constructed. Failure markers reject admission.
- Checkpoints are loaded through the singly-linked, SHA-256-bound,
  `weights_only=True` loader. The intervention envelope must have the exact outer
  schema and policy/seed/step/source/parent/common-action bindings. Its inner
  legacy state must have the exact checkpoint schema, configuration, original
  source, split, parameter layout, filter fields/counter and expected evaluation
  prefix, and its complete tensor tree must be finite.
- The old accepted analysis summary is bound by the hard-coded
  `df4f1516...09480` SHA-256. The fixed panel is exactly
  `[9, 33, 32, 49, 11]`. For each seed, the generated pair/sum grid, train/test
  IDs, stratified fit/evaluation IDs and all 20 row permutations must equal the
  corresponding accepted legacy step-1500 NPZ members in dtype, shape and value;
  the scalar/raw and probe-split receipts are also bound. The runner then calls
  the unchanged accepted activation, probe, behavior and saved-array analysis
  helpers with the original `RECIPE` constants.
- The code performs inference only for the admitted 35 new action states. It
  never loads an archived native model checkpoint, constructs an optimizer,
  calls backward, takes an optimizer/filter step, or invokes training. The live
  batch cannot be admitted until its exclusive `batch-complete.json` exists, and
  all admitted batch/prior receipts are rehashed before successful completion.
  Models are newly constructed copies; activation extraction is the previously
  fixture-audited nonmutating helper, with an additional CPU/all-CUDA RNG guard
  here.
- Per-state activations, behavioral values, both probe families, fixed-panel
  summaries, symmetry/margin values and full checkpoint provenance are retained.
  Nonfinite model/activation state fails directly; non-JSON-finite downstream
  values fail exclusive JSON serialization. Existing 2000/2500 evaluation rows
  independently constrain reproduced behavior within the frozen tolerances.
- Output must be a new large-volume descendant. Writes are atomic and exclusive;
  raw, scalar and analyzed-state artifacts are receipted. Prospective NPZ and
  exact JSON sizes enforce a 3 GiB output ceiling and 1 GiB free-space reserve.
  The cooperative deadline is 19 minutes, CPU math is pinned to one thread, and
  peak RSS is recorded and checked against 16 GiB.

## Validation and launch condition

All 7 synthetic fixtures passed independently in 0.104 seconds with the four
CPU math backends pinned to one thread. Python compilation and `git diff
--check` also passed. Fixtures cover exact roster, per-seed commit preservation,
source and receipt corruption, envelope separation, accepted old-source pins,
CUDA environment binding, and exclusive/bounded output.

The PASS is conditional on waiting for the one live training batch to finish and
then freezing this source before measurement. The in-process memory/time checks
are cooperative, so the future acquisition must also use and verify its planned
external 16 GiB/no-swap/20-minute service bounds. Batch completion and input
receipt validation at that later time remain acquisition facts, not claims made
by this source-only review.

The review-changes workflow materially guided the fail-closed provenance and
environment checks above; no style-only findings were treated as blockers.
