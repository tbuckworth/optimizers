# Independent selectivity audit implementation

## Readiness — 10 September 2026

The existing checker is complete and ready for main's final source review,
freeze and once-only execution after acquisition completion. This is not a
scientific audit PASS. No scientific arrays, checkpoints, MNIST bytes, model
objects or producer helpers were loaded during preparation. Main owns both
service launches and their admission checks.

- Checker: `scripts/audit_spectral_selectivity_boundary.py`, SHA256
  `a992842846c767c0af44cf41a19fb3042b4aebc951089d78469e26b0f3465527`.
- Fixtures: `tests/test_audit_spectral_selectivity_boundary.py`, SHA256
  `146cf1727e3e315b12ea0f6723aedfbac830870914c530457a5b454bf1611a54`.
- Producer schema inspected: SHA256
  `c851cb711e85a07aacad891cfd1122d5bf1a3f7d406142d1b98cca4cb2afac0f`.
- Final fabricated-only run: **18 tests PASS, 0.771 seconds**. Whitespace
  check passes. These tests do not benchmark the eventual full-audit runtime.

The checker requires the exact 245 receipted artifacts plus completion,
36 trajectories, 756 saved evaluation rows and 36 native diagnostic anchors.
It independently rebuilds all 86 endpoint fields, all corresponding warmup
changes, paired policy contrasts, cue interactions and three-seed summaries.
Rare/actually-wrong membership, sham quotas, input normalization and patches
are rebuilt from the saved plan's prescribed RNG streams and pinned training
IDX bytes, only when main executes the audit. No official test data is read.

The fixtures cover receipt/hash/path/symlink and JSON/NPZ safety failures;
CPU restricted tensor decoding with the documented nested RNG types; scalar
tensor hash shape; paired plans and rare exclusion; patches; CE/macro/micro/cue
denominators; all endpoint fields and absent-wrong nulls; nonorthogonal native
action versus an ideal projector; zero-gradient definitions; numerical rank;
rounded decay; Adam moments, counters and ordering; full saved diagnostic
schemas at updates 101 and 2,000; exact final-state identity after clearing
gradients; and norm-control zero/tolerance cases. A three-coordinate fabricated
state exercises diagnostic arithmetic without constructing or stepping Adam.

Two pre-execution defects were corrected transparently. Source inspection found
that `ascontiguousarray` changed scalar tensor hash shape from `[]` to `[1]`;
the inherited encoding now preserves original shape. The first fixture run
then had 15 passes and two errors, both because Python Boolean values satisfy
`isinstance(value, int)` and the exact-value guard rejected matching booleans.
The assumption-debugger trace isolated that shared type guard; explicit Boolean
dispatch and a direct regression fixture fixed it. Neither involved scientific
data, altered producer code or an outcome-dependent tolerance change.

## Fixed pre-result numerical tolerances

All values below are fixed before acquisition or scientific readout. They are
arithmetic admission checks, not scientific-effect thresholds.

| Quantity | Required agreement |
| --- | --- |
| Receipts, IDs, labels, plan arrays, input bytes, counts, roster, nulls and state hashes | Exact; no tolerance or dropped rows. |
| Float64 CE sums/means, paired summaries, utilities, raw gradient energies and coherence | Absolute `1e-10` plus relative `1e-10`. |
| Native probe-action energy/retention and FP32-mean quantities | Absolute `1e-7` plus relative `5e-4`; raw float64 energies/coherence are additionally checked at the tighter tolerance above. |
| Saved native training action | L2 error from independently calculated float64 `V(V^T g)` no larger than `3e-5 * ||g||_2 + 1e-12`; identity fallback stays identity. |
| Numerical-span basis diagnostics and movement decomposition | Absolute `1e-10` plus relative `1e-8`; rank is exact under `eps64 * max(V.shape) * largest_singular_value`. |
| Norm-history arithmetic | Absolute `1e-12` plus relative `1e-10`; delivered/native norm mismatch must separately satisfy the producer's `10 * eps32` bound, with exact explicit-zero rules. |
| Adam moment recurrence | Each coordinate's error is bounded by `16 * eps32` times the sum of absolute recurrence terms, plus `1e-30`. |
| Adam parameter recurrence | Each coordinate's error is bounded by `16 * eps32 * (abs(theta_before) + abs(ideal_adaptive_move)) + 1e-30`. |

Native arithmetic permits CPU/CUDA FP32 reduction differences; it does not
silently substitute orthonormal span projection for the saved `V(V^T g)` action.
Wrong-minus-true subtraction remains FP32. The movement split reproduces the
rounded FP32 decay multiplication before separating adaptive motion; the Adam
identity check independently compares its ideal recurrence with an explicit
roundoff bound. No gradient/utility sign must agree with finite CE change.

CLI: `/usr/bin/python3 scripts/audit_spectral_selectivity_boundary.py --execute
--acquisition-dir <acquisition-001> --output-dir <new audit-001>
--expected-complete-sha256 <SHA> --expected-manifest-sha256 <SHA>
--expected-sources-json <main-frozen source-pin JSON>`.
The exact service is `spectral-selectivity-boundary-audit-001.service`.
It enforces one CPU, 4 GiB/no swap, five minutes hard and 250 seconds cooperative.
Receipts, data, producer pins, checker sources and the main-supplied pin file
are rechecked before reporting PASS. Output is exclusive and at most 100 MiB.

## Preserved preparation history

Original status: preparation only, 9 September 2026. The producer had not yet declared
its artifact schema stable. No scientific arrays, checkpoints or MNIST bytes
have been opened, and no acquisition or scientific audit has been run here.

The checker will own its plan reconstruction, statistics, contrasts and saved
diagnostic arithmetic; it will not import the producer or its numerical helpers.
All 36 trajectories and all 36 native diagnostic events are indispensable
acceptance checks, with no outcome-dependent exclusions. The prospective limit
remains one CPU, 4 GiB RAM, 250 seconds cooperative / five minutes hard, no GPU,
and at most 100 MiB of new audit output.

The producer source and inherited snapshot serialization were inspected as
source only. Required conventions include FP32 subtraction for wrong-minus-true
gradient residuals, FP32 means for native mean action, a float64 numerical span,
and the rounded FP32 AdamW decay operation. An ideal float64 decay vector must
not be mistaken for the producer's actual rounded decay decomposition.

Safe deserialization follows the current primary documentation:
[PyTorch 2.11 torch.load](https://docs.pytorch.org/docs/2.11/generated/torch.load.html)
with explicit `weights_only=True` and `map_location='cpu'`, and
[NumPy 1.26 load](https://numpy.org/doc/1.26/reference/generated/numpy.load.html)
with `allow_pickle=False`, closed archive handles and bounded archive members.
The [AdamW state contract](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)
requires explicit parameter-order checks: serialized optimizer IDs alone do
not identify model parameters. No model object or optimizer is constructed.

This corroborates recorded arithmetic and provenance, not an independent proof
that every saved logit or per-example gradient came from the asserted model.
That connection rests on reviewed producer source, preservation fixtures and
hash-bound artifacts. The first 32 actually wrong probes are a fixed convenience
subset in class-blocked training-ID order, not a representative wrong-label
sample or an outcome-selected subset.
