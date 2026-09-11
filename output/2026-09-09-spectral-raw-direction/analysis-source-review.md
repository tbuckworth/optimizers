# Independent source review: raw-direction paired JSON analysis

9 September 2026. **PASS for final source freeze and later bounded execution;
no scientific or launch-blocking source defect was found.** This review is not
an analysis result or launch receipt. It did not read real scientific JSON/NPZ/
checkpoint data, calculate saved-data results, run inference/training, or touch
the live acquisition or frozen measurement sources.

Reviewed current SHA-256:

- `experiments/analyze_grokking_raw_direction_results.py`:
  `e45edf6c891319d3676baa54eee8de37a0a445185157c28505ff7fe16395ace3`;
- `tests/test_grokking_raw_direction_analysis.py`:
  `68f51d376069ac3f12daa08f65e28d6f234206f0389683f82329af55c0fbb3ee`;
- paired-analysis protocol:
  `12860d36b39a831e80a9d7041385da0289f837328e863b7d710ad32582b38e1d`;
- implementation note:
  `5922f51130e729e708aa6a3a8f563738bb5ca9e8e53e9e90020e830fd465c01c`;
- launch plan:
  `a418e15a177f73db895b937fdacb8bff29fee55f3da76830bbc5f2d687f1d046`.

The sole imported implementation is the unchanged standard-library action
analyzer, SHA-256
`e0a57f6ac5dfc0e6fb5638c0777efe1de5fba7b455c2421880514cfe28d2c628`;
it has no worktree diff.

## Fixed estimand and arithmetic

The new input roster is exactly 15 ordered identities: seeds 100--104,
`raw_norm_matched`, and steps 1501/2000/2500. All 15 rows are validated before
endpoint selection, so a malformed 1501 state cannot be hidden merely because
it is not contrasted. Comparisons are restricted to steps 2000 and 2500 and
use all five paired seeds for:

- raw minus archived norm-matched, the primary contrast;
- raw minus archived native, secondary;
- raw minus archived orthogonal, secondary.

This yields six fixed contrast rows. Each contains the unchanged 14 metric
definitions and units. The three primary metrics remain held-out CE (lower is
better), held-out correct-class margin (higher), and selected-five final-hidden
held-out R² (higher). Accuracy stays a fraction. Descriptive metrics have no
invented favorable direction.

The frozen `_paired_summary` retains each seed's left, right and left-minus-right
difference. For a complete five-seed metric it computes the arithmetic mean,
sample SD with denominator four, SE=SD/sqrt(5), and positive/zero/negative
counts. The adapter derives favorable count only from the declared lower/higher
direction. It adds no p-value, equivalence flag, composite score, endpoint
extension or pseudo-replication.

Required metric values reject missing keys, booleans and nonfinite numbers.
Only the four explicitly optional secondary symmetry ratios may be `None`. If
either member of any such pair is undefined, the five paired records and
defined count remain visible, while mean/SD/SE, all sign counts and favorable
count stay `None`; the code never averages a favorable four-seed subset.
Aggregate overflow is also rejected.

## Archived and new provenance binding

The archived action summary is pinned to SHA-256
`249a8b9f5b680f72f7e9a402f56a9a4734814d99dc7d8097442eac4c965aa237`,
schema and its exact ordered 35 action rows plus 15 archived-native rows. Its
raw/scalar/checkpoint receipt identities are tied to each saved row. The code
also verifies the accepted original representation-summary identity, old
35-state measurement completion and manifest, their measurement/action source
maps, fixed panel, old action-batch completion
`22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45`,
and all five archived seed completions. It reuses those already aggregated rows;
it never calls the old analyzer's `main`, repeats an old aggregation, or opens
an old model/array payload.

New measurement admission requires exact 15-state manifest/completion rosters,
successful status, no failure marker, the same prior-summary/recipe/fixed-panel
contracts as the accepted old measurement, and the complete expected source
union. The acquisition map must equal the accepted old scientific map plus all
15 explicitly listed new/frozen acquisition paths (19 unique map entries after
overlap). The measurement map must equal the accepted
old measurement map plus that acquisition map and all five collector/guard/
fixture/protocol paths. Every map is checked against current file bytes.

The raw batch manifest/completion and each seed manifest/completion are parsed
and receipt-bound. The analyzer requires the fixed five seeds, exact source and
resource metadata, seed-100 admission chain, 1,000-step identity, three capture
receipts, nine-artifact disk roster, no failure markers and fixed checkpoint
identities. Each new checkpoint is tied to the measurement result. Its original
parent receipt/metrics hash and fork scientific hash must agree both with the
new seed completion and the corresponding accepted archived seed/fork.

For each state, raw NPZ content is streamed only for SHA-256/size/path receipt
verification; no member is loaded. Scalar and analyzed JSON use their distinct
schemas. Shared behavior fields and selected/null probe summaries must agree,
while analyzer-derived margin, symmetry and fixed-panel fields are retained for
the separate independent arithmetic audit rather than incorrectly requiring
whole-document equality. Scalar, analyzed, raw, checkpoint, seed completion,
seed manifest, acquisition source, parent/fork and per-seed recipe/split
contracts are all cross-bound. The summary omits only bulky `full_symmetry`;
the 15 scalar and 15 analyzed source files are copied byte-identically and
receipt-hashed.

All admitted measurement/acquisition/archived receipts and source maps are
rechecked immediately before completion. The output interpretation preserves
the archived-reference/CUDA limitation and explicitly disclaims equivalence,
mediation, pre-fork, semantic-usefulness, safety and speed conclusions.

## Standard-library and resource boundary

The adapter and its only imported module use the Python standard library. A
fresh-process check confirmed neither `torch` nor `numpy` enters `sys.modules`.
There is no checkpoint loader, model, inference routine, NPZ parser or array
calculation path.

The inherited output writer requires a new descendant of the large-volume
storage, refuses an existing file via exclusive creation, enforces 100 MiB
output plus 1 GiB free reserve, and preserves partial output on failure. The
in-process resource check enforces a five-minute cooperative limit, 2-GiB peak
RSS and zero process swap residency, including during receipt rechecks. The
launch plan correctly reserves the hard controls for the external one-shot
service: 2 GiB memory, zero swap, 100% CPU quota, 300-second runtime,
`Type=exec`, `KillMode=control-group`, and `Restart=no`. It requires successful
terminal 15-state measurement, exact fixed arguments, absent `analysis-001`,
recorded unit/PID/ExecStart/interpreter/properties/source hashes, and no retry.
The plan accurately states that there is no separate in-process service guard.

## Synthetic verification

Command:

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 python3 -m unittest discover -s tests \
-p 'test_grokking_raw_direction_analysis.py' -v
```

Result: **11/11 fixtures passed in 0.477 seconds**. Inputs were tiny synthetic
JSON and deliberately invalid opaque checkpoint/NPZ bytes. Coverage includes
fixed arithmetic/units/favorable counts, mixed signs and zero mean without an
equivalence inference, exact roster/order, invalid 1501 values, undefined-ratio
semantics, realistic differing scalar/analyzed schemas, raw hash mutation,
analyzed shared-field mutation despite rewritten receipts, checkpoint swap,
failure markers, incomplete source union, and import without NumPy/PyTorch.

## Final-freeze condition

Current `HEAD` `5a83c132c4cb3b9cab9e8da62be8f63186e83c6c` is explicitly a WIP
snapshot, not scientific acceptance. The final analysis fixture and supporting
launch documents are not all in that commit. Main must commit and record the
exact reviewed source/test/protocol snapshot before execution, verify the
measurement is successfully terminal, and inspect the actual service controls.
The analyzer records its four-file analysis source map and rejects any change
during execution; committed-HEAD equality is an external main-agent launch
check, transparently documented rather than falsely claimed as an in-process
guard.

Verdict boundaries: this PASS approves only final freezing of the bounded
JSON analyzer. It does not approve analysis before measurement completion,
source or receipt mismatches, reuse of an existing/partial output, retries,
real-data calculation in this review, or scientific acceptance before the
separate saved-data/first-action audit.
