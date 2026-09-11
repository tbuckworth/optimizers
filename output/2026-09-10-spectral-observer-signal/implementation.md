# Saved-vector signal accounting: source readiness

Codex — Spectral Optimizer Investigation · 10 September 2026.

## Source and fabricated checks

- Analyzer (artifact not distributed in this public snapshot):
  `7bad0a6bdd73cd7c978c47988dfc5b9705bfa9a0cd2a7e0074617138de80eba8`.
- Fixtures (artifact not distributed in this public snapshot):
  `ad9dc0e94cdf4f9a636c9a18ea76369b3c50a17feb76e7ef6bf41e54896a1c16`.
- [Protocol](protocol.md):
  `2f1992198db3e5c94321ccdd2ec6415369f503887a785c20d1c505e1cde61ec6`.

All **13 fabricated tests pass**, latest unittest runtime 0.122 seconds.
Initial fixtures also passed; main then identified that the original
`2^24−1` cast fixture was exactly FP32 representable. The final fixture uses
`2^25−1=33554431`, which distinguishes FP64-before-subtraction from incorrect
FP32-first subtraction. The producer formula itself was already FP64-first.
Main's entry-clock clarification is incorporated before NumPy import.

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -p test_spectral_observer_signal.py -v
```

Coverage includes positive/negative/zero products and undefined zero cosines;
compensated versus ordinary reduction; near-cancellation and the constituent-
sum identity bound; deliberate numerical failure with preserved evidence;
raw/zero identities; both probes and all action contrasts; authoritative
independent J versus a deliberately wrong producer copy; CE sign/group joins;
all-seed summaries and missing/duplicate rosters; wrong hash/size, symlink and
escaping receipts; exclusive capped JSON/failure output; and inert imports.
The loader fixture supplies a stub Torch module with realistic saved archive
keys and inaccessible incidental members. It does **not** import real Torch,
open even a fabricated `.pt`, instantiate a model or execute an optimizer.

## Exact inputs and extraction boundary

The source fixes the already accepted acquisition, completion, manifest and
audit hashes. It checks the full PASS/zero-error/33,032-check audit, original
full counts, matching completion/source/receipt maps and 122 original artifact
receipts. The only tensor extraction roster is:

- three `oracles-sSEED.pt`: `groups.majority.mean_gradient` and
  `groups.rare8.mean_gradient`;
- six `common-action-sSEED-CELL.pt`: `g_star`;
- 12 `observer-sSEED-CELL-SCHEDULE.pt`: `post_action`, the accepted O151 action.

These are exactly **21 archives / 189,384,513 recorded bytes**, confirmed from
accepted receipt metadata only. Every selected path must be a contained regular
file matching size and SHA before loading and at completion. Each archive is
loaded exactly once, in fixed seed/cell/schedule order. Parent identity and
common-gradient I9 hash bind probes, histories and accepted readouts. The same
zero-readout receipt record must be shared across both cells for each seed.

Only explicit admitted execution imports Torch, inside `restricted_vectors`.
The loader requires Torch 2.11.0+cu128, CUDA hidden, and uses
`torch.load(map_location='cpu', weights_only=True, mmap=True)`. Selected members
must be finite CPU FP32 vectors of length 50,890; copied NumPy arrays survive
after the archive object is released. Mmap avoids intentionally materializing
incidental image/model/optimizer/observer tensors. Their contents are not
inspected. No model, gradient, SVD, observer recurrence or Adam step is invoked;
the old audit is never imported or rerun.

## Frozen arithmetic and control semantics

All selected vectors are converted to FP64 before subtraction. The reported
dot is `math.fsum(float(x_i)*float(y_i))`; an independent `np.dot` reduction
must differ by no more than

```text
T_dot = 128*eps64*sum(abs(x_i*y_i)) + 1e-12.
```

For `q·(x−y)`, compare the direct difference-vector product with the difference
of the two separately compensated products. The frozen comparison is

```text
T_pair = 128*eps64*(S_direct + S_left + S_right) + 3e-12,
```

where each S is its corresponding absolute-product sum. These are operational
consistency thresholds, not general worst-case reduction theorems or effect-
size success gates. Failed checks raise with their numeric evidence retained
in the failure footer. No tolerance adjustment or automatic retry exists.

Dot values, ordinary reductions, discrepancies, thresholds, both vector norms
and signed cosines are retained. The raw sign is always reported. A directional
sign is marked resolved only outside `±T_dot`, or outside `±T_pair` for a
difference product; otherwise it is `roundoff_unresolved`. Exact computed
zero and zero-vector controls retain numeric zero, raw sign `zero` and
undefined cosine where appropriate. No unresolved sign supplies support for
a directional hypothesis. No division by input alignment or conversion-
efficiency statistic is calculated.

All actions include explicit identities `F_raw=B`, `F_zero=0`, `K_raw=0`,
`K_zero=−B`, checked exactly. Existing J comes from the audit's top-level
`checked_readouts`, joined by exact physical ID, **not** its nested producer
report copy (which can differ at the last bit). U comes from `checked_cases`
action improvements: `rare_ce` or `majority_macro_ce`. Actual Adam vectors are
not reopened or recomputed. J total/decay/adaptive, movement and native probe
accessibility are identified as accepted joins. CE utility is before-minus-
after; accuracy utility is after-minus-before. Training/held-out/probe
populations remain explicitly distinct.

## Stable output schema

Schema: `spectral_observer_signal_v1`.

Success writes nine JSON files: `manifest.json`, six
`case-sSEED-CELL.json`, `result.json`, `complete.json`. A failure instead writes
`failed.json` and preserves any earlier artifacts. No tensor copies, plots,
models or dense parameter-squared arrays are produced.

Each case has:

```text
seed, cell, vector_bindings
probes[majority|rare8]:
  B: alignment record
  actions[native_interleaved|native_grouped|raw|zero]:
    F, K: alignment records
    accepted_join: J{total,decay,adaptive}, E_over_zero,
      U_heldout_ce, U_train_true_ce, heldout_accuracy_change,
      train_true_accuracy_change, U_train_assigned_ce, U_train_wrong_ce
    accepted_probe_accessibility, accepted_movement,
    physical_id, accepted_state_receipt, accepted_logit_receipt,
    control_identity
contrasts[12]:
  seed, cell, probe, contrast, left, right, primary,
  D_filter: alignment record, D_Adam{total,decay,adaptive},
  D_U_heldout_ce, D_U_train_true_ce, D_heldout_accuracy
```

`result.json` contains these six cases, six `label_interventions`, complete
`summary`, exact counts, units/scope text and source/input/audit bindings.
The summary has four `input` panels, 16 `actions` panels, 24 `contrasts`
panels and two `label_intervention` panels. Every panel retains seed order
202609121/122/123, values, mean, sample SD/SE and raw signed counts. New
alignment panels additionally retain resolved/ambiguous sign counts. Undefined
cosines and absent Clean wrong-label metrics remain null; incomplete triplets
do not receive a pooled mean or sign count.

Counts are six cases, 12 B, 24 native F, 24 native K, 48 logical action/probe
joins, 72 all-control contrasts (12 primary grouping rows), six label-input
contrasts, 21 accepted physical / 24 accepted logical readouts and 21 loaded
archives. These are not additional independent replications.

## Later command and enforced limits

After main creates an unused mounted large-volume parent and freezes the
seven-entry source map (new analyzer, fixtures, protocol, this note, and the
three old decision/design/review files), the exact CLI shape is:

```text
/usr/bin/python3 scripts/analyze_spectral_observer_signal.py \
  --execute \
  --output-dir /tmp/spectral-experiment-artifacts/MAIN_NEW_EXCLUSIVE_PARENT/analysis-001 \
  --expected-sources-json output/2026-09-10-spectral-observer-signal/expected-sources.json
```

`MAIN_NEW_EXCLUSIVE_PARENT` is an explicit placeholder for main's newly
allocated directory; it does not currently exist by this agent's action.
Main must run that command inside `spectral-observer-signal-001.service`
with Type=exec, Restart=no, KillMode=control-group, CPUQuota=100%, MemoryMax=
4 GiB, MemorySwapMax=0 and RuntimeMax=180s. Source verifies these actual
service/cgroup values, CUDA-hidden and all four math-thread environment
variables equal to 1, NumPy 1.26.4, and the `/private-artifacts/storage` `/dev/RECONFIGURE_FOR_LOCAL_STORAGE` mount.
No-execute invocation stops before admission or scientific input access.

The 150-second cooperative clock starts before NumPy import and includes
admission, hashing, loading, arithmetic and output. The hard service clock
covers the full process. Source/test/protocol/note pins must match committed
bytes and are rechecked before success. Output is capped at 100 MiB with a
separate 1 MiB footer allowance inside that cap; admission requires the full
cap plus 1 GiB free disk, and subsequent checks preserve at least 1 GiB.
Completion/failure records preserve actual resource admission, elapsed time,
RSS, input receipts and source pins. No in-process retry or resume path exists.

The research-workflow guidance informed reproducibility and claim boundaries;
the user's autonomous, saved-evidence/no-rerun scope overrides its usual
fresh-replication and approval gates. This accounting can locate a local
alignment mismatch, not identify semantic selectivity or long-run mediation.
