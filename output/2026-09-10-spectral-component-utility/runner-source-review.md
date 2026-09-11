# Component-utility producer: independent source review

10 September 2026 — **PASS for the reviewed producer source and fabricated
fixtures. No unresolved producer defect found.** This is not acquisition
admission, a real-input verification, or a separate acceptance of the guard
and saved-array auditor, which have their own review tracks.

## Scope and resolved finding

Read the entire [producer](../../experiments/spectral_component_utility.py), its
[tests](../../tests/test_spectral_component_utility.py), [protocol](protocol.md),
archive contract (artifact not distributed in this public snapshot), [runner preparation](runner-preparation.md)
and parent inventory metadata (artifact not distributed in this public snapshot). Rechecked the relevant
accepted helper APIs, original producer/readout schema, transformations and
guard interface. No real data, checkpoints, old archive arrays, model states,
GPU calls, scientific measurements, attempts or units were consumed/created.
Only this review note was authored by the reviewer.

**Low severity, resolved — a nonregular input could block before validation.**
`experiments/spectral_component_utility.py:88` initially opened inputs with
`O_RDONLY|O_NOFOLLOW`. A FIFO at the named path could block in `open` before
the regular-file check ran, wasting the bounded unit rather than rejecting the
input. Main added `O_NONBLOCK`, matching the accepted restore loader's behavior.
The final fabricated FIFO rejection fixture at
`tests/test_spectral_component_utility.py:116` passes. The corrected source was
reread. No scientific arithmetic changed.

Main also added NPY-header preflight while this review was underway. The final
reader checks declared shape/dtype/payload length before NumPy allocation,
in addition to ZIP member counts, expanded byte limits, object rejection and
finite values. Its fabricated billion-element header is rejected without
allocating that array. This is included in the reviewed version, not an
unresolved request.

## Scientific and archive correspondence

- **Fixed inputs and comparison:** all twelve native parents remain in the
  registered seed → none/translate → h100/final56304 order. All three seed
  panels are fixed before loading a parent. The accepted stream-40/41 selector
  is used without balancing or outcome-dependent choices. Image-major grids
  enumerate the 25 `(dy,dx)` shifts, including original index 12. Both action
  batches are translated even at none-trained parents; raw pixels are
  translated before the original FP32 normalization.
- **Baseline and derivatives:** each parent is restored privately with the
  strict strong-schema adapter and exact state digest checks. The first 500
  reporting predictions use the original 500-example chunk convention. Actual
  and expected logits are saved before an equality failure, and no action is
  reached on mismatch. Baseline S/F/C/direct-L/H_O/H_T gradients use the accepted
  differentiable objectives; scalar and gradient-sum tolerances are enforced.
  Only detached CPU arrays/gradients and scalar metrics escape
  `baseline_gradients` (line 268), releasing all local forward-graph references
  before private actions. Saved parent `.grad` buffers are not accumulated into.
- **Actions and readouts:** each draw computes one FP32 mean assigned-label CE
  gradient, supplied to both accepted private raw/native actions. No optimizer
  reset, second native observation, step normalization or training continuation
  is introduced. Every raw/native/decay endpoint at full and tenth fraction is
  evaluated. Materialized FP32 points, FP64 actual deltas, same-fraction decay
  accounting, all six signed utilities and all original/per-view/wrong-subset
  metrics match the contract. State digest equality is checked again afterward.
- **Reductions:** finite effects are baseline minus endpoint; native-minus-raw
  contrasts subtract those effects. Finite data effects compare to the matching
  decay endpoint. Both draws are averaged within each seed/parent using
  `math.fsum/2`. All eight mode/stage/fraction cells remain visible. The primary
  summary is exactly translated/final/full, all three seeds, H_O and C—no
  alternative state, favorable fraction or pooled twelve-parent statistic.
- **Inventory:** each parent writes 31 receipted artifacts: three baseline
  artifacts, four raw/native action artifacts, twelve paths and twelve endpoint
  logit sets. Twelve parents plus three panel files and provenance give 376
  receipts before `results.json`, whose payload lists exactly those receipts.
  The implemented upper bound is 6,413,406,496 bytes plus the 1,048,576-byte
  failure reserve, below the 8,589,934,592-byte cap. This is a prospective byte
  bound, not measured full-rank runtime or memory.

## Binding, integrity and resource lifetime

`verify_manifest` (line 148) requires the exact old/new source, test and document
set; verifies working-file hashes and corresponding bytes at the stated full
Git commit; and binds the inventory's own hash. Old scientific source hashes
must remain unchanged. `verify_input_metadata` (line 167) connects the inventory
to the original complete results and matching PASS audit, including archive,
source/data pins, native checkpoint/readout receipts, fixed roster and plan
receipts. Plan arrays/header hashes, labels against the pinned IDX labels,
fixed assignment law and counts are checked before panel selection.

Execution requires explicit manifest arguments, an empty real output directory
under the exact large-volume attempt naming convention, verified mount/free
space, a systemd invocation identity and exclusive `attempt.json` creation.
The producer calls the separate guard before scientific loading and records its
returned runtime identity/settings in provenance. Failure is retained through
the guard rather than retried. The final source/data hashes are rechecked.

One parent is scoped to one `acquire_parent` call. Updated private models are
discarded; post-bases/moments are archived and dropped before endpoint evaluation.
Only the three endpoint vectors survive that phase. Endpoint models are private
copies evaluated under `no_grad`. The full results list retains scalar metadata
and receipts, not twelve parents' tensors/bases. The current guard API matches
the producer's `Run`, `check`, `save`, `configure` and failure-footer calls;
complete guard/cgroup enforcement remains separately reviewed.

## Verification receipt and limits

Independent final command, exit 0:

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 60s python3 -m unittest discover -s tests -p 'test_spectral_component_utility.py' -v
Ran 16 tests in 1.212s
OK
```

The tests include a fabricated linear-model derivative check and mocked
`acquire_parent` success with 31 artifacts, two pair calls, four actions,
twelve endpoints and four contrasts, followed by saved baseline-mismatch
failure before actions. Source reads/restoration are mocked in that integration
fixture; it does not establish real checkpoint compatibility. The final added
fixture passes actual canonical fabricated CPU raw/native actions at h100 and
final56304 through the independent NumPy action checker, and both full/tenth
paths through its path checker. The synthetic tracker starts at rank two; no
scientific parent is read. Baseline metrics also agree with the independent
NumPy reconstruction. This checks the helper/auditor interface, not the full
archive auditor or GPU fidelity. The producer source is unchanged. Earlier
14-test and 15-test runs passed (the latter in 0.272s). `git diff --check` passed.

Reviewed SHA-256 values:

```text
experiments/spectral_component_utility.py
  9078630f867257d716027c670d6382119ed49b7f807073c8c107e4dc57534ffe
tests/test_spectral_component_utility.py
  1c05e9a869289de7afb3ba1e9f7f5aaf03700d392427fe878d81b7f5927cb635
experiments/spectral_component_utility_audit.py (integration-fixture dependency only)
  9e938a3fd2720a95124a422a18a6470f114d0b7a0b64d641c7f6567dc35cf897
protocol.md
  f8f5ff18670cba0ef755459b774a1613a135c5de0b423c8b5c6d7bb6aa1b1efc
archive-contract.md
  d4de9f601b6a7223b4b7b931714815cd4d7eb9223bee38f20a1d702be3e8d4d5
runner-preparation.md
  a7bd9d5a97e529df4fcb0535178400a83be8c30e6c65259140acf15bed7811ed
parent-inventory.json
  0052af5894abfce52e8ba6d60fec472ae6350bbe6a85bc6c2501a55981c23297
```

Review-changes guidance shaped the severity/line-specific finding; the
best-practices check used versioned primary documentation. NumPy's
[1.26 load contract](https://numpy.org/doc/1.26/reference/generated/numpy.load.html)
supports explicit pickle rejection and context-managed NPZ closure; header and
resource checks remain additional application responsibilities. PyTorch's
[autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html)
and [copy_](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.copy_.html)
contracts support non-accumulating derivatives and copying the recorded CPU
point into the private model without requiring a same-device source.

Not established here: the final committed launch manifest, independent guard
and auditor acceptance, fresh hardware/cgroup admission, real parent hashes
and baseline equality, GPU numerical fidelity, full-rank memory/runtime, or
successful complete acquisition/audit. Those remain separate pre-execution
and terminal checks; this PASS does not relax them or authorize a launch.
