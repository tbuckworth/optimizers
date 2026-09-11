# Native layout inspection: failed, no retry

Codex / Spectral Optimizer Investigation, 7 September 2026.

The single approved engineering observation failed. No native source/environment
document was retained, so the proposed paired metadata size comparison was not
run and no new native RNG length or storage-fit claim is available. This is an
engineering failure, not evidence for or against the optimizer hypothesis.

## What happened

The source freeze was e806973c88b3e51e472b0a42a15ac2798be44031. Parent verified
a clean worktree, the exact committed helper, the documented mount,23,596 MiB
free on the exact local RTX3090, and only the two known unrelated compute apps.
Inspection-only GO preceded this exact command:

```text
PYTHONDONTWRITEBYTECODE=1 python3 output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/native_layout_inspection.py --inspect-native-layout --expected-commit e806973c88b3e51e472b0a42a15ac2798be44031 --expected-gpu-uuid GPU-994b97cd-d768-1f03-78fb-6a70e3cc1e0c
```

Tool handle56037 terminated with exit1. Controller PID989661 and worker PID989705
are terminal. The marker was published at00:11:11.343800204 UTC and failure file
at00:11:14.073744164 UTC; these are filesystem publication times, not a complete
measured CPU/wall resource record. No retry or alternate directory was used.

The fixed root is
`/tmp/spectral-experiment-artifacts/spectral-i7-native-layout-inspection-001`.
It contains exactly the676-byte marker and594-byte failure file:1,270 logical
bytes. Both remain there and have byte-identical repository copies:

- Marker (artifact not distributed in this public snapshot), SHA256
  88a825ca918a04d3417c319af664184995c8a391ffcb75589c0a8dccb069d83e.
- Failure (artifact not distributed in this public snapshot), SHA256
  87ed567200f25761eeee13ff02ccd2c126a8cd52363961cc94fd2d50534260a9.

The failure record confirms the worker was absent from the GPU after exit.
Read-only process/device checks agree; unrelated desktop applications were not
changed. The singleton now exists and is consumed: absence of a result does
not authorize a repeat. Initializing CUDA was authorized, but the retained
failure does not identify whether initialization succeeded or the precise
stage reached. No successful runtime/peak-memory observation is claimed.

## Diagnostic gap and evidence-backed candidate

I missed a diagnostic defect in review. The controller raises its generic
nonzero-exit error before decoding the worker's bounded structured failure.
Consequently the detailed worker exception was discarded, not saved in another
log. The retained cause is only `native inspection worker exited unsuccessfully`.
It cannot establish the exact historical failure stage or reason.

Hypothesis-first follow-up stayed CPU-only. A clean-source prefix check passes
explicit Torch configuration and all43 scientific source bindings with CUDA
hidden/uninitialized at the same commit. It observes Torch2.11.0+cu128,
NumPy1.26.4 and cuDNN91900. This rules out a reproducible failure in that checked
CPU prefix under the diagnostic process; it does not replay the native worker
or prove every original prefix/environment detail was identical.

A concrete source/API mismatch is present: source_environment._uuid accepts
only strings or bytes-like values. The pinned PyTorch C++ implementation binds
device.uuid to its _CUuuid class and supplies string conversion for that class.
CPU introspection confirms the installed class exists and is not a subclass
of those accepted types. Its local type stub instead annotates uuid as str;
that annotation is not the runtime binding. See the exact
[pinned PyTorch implementation](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/csrc/cuda/Module.cpp#L979).

A synthetic string-convertible UUID protocol double is rejected by the current
collector with `CUDA device has no stable UUID`, without CUDA initialization.
This is a confirmed code/API incompatibility and a plausible proximate cause,
not a recovered native exception or an actual device-UUID observation. The
missing worker detail prevents conclusive attribution of this attempt.

No source/environment validator, production optimizer, frozen helper, tolerance
or gate was changed after the failed observation. The compatibility and failure-
reporting fixes remain future CPU-only engineering work with their own review.
Nothing in this report authorizes another native inspection or a scientific run.

## What remains known

The preceding current-envelope component projection remains945,341,202 bytes
(901.547625 MiB); it is incomplete and is not superseded. The new pure-schema
receipt/phase-record bound (artifact not distributed in this public snapshot) accounts for a
separate13,684,358-byte scoped subtotal, including reserved failure capacity.
Neither subtotal proves the unchanged1-GiB complete-attempt cap feasible.
Native RNG layout, repeated real provenance overhead, CPU-auditor metadata,
remaining native fields, whole-root writer behavior and all-phase timing/peaks
remain unresolved. The offline comparator is implemented/tested but unmeasured.

Pre-execution verification passes338 I7 tests,17 repository tests and six
reminder tests. Four focused comparator tests pass after final label edits.
These CPU tests did not expose the native UUID mismatch and were never a native
execution certificate. Full verification order and handles are retained in
native-layout-inspection-checks.json. No optimizer-science conclusion changed.

Next safe work: preserve bounded worker failure/stage diagnostics and add a
pinned-type CPU regression/strict UUID normalization proposal. Reassess native
inspection authority only in a separate explicit decision; do not bypass the
consumed singleton. The research goal and original two-hour reminder stay active.
