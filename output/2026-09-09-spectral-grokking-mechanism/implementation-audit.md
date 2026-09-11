# Independent implementation audit: saved-checkpoint representation probe

9 September 2026 · Pre-acquisition review

## Verdict

**Ready for the fixed six-state calibration stage only**, conditional on recording
the promised launch-time systemd checks (16 GiB `MemoryMax`, zero
`MemorySwapMax`, 30 minute `RuntimeMaxSec`, whole-cgroup termination and no
restart) before the first checkpoint is opened. I found no remaining material
construct, leakage, provenance, mutation, overwrite or roster defect in the
reviewed implementation.

This is not approval to run the remaining 144 states immediately. That stage is
admissible only if the complete, hash-bound calibration bundle passes the four
frozen per-arm construct margins and its observed runtime, maximum memory and
output size remain within the registered bounds. A failed calibration invalidates
or motivates diagnosis of the ruler; it is not a kill criterion for the optimizer
idea.

I did not load a scientific checkpoint, run inference on a trained state, restart
training or inspect any calibration outcome in this audit.

## Reviewed contract and implementation

I reviewed the protocol and design against:

- `experiments/grokking_model.py`
- `experiments/grokking_confirmation.py`
- `experiments/grokking_representation.py`
- `experiments/measure_grokking_representations.py`
- `tests/test_grokking_representation.py`
- `tests/test_grokking_representation_runner.py`
- the committed confirmation summary and all 15 raw metric manifests

The numerical construct now matches the frozen protocol:

- `final_hidden` is hooked at the output of `ln_final`, immediately before
  `unembed`. The control is the input to `ln1`, with both token positions
  concatenated in order. Logits are independently hooked and required to equal
  the model return for every batch.
- Probe rows are only the never-trained pairs. They are sorted by `(a,b)`, split
  once per training seed within each true-sum stratum, and reused for both
  representations and every arm/checkpoint. Validation rejects duplicates,
  overlap, omissions and out-of-range indices.
- Features and targets are centered using fit rows only. One fit-feature RMS
  scalar is used. The fp64 ridge solve is
  `(Z_fit^T Z_fit/n + 0.001 I) B = Z_fit^T(Y_fit-Ybar_fit)/n`; restoring
  `Ybar_fit` gives the unpenalized intercept. R-squared pools cosine and sine,
  and its fit-target mean is also the evaluation reference. Values are not
  clipped. Frequency selection uses fit R-squared only, with smaller frequency
  winning exact ties.
- Each null globally permutes the never-trained **row sums**, rather than a
  113-class codebook. The same 20 seeded permutations are passed explicitly to
  both probes for all states of a seed. The raw NPZ archives the full permutation
  matrix and exact split indices; scalar JSON retains permutation hashes,
  selected frequencies and all 56 fit/evaluation scores for every null.
- Zero-RMS features yield an intercept-only predictor. Non-finite inputs and
  invalid Gram systems fail closed.

The construct fixtures cover hook/forward parity, multiple inference batches,
state/gradient/mode nonmutation, stratified disjoint coverage, genuine row-null
behavior, synthetic Fourier recovery, an independent-label memorization failure
inside its finite null envelope, fit-only feature preprocessing and target
selection, affine scale/offset invariance, finite degenerate outputs, and an
independent fp64 normal-equation calculation.

## Checkpoint provenance, pairing and non-training behavior

The runner pins the confirmation summary hash, then verifies the full 15-run
roster and each raw metrics hash. For every requested state it checks the
checkpoint receipt hash, singly-linked regular-file loading through the existing
`weights_only=True` loader, checkpoint schema and step, full recorded config and
config hash, source identity and hash, regenerated data split identity, parameter
layout, and the complete 50-step evaluation prefix. It also re-hashes the original
model, confirmation harness and filter sources against the state metadata.

The raw manifests contain exactly five seeds, three arms and ten checkpoint
steps (150 states), with one common split per seed across arms. The confirmation
corpus records two repository commits because acquisition crossed the results-only
commit `e509cade`; the scientific model, harness and filter file hashes are common
across all 15 runs. The runner correctly binds each checkpoint to its own complete
record rather than incorrectly requiring one repository commit string.

No optimizer is constructed or stepped, no backward pass exists, and optimizer
or filter state is not restored. A fresh model receives only the recorded model
state and is used under `no_grad`/evaluation mode. The extraction helper removes
its hooks and restores every module's prior mode even on failure; fixtures verify
parameters, buffers and pre-existing gradients remain byte-equal.

Behavioral CE/accuracy is recomputed on the archived logits and compared with the
corresponding checkpoint row before a scalar result is accepted. Checkpoints are
paired by the fixed training seed and verified data split. Checkpoints are repeated
states, not independent replicates.

## Stage, output and resource safety

The stage rosters are exactly six calibration states and 144 disjoint remaining
states. Output must be a new descendant of `/tmp/spectral-experiment-artifacts`; the runner
rejects existing paths, missing parents and paths resolving outside that volume.
Manifest, NPZ, scalar result, completion and failure files use exclusive-create
semantics. Pre-write byte bounds include NPZ overhead and reserve 1 GiB of free
volume space. The initial free-space check requires the registered 10 GiB output
allowance plus 1 GiB reserve.

The remaining stage requires the exact six result receipts, hashes every result
and activation archive, requires identical recipe/source/environment metadata,
and recomputes the frozen calibration gate. It therefore cannot silently
reacquire the six calibration states or proceed on a directory name alone.

Hard RAM, swap and wall-clock enforcement is deliberately external to Python.
The systemd property readback and final unit resource accounting must be retained
with the calibration launch record. The runner's 1,740-second cooperative deadline
is a secondary between-state guard, not a substitute for that cgroup.

## Checks run without scientific acquisition

At the final reviewed source versions:

```text
PYTHONPATH=. python3 -m unittest discover -v -s tests \
  -p 'test_grokking_representation*.py'
12 tests passed

python3 -m py_compile experiments/grokking_representation.py \
  experiments/measure_grokking_representations.py \
  tests/test_grokking_representation.py \
  tests/test_grokking_representation_runner.py
passed
```

A synthetic full-size probe fixture (8,939 rows, 256 features, 56 frequencies,
20 nulls; no trained activation) completed in 2.74 seconds with 722,500 KiB
maximum resident memory and zero swaps. This only checks numerical scale; it is
not a forecast of checkpoint loading/inference cost and is not scientific
evidence.

Reviewed SHA-256 identities:

```text
906ee280f42756351ef1439abf6061f7d2401272dae9006e99eb45ba3636b22f  experiments/grokking_representation.py
a8056c9d45b35cd5f188f6a3e64ecfb6896fef1734498e9d7365500c10046423  experiments/measure_grokking_representations.py
1c97bf056663dadd18b79d14278e93c114f39da6b1363ffe90f1a443d6cce27e  tests/test_grokking_representation.py
f06ffdd3a0ea0794c3a8ee1d4bb44d37decbb2518f465676947756fe953583ac  tests/test_grokking_representation_runner.py
b318c6a5608165e3e8ebef51fdd8c62947ca2ef903514434f3f59f27d5f813d6  output/2026-09-09-spectral-grokking-mechanism/protocol.md
```

Any change to these pinned measurement sources after calibration requires a new,
separately labelled acquisition stage; it must not overwrite or be pooled
silently with the reviewed calibration bundle.

## Interpretation limits retained

The selected-five score is cross-fitted decodability, not evidence that the
network uses the decoded feature or implements a five-frequency Fourier circuit.
The maximum of 20 row permutations is a descriptive finite envelope, not a
multiplicity-corrected significance threshold. Calibration uses one training
seed and only validates the ruler. Later arm/checkpoint trajectories remain
observational evidence from already-diverged runs; temporal ordering alone is
not causal mediation.
