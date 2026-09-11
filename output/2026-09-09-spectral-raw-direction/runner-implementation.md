# Raw-direction acquisition runner implementation

9 September 2026. Implementation and synthetic fixtures only. **No scientific
checkpoint was loaded, no training/inference acquisition was launched, and no
completed experiment was restarted.** Source/protocol review and main-agent
execution admission remain prerequisites. No commit was made by this leaf.

## Files

- `experiments/grokking_raw_direction.py`: one fixed `raw_norm_matched` branch
  from each original legacy step-1500 state, exactly 1,000 new updates and
  captures at 1501/2000/2500.
- `experiments/run_grokking_raw_direction_batch.py`: all five seeds 100–104,
  sequentially, using seed 100 only for resource/source/receipt admission. No
  scientific outcome gate, retries, replacement seeds or additional policies.
- `tests/test_grokking_raw_direction.py`: synthetic state/Adam/receipt/batch
  fixtures. Tests use tiny synthetic tensors and temporary files only.

The old action runner is never invoked. Its trusted snapshot, scientific-state
hash, Adam decomposition and environment-validation helpers are imported
unchanged; every actual update calls the original `train_step` exactly once.
The complete old source map must equal the hash-bound accepted action-batch
map at runtime. The old parent checkpoint receipt, source metrics hash and
scientific-state hash must also agree. Source pins include all new scientific
code/tests/protocol plus the imported old scientific sources and their pins.

The first-step tensor artifact records full before/after model, optimizer,
estimator and RNG state, raw gradient, numerical Q, four separately named
action tensors and the actual Adam decomposition. Only the new raw action is
delivered; hypothetical native/orthogonal/norm-matched tensors do not imply
extra optimizer updates. Basis retention is disabled after this capture.
Every one of the 1,000 history rows includes branch-local action diagnostics
and actual/adaptive/decay movement norms plus decomposition residuals. Later
full Adam diagnostic arrays are discarded rather than saved. Original
50-step evaluation placement and RNG checks are preserved.

Each seed completion binds every produced JSON, trajectory snapshot,
checkpoint and first-step tensor file through an exact nine-artifact roster;
the batch completion binds each seed completion and its own manifest. Source,
receipt and exact history/capture rosters are checked for admission. A shared
prospective writer enforces 20 GiB total output and a 1 GiB free-space reserve,
using unchanged atomic exclusive-create writers. Cooperative seed limits and
subprocess timeouts enforce 3 hours per seed / 12 hours per batch; the external
16 GiB/no-swap/one-CPU service guard remains main-agent launch responsibility.
Partial output and failures are preserved; when a full disk prevents the
bounded failure receipt, its complete JSON is printed to the raw service log.

## Synthetic fixture result

Command:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -p test_grokking_raw_direction.py -v
```

**9 tests passed in 4.128 seconds** on the final implementation pass before
independent review. They cover bitwise complete-state restoration, unchanged
filter configuration/mutable schema, authentic carried Adam state, exactly
one original synthetic training step, complete first action tensors, basis
release, fixed new-runner commands, no metric gate across the five-seed mocked
batch, no restart on deadline, exclusive output, finite-state/space guards,
complete receipt/history binding and exact archived parent/source admission.

An initial test invocation used module notation although `tests` is not a
Python package; it failed import before running any test. Switching to the
repository's discovery form above resolved that invocation error. This was
not a scientific acquisition failure. The expected timeout fixture prints
synthetic failure JSON and passes; no real job timed out.

Source SHA-256 at this check (not an execution receipt):

```text
runner 0367b536002c7bcb2996e1401027926138488c8a9c65d95fa9f4f99bbae59d31
batch  0cd2f527a3a2c8a7dd22ce9c2b2504b86264ab04496caf921667c7f89f6d8b25
tests  cfbab6b1c1458f19f65c96ce486e3237f1600ba9c35af20abf782333e5c84bab
```

These tests validate implementation plumbing, not numerical reproducibility
of CUDA training or any useful-learning hypothesis. Real input admission,
resource provisioning, acquisition, measurement and independent saved-data
audit have not been performed by this implementation leaf.

Main-agent prelaunch addendum: the two operational guard/fixture source pins
were added after the historical hashes above. Final runnerSHA256 is
`dbf5ad6d33c75f27c3a016499d60fa9a36e503543b946dcd313a707d91804f5a`;
the unchanged batch/runner-test hashes still match. Independent assembled
source review PASS, main all72grokking fixturesPASS in5.427seconds, parent
admissionPASS, and KB lintPASS. No acquisition occurred during these checks.
