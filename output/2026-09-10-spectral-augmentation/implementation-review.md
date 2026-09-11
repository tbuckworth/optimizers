# Implementation and source checks

Codex — Spectral Optimizer Investigation · 10 September 2026

Prospective checks only at this checkpoint. No augmentation acquisition or
scientific analysis has run. Existing completed experiments are unchanged.

## Current primary documentation

The best-practices skill was used to check the implementation against official
documentation, not to introduce an approval workflow or update dependencies.

- Explicit PCG64 with integer-sequence SeedSequence entropy is supported by
  [NumPy](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html).
  The [compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)
  does not promise identical results after arbitrary version, call-shape or
  call-order changes. Save concrete gates, centers and hashes as specified.
- The fixed square uses the geometry in the authors'
  [Cutout code](https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py).
  Our gate, size, input normalization and MLP differ from their benchmark;
  the protocol says so. Targeted/opposite match each other in area, not Random.
- The installed scientific environment is pinned to PyTorch 2.11.0+cu128,
  so the [2.11 AdamW documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)
  is the relevant API reference. Explicit `foreach=False, fused=False` retains
  the established single-tensor path and avoids relying on CUDA defaults.
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
  supports controlled RNGs and deterministic algorithms, not a guarantee across
  releases, hardware or CPU/GPU. Pin environment, flags and restored state;
  don't promise bitwise cross-platform replication.
- [The allocator fraction API](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
  bounds the caching allocator, not every CUDA/driver allocation reported by
  `nvidia-smi`. The protocol explicitly calls its GPU limit an allocated-memory
  guard; retain separate process/GPU admission checks and resource receipts.

Verdict for these API choices: **consistent with the relevant documentation**.
This is not a result, performance guarantee or certificate that an as-yet
unreviewed acquisition runner enforces every contract.

## Fabricated tests completed so far

Main read both pure helper modules and fixtures. The first test command used
package-style `tests.*` imports, but this repository's tests are not an importable
package; two loader errors occurred before any fixture ran. Corrected to:

```
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -p 'test_spectral_augmentation*.py' -v
```

All **9** then-existing mask/contrast tests passed in 0.308 seconds. Independent
review also ran them with one-CPU affinity and CUDA hidden: PASS. These tests use
fabricated images/scalars only, no MNIST, model, checkpoint, Torch or GPU.
Mask probabilities were independently checked by exact enumeration of all
784 centers; no sampled training outcome was consulted.

The contrast fixture explicitly tests negative I_Q despite absolute native
cue worsening. Missing Clean wrong-label metrics remain undefined rather than
being converted to zeros. Roster duplicates/missing cells, changed endpoints,
inconsistent warmup outcomes and nonfinite scalar metrics fail closed.

The [design review](design-review.md) supports proceeding with implementation.
Its requested I_Q wording correction and full logical-logit byte count are
incorporated into the protocol. Source/runner admission and its fixtures remain
the next implementation checks, not a requirement to redo old scientific work.

## Completed source and fixture review before freezing

The main agent read the complete acquisition runner, independent NumPy auditor
and all four new fixture files. The [independent source review](source-review.md)
passes the runner/masking/contrast contracts. Main resolved the auditor's
manifest field to `batch_size` before execution and checked the explicit
restored-state binding. The audit has a 540-second cooperative deadline so
the outer 600-second hard deadline leaves failure-footer time. Its PASS receipt
is written after the summary, not before. No model tensor is deserialized:
checkpoint bytes are hashed; state/first-gradient hashes remain producer claims
whose pairing is checked, not independently recomputed internal states.

The final combined suite passed **29 tests in 6.348 seconds** with CUDA hidden,
single-thread math and fixture scratch under `/tmp/spectral-experiment-artifacts`. This includes
a complete 315-artifact/72-trajectory/1,512-evaluation **fabricated** archive.
No training IDX, actual model, scientific archive or GPU was used in these tests.
The auditor independently rebuilds raw metrics using float64 NumPy log-sum-exp;
the contrast function is shared and independently source/fixture-reviewed.

Final acquisition source SHA256:
`7a4f6aab337551caea54d53333f48284e82a72702319238feffe234919333e2a`.
Final auditor source SHA256:
`8f2bc33c22956177ce5223c1c20e6edb7cff5107385b8ab2e30c0023075647a9`.
The acquisition will require all new scientific sources/tests plus accepted
base/core/filter/protocol bytes to match the current commit. Neither code nor
protocol will be edited during acquisition. Plotting and HTML delivery scripts
are separate consumers; they cannot launch or restart science.

The exact conservative artifact inventory is **1,743,581,264 bytes**, including
907,200,000 bytes for all logical logit histories. Headroom below 2 GiB is
403,902,384 bytes, with a separate 1 MiB failure reserve. GPU/host/time/capped
write guards are implemented and fixture-checked; actual admission is still
checked by the once-only execution, not inferred from these fixtures.
