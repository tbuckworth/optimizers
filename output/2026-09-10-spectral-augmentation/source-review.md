# Augmentation implementation: source review

Codex — Spectral Optimizer Investigation · 10 September 2026

## Scope and current disposition

**PASS — no material implementation issue found.** Reviewed the mask helper, scalar-analysis helper, complete 458-line acquisition runner, their fixtures, and the exact reused base/core functions. This is prospective implementation review, not experiment execution or revalidation of prior scientific results. The fixed protocol, as corrected after design review, is the comparison anchor. No additional arm, research gate, or user-approval step is requested.

Read both new helper modules and their fabricated tests completely. Independently ran their nine tests with one-CPU affinity, single-thread math settings and `CUDA_VISIBLE_DEVICES=''`: **9 PASS in 0.293 seconds**. These tests use only fabricated NumPy images and scalar records; no MNIST, Torch model, scientific array, training state, inference, or experiment was loaded or executed. The main agent reported two earlier test-loader errors from treating `tests` as a package; those involved no scientific work, and the reviewed invocation uses unittest discovery.

## Checks completed

- Mask RNG is separate from the global RNG and from data/cell streams. Gates and centers are indexed by batch occurrence, generated without labels or model outputs, and reused across modes. The source applies masks to copies, preserves floating dtype, handles duplicate images independently, includes uncued images, and never inserts a cue itself. Integration must insert the cue first.
- Half-open clipped random rectangles, inactive sentinel rows, fixed targeted/opposite locations, and geometric coverage counts agree. Fabricated all-one images directly verify the 25 full-coverage and 49 overlap counts among 784 centers, and the 43,264 total erased pixels. Counts are explicitly geometric, not changes in originally nonzero pixels.
- The scalar analyzer requires the exact 72-member seed/cell/policy/mode roster, rejects duplicates and incomplete rosters, joins by seed rather than input order, and fixes endpoint/warmup steps to 2,000/100. Warmup metrics must agree within each cell's forks.
- Original CE and accuracy values retain their direction; no composite outcome is created. Absolute endpoints, changes from warmup, mode-minus-none changes, native-minus-raw contrasts, optimizer interactions, targeted-minus-opposite contrasts, and association contrasts are separate.
- The cue difference-in-differences sign is correct, including a fabricated case where relative improvement accompanies absolute harm. The output explicitly warns that negative `I_Q` alone does not establish native cue reduction.
- Clean's absent actually-changed-label metrics remain `None`, not zero. Undefined-versus-defined mismatches in contrasts are rejected. Scalar inputs must be finite non-boolean numbers; summaries retain all three seed values, means, extrema, and sign counts, not pseudo-replication or significance claims.
- These helpers import neither the old training module nor a model, and do not monkeypatch any accepted experiment. Receipt/logit verification is explicitly outside the inert scalar analyzer's contract, not falsely claimed by its output.

## Inspected helper hashes

| File | SHA256 |
| --- | --- |
| `experiments/spectral_augmentation_masks.py` | `3a17d98db89754e118b52722392fc3108d4de330211a0fe25e5b1f4471166f0f` |
| `tests/test_spectral_augmentation_masks.py` | `0ed9202dc225cf60e33cd11a14f027187402c2a434396068d1bd319defceb83b` |
| `experiments/spectral_augmentation_analysis.py` | `b9ac077d0d04c30046a6aeb02b1426b172e00d4fc2b90783fa401f1c9fab95f6` |
| `tests/test_spectral_augmentation_analysis.py` | `bbf8f340bb596b515073d2c05f58d1dba9a6f758247c14218acd4f8ebb1e7b1b` |
| `output/2026-09-10-spectral-augmentation/protocol.md` | `4eaeb66a4d87ad0f59f5164449deff3d3dee6e81df659f6b757d8a60b5c091be` |

The main agent separately owns official-library documentation validation. The remaining sections record the completed runner review. Source-review PASS is not an acquisition or launch certificate.

## Complete runner and integration review

Read `experiments/spectral_augmentation_boundary.py` completely, alongside the reused `training_update`, `evaluation_row`, `predict`, `validate_bounds`, model/optimizer/tracker constructors, snapshot, digest, and restore implementations. Read all runner fixtures before executing them.

- **Full fork identity and order.** The exact 72-member roster is constructed prospectively, with a fixed alternation of policy-first position. Each seed uses one clean, unmasked, common-only warmup. Every branch restores the same full model/Adam/observer/RNG snapshot and checks its digest before raw delivery discards the unused observer. No state clock is silently reset or coerced. The first raw-gradient hash is required to agree across policies for every cell/mode pair.
- **No old-source mutation.** The new runner imports accepted pure/scientific functions without changing their constants, callbacks, or policy semantics. Its wrappers do not monkeypatch the old experiment. Temporary test mocks intercept updates/restores only inside fabricated fixtures; no real optimizer or observer update is executed there. Read-only git comparison found no change to the accepted base, canonical filter, or core files.
- **Actual occurrence inputs.** CPU NumPy FP32 normalization and canonical cue insertion precede the saved occurrence rectangle erasure and CUDA transfer. Assigned targets are indexed by the same batch positions. Every mode includes rare and uncued images; repeated examples can receive different masks. Source arrays and mask/data plans are checked for mutation. Cell pixels, assigned targets, patch masks, batch arrays and mask arrays are separately hash-bound.
- **Evaluation semantics.** The held-out inputs are fixed unpatched and visible-white-patched images, without training erasure. Training evaluations use fixed cell inputs without erasure, with both true and assigned targets. Initial and warmup predictions are reused only for their identical states and cell-specific inputs; state checks confirm these evaluations do not alter the snapshot. The runner calls the accepted evaluation function whose exact schema matches the new scalar analyzer, including the empty Clean wrong-label group.
- **Whole roster and outputs.** Every continuation has exactly 1,900 action rows and evaluation steps 0, 100, …, 2,000. The final ordered results must equal the prescribed roster. Initial/warmup/end states, all logit histories, action logs, per-cell counts, occurrence coverage and pairing receipts are retained; no selected endpoint or success-dependent continuation exists. The runner does not execute the old summary/analysis entry point. New scalar analysis is a separate consumer of the raw evaluation rows.
- **Provenance.** The runner requires the accepted base SHA, canonical filter/core pins and all directly used new sources/tests/protocol. Each source must match bytes in the current git commit before execution. Both accepted IDX hashes are checked before any model update; source/data hashes are reread after acquisition. Artifact writes are exclusive direct children with size/SHA receipts. Completion enumerates prior receipts and the complete result binding.
- **Once-only and resources.** `--execute` is required and no seed/step/batch/resume overrides exist. Admission requires an empty, resolved parent on the exact large-volume mount, a new `acquisition-001`, sufficient disk, and an exclusive cross-directory attempt receipt. The fixed systemd unit must have the specified cgroup limits, 30-minute hard timeout, no restart, and control-group kill semantics. GPU admission rejects non-allowlisted clients; the main agent still owns the fresh live-client/handle check. Cooperative deadline, host peak, GPU allocation and free-space checks run around updates/writes. Failures preserve partial artifacts and emit a failure receipt rather than retrying.

## Bounded inventory and fabricated tests

Independently summing the source's dimension/byte arithmetic gives **1,743,581,264 bytes**, leaving **403,902,384 bytes** below the 2-GiB cap. This includes 907,200,000 bytes of full logical logit histories, an overestimate for all 78 initial/warmup/final snapshots, shared logits, plans, JSON allowances and ZIP overhead. Each action row is explicitly limited to the inventoried 1,024 bytes. A separate 1-MiB failure reserve is enforced. This is a conservative prospective inventory, not measured output size or a promised runtime. The capped writer enforces the actual total regardless of the estimate.

The full fabricated suite passed independently: **19 tests in 0.324 seconds**, with one-CPU affinity, one-thread math settings, and CUDA hidden. It checks roster/inventory, refusal without execution, absence of overrides, actual cue-then-erasure pixels on tiny fabricated arrays, unchanged targets, mocked call order with diagnostics forbidden, repeated-occurrence denominators and empty cued populations, full-state digest mismatch rejection, capped/short writes, an in-memory NumPy archive round-trip, fake GPU/cgroup admission, and the cooperative deadline. No model was constructed; no inference, backward pass, Adam step, covariance update, scientific file, or acquisition was executed.

Command used:

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -c 'import os,unittest; os.sched_setaffinity(0,{min(os.sched_getaffinity(0))}); suite=unittest.TestLoader().discover("tests",pattern="test_spectral_augmentation_*.py"); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

## Final additional hashes and theoretical read-back

| File | SHA256 |
| --- | --- |
| `experiments/spectral_augmentation_boundary.py` | `7a4f6aab337551caea54d53333f48284e82a72702319238feffe234919333e2a` |
| `tests/test_spectral_augmentation_boundary.py` | `ff8b4b93547e9a8d8de06321c4b1be33fb1f4c7a3f382caa1bf648920ed7f96f` |
| Final reviewed `protocol.md` | `4aa4e7d9d5ba8ad718648980010412ed2f26e3e28abd8301ae9ac60603de4e9c` |

The final protocol adds a correct favorable toy calculation: cue occurrence probability 0.1 becomes 0.05 under half erasure, reducing `q(1−q)` from 0.09 to 0.0475. Within-example augmentation variance can increase while the between-example term decreases enough to reduce total variance. This is appropriately conditioned on an additive constant cue-gradient model and does not predict the actual evolving neural covariance. It changes no execution choice.

Main-agent source freeze, fresh resource/once-only admission, and the separately prepared saved-data checker remain ordinary implementation/execution responsibilities. This review does not execute them or certify unreviewed subsequent source changes.
