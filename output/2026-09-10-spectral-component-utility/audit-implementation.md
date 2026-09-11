# Component-utility saved-array auditor

The independent auditor is ready for main's final source review and freeze. This is an implementation handoff, **not a scientific audit result or acquisition admission**. Only fabricated CPU arrays and temporary fixtures were used; no real checkpoint, scientific array, IDX input, GPU, model, or optimizer was loaded or executed.

## Scope and interface

`experiments/spectral_component_utility_audit.py` imports only the standard library and NumPy. It does not import producer, action, objective, restoration, or observer calculations. Import and default CLI are inert. Execution requires every bound argument:

```text
--execute
--input-dir ABSOLUTE_ACQUISITION_DIRECTORY
--output ABSOLUTE_NEW_JSON_OUTSIDE_ACQUISITION
--expected-results-sha256 SHA256
--expected-sources ABSOLUTE_MANIFEST_JSON
--expected-sources-sha256 SHA256
--attempt ABSOLUTE_ATTEMPT_JSON
--expected-attempt-sha256 SHA256
--inventory ABSOLUTE_PARENT_INVENTORY_JSON
--expected-inventory-sha256 SHA256
```

There are no CLI scientific dimensions, seed, threshold, or roster overrides. Internal small layouts exist only for fabricated parent-archive fixtures. Production fixes 12 parents, 24 pairs, 48 raw/native action records, 144 endpoints, 12 baseline readouts, three panels, and exactly 376 receipted artifacts plus final `results.json`.

The PASS output has `independent_summary` matching the producer's `primary`/`cells` schema, `checked_parents` with recalculated metrics/effects/contrasts, exact `counts`, `checks`, empty `errors`, input/source/attempt hashes, acquisition-relative receipts, and descriptive array residuals. The primary remains the three translated final parents at full displacement, restricted to H_O and C. Two draws are averaged within each seed; no seed pooling, sign gate, selector, or inferential claim is added.

## Independent checks

- Exact artifact roster, direct-child paths, byte sizes/hashes, source manifest and committed/current source bytes, original PASS/result/branch/receipt bindings, attempt identity, acquisition guards and byte inventory.
- Original pinned plan roles and assignment law; streams 40/41 panel reconstruction; role labels, IDs, shifts, actually-wrong masks and source plan header hashes/counts. Accepted original audit/data-pin binding is reused. The auditor does not freshly read IDX inputs or repeat the old training-plan analysis.
- Original first-500 reporting baseline equality; parent immutability digests; exact shared parameter vectors, inherited moments and clocks across the private actions, and identical current gradients within raw/native pairs. Raw observer clocks stay unchanged; native advances once. Adam advances once in both.
- Independent post-basis Q(Qᵀg), with an absent-basis identity bypass, and canonical AdamW array equations from the saved moments/gradient/counter. No covariance update, eigendecomposition, observer history, or optimizer execution is replayed.
- Exact rounded FP32 decay and full/tenth materialized paths; FP64 differences of stored endpoints, norms and compensated signed gradient–displacement products. The full endpoint is not reconstructed through a rounding-prone subtraction/addition.
- FP64 per-view class-0 gauge, S/F/C and independently evaluated direct assigned-view L, mean-logit true/uniform terms, original/per-view H_O/H_T and prediction statistics. Empty wrong subsets retain null mean scores and zero sufficient statistics. Every finite/linear/decay/data effect, contrast and registered summary is recalculated.

Saved arrays cannot independently establish that the supplied neural logits or six parameter gradients were computed correctly. The differentiated component closure and hash-bound producer/restoration checks constrain this boundary; they are not model/autograd replay or independent identification of a causal mechanism.

## Frozen numerical and resource rules

Scalar metrics/dots use absolute 1e-10 plus relative 1e-10 tolerance. Adam/projection comparisons retain **elementwise** absolute 1e-7 plus relative 5e-5; L2 and maximum residuals are additionally reported, not alternative acceptance rules. The gradient-sum residual has the separate bound 1e-6 + 5e-5‖g_L‖. Linear component closure uses the corresponding gradient bound times the actual relevant displacement norm, plus scalar roundoff; FP32 gradient closure does not imply exact scalar closure of its dot products. No tolerance was calibrated on scientific data.

Numeric NPZ reads are capped at 256 MiB, including expanded members. Headers are checked before allocation; object/structured arrays, unsafe or duplicate members, inconsistent declared payload sizes, symlinks, and receipt mismatches are rejected. `O_NOFOLLOW|O_NONBLOCK` and opened-descriptor regular-file checks prevent the earlier path inspection from being the only file-type guard. Parsing consumes the exact bounded hash-verified payload. One large basis is processed and released at a time; no P×P matrix is formed.

The explicit audit service is `spectral-component-utility-audit-001.service`: one CPU, 2 GiB, zero swap, 900-second cooperative and 20-minute hard limits, no restart, control-group kill. CUDA must be hidden, NumPy must be 1.26.4, and four thread variables must be `1`. Output is exclusive, at most 64 MiB, and outside the immutable acquisition. Actual full-archive runtime and memory have not been measured; these remain enforced limits, not a performance promise.

## Fabricated verification and receipts

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= timeout 60s python3 -m unittest discover -s tests -p 'test_spectral_component_utility_audit.py' -v
```

Initial 12 fixtures passed in 0.232 seconds; expanded 16 passed in 0.258 seconds; final **18 passed in 0.266 seconds**. There were no fixture failures. Tests cover independent scalar formulas, m=1 and empty-subset behavior, offset stability, Adam/projection/clocks, no-basis bypass, FP32 path rounding and compensated dots, all-seed summary signs, a complete tiny parent archive and targeted tampering, roles/assignment reconstruction, committed/current sources, provenance/resource consistency, numeric-header attacks, file-type races, and inert invocation. The tiny archive uses fabricated constant predictions and is an interface/arithmetic test, not simulated optimizer efficacy. Scoped `git diff --check` passed.

The best-practices validator was used before implementation. Official NumPy 1.26 documentation supports [non-pickle loading, bounded headers and closing NPZ handles](https://numpy.org/doc/1.26/reference/generated/numpy.load.html); its [random compatibility policy](https://numpy.org/doc/1.26/reference/random/compatibility.html) motivates pinning the exact generator, version, call order and draw shapes. The additional admission and arithmetic checks are project requirements.

- Auditor SHA256: `9e938a3fd2720a95124a422a18a6470f114d0b7a0b64d641c7f6567dc35cf897`.
- Fixture SHA256: `877a056b68c7a9c1c5c4c20cf2e7f8deb1051853701994b062fc38de6548e94e`.

Main owns final review, source freeze, and any once-only acquisition/audit launch. The leaf changed only its new auditor, fixture and this note; it made no commit or external delivery.
