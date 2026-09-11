# Calibration producer — source and fabricated-fixture freeze

**No actual model, tokenizer, decoder, archive inspection, calibration stage or evaluation stage has run in this task.** Root owns independent source review, current resource checks and any new release. All earlier collection/preflight stages remain consumed.

- [calibrate.py](calibrate.py), SHA256 `79b1d1a042a717ea6d691234d25c2dbcfbfb7e87bfd4224105afb3b1bf0bf37d` (317 lines).
- [test_calibrate.py](test_calibrate.py), SHA256 `200ee1df55933461f57adadaacd3174b2776de227b6df63d2f591b1ba86620b8` (312 lines).
- Main protocol remains `4d80a9eef24e32a475c1d5016f57ca05419cacb45f2b27504f9ca7f916fae762`; main pure patterns implementation remains `524efc43e54c9549e66586f8a8dcc9db8542979637a56d840c3739f1c156a789`. Both were read in full. No estimator, split, weighting, sign, variance threshold or success-criterion change.

## Isolation and provenance

The new exclusive `calibration/` requires the explicit root GPU release, exact reviewed source SHA, one-thread/offline environment, and disabled implicit HF credentials. Its read set includes only the 64 calibration dataset/token rows and their 32 pairs, original directions/reference files, code/cache pins and metadata receipts/manifest. It never opens evaluation datasets/tokens/features/keys. The complete preflight's input-hash map is compared as metadata only. A fabricated file-open sentinel verifies this distinction with the evaluation file absent.

Fully read immutable helper sources: this study's `preflight.py` (`05ce0a8a…`), previous natural-addon `forward.py` (`23c9c7cf…`) and runtime amendment (`679834b6…`). Only their pure validation/runtime/token-record/model-loading/capture helpers are reused, not old stage entrypoints or their old fixed-32-row score exporter. The current Aug 31 Python build, numerical packages, seven distro versions and two host binary digests must equal the completed preflight before and after processing.

The actual original `acquire.py:decode`, upstream `jlens/hf.py`, `jlens/lens.py` and package initializer were read in full. Their scientific path is preserved: pinned BF16/eager/eval/no-grad model; frozen last-input-position block-11 capture; FP32 J-Lens transport; BF16 head cast **before** final normalization; one `.float().topk(12)` per signed pattern; each selected token ID decoded individually with no filtering. The local Qwen implementation is also pinned (`aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048`), alongside model/adapter/cache/lens hashes. The original tokenizer's null BOS means there is no evidence the earlier default mutated BOS behavior; the new already-tokenized adapter explicitly uses `force_bos=False`.

The [best-practices check](https://docs.pytorch.org/docs/2.11/generated/torch.load.html) confirms restricted checkpoint loading; the pinned lens method uses `weights_only=True` and CPU mapping. NumPy direction loading uses `allow_pickle=False` with a closing context manager as [documented](https://numpy.org/doc/stable/reference/generated/numpy.load.html). [PyTorch does not guarantee stable tied top-k indices](https://docs.pytorch.org/docs/2.11/generated/torch.topk.html); actual IDs/order/full logits/cutoff counts are retained once, never reranked or decoded again for tie selection. These APIs were checked against current official documentation and the exact local implementations.

## Artifact contract

`inputs.json` preserves all 64 records and 32 pairs. `features.npz` retains `activation_11` FP32 `(64,1,1024)`, original `u32` FP32 `(1024,4)` and `source_mean64` FP64 `(1024,)`. Calibration joins are verified before the singleton-position dimension is removed for fitting.

`patterns.npz` contains every unchanged patterns.py output: `patterns64`, `cross64`, `score_energy64`, `signed_inputs32`, `calibration_gaps64`, `max_single_pair_energy_share64`, `direction_energy_fraction64`, and `unit_score_identity64`. The gap here is the protocol's direct FP64 `Δh @ U`, not a replacement evaluation scoring direction. Invalid/zero-energy axes stop the entire fit; no alternative pattern or adaptive cutoff is introduced. Captured states are saved before fitting, so a failed fit retains its evidence.

`decoder.npz` preserves `inputs32 (8,1024)`, `logits32 (8,248320)`, `top_ids64 (8,12)`, and `top_logits32 (8,12)`. `readouts.json` has schema `jlens_pattern_calibration_readouts_v1`, ordered PC1+/− through PC4+/−, exact tokens/IDs/scores and `cutoff_logit`, `strictly_above_cutoff_count`, `cutoff_tie_count`, `selected_at_cutoff_count`.

`original-interpretations.json` is a byte-identical copy of the frozen original reference JSON, SHA `0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`. `references.json` has schema `jlens_pattern_calibration_references_v1`, `arms: {U: {axes: [...]}, P: {axes: [...]}}`. Each axis row has `axis`, `positive_reference`, `negative_reference`; each pole contains exactly `example_prefix` and `direction_tokens`. Both arms use the same original C prefixes; U uses unchanged A lists and P substitutes only the new 12-token lists. Original B fields remain untouched in the byte-preserved archive, not shown in this comparison.

The complete receipt uses `jlens_pattern_calibration_calibration_receipt_v1`, seven output-file hashes/sizes, all source/input/cache/runtime pins, 64-forward/8-decode counts, zero evaluation/retokenization/refit counts and model/lens parameter-version invariance checks. Its 8 GiB PyTorch allocator bound is not a total-process GPU cap. Root separately enforces CPU1/8 GiB host/no swap/900 seconds. Failures are retained in the exclusive stage; no automatic retry.

## Fabricated validation

`env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 timeout 60s /usr/bin/python3 test_calibrate.py`: **11 tests PASS in 0.104 seconds after imports**.

Fixtures use only generated arrays, temporary files and tiny CPU modules. Coverage includes all 64 captures, immutable masks/last position, independent matrix-form moments, every signed input/full logit, exactly eight top-k calls, all-tied logits and Unicode/whitespace/empty token strings, actual pinned HFLensModel BF16 norm input, unchanged original bytes and control/example fields, calibration-only file access, old-32-row/wrong-role/offset rejection, zero-energy stop before decoder loading, model/lens mutation and nonfinite rejection, safe immutable direction loading, source/release/dangling-stage guards, and full fabricated success/failure receipts.

The added full-wrapper fixture initially let the production `cuda` argument reach its tiny CPU model; CUDA was hidden and initialization failed before any real model or scientific data access. Only the test harness was corrected to assert that production argument then route the fabricated calculation to CPU. Producer bytes were unchanged. This is a recorded fixture error, not an actual calibration attempt.

The normal source hook then flagged only `CALIBRATION_TOKENS_SHA` on line 17. Worker independently rehashed the committed calibration-token file and root corroborated the public artifact digest. The exact source line now has a `gitleaks:allow` comment; no scanner rule, receipt or data was changed and no hook was bypassed. Removing only that comment exactly reproduces the root-reviewed producer SHA256 `a5310cc8086e08a76a54296a77314cf13282d01af0946bdfda3c6ab9521fb4cf`. Final producer hash above includes this comment-only correction.

Recommendation: ready for root's final exact-byte review and separately admitted once-only calibration stage. Finite 32-pair estimation, population mismatch and nonlinear/tied vocabulary readout remain interpretation risks, not grounds to alter this frozen recipe.
