# Content × template forward adapter: source freeze

Source only, 11 September 2026. No real tokenizer, model, scientific array or
measurement was loaded; actual stages require separate main-agent release.

## Scope and contract

The main protocol (artifact not distributed in this public snapshot)
is pinned to `620565ecc6c78c64ac395efee947e20a97772bcb6ccc804c645ccfc198d350f3`.
Dataset/pair hashes are the committed main `81bf8ae` inputs. Model revision,
weight/config/tokenizer pins, runtime versions, clean upstream adapter, canonical
U32/mean and post-block 11 final-position capture are unchanged. No lens loading,
decoding, labels in model input, generation, refitting or judging is introduced.

New `token-preflight/` and `forwards/` paths are exclusive, including dangling
symlinks; an attempt precedes input reads and failures remain consumed. All 16
texts must pass the 1–96-token bound, without truncation or partial output.
Forward execution binds the preflight receipt, source, inputs, IDs and masks;
no tokenizer is loaded again. Hash checks surround reads/stages. Runtime stays
offline, frozen/eval and `no_grad`, with parameter-version checks. Main's launcher
owns host/time limits; the 8 GiB PyTorch allocator setting is not a whole-process
GPU-memory guarantee.

Output layout remains `features.npz` containing activation_11 (16×1024 float32),
scores64 (16×4), gaps64 (8×4); scalar `scores.json`; all pair metadata and gaps in
`gaps.json`; full records in `inputs.json`; and hashes/runtime/resource receipts.
Scores use `(float64(h32) - source_mean64) @ float64(U32)`. Gaps subtract these
saved scores in O/P order, not a separately rounded activation-difference product.
No score threshold or semantic success selection exists in this adapter.

## Documentation validation

The best-practices-validator check retained the established contract:

- [Transformers 5.5 tokenizer API](https://huggingface.co/docs/transformers/v5.5.0/en/main_classes/tokenizer): special-token addition defaults on, so it is explicitly disabled; padding and truncation are also explicitly disabled.
- [Current Transformers model API](https://huggingface.co/docs/transformers/main_classes/model): `local_files_only` and revision selection support the pinned offline load. The version-5.5 model page was unavailable through the browser; current docs are corroboration, not an upgrade instruction or proof of version identity. Runtime still asserts installed 5.5.0.
- [PyTorch 2.11 no_grad](https://docs.pytorch.org/docs/2.11/generated/torch.no_grad.html) disables gradient recording; parameters are separately frozen. [Inference-mode documentation](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.inference_mode.html) makes the separate eval requirement explicit. No switch to inference mode was made.
- [NumPy 1.26 load](https://numpy.org/doc/1.26/reference/generated/numpy.load.html): retain `allow_pickle=False` and the NPZ context manager; access only `u32` and `source_mean64`.

## Fabricated verification

Command, from the worker root:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/python3 output/2026-09-11-j-lens-content-template/test_forward.py
```

13 tests PASS, 2.584 seconds. Fake text/tokenizer/model and temporary synthetic
arrays only. Coverage includes inert import; ordered IDs/pairs/poles/templates;
word counts/endings; malformed token IDs/masks and late whole-panel rejection;
post-block/final-position/nonmutation capture; canonical score arithmetic and
positive/zero/negative O−P gaps; finite/dtype/norm checks; duplicate/nonfinite JSON;
exclusive/dangling and consumed failure guards; preflight tamper binding; missing
preflight preventing archive/model access; and fake end-to-end output hashes.
Independent source review caught an initial bare-list assumption incompatible
with the new rows/pairs wrappers. Both stage loads and fabricated fixtures were
corrected before any real execution; regression tests reject the old containers.

SHA256 forward.py: `16d53d9da76269537d8743ad92387bc8fd36f6d89c7d7dba28b86a818c877b7e`.
SHA256 test_forward.py: `c623845c6649a1c9347ff0a8a2edd1289cf05a847ab848214047c25d6d1bf51e`.
