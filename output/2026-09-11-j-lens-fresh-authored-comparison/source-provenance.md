# Fresh-authored verb-only source preparation

New code only; no old entrypoint or consumed stage is called. Main owns actual
stage admission. This leaf has not run a real tokenizer, neural forward,
saved-array checker, reader, or grading stage for this panel.

## Fixed contract

All 32 `fa01`–`fa08` active/passive O/P rows and all 16 pairs remain unchanged.
Literal source pins bind raw author, exclusions, protocol, derivation receipt,
dataset/pairs, canonical mean/U32, original reference export and model metadata.
All 14 literal input file hashes were verified without deserialization.
The cached model revision/weight pin, adapter clean revision, installed runtime,
BF16 eager eval/frozen/no-grad single-row forward, and post-block 11 hook are
unchanged. Only `verb` is captured; the period check validates full input
tokenization but does not capture or score that endpoint.

`token-preflight/tokens.json`: `jlens_fresh_authored_tokens_v1`; records contain
exact text, spans, IDs, all-one masks, offsets and singleton
`captured_positions` / `selected_substrings` dictionaries keyed `verb`.
Preflight receipt: `jlens_fresh_authored_token_preflight_receipt_v1`.

`forwards/features.npz`: `activation_11` float32 `(32,1,1024)`, `scores64`
float64 `(32,1,4)`, `gaps64` float64 `(16,1,4)`, `locations=['verb']`.
`scores.json`: `jlens_fresh_authored_scores_v1`, `locations.verb` contains all
four axis/value mappings in fixed row order. `inputs.json` uses
`jlens_fresh_authored_inputs_v1`; `gaps.json` retains all ordered O-minus-P
pair metadata and singleton-role gaps. Forward receipt:
`jlens_fresh_authored_forwards_receipt_v1`.

The checker requires explicit packet-manifest SHA, response-lock SHA and immutable
Git lock commit. The public verifier checks all four 64-choice responses and all
five committed blobs before any measurement receipt, scientific hash or array
access. Frozen design/reference hashes and exact judged preflight/token hashes
are also bound to the measurement. The gate is checked again before output.
Independent scalar `fsum` corroborates 128 projections; exact subtraction of
saved scores corroborates 64 gaps. All signs/zeros and template changes remain
descriptive: the primary comparison belongs to the separately locked A/C reader
grade, not an all-positive gloss criterion.

## Checks and documentation

Synthetic tests use fabricated rows/tokens, a tiny CPU-only fake model and
temporary arrays only. They cover singleton capture, shapes/dtypes/joins,
split-verb boundaries, late whole-panel failure, tampering, no-overwrite/dangling
guards, source pins, lock-before-array order and judged-token binding. An initial
stale schema assertion in the copied test failed and was corrected before freeze;
it did not affect producer code or any actual data.

Final synthetic command: `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 timeout 60s /usr/bin/python3 -m unittest -v
test_forward test_check` from this directory: 23 tests PASS, 2.880 seconds
(4.542 seconds command wall time). The main judging verifier is pinned to
SHA256 `50de1fff65d4fc50f0aaea0bf901e9096a01fcc62d0c388aba774017d8f942bf`.

The best-practices-validator skill guided explicit API checks against
[PyTorch 2.11 no_grad](https://docs.pytorch.org/docs/2.11/generated/torch.no_grad.html),
[NumPy 1.26 load](https://numpy.org/doc/1.26/reference/generated/numpy.load.html),
and [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer).
Explicit eval/frozen parameters accompany no-grad; tokenization explicitly
disables special tokens, padding and truncation and requires fast offsets;
NPZ reads use `allow_pickle=False` and context managers. The complete pinned
`jlens/hf.py` and `jlens/__init__.py` were reread: only the unchanged text-module
forward with `use_cache=False` is used, not encoding or lens decoding.

The 8 GiB PyTorch allocator bound is not a hard total-process GPU-memory cap.
Main supplies CPU/host-memory/time admission. Cloud spent $0, reserved $0,
ceiling $100. No new paid work is selected here.
