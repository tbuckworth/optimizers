# New-verb transfer source preparation

11 September 2026. Source and fabricated fixtures only; actual tokenizer,
forward and checker stages are **unrun**. No actual-stage directories created.
Main owns source acceptance, admission, one-run execution and reporting.

[forward.py](forward.py) is a new import-inert entrypoint, adapted after reading
the complete [previous endpoint source](../2026-09-11-j-lens-endpoint-location/forward.py)
(SHA256 `f6b6289d484d2e832982b7a0f78cd60d6010b9d45ef80bc5724ab5cac36dda1e`).
It does not import or invoke that producer. The complete pinned upstream
`jacobian-lens/jlens/hf.py` was inspected: the chosen text forward disables
cache; wrapper construction freezes/evals parameters; its truncating `encode`
method and unembedding method are not called.

Changes are deliberately limited: 32 namespaced rows nv01–nv08, 16 fixed O/P
pairs, new schemas/paths/release variables and frozen inputs; all old-full-text
loading, suffix matching and old-token-prefix constraints are removed. The
unchanged model revision, weight/config/tokenizer hashes, clean adapter commit,
canonical U32/mean and runtime versions remain asserted. Full sentence strings,
span metadata and pair ordering are byte-bound by the three main hashes in
the protocol (artifact not distributed in this public snapshot).
The accepted roster (artifact not distributed in this public snapshot) is also pinned.

All 32 rows must pass exact wrapper/ID/metadata/5-or-7-word/ASCII validation.
Fast-token offsets must cover the full verb contiguously, end exactly at its
boundary, and include no future lexical content; leading whitespace is allowed.
The final token must cover the period and end at string length. No truncation,
padding, special-token additions or dropped cases. Save full IDs/masks/offsets,
both positions and selected substrings; freeze source/input/token receipt
identity before the model stage. Any failure consumes its exclusive stage.

Each of 32 forwards captures both positions from one unmodified post-block11
output; order is `verb`, `sentence_end`. NPZ contains activation_11
(32×2×1024 float32), scores64 (32×2×4), gaps64 (16×2×4), and named locations.
Scalar JSON schema is `jlens_new_verb_transfer_scores_v1`, with `locations`
mapping each role to four `{axis, values}` records in fixed ID order. Gap JSON
keeps all 16 pair metadata records and both four-axis differences. No old
scores, new fit, decoder, judgment or comparator is loaded.

[check.py](check.py) is separately inert and hashes the new producer before
importing its reviewed JSON/roster/token-binding helpers. It never calls their
runtime, tokenizer, model, capture or producer scoring functions. After a new
exclusive attempt and all receipt/input/output pins, it checks saved shapes,
finite values, locations, full input records and scalar exports. Independent
`math.fsum` reconstructs all 256 projections (fixed absolute arithmetic tolerance
1e-12); subtraction of the saved score64 rows verifies all 128 gaps exactly.
It retains all gaps and all axes' positive/zero/negative counts, both-template
positives and template sign changes. The sole primary Boolean is all 16 verb
PC4 gaps strictly positive. Zero is retained and fails that strong prediction.
Shared integrity helpers are not independent semantic/model validation.

## API check

The best-practices-validator review checked the current official
[fast-tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)
and installed Transformers 5.5.0 `tokenization_utils_base.py`: offset mapping
returns character spans for fast tokenizers; special tokens default on and are
explicitly disabled. The forward contract remains eval/frozen plus
[PyTorch 2.11 no_grad](https://docs.pytorch.org/docs/2.11/generated/torch.no_grad.html),
not a new inference-mode variant. [NumPy 1.26 load](https://numpy.org/doc/1.26/reference/generated/numpy.load.html)
supports the retained `allow_pickle=False` NPZ context managers. Actual runtime
remains pinned; no installation, upgrade, network acquisition or model load.
Host/time/CPU limits remain the main launcher's responsibility; the 8 GiB
allocator bound is not a total-process GPU cap.

## Fabricated checks and source pins

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/python3 -m unittest discover -s output/2026-09-11-j-lens-new-verb-transfer -p 'test_*.py'
```

19 tests PASS in 3.126 seconds. Only fake sentences/tokenizers/models and
temporary synthetic arrays. Coverage includes rejection of old 16-row inputs,
joins/spans/offset boundaries, split verbs, invalid/overlong IDs and masks,
late whole-panel rejection, both simultaneous captures/dtypes, no mutation,
score/gap orientation, exact zeros, strict JSON, pin-before-import/array guards,
preflight tampering and consumed/dangling stages. Fake end-to-end forward and
checker receipts/output hashes also pass. No results for the actual roster
were generated, inspected or scored.

SHA256:

- forward.py: `f59d6a0da7bd6280bd9a58f4688628ab1699389fa1b9d2e7fd73422a6808fb49`
- check.py: `7f74d36d9549b1ad76a02d1ce09c4aed6407c911ad580fd037eb8b7ba4f1924f`
- test_forward.py: `de7b119737f3a50e56946212d3cb9b8daf2e00b400ef218eec2ef0609e784bc3`
- test_check.py: `dc142b42f799e7d7d2fce29b1591958aa9ae761963dbbdf4fb4e1e26344318e3`
