# Independent-content forwards: source feasibility

Codex — Spectral Optimizer Investigation · 11 September 2026

**Feasible with a small new forwards-only entrypoint; not implemented or
admitted here.** Reuse all four canonical directions, the old fit mean and
the unchanged A/B/C references. Load the cached model once, capture each new
frozen text once, export its four scores, and stop. No lens loading or token
readout decoding is needed. Corpus selection and reader replication remain
main-owned decisions.

Current-state anchor: main commit `9336a43`,
research state (artifact not distributed in this public snapshot).
Source worktree is clean at `c3fbff02378795aa926a545aa9a546a82812d9a7`:
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree`.

## Existing implementation contract

The frozen acquire.py (artifact not distributed in this public snapshot)
has SHA256 `076aeda34c1960744a4107e5b91461861017d0673c8fdb91664a70a951439786`.

- `load_runtime`: `/usr/bin/python3`, Python 3.12.3; NumPy 1.26.4,
  Torch 2.11.0+cu128, Transformers 5.5.0, huggingface-hub 1.8.0. Model load
  uses the fixed revision, `local_files_only=True`, `trust_remote_code=False`,
  bfloat16 and eager attention. Explicit eval/frozen parameters and no-grad
  capture are distinct requirements. No install, upgrade or network fallback.
- `capture`: each row has an ID and exact `prefix` string. It calls the
  tokenizer directly with `add_special_tokens=False, truncation=False`,
  rejects zero or more than 96 tokens, and forwards a single unpadded `[1,T]`
  integer tensor. It uses no chat template, completion, generation or loss.
- The hook is on `model.layers[11]`: zero-based post-block layer 11, the
  twelfth of 24 text blocks, width 1024. It saves `h[0,-1]`, expects exactly
  one hook invocation, returns None without changing output, and removes
  the hook in `finally`. Saved float32 values are exports of the model's
  runtime activations, not a claim that inference ran in float32.
- The pinned HF adapter (artifact not distributed in this public snapshot)
  forwards its text module with `use_cache=False`. Keep this already checked
  full-forward path; do not add early layer termination, batching/padding,
  alternative attention kernels or a different model wrapper to this task.
  Do not call the adapter's `encode`, which enables truncation by default.
- `parameter_state` checks eval mode, frozen/gradient-free parameters and
  parameter identity/version invariance. Preserve those checks around capture.
- `run_stages` defines scores as
  `(h32.astype(float64) - source_mean64) @ u32.astype(float64)`.
  Preserve this exact four-column object and score definition; never center
  using the new panel, renormalize activations, refit PCs or choose new signs.

## Frozen objects and cache

Paths below the worktree's `output/2026-09-10-j-lens-fresh-content/`:

| Artifact | SHA256 / required use |
|---|---|
| `preparation/directions.npz` | `47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa`; later read only `u32` (1024×4 float32) and `source_mean64` (1024 float64) |
| `preparation/selection.json` | `aac75625b67d9a90ba351d979f3e1a77cd75f69706a8b3113e071630de023f4d`; axis/reference provenance, no reselection |
| `references/interpretations.json` | `0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`; unchanged exact A/B/C material |

These objects were already certified by the prior acquisition/audit; this
assessment did not open the NPZ numerically. A later admitted implementation
must pin-check before/after context-managed `allow_pickle=False` loading and
verify finite shapes/dtypes. The old `load_inputs` is not reusable unchanged:
it assumes 24 authored IDs and loads now-unneeded decoder inputs.

The model cache root is `/private-artifacts/storage/cache/huggingface/hub`.
`Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17` has snapshot directory
`models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17`.
The cached shard `model.safetensors-00001-of-00001.safetensors` exists and
has 1,746,942,600 bytes (metadata checked, weights not opened here). Its
previously certified SHA256 is
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`;
recheck it at admission, not by downloading another copy.

Reverified small-file digests:

| Snapshot file | SHA256 |
|---|---|
| `config.json` | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |

The cached tokenizer configuration declares Qwen2Tokenizer, no BOS token,
`add_bos_token=false`; explicit no-special-token calls remain authoritative.
The wrapper can otherwise set add-BOS for tokenizers exposing a BOS token.
Retain exact input IDs and position indices so the actual behavior is auditable.
Model configuration advertises 262144 positions, but that is **not** permission
to exceed the prior 96-token operational cap or extrapolate resource guarantees.

The adapter source checkout is
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/jacobian-lens`, verified
clean at `581d398613e5602a5af361e1c34d3a92ea82ba8e`; retain explicit path/import
provenance rather than assuming an installed jlens package. The historical
lens revision is `0731326edff4ae730ffc5356fe1a4728c748b3a6`, file
`qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`, digest
`aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48`.
It supplied the retained references; the new forwards-only lane need not load it.

## Natural-text compatibility and the main design choice

Mechanically, one independently sourced English sentence or contiguous
sentence prefix is compatible if its exact frozen string tokenizes to 1–96
tokens. Source URL/document ID and character offsets should remain metadata,
not be prepended to model input unless also part of the fixed reader stimulus.
Process samples separately: no preceding article context carries between rows.

The old authored examples ended shortly after a verb; natural complete
sentences may end on punctuation, quotes or abbreviations and have longer,
more variable context. Last-position capture remains mechanically identical
but the linguistic measurement site changes. A score at a final period is
not automatically the same semantic measurement as one at a verb. Likewise,
a title, HTML fragment, truncated sentence or orphaned pronoun can change
what readers and model infer. Domain/source variation is another intended
transfer challenge, not something the previous small framing analysis removes.

**Recommendation:** before any new forward, main should fix one deterministic
text-span/endpoint rule, exact normalization policy, token eligibility cap,
source-based selection order and pairing rule; preserve the strings the
reader will actually see. If complete natural sentences are chosen, retain
their punctuation and describe the resulting distribution shift. Do not
choose endpoints or strip punctuation after inspecting activation outcomes.
A tokenizer-only eligibility check can precede GPU work once authorized;
the resulting roster must be frozen before measurements. Reject an invalid
frozen panel rather than silently truncate, drop or substitute rows mid-run.

## Minimal new entrypoint and resource envelope

Copy/adapt the small guarded runtime/capture/score-export portions into a new
import-inert module under this new study. Do **not** call old `acquire`,
`run_stages`, `decode`, preparation or any old CLI. Remove lens loading and
the entire reference-decoding stage. Hash-pin unchanged reference bytes
before and after the run instead. Replace hard-coded old IDs/counts with the
new protocol's fixed roster; save original text IDs, exact strings/input IDs,
positions, h32, four score64 columns and fixed pair gaps, plus source/cache/
runtime/parameter-invariance receipts. New independent stage/schema names
must not accidentally admit the old scalar key during later grading.

Use new exclusive attempt/output paths and a uniquely named new unit, with
no resume/retry on ambiguity. Fabricated tests should check unchanged hook
position, no-special-token/no-truncation behavior, long-input rejection,
exact roster coverage, frozen score definition and before-read guards.

Prior [measured run](../2026-09-10-jlens-fresh-content/acquisition-review.md):
24 short prefixes plus 15 decodes took 22.02 seconds including startup;
GPU peaks were 1.549 GB allocated / 1.615 GB reserved, host peak 2.860 GB.
These are an anchor, not a promise for an unselected count/length of new spans.
The prior conservative envelope was one RTX3090, one CPU, 8 GiB host,
no swap, 600 seconds, stop timeout 10 seconds, no restart, and an 8 GiB
PyTorch allocator bound (not a total-process GPU-memory hard cap). Recheck
live host/GPU headroom and freeze the new count/token cap before admission.
No paid resource or download is necessary for the model path.

## Consumed handles: never repeat

- Worktree `output/2026-09-10-j-lens-fresh-content/preparation-attempt.json`
  and `preparation/` are complete.
- Same directory: `acquisition-attempt.json`, `acquisition-receipt.json`,
  `references-attempt.json`, `references/`, `fresh-features-attempt.json`
  and `fresh-features/` are complete, immutable and not new output targets.
- `j-lens-fresh-content-20260910-MMxZIs.service`, invocation
  `dd48de61c4f14a298d22f247cf6da0e9`, completed successfully. Its recorded
  active/exited state is RemainAfterExit, not an invitation to restart.
- The older completions acquisition, saved-fit geometry analysis, fresh/
  within-topic packaging, response locks, grading and delivered reports are
  also consumed. No old readers, scalar outcomes or payloads are regenerated.