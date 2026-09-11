# Frozen endpoint/location protocol and implementation decision

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

Main accepts the complete [proposed design](proposed-design.md) and its exact
16 strings/spans for bounded implementation under existing autonomous authority.
dataset.json (artifact not distributed in this public snapshot) and pairs.json (artifact not distributed in this public snapshot) freeze those strings,
positions, metadata and order. No new user approval gate. This decision does
not start a tokenizer or model; source review and fabricated tests come first.

Primary: all eight fixed PC4 O−P gaps at the verb-final subtoken strictly >0.
Zero fails. Sentence-end is descriptive, not an alternative success condition.
Retain every gap and signed shift versus prior full-prefix end, all four axes,
all three locations, and per-template signs. Four reused contents expressed
twice, no independent-confirmation/p-value claim. Prior1/8result is unchanged.

Execution: one cached frozen Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17 load,16 NEW shortened text forwards,
post-block11 simultaneous captures at verb-final subtoken and first-sentence
last token, converted to h32. Use unchanged canonical U32/source_mean64 and
float64 scoring, exactly as specified in the design. No decoder/PCA/judge/
training/network/paid call. Worker root output/2026-09-11-j-lens-endpoint-location/
gets new guarded entrypoint and exclusive stage directories. Never invoke old
consumed stages or repeat old full-prefix forwards.

Whole-roster CPU preflight: exact hash-pinned old and new inputs;5-word active,
7-word passive new strings obtained only by removing the exact old suffix.
Fast offsets with complete contiguous verb coverage, final verb subtoken ends
exactly at verb boundary, no future lexical content, allow leading whitespace.
Final token covers terminal period and ends at exact string length.1–96tokens,
no special tokens/padding/truncation; all-one masks. Short IDs must equal the
old frozen full-input ID prefix, with strict shorter length. Save offsets,
selected indices/substrings and exact inputs. Any failure stops wholepanel;
no guessed indices, dropped or substituted cases. Freeze receipt before GPU.

Resource bounds: /usr/bin/python3, CPUquota100%, CPUpreflight4GiBhost/no swap,
CUDA hidden/TimeoutStartSec120. GPUforward8GiBhost/no swap/RuntimeMax600,
8GiB PyTorch allocator bound (not process cap), Restartno/stop10sec. Check current
live handles and GPU/host headroom before launch. Model modes/gradient absence/
tracked parameter versions unchanged. Source/input/protocol hashes in receipts.

Main read the position reasoning (artifact not distributed in this public snapshot). This is not fixed
meaning with only position changed: causal prefixes contain different actors/
objects, and removing the ending removes time information. Token IDs must match
prefixes, but bit-identical BF16 states across different-length runs are not
asserted. Even success is a local lexical/contextual correspondence, not an
abstract causal feature or explanation of the old external-text19/24 result.

Source documentation: official fast-tokenizer offset mapping and char_to_token
at https://huggingface.co/docs/transformers/main_classes/tokenizer; installed
5.5.0 docstring matches. Main inspected Qwen3.5 text causal mask, triangular
gated-delta computation and causal convolution. Continue using eval+no_grad
and pinned jlens.from_hf(...,None,force_bos=False) textforward use_cache=False.

No stages have run at protocol freeze. Paidspent/reserved$0/$100. J-Lens-only;
optimizer/familiarity deferred. Full research goal/two-hour reminder active.
