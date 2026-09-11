# Frozen PC4 content × template measurement

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

This implements the exact 16-prefix [accepted design](design-decision.md),
not a new design or a restart. dataset.json (artifact not distributed in this public snapshot) and pairs.json (artifact not distributed in this public snapshot)
fix order and membership: four contents, active then passive, O then P.
Main checked all strings against [proposed-design.md](proposed-design.md).

## Prediction and analysis fixed before measurement

Primary: PC4(O) − PC4(P) > 0 in **all eight** pair/template cells.
Show every signed difference, positive/zero/negative counts, and each content's
sign consistency across templates. Zero fails the strict prediction. Keep all
four original axes; PC1–PC3 descriptive, no new control or revised criterion.
No p-values, exclusions, replacements, sign flipping, new judge, or PCA refit.
Four content contrasts expressed twice are not eight independent content pairs.

Measure post-block layer 11 (zero-based), final input token, width 1024,
from cached Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17. Same local J-Lens text adapter,
force_bos=False, eval/frozen parameters, no gradients, cache or generation;
BF16/eager model; convert last-position activation to float32 h32.
No lens decoding or vocabulary projection. All scores are exactly
z = (float64(h32) − source_mean64) @ float64(U32).
Use unchanged canonical directions archive SHA256
47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa,
keys u32 (1024×4 float32) and source_mean64 (1024 float64).

New worker entrypoint/output root only:
/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-content-template/.
Source and fabricated tests must be reviewed and committed before admission.
Each actual stage gets an exclusive directory and an attempt receipt before
work; never overwrite or rerun a consumed stage. Stop on any provenance drift.

One CPU tokenizer preflight covers the entire unchanged roster. No added special
tokens, chat wrapper, truncation, or padding; exact input IDs and all-one masks.
Every input must have 1–96 tokens; any failure stops the whole panel. New source
must validate the exact roster and 8-word active / 10-word passive strings.
Freeze token receipts before one model load and 16 sequential text forwards.
Save all activations, all 64 scores, all 32 differences, input/source/model
hashes, parameter immutability and resource receipt. Main checks saved scalar
projections/gaps independently without another model call.

## Runtime and authority

User-authorized autonomous J-Lens follow-up, local desktop only, no training,
network acquisition or paid compute. Previous 24-prefix run used about 12s
and 1.55 GB peak PyTorch allocation; check actual headroom before launch.
CPU preflight: one core, 4 GiB host, no swap, 120s timeout, CUDA hidden.
Forward: one core, 8 GiB host/no swap, 8 GiB PyTorch allocator ceiling, 600s
runtime cap, no restart. Record current units and GPU before admission.
No optimizer/familiarity experiment is resumed. Paid spent/reserved $0/$100.

## Scope and interpretation

## Implementation documentation check

The tokenizer API defaults to adding special tokens; explicitly disable it,
along with padding/truncation. Source:
https://huggingface.co/docs/transformers/main_classes/tokenizer
Inference-mode disabling gradients does not itself set eval mode; use both.
Source: https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.inference_mode.html
Check installed source when documentation versions differ. No new library or
API integration is needed; preserve the previously measured forward contract.
