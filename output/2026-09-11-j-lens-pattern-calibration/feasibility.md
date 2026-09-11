# Fixed-score pattern calibration: saved-source feasibility

11 September 2026 · metadata/source inspection only · no acquisition or scientific stage admitted

**By article ID, the proposed 64 calibration + 32 evaluation prefixes fit the remaining pool.** Technical lead/prefix/token eligibility remains unknown. The cached decoder and unchanged model contract are available; there is no identified cache/API blocker, but normalization, tie handling, split and pair-distribution weighting must be fixed prospectively. This is description calibration for the old scores, not PCA refitting.

## Complete prior-request exclusion and capacity

| Original owner topic | Catalogue IDs | Previously requested IDs | Remaining IDs | Maximum disjoint within-topic pairs |
| --- | ---: | ---: | ---: | ---: |
| Astronomy | 53 | 13 | 40 | 20 |
| Cooking | 37 | 11 | 26 | 13 |
| Football | 6 | 6 | 0 | 0 |
| Programming | 139 | 32 | 107 | 53 |
| Total | 235 | 62 | 173 | 86 |

First-root ownership remains astronomy → cooking → football → programming, assigned before exclusions. The frozen catalogue happens to have no duplicate IDs across these groups. The 86-pair bound leaves one programming article unpaired; it is capacity arithmetic, not an actual ordering or selected split. Thirty-two calibration pairs plus sixteen evaluation pairs require 48 disjoint pairs, fewer than 86. No four-topic balanced design is possible; even 32 total texts per each of the three remaining topics would exceed cooking's 26-ID capacity. A fixed unbalanced frame or different declared allocation is required, not post-result balancing.

Scoped enumeration of J-Lens output directories found **67 request files: 4 category-list requests and 63 article-request records**. The latter represent **62 distinct article IDs/network events**: 28 from the first Wikipedia acquisition and 34 from the natural add-on. The resumed Spiral-arm first request is a copied response record, not another GET; its page ID is excluded once. Old technical redirect skips 69730468, 78241963, 6427906 and 47618653 remain excluded. Add-on K005 requested both 57077104 (technically eligible) and 74785139 (redirect); the entire pair was rejected, so **both** still enter exclusions. Selected-only exclusion would be wrong.

The four raw category responses were checked against compiled catalogue member arrays; each request's referenced response bytes were hashed and size-checked without parsing article extract text. Historical receipts bind the inspected request/response files. The old 28-ID union was independently compared with the prior frozen inventory, whose builder was not imported or executed. Source hashes were rechecked after enumeration. inventory.json (artifact not distributed in this public snapshot) preserves all 62 excluded IDs, per-topic remaining IDs, all 63 request records, the copied event, and source paths/hashes/sizes. It contains no new candidate hash-order, pairing or split.

Both saved attribution reviews revisit the already-selected article/history/talk URLs (72 old; 96 add-on). They identify no separately opened additional candidate article; linked translation sources/older revisions were explicitly not opened. Catalogue titles, API documentation, source code examples and copied report payloads are not new article-request events. This is a scoped persisted-record audit, not a full browser-history audit or a pretraining-unseen claim. The catalogue is a captured source frame, not guaranteed-current Wikipedia membership. Collection can still fail through redirects, duplicate prefixes, short/malformed leads or tokenizer limits. A bounded whole-pair technical exclusion/failure rule must be frozen before any request; the ID surplus is not a promise of 96 eligible leads.

## Pair-gap calibration: exact but a different estimand

For 32 disjoint within-category calibration pairs, define Δh_i=h_left−h_right and Δz_ij=u_jᵀΔh_i. Then

```text
d_j = [Σ_i Δh_i Δz_ij] / [Σ_i Δz_ij²]
```

is the zero-intercept least-squares activation-pattern slope for that fixed score. Equivalently it is the covariance pattern of the empirical distribution containing both +Δh_i and −Δh_i with equal weights. With a positive denominator, u_jᵀd_j=1 in exact arithmetic. Reversing any pair's orientation leaves both sums unchanged. The old centering vector cancels in the mathematical pair difference; keep the old centered-score definition for evaluation, and freeze whether recorded pair gaps are authoritative subtraction of saved scores to avoid near-zero last-bit ambiguity.

This is **not** pooled raw-text covariance, nor automatically the centered covariance within a named topic. Under an additional independent same-category sampling model, E[ΔhΔhᵀ|g]=2C_g; deterministic finite-catalogue pairing does not by itself justify that population claim. The contrast distribution and category mixture are determined by the prospectively fixed pairing/split. It avoids estimating category means but does not remove category weighting or model-reading confounds.

The squared-gap denominator means high-magnitude calibration contrasts can dominate the pattern, whereas the proposed held-out sign accuracy gives every pair equal weight. This is an estimand trade-off, not an algebra error. Thirty-two high-dimensional contrasts can yield an unstable pattern, particularly with small score variance. Freeze nonfinite/zero/small-variance failure handling and any weighting rule before results; do not trim small evaluation gaps or choose a better-looking pattern afterward. Estimation requires only an m×4 cross-product and four sums of squares: O(mk), with m=1024 and k=4. No eigendecomposition, full covariance, training or new u is necessary. This choice is compatible with keeping the same original example panels and all four old score targets.

## Exact cached decoder and source contracts

Worker root W:
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree`.
Adapter root J:
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/jacobian-lens`.
Cache root H:
`/private-artifacts/storage/cache/huggingface/hub`.

- J is clean at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. `jlens/lens.py` SHA `e231e7d3a6c8e8f7791b53705a34342d0bba376a127a82376eaf6ec30ca11808`; `jlens/hf.py` SHA `228cf078e4586a7b7f61a6f5064403b8960de337afd19256efa56f04d53e3222`; `jlens/protocol.py` SHA `7217a3bc2c1cd01fed3e6b00c53effbd957effddc49fe073da89cb10d1b1a344`.
- Lens file: H/`models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`. Existing file stat: 48,242,373 bytes. Frozen content SHA **from the successful original decoder receipt**, not a new weight read: `aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48`. That receipt records 233 fitted prompts. Checkpoint tensors were not opened here; a future admitted load must verify content hash, layer 11 availability, width 1024 and finiteness.
- Model: Qwen/Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`; H/`models--Qwen--Qwen3.5-0.8B/snapshots/<revision>/model.safetensors-00001-of-00001.safetensors`, previously verified SHA `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. No model weights were loaded or rehashed in this task.
- Original signed readouts: W/`output/2026-09-10-j-lens-fresh-content/references/interpretations.json`, SHA `0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`. Original decoder source `acquire.py` in that study, SHA `076aeda34c1960744a4107e5b91461861017d0673c8fdb91664a70a951439786`; its reference receipt SHA `d4d1ba26c84aa5ce65efc8068d26b7b6c5ee969af7e20db27e418225f6779e92`. These completed stages must never be invoked again.
- Old score basis: W/`output/2026-09-10-j-lens-fresh-content/preparation/directions.npz`, SHA recorded in receipts `47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa`; members `u32` (1024×4) and `source_mean64` (1024). The archive was not opened here. Preserve those realized float32 u columns, not re-normalized replacements, for all new score targets.

**API path:** load the exact local checkpoint with `JacobianLens.load(local_path)`, which uses `torch.load(...,map_location="cpu",weights_only=True)`; its constructor casts Jacobians to float32. For a saved float32 vector on the chosen device, `lens.transport(vector,11)` computes vector @ J_11.T. Then the existing model adapter's `unembed` casts to the LM-head dtype/device, applies final norm and the LM head. The old acquisition used `model.unembed(lens.transport(tensor,11)).float()`, then `topk(12)` and individual `tokenizer.decode([id])` strings. No prompt forward is required just to decode a vector, although the head/final norm and tokenizer must be available in an admitted process. Do not use `lens.apply` or adapter `encode`: they tokenize/truncate/run a prompt. Do not use the network-capable `from_pretrained` lens fallback when an exact local file exists.

## Precision, normalization and runtime hazards

1. Estimate unnormalized d in float64 and preserve it. For the proposed display, normalize d in float64 then cast to float32; construct −d by the stated sign rule and save the actual input tensors. Input normalization changes the per-score-unit scale, so uᵀd=1 applies to the unnormalized pattern, not its unit-length decoder input. Fail on zero/nonfinite norms. Decoder directions must never replace old u in held-out scoring.
2. J transport is float32; the adapter casts its result to **BF16 before** final RMS normalization/LM-head projection. Casting the returned logits to float32 cannot recover BF16 precision. Installed Qwen source computes RMS statistics in float32 with epsilon 1e−6 and multiplicative (1+weight), then returns to input dtype; its head has no bias. Exact linear-surrogate or scale-invariance claims must therefore not be substituted for actual decoder output. Parameter/mean biases are not added to the pattern vector.
3. Original top-k ordering came from `torch.topk`, with potentially tied BF16 logits; there is no explicit original token-ID tie breaker. Preserve all old reference strings/ranks unchanged. A new reference tie policy—including the 12th-token boundary—must be fixed before new decoding, and any deterministic token-ID tie rule must be reported as prospective rather than retrospectively applied to old lists. Preserve duplicates, multilingual strings, leading whitespace and empty/fragment tokens, without selected translations.
4. Tokenization/forward contract remains exact 16-word prefixes, no BOS/additional specials/truncation/padding, stored IDs/all-one masks/offsets, ≤96 tokens, single-row BF16 eager/eval/no_grad, unmodified post-block 11 at the final actual subtoken, float32 saved h and float64 old-u scores. `jlens.from_hf(...,force_bos=False)` avoids adapter token mutation. A distinct new entrypoint/stage is necessary: old producers hard-code the consumed 32-row roster and receipts.
5. Runtime metadata observed now without importing torch/transformers: `/usr/bin/python3`, Python `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`; NumPy 1.26.4, torch 2.11.0+cu128, transformers 5.5.0, huggingface-hub 1.8.0. Binary hashes independently checked: Python `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`, libc `3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf`. Installed Qwen modeling source SHA `aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048`. The original reference decoder used the older June Python build; do not claim exact numerical reproducibility across that OS update. The current forward-amendment receipt records the explicit transition, rather than silently accepting it.

Current inherited pins and package/binary requirements are documented in W/`output/2026-09-11-j-lens-natural-addon/forward.py` (SHA `23c9c7cf77f2239bd16b219c440fbf056628730cd2bdf51f66b812ab0e8f580c`) and `forward_runtime_amendment.py` (SHA `679834b6ff4b2e6b37ef4d60c774c994dcf7bffdf4252d388945034d7e70f409`). Their tokenizer/config/merge/vocabulary/model-index SHA pins should be copied as immutable data into a new reviewed source, not recovered through executing an old stage. Live package/source/cache integrity and resource admission still need verification immediately before any future run.

## Resource and decision boundary

A full 96-prefix measurement would retain 384 KiB of raw float32 1024-wide h values (plus metadata/scores), and the 1024×4 float64 cross-product is 32 KiB. Eight new signed pattern decodes suffice if the original eight signed lists are reused unchanged; this is not an instruction to re-decode the originals. Model memory dominates: the last 32-prefix lane used about 1.55 GB allocated/1.62 GB reserved on the RTX3090 with an 8 GiB allocator limit; that is historical evidence, not a fresh admission or total-process cap. No runtime estimate is guaranteed by linearly scaling its 13-second duration.

Root must still freeze the split/pair order and category weighting, request budget/technical-stop rules, minimum variance, normalization/tie policy, example/display comparator, unchanged-label scoring and reader locking. Calibration and evaluation must be article-disjoint from each other and all prior requests, with no evaluation-dependent pattern choice. The source catalogue has sufficient ID capacity for that decision; technical data eligibility and interpretive benefit are untested. No source/helper stage, tokenizer, archive deserialization, model/decoder, reader, checker or grader ran here. No network calls or children were used. The best-practices skill was applied to pinned upstream/installed source within the explicit no-network boundary, not claimed as a fresh web-doc check.
