# Calibration implementation checks — 11 September 2026

Source/API review before the new calibration implementation is released. This
does not amend the protocol, produce neural results, or release evaluation.

- The pinned local `jlens/lens.py` loads the cached checkpoint with
  `torch.load(..., map_location="cpu", weights_only=True)`, stores Jacobians in
  FP32, and transports `residual @ J.T`. These arguments agree with the
  [PyTorch 2.11 loading documentation](https://docs.pytorch.org/docs/2.11/generated/torch.load.html).
- The pinned `jlens/hf.py` casts the transported residual to the head's dtype
  **before** final normalization and unembedding. Preserve that BF16 path;
  an FP32 normalization substituted here would be a different decoder.
  The old actual `acquire.py:decode` calls this same pipeline, converts its
  logits to FP32, calls `.topk(12)` once and decodes each selected ID separately.
- [PyTorch documents unstable tied top-k indices](https://docs.pytorch.org/docs/2.11/generated/torch.topk.html).
  Preserve each actual result and all vocabulary logits. A saved-array check
  must accept any admissible selection/order among equal scores; it must not
  call the decoder again or substitute a tie-breaking rule.
- Array readers use `allow_pickle=False` and a context manager, as documented
  in [NumPy 1.26](https://numpy.org/doc/1.26/reference/generated/numpy.load.html).
- The old import-inert `forward.py:capture` observes post-block 11 at each
  record's frozen final input position, removes the hook in `finally`, uses
  `no_grad`, and returns FP32 states. Its generic capture loop may be reused
  only after new calibration-specific 64-row/32-pair joins; its old stage
  entrypoints and hard-coded 32-row score exporter must not be used here.
- `from_hf(..., force_bos=False)` avoids tokenizer changes explicitly. The old
  default would only change BOS behavior if the tokenizer had a non-null BOS;
  the frozen tokenizer has none. There is no evidence of an old BOS change.
- The existing runtime helper pins the actual August-build Python, numerical
  packages and host binaries. Recheck those pins, cached model/lens bytes and
  GPU/host headroom before launch. An 8 GiB PyTorch allocator bound is not a
  total-process GPU limit.

Scientific verification is separate: independently reconstruct pair moments
from saved FP32 states and fixed U in FP64; verify every pattern, denominator,
diagnostic, exact normalized signed input, top-k membership, and unchanged
control/example references. This checker must not import the producer or fit
implementation, load a model/lens/tokenizer, or access evaluation states.

The source facts above were checked against the actual pinned files, not
assumed from a paper's idealized linear formula. Whether calibrated descriptions
help readers remains the later held-out empirical question.
