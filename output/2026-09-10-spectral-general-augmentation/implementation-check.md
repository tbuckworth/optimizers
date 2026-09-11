# Implementation check before acquisition

Codex · 10 September 2026 · prospective, no model results

**Assessment:** aligned with the installed framework and the intended pairing.
No extra dependency or optimizer change is needed.

- Local versions verified: PyTorch 2.11.0+cu128, NumPy 1.26.4. Explicit seeded
  NumPy generators and deterministic PyTorch settings follow the version-matched
  [reproducibility documentation](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
  This controls this fixed platform, not bitwise reproducibility across releases.
- Explicit AdamW arguments preserve the accepted single-tensor implementation
  and decoupled weight decay. `foreach=False`, `fused=False` avoid an implicit
  CUDA implementation selection; see [AdamW 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).
- Exact integer translation with zero padding is implemented directly on NumPy
  pixels. It has the horizontal/vertical integer-displacement and outside-fill
  semantics described in [Torchvision affine](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.affine.html),
  without adding Torchvision or interpolation to the acquisition dependency graph.
  A separate pixel-loop fixture oracle checks every displacement.

**Concerns addressed:** augmentation RNG cannot consume the example-sampling RNG;
augmentation must start inside the spectral warmup, not afterwards; pairing at
step 100 is within augmentation mode only; no hidden multiview compute; no
best-checkpoint selection; preserve absolute outcomes beside interactions.
Remaining concerns are scientific boundaries, not implementation failures:
label preservation is plausible rather than guaranteed, and a fixed hard-rank
recipe may underfit. Training-only visual review is required before admission.
