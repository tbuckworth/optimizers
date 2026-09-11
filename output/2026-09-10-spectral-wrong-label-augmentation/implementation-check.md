# Prospective implementation check

Codex · 10 September 2026

The unchanged translation, AdamW, deterministic framework configuration and
bounded writer are reused from the accepted ordinary-augmentation run, rather
than changing the optimizer. Its [implementation check](../2026-09-10-spectral-general-augmentation/implementation-check.md)
records the primary version-matched [PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html),
[AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html) and
[translation semantics](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.affine.html)
documentation verified earlier in this continuation. No new external API or
dependency is introduced. Framework/platform determinism is not a claim of
cross-version reproducibility.

The new code must change targets, not their identity across augmented views.
Independent RNG streams keep corruption from changing the existing sampling
and transformation draws. Exactly80% actually wrong is deliberately distinct
from nominal80% random replacement. Clean held-out metrics and wrong-target fit
have separate labels/denominators. Source review and fixture results are
recorded in preflight.md before admission; this design note is not itself a
claim that unreviewed code has passed.
