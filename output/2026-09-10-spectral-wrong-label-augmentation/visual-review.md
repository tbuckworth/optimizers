# Inherited ordinary-translation visual review — PASS for unchanged transform

Codex · 10 September 2026 · before acquisition

This extension uses the exact unchanged integer-translation function already
reviewed in the [ordinary experiment](../2026-09-10-spectral-general-augmentation/visual-review.md).
The prior fixed30-image training-only preview and pixel-loop oracle showed the
intended ±2-pixel zero-fill translation, with no interpolation or wrapping.
No further transformation tuning or image exclusion is selected here.

New seeds select new train/held-out pools and random shifts. This is inherited
review of transform behavior, not a claim that every new view was inspected or
every cropped digit is unambiguous. Deliberately wrong assigned labels are the
new intervention; they are not a transformation labeling bug. The full pinned
raw image/true-label sources and actual corruption/view plans are retained.
