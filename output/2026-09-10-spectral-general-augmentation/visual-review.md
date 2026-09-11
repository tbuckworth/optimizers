# Training-only transformation review — PASS

Codex · 10 September 2026 · before any optimizer acquisition

Inspected all three full contact sheets: first (artifact not distributed in this public snapshot),
second (artifact not distributed in this public snapshot), third (artifact not distributed in this public snapshot). They show the prospectively
selected first three training IDs per class for seed 202609141, each original
plus eight extreme/cardinal translations. Directions, zero-filled borders,
and fixed image intensity scale agree with the specified transform. No wrapping,
interpolation blur, rotation, or reflection is visible. The handwriting remains
plausibly the same digit under translation; some original handwriting is already
ambiguous. This is a plausibility check, not a guarantee for every image.

Numerical input check (artifact not distributed in this public snapshot), all 5,000 training images × all 25 shifts:
mean per-image lost-intensity fraction **0.053118%**, largest individual loss
**6.876851%**. This measures cropped pixel mass, not semantic information loss.
No examples were excluded or labels changed; no transform strength was tuned.
No held-out image outcomes or optimizer/model results were inspected.

Decision: retain the prospectively selected ±2-pixel zero-filled translation.
This review and the exact preview artifacts are bound into the source freeze.
