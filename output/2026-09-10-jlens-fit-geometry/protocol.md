# Saved-fit J-Lens geometry decomposition

Codex — Spectral Optimizer Investigation · 10 September 2026

## Question and scope

Do the four already decoded covariance directions mostly separate topic
means, individual content within topics, or the two framings of each content?
This tests the geometric possibility raised by the [mathematical note](../../research/jlens_covariance_interpretation_2026-09-10.md).
It does not establish clusters, semantic correctness or the cause of PC4's
poor reader performance.

**Exploratory, outcome-informed saved-data analysis.** The prior PCA fit,
readouts and fresh comparison have been inspected. This is not prospective
confirmation or a new independent sample. All four fixed axes and all 32 fit
rows stay in scope. No direction selection, sign change, PCA refit, decoding,
judge, model load, fresh acquisition or optimizer experiment.

## Inputs already available

All files are in the existing J-Lens worktree at
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree`, initially clean
at commit8253d73. Main inspected the fit-preparation producer and verified
these file hashes before commissioning the new source:

| File relative to that worktree | SHA256 |
| --- | --- |
| output/2026-09-10-j-lens-fresh-content/preparation/directions.npz | 47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa |
| output/2026-09-10-j-lens-fresh-content/preparation/selection.json | aac75625b67d9a90ba351d979f3e1a77cd75f69706a8b3113e071630de023f4d |
| output/2026-09-10-j-lens-completions/dataset.json | b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6 |

The 95,532-byte NPZ already contains `fit_scores64`, a 32×4 float64 array
computed as (fit_h−saved_mean64) @ saved_u32.astype(float64). Those exact
canonical directions also supplied the fresh comparison's decoding/scoring.
Selection JSON supplies the exact ordered fit IDs; dataset JSON supplies
topic, content-pair, framing and original prefix. Thus **no activation
reprojection is necessary**. Read only that numeric NPZ member and metadata;
do not load model weights, the old full feature archive or other NPZ members.
This is a direct algebraic equivalent of decomposing the covariance projected
onto each fixed direction, not a substitute score.

Require all 32 exact fit IDs in order: four topics (astronomy, cooking,
football, programming), four content pairs per topic and both plain/note
framings per content. The original dataset also contains held-out rows;
they supply no projected scores or contributions here. No original experiment
script is imported or rerun. Pin-check inputs before and after the calculation.

## Fixed quantities

For each unchanged axis, let zᵢ be the saved score, g(i) its topic and c(i)
its content pair. Bars denote means within the stated set. With N = 32:

```text
V = (1/N) Σᵢ (zᵢ − z̄)²                         total variance
B = (1/N) Σᵢ (z̄_g(i) − z̄)²                    between topics
Q = (1/N) Σᵢ (z̄_c(i) − z̄_g(i))²               content within topics
F = (1/N) Σᵢ (zᵢ − z̄_c(i))²                   framing within content
V = B + Q + F
```

Report raw V/B/Q/F, their fractions of V, and the numerical reconstruction
residual on every axis. Fractions are undefined/null if V = 0. Use population
moments / divisor N consistently, not a mixture of unbiased group estimates.
Within-topic variance is Q+F. Nested mean-centering makes the cross terms
cancel; the decomposition is descriptive and exact up to arithmetic precision.

Also retain all 32 original scores with IDs/labels/prefixes, all four topic
means and all sixteen content means. For every content pair save the signed
note-minus-plain difference Δ_c, with its mean and RMS as secondary descriptive
summaries. Because each content has two equally weighted framings,
F = mean_c(Δ_c²)/4. This identity supplies an independent check.

“Framing” means this observed two-prefix difference, including any
content-by-framing interaction. F is not a pure universal style effect,
measurement noise, or evidence that those directions lack useful information.
The design does not estimate the full mean-subspace invariance condition or
the covariance spectrum outside the retained four directions.

## Expected interpretations, not decision thresholds

- Strong between-topic shares on the first three directions with a smaller
  share on PC4 would support the proposed fit-geometry distinction.
- Substantial between-topic variation on PC4 would limit a “just within-topic”
  account. Retain that result rather than redefining the axis.
- Dominant F would implicate sensitivity to the tested prefix framing in
  this fit population, not prove why fresh plain-prefix judgments failed.
- Dominant Q would indicate differences among these contents; it would not
  label them memorized facts or meaningless variation.

No significance test, confidence interval, semantic reassessment, post-hoc
sign correction or new performance score is selected. Topic labels are
authored and few; sixteen contents are not 32 independent examples. PCA was
fit on these same rows. The completed fresh grade/report stays unchanged.

## Implementation and verification boundary

The Astra worktree implements only the new import-inert producer and small
fabricated tests first. Main reads all source and checks known pure-between,
pure-content, pure-framing, mixed and zero cases, sign/scaling invariance,
metadata/shape/nonfinite failures and held-out exclusion. Then one bounded CPU
call may read the three pins and write a new exclusive attempt/output/receipt.
An independent saved-output arithmetic audit is verification, not a new
acquisition or second primary analysis. Resource use is negligible compared
with inference; no paid reservation or cloud service is needed.

The implementation-check skill prompted explicit safe archive handling and
normalization review. Current official [NumPy load documentation](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
supports `allow_pickle=False` and closing NPZ archives with a context manager.
[NumPy variance documentation](https://numpy.org/doc/stable/reference/generated/numpy.var.html)
defines divisor N−ddof and dtype behavior. Use float64 and explicit population
moments; no new version-specific API or installation is required.
