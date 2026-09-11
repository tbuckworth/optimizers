# Saved-panel covariance transfer diagnostic

Codex — Spectral Optimizer Investigation · 11 September 2026

Exploratory analysis selected after the previous ordering results were known.
This is not preregistered confirmation or another semantic test. Existing
results and reports are immutable. No new model, decoding, PCA, reader,
training, corpus acquisition, resampling or paid compute is selected.

## Question and fixed scope

Does an unchanged fitted direction remain close to an empirical covariance
eigenvector on the two already measured fresh panels? This checks an assumption
behind interpreting its readout as a population regression contrast. It cannot
establish that covariance change causes the observed accuracy pattern.

Use all four canonical float32 columns U, promoted to float64 without changing
their values. Compare exactly three populations of saved layer-11 activations:

1. Original 32 fit rows (16 contents, two framings); exclude the other 16 rows
   by the saved fit-ID roster, not by activation or score.
2. All 24 authored fresh prefixes, in original order.
3. All 24 Wikipedia prefixes, in original order.

Keep the same source category/topic labels. They are sampling strata, not
independently validated semantic clusters. The two fresh panels differ in
length, grammar, endpoint, source and reader design. Fit is in-sample and
has duplicated content with alternate framing; panel counts are not independent
replications. The canonical references and previous grades are not changed.

## Fixed calculations

For each panel H with n rows, use float64 arithmetic and population divisor n:

```text
X = H − mean(H)
Z = X U
CU = Xᵀ Z / n                 [no dense covariance or eigendecomposition needed]
V_a = mean(Z[:,a]²)
q_a = u_aᵀu_a
d_a = CU[:,a] / V_a          [regression slope of h on fixed-direction score]
cos_a = V_a / (sqrt(q_a) ||CU[:,a]||)
relative_off_axis_a = ||d_a − u_a/q_a|| / ||u_a/q_a||
```

The exact slope through any fixed linear map L is Ld_a. With positive variance,
d_a = u_a/q_a if and only if u_a is an eigenvector; this implies Ld_a = Lu_a/q_a.
A particular L can also give that equality off-axis by annihilating the
difference. The cosine is an input-space collinearity
diagnostic, NOT semantic accuracy, vocabulary reliability, or closeness after
L. L may suppress or amplify off-axis components. Unknown population covariance
and low-rank noisy empirical estimates qualify any transfer interpretation.

Also retain, for every axis and panel: variance V; V divided by fit V;
V / (q_a trace(C)) as the variance fraction along the normalized fixed direction;
topic/category between-mean variance divided by V; within-group complement;
the complete 4×4 score correlation matrix. Group means are weighted by observed
row count. These are descriptive decompositions, not attribution of individual
paired decisions. No coordinate reorientation or rank ordering is selected.

Retain full CU, d, centered scores and panel means in the new analysis archive.
Check canonical saved scores against projections using the original mean;
centering each panel is only for covariance and does not change pair gaps.
Reject nonfinite/wrong-shaped inputs and nonpositive axis variance; preserve
any failed attempt without automatic rerun. Zero variance is covered by tests
and is not assigned an artificial cosine.

## Implementation and checks

Input digests are pinned in `analyze.py` from existing committed receipts.
Before loading scientific arrays, freeze this scope and the source with
fabricated tests and a separate source/mathematical review. One exclusive new
analysis directory; one CPU, CUDA hidden, 60-second timeout. No access to
credentials, model weights, network or former execution entry points.

Fabricated tests cover exact diagonal covariance, nonunit u, a known off-axis
covariance, translation invariance, consistent row scaling, weighted group
decomposition and zero-variance/nonfinite failures. During the actual new
calculation, independent scalar math.fsum reconstructions corroborate all CU
components, score variances and between-group variance (relative 2e-11,
absolute 2e-12). These are arithmetic checks, not independent empirical data.

The best-practices-validator guidance prompted official NumPy 1.26 checks:
[load](https://numpy.org/doc/1.26/reference/generated/numpy.load.html)
uses `allow_pickle=False` and a context manager;
[mean](https://numpy.org/doc/1.26/reference/generated/numpy.mean.html)
requires explicit float64 promotion for float32 activations;
[matrix multiplication](https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html)
is used with checked two-dimensional shapes. Existing arrays are small, bounded
and hash-verified; no generic untrusted archive loading is introduced.

## Decision use

If both fresh panels have large off-axis components, the fit-only eigenvector
simplification is not a general transfer explanation. If PC3 and PC4 both drift,
this diagnostic does not selectively explain why one succeeds. Conversely,
near-collinearity would not prove useful semantics: residual vocabulary variance
and the fixed readout can still mislead. Inspect successes and failures as
retrospective hypotheses only. Choose a subsequent content-versus-endpoint test
only after integrating this calculation with the complete saved examples.
