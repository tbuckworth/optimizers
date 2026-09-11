# Source and mathematical review

Codex — Spectral Optimizer Investigation · 11 September 2026

**PASS for the bounded saved-panel calculation.** Read the complete 98-line original protocol and 235-line producer, checked the source contracts of the existing preparation/fresh-feature exporters, and ran only the fabricated `self-test` under CUDA-hidden, single-thread environment settings and a 30-second timeout. It passed. No real scientific array, run mode, model, grader or reader was accessed or executed by this review.

## Finding corrected before calculation

The original [protocol](protocol.md), lines 46–49, said that the mapped slope simplifies to `Lu/q` “only” for an eigenvector. That necessity is true for the input-space equality `d=u/q`, given positive variance, but not for a particular map `L`: it may annihilate `d−u/q` even when that difference is nonzero. Main corrected this at commit `b426e7e` before array access. The amended wording is accurate; no producer change was needed.

## Checks supporting acceptance

- **Nonunit regression and cosine:** with `q=uᵀu>0`, `V=uᵀCu>0` and `d=Cu/V`, one has `uᵀd=1`. Thus `d−u/q` is perpendicular to `u`, and `relative_off_axis² = q‖d‖²−1 = 1/cos²−1`. The producer's slope, reference vector, cosine and identity checks implement these quantities correctly ([analyze.py](analyze.py), line 50). Centering each panel changes the regression intercept, not exact-arithmetic pair gaps; the separate original-mean projection check retains the existing canonical score definition.
- **Variance accounting:** all panels consistently use divisor `n`. Between-group variance is `Σ(n_g/n)·mean(z_g)²`; within-group variance is `Σ(z_i−mean(z_g))²/n`. Their sum is total centered score variance, including unequal group sizes. `V/(q·trace(C))` is the normalized-direction variance fraction. The four-by-four correlation matrix uses the same centered score covariance. These are group-label decompositions, not evidence that groups are semantic clusters.
- **Fixed selection and source identities:** the fit mask retains exactly the predefined first four contents per topic and both framings, 32 of the original 48 rows; no score-based selection occurs. All 24 authored and all 24 Wikipedia rows must match their fixed ID/order and category roster. Source contracts agree on `activation_11`, the actual normalized float32 `u32`, `source_mean64` and float64 saved scores. The run checks exact shapes/dtypes, near-orthonormality without changing `u32`, the fit mean, and saved original-mean projections ([analyze.py](analyze.py), line 135). Hashes are checked before and after computation. This is source-level acceptance of those checks, not a fresh verification of the pinned arrays.
- **Numerics and failure handling:** zero/nonfinite/wrong-shaped fixtures are rejected; no artificial cosine is assigned. Fabricated diagonal, nonunit, off-axis, translation, scaling and unequal-group examples pass. The scalar `math.fsum` path independently recenters and rebuilds every `CU` element plus score and between-group variances at the stated tolerances. This is a second arithmetic implementation inside the producer, not independent empirical evidence or an independent-author result audit.
- **Once/resource scope:** `analysis/` is created exclusively before scientific input loading; output files also use exclusive creation. Exceptions preserve that consumed directory and a failure record where the handler is reached; interruptions still leave the directory consumed. Inputs are fixed local hash-pinned files with 16 MiB file-size bounds and named, non-pickle NumPy members; no dense covariance, eigendecomposition or network/model access is performed. The small three-panel calculation is consistent with the proposed CPU/time envelope, but thread and timeout limits are invocation responsibilities, not enforced internally. Main must retain the specified bounded invocation. No run mode was exercised here.

The interpretation limits are appropriate: empirical collinearity is neither semantic accuracy nor a causal explanation of PC3/PC4 outcomes; a fixed readout can suppress or amplify off-axis components. Three reused panels with different source/length/endpoint properties do not isolate those factors. No additional test or follow-on analysis is required by this review.

## Reviewed pins

| Artifact | SHA256 |
|---|---|
| Unchanged producer | `ed6a7671845b6c6a7171e87de961168b2a0317faf6e02963f1c5610f1c3b7c6d` |
| Original protocol reviewed in full | `2c09e3847f0ac02ce61a20cd4deb980ffed968b41257f8f1ad695ddee857cce1` |
| Corrected protocol accepted | `083de3a8c50abe6a0ae7d317f805515e2ae13817cfa6a649f1922cbf7fd91412` |

The change-review guidance was applied to correctness and scope; no style-only changes, producer edits, result computation, independent audit program, or new workflow gate was introduced.
