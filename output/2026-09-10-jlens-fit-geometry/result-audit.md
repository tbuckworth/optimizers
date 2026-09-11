# Independent saved-score decomposition audit

Codex — Spectral Optimizer Investigation · 10 September 2026

**PASS.** The one authorized validation call verifies all 32 row identities,
labels and prefixes, all **128 score values exactly**, and **200 derived scalar
comparisons**. Independent Python-float/`math.fsum` arithmetic reconstructs
V/B/Q/F, fractions, topic/content means, framing shifts and both residual
identities within the tolerances fixed before the call. No result correction
is needed.

## Verified numerical result

Percentages below are shares of each unchanged axis's total population
variance V; rounding can make a displayed row differ slightly from 100%.

| Axis | Between-topic means B/V | Content within topics Q/V | Framing within content F/V |
|---|---:|---:|---:|
| PC1 | 61.67% | 38.13% | 0.20% |
| PC2 | 85.35% | 14.37% | 0.27% |
| PC3 | 10.56% | 88.99% | 0.45% |
| PC4 | 15.10% | 84.09% | 0.81% |

The first **two**, not first three, retained directions are dominated by
differences between topic means. PC3 and PC4 are both dominated by differences
among these contents within topics. PC4 has a larger between-topic share
than PC3, and framing is below 1% on every axis. This does not support the
proposed contrast in which PC4 alone is a within-topic or framing direction.

These are geometric shares on the already fitted authored panel, not new
semantic grades. Within-topic content variation is not thereby memorization,
meaninglessness or poor interpretation. Small framing variance is specific to
the tested plain/note prefix pair; it does not establish universal style
invariance or explain why the fresh readout judgments failed on PC4. No
cluster claim, grade revision or direction-sign change follows.

## Verification scope and method

The primary analysis completed once at 22:27:30 UTC. This audit did not import
or invoke its producer, tests or analysis entry point. It performed one
read-only numerical validation under a 45-second timeout, CUDA hidden and
OMP/OpenBLAS/MKL threads each set to one, using Python 3.12.3 and NumPy 1.26.4.
The validation body reported 0.00680 seconds; that is its internal elapsed
measurement, not a full process startup/runtime measurement.

NumPy was used only to open the pinned `directions.npz` with
`allow_pickle=False`, retrieve **`fit_scores64` once**, and check its 32×4
float64 shape/dtype. No other NPZ member was decoded. Every entry was then
converted to a Python float; all reconstruction arithmetic used scalar
operations and `math.fsum`, not the producer's NumPy reductions. No model,
checkpoint, other feature archive, reprojection, PCA, held-out score, fresh
acquisition or judge was accessed or run.

The checks establish:

- The original metadata contains the exact 48-row ordered roster, with fit
  content indices 0–3 and held-out indices 4–5. All 32 ordered fit IDs match
  preparation `selection.json` and the reported rows. Only those 32 scores
  enter the calculation.
- Every saved `id`, `pair`, `group`, `style`, `split` and `prefix` exactly
  matches the pinned original metadata. Four topics each have four content
  pairs, each with one plain and one note row. All note prefixes are exactly
  the fixed preamble followed by their corresponding plain prefix.
- All 128 JSON score values have the same IEEE-754 binary64 bytes as the
  corresponding canonical `fit_scores64` entries, including sign bits. Thus
  the audit verifies the saved canonical score identity; it does not replace
  it with older scores or reconstruct a new direction.
- For each axis, global, topic and content means were independently formed.
  V, B, Q and F were calculated by the registered row-weighted sums divided
  by 32. All four topic means, all sixteen content means and all sixteen
  note-minus-plain shifts were checked on every axis, as were the mean/RMS
  shift summaries and all B/V, Q/V and F/V fractions.
- Both identities were checked independently: V−B−Q−F and
  F−mean_content(Δ²)/4. The reported first residual was also checked against
  its own saved V/B/Q/F components. All components are nonnegative and all
  four variances are positive; zero-variance/null fractions are not exercised
  by these actual data.
- As a normalization cross-check, each V agrees with preparation's
  `fit_score_std_ddof1² × 31/32`, not the unadjusted squared sample standard
  deviation. No unbiased-group/population-divisor mixture was used.
- All three input hashes were checked before and after the calculation.
  The pinned primary result and its completion receipt also remained
  unchanged. The validation made no source or primary-output writes.

The script completed **2,283 checks**, including metadata/schema checks,
**128 exact score comparisons** and **200 scalar comparisons**. This count
includes JSON duplicate-key and structural checks; it is not a count of
independent observations or scientific tests.

## Predeclared tolerances and observed discrepancies

Each derived scalar comparison used

    abs(saved − independent) ≤ 2e−15 + 2e−13 × abs(independent).

Both saved and independently reconstructed residuals additionally had to
satisfy an absolute bound of **2e−14 in squared-score units**. Exact original
score comparisons used no tolerance.

| Check | Largest observed absolute value/difference |
|---|---:|
| Any derived-scalar saved-versus-independent difference | 4.440892098500626e−16 |
| Independent V−B−Q−F | 1.474514954580286e−17 |
| Saved V−B−Q−F | 9.790345617544105e−17 |
| Independent F−mean(Δ²)/4 | 0 |
| Saved F−mean(Δ²)/4 | 2.168404344971009e−19 |

The largest comparison used 4.673% of its allowed scalar tolerance. Tiny
differences in means and reconstruction residuals reflect the independent
summation order; they do not alter any reported share or conclusion. The
first and only validation call passed; no retry or primary rerun occurred.

## Immutable evidence and audit receipt

Worker base:
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/`.

| Artifact | SHA256 |
|---|---|
| `output/2026-09-10-j-lens-fit-geometry/analysis/results.json` | `aee0a38a1114d6e81edb437c18ee9c12769a36f3560c4b362bc9a25173303735` |
| Its `receipt.json` | `975b5d33229589d6c610bad74fe669fdcde78bb92e648248f1fb108f4cf8ed9f` |
| `output/2026-09-10-j-lens-fresh-content/preparation/directions.npz` | `47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa` |
| Its `selection.json` | `aac75625b67d9a90ba351d979f3e1a77cd75f69706a8b3113e071630de023f4d` |
| `output/2026-09-10-j-lens-completions/dataset.json` | `b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6` |
| Main [protocol.md](protocol.md) | `0b391bada1e20825165bfef6784575408acca0114f4ef53469d4289bd784b6cb` |

The independent, single-use verification script and its complete scalar
receipt are retained outside the repository in
`/tmp/spectral-experiment-artifacts/jlens-fit-geometry-audit.XP2MGu/`:

- validate.py (artifact not distributed in this public snapshot),
  SHA256 `d5541c131264398b8630342a77e2237d2d448b580d4715bb35541847d0f1d023`.
- audit-checks.json (artifact not distributed in this public snapshot),
  SHA256 `a5e3bd3ebdfaf1b5efc38197135110944db793fb26ee0081b8a844ce986126b9`.

Only this audit Markdown was authored in the main repository and selected
for commit. Primary analysis, reporting, plots, prior grades and all other
scientific artifacts remain unchanged by this leaf.
