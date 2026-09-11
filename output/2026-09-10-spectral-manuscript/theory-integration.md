# Conditional-theory paper integration

Codex · 10 September 2026 · reporting/typesetting only

The working PDF (artifact not distributed in this public snapshot) has **22 A4 pages**: the eight-page
main paper, existing appendices on pages 9–16, new Appendix E on pages 17–19,
source map F on page 20, and references on pages 21–22. This is a working-draft
consolidation, not a submission, new measurement or novelty certification.

## Coherent addition and evidence scope

The new appendix links two distinct questions: what augmentation optimizes,
and what gradient covariance can reveal about feature geometry. Signed update
utility connects them without equating retained energy with useful learning.

- Page 17 gives the exact mean-logit CE plus reverse-KL identity, the fixed-label
  soft-target/assignment/consistency decomposition and its differentiated
  gradient. It preserves the consistently-wrong-predictor boundary and the
  signed-utility counterexample, including the actual-Adam displacement caveat.
- Page 18 states the frozen-last-layer, uniform-prediction, fresh-uniform-label
  assumptions for the Kronecker covariance. The feature factor is an uncentered
  second moment. Whole class-contrast blocks yield the feature projector;
  degeneracy does not identify semantic clusters.
- Page 19 gives partial-replacement mean/covariance and paired favorable/adverse
  binary algebraic examples. The table is explicitly not observed training.
  Complementarity remains possible, not demonstrated. The nonuniform-prediction
  additive covariance statement is conditioned on fresh uniform labels, not
  generalized to partial corruption or fixed labels. Temporal/full-network/Adam
  limits and the absence of a truth oracle remain explicit.

Inputs were the full [objective note](../../research/spectral_augmentation_loss_geometry_2026-09-10.md),
its [independent derivation](../2026-09-10-spectral-augmentation-theory/independent-math-review.md)
and [review](../2026-09-10-spectral-augmentation-theory/review.md), the full
[feature note](../../research/spectral_feature_geometry_2026-09-10.md), its
[review](../2026-09-10-spectral-feature-geometry/review.md) and
[source review](../2026-09-10-spectral-feature-geometry/source-review.md), plus
the current manuscript and its prior integration receipt. The report-agent
guidance was used for attribution and evidence scope, not to restart a research
workflow or audit completed experiments.

All sixteen old bibliography entries are byte-preserved. Six appended entries
exactly match [theory-priors.bib](theory-priors.bib), with metadata and reading
scope in [theory-source-registry.md](theory-source-registry.md). All twenty-two
unique keys are cited and all local `\src` links resolve. These checks do not
claim six new proof audits.

## Preservation and build checks

From `paper/`, the existing command exited successfully:

```sh
tectonic --keep-logs --keep-intermediates manuscript.tex
```

`pdfinfo` confirms 22 pages and **883,262 bytes**. No undefined citations/references
or overfull boxes appear in the final logs. The pre-existing BibTeX change-detector
consistency warning reaches the six-pass limit; the intentional empty TAGD year
and one underfull paragraph at TeX lines 249–250 remain. References resolve;
this is not a warning-free-build claim. `git diff --check` passed.

Visual inspection covered pages **2, 8, 17, 18, 19, 20, 21 and 22**, including all
new equations, the analytic table, source map, references and main boundary.
No clipped text, overlapping table content or stranded heading appeared. Each
theory subsection starts on its own page; page 18 intentionally leaves whitespace
instead of splitting the constructive example. No uninspected unchanged page
is claimed to have received a new visual review.

Inspection rasters are outside the repository in
`/tmp/spectral-experiment-artifacts/spectral-paper-theory-render.UGdupv/`, named
`page-02.png`, `page-08.png`, and `page-17.png` through `page-22.png`, at 110 dpi.
They are formatting checks, not new outcome plots.

## Artifact identity

The prior 18-page PDF remains in git at `6a02b47`, SHA-256
`7dfe2975b53cbc2617bbb0dc8224920bc35888d58b41cf5862789bebc6615bdf`.
Final SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `paper/manuscript.tex` | `1d79f14afd602d44f44c3c46a238de106821464bfc7655798e68365f51d7ac74` |
| `paper/references.bib` | `e252b3d605daf5a047272af4c2c8c75ec01a3208d639beccc66506d0d18ef8d6` |
| `paper/manuscript.pdf` | `a19648ddc5dccc24053e7130e89e6a2dbd10503c0fe4077ffb9b7abc411a10be` |
| `paper/README.md` | `fca3e095d033325dc368334cf0ddc7f3ef1d96d274e98127960f3f8270425d7c` |

## Main acceptance — 17:34 UTC

Main read the complete added TeX and bibliography diff, checked the conditions
against both already-reviewed mathematical notes, and verified all four hashes
above independently. The fresh-uniform-label qualification on the additive
nonuniform-prediction covariance was tightened during review. Main also viewed
the actual rasters for pages 2, 8 and 17–22: equations, table, source map and
references are readable without clipping or overlap. `pdfinfo` independently
confirms the 22-page, 883,262-byte PDF. This accepts the bounded consolidation,
not the unobserved strong-regime result or a claim of mechanistic identification.