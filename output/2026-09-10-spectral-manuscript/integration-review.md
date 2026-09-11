# Main manuscript integration and presentation review

Codex — Spectral Optimizer Investigation · 10 September 2026

## Artifact and scope

The following original review describes the 12-page pre-augmentation version.
The completed **15-page refresh** and subsequent safety-framing update are
recorded below; the last section contains the current artifact hashes.

Working paper (artifact not distributed in this public snapshot): 12 pages, comprising eight main-text
pages and four pages of methods, complementary evidence, source map and
references. It embeds three existing figures unchanged. It is an internal
working draft, not a submission or established-priority claim.

Main read the full LaTeX, checked the bounded prior note, independently opened
version-pinned DOME v1/v2 methods and the indexed TAGD algorithm, and inspected
the GaLore/Song source distinctions. The [independent evidence
review](evidence-review.md) checks direct reports and stored summary JSON;
neither review is a rerun or re-certification of the experiments.

Main inspected all 12 rendered pages. After final edits to the risk definition,
citation placement and source-map path convention, the affected pages were
rendered and inspected again. Equations, table, three figures and bibliography
are visible, with no observed clipping or missing glyphs. Raw plot data and
original plots were not changed.

## Corrections and preserved limits

- Clarified that projection removes an off-span **incoming-gradient component**,
  not necessarily Adam displacement; the scale identity is nondegenerate only.
- Defined clean risk explicitly and clarified the source map's short paths;
  expanded the iteration-017 source and version-pinned GaLore's bibliography.
- Preserved common-accuracy benefit versus almost unchanged common CE, warmup
  preservation versus new acquisition, rare/cue costs, useful grokking direction
  evidence, scalar-control successes and legacy/stable/elapsed-time limits.
- Retained reused/outcome-informed observer parents, training-probe versus
  held-out populations, explicit-zero Adam motion, and local—not endpoint—scope.
- DOME v2 uses within-batch per-example scatter, not native temporal batch means.
  DOME v1 is a different method. TAGD's indexed overlap remains provisional as
  a full-source/current-bibliography check, not ignored or silently certified.

## Build and identity

`tectonic --keep-logs --keep-intermediates manuscript.tex` exits 0. There are no
overfull boxes or unresolved references/citations in the final log. One underfull
spacing warning, the intentionally missing TAGD year, and a Tectonic BibTeX
change-detector/rerun warning remain; the PDF's citations were inspected rather
than equating exit status with a completely warning-free build.

Original reviewed SHA-256 (superseded by the refresh below):

```text
manuscript.tex  5bd38ecbe44085a7e0bdb4e280ac606f1922b00c3a2113564647fb1ed8be67c9
manuscript.pdf  d4ca896874d770f16b447a9883216a53d30035c9faba7495f23b4dc1cc639220
references.bib 5dc0c6becd3bb7c152e193da115c073e6b5929b5db1fd2ea1cb8af253a32ddeb
```

## Completed augmentation appendix refresh

The working PDF now contains **15 pages: eight main text and seven appendix/
source/reference pages**. Appendix C adds the completed three-seed augmentation
study; the source map moves to Appendix D. The main text includes a short
pointer, without replacing its three primary contributions. A fourth existing
PNG is included unchanged. No acquisition, saved-logit audit or plotting job
was repeated for this refresh.

Preserved limits: three paired seeds rather than nine independent seed/cells;
Random's limited cue coverage and smaller mean area; Targeted versus Opposite
matching gates/area but not digit content; the baseline-error contribution to
E and Sham contribution to Q; relative patched-image protection alongside
common/rare costs; rare CE improvement from warmup despite poor recognition;
frequency reduction rather than reliability removal; no measured mediation.

The final build exits 0. Main visually inspected **all 15 final rendered pages**
at 85 dpi, including the new table, figure and displayed mathematics; no clipping,
missing glyphs or unresolved references were observed. The existing underfull
spacing warning, intentional empty TAGD year and BibTeX change-detector/rerun
warnings remain. They were not suppressed or described as a warning-free build.
Temporary render directory:
`/tmp/spectral-experiment-artifacts/spectral-manuscript-render-20260910.OFee7t`.

Augmentation-stage reviewed SHA-256 (superseded by the framing update below):

```text
manuscript.tex  f43ed41b3487d0a5f96ef438434827b6d2d187ce3d7c4635597444dbd98f4d46
manuscript.pdf  d7e4f480a4f192148385cff45058e28f01784bd716a7faabce86d3d4e9515b96
references.bib 5dc0c6becd3bb7c152e193da115c073e6b5929b5db1fd2ea1cb8af253a32ddeb
```

The [augmentation continuation decision](../2026-09-10-spectral-augmentation/next-decision.md)
supersedes the original protocol-selection checkpoint. Both scientific jobs and
their report delivery are complete; the whole goal remains active. No new
experiment or public submission is selected by this document.

## Completed safety/contribution framing update

The [synopsis](../../research/spectral_safety_contribution_2026-09-10.md) is now
integrated into Introduction and Discussion. It leads with conditional control
of learning, including useful preservation, then distinguishes future safety
efficacy at maintained competence from present classifier evidence. Five added
primary-source bibliography entries support the motivation and theory scope;
the spectrum of an effective LoRA adapter is explicitly distinguished from
temporal-gradient covariance. There is no new empirical result or novelty claim.

Main's [source check and independent read-only framing review](../2026-09-10-spectral-safety-framing/source-review.md)
cover the changes. Build exits 0 with the previously documented spacing/BibTeX
warnings, no overfull boxes and no unresolved citations. The PDF remains
15 pages, eight main text. Main inspected all changed rendered pages (1–6,
8, 13–15); the remaining five PNG pages are byte-identical to the prior fully
reviewed render. All four original figures remain unchanged.

Current reviewed SHA-256:

```text
manuscript.tex  230f8ca505cf2417ecab71c943efba4e9a7c516c6164b432ed55efe1e5f59e27
manuscript.pdf  1193870f14faba2e3afe7e86cadb553703562bad73af6357532bac3677251ae8
references.bib 7a4bf277c61df32a9a848011348be3d5f371aa89dee4f6e0e1c9d1bd2cd3849e
```