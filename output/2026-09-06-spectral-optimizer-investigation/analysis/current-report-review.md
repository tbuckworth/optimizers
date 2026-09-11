# Current report review — 7 September 2026

## Scope and scientific review

Reviewed the new cohesive synthesis against the original investigation's
final correction record, all six focused continuation result reports, and the
two independently reviewed September 7 mathematical reports. This is a
document-consistency review, not a repeated raw-data audit or new experiment.
The report-writing leaf owned only the new Markdown; main owned integration
and publishing. A separate mathematical reviewer examined the high-risk claims.

The following boundaries pass review:

The independent mathematical reviewer raised one clarity correction: the
paired-batch cross-moment identity must use the fixed pre-observation observer
mean, with conditionally independent gradients from the same fixed-training-
objective batch distribution. Main explicitly defined m=m_(t−1), d=bar_g−m,
and that conditioning. The reviewer otherwise passed the scientific boundaries.

Main also made the ideal/native recurrence distinction, orthogonal-projector
MSE assumptions and independent signal/noise assumptions explicit; specified
the local smoothness condition for the Hessian identity; added the uniform-
prediction antiparallel-mean connection; and limited proposed one-step falsifiers
to their actual estimands. Forty-eight malformed inline-math delimiters in the
draft were corrected without changing their mathematical content. Direct
companion citations and the figure's aggregation/denominator caption were added.

Existing primary-source claims were reused, not expanded into a new literature
survey. Additional source checks confirmed the stated scope of
[Song, Ahn and Yun](https://arxiv.org/abs/2405.16002) and
[Feldman](https://arxiv.org/abs/1906.05271).

Reviewed current narrative SHA-256:
`d9f65126ef9d8c3f223cac15caf4148349fe5500f49b6d72c9a99b8b0e4f407a`.

## Build review and validation

The best-practices check is in [the plan](current-report-plan.md). A separate
builder reviewer verified that imports load helper definitions only and that
all writes resolve to the new edition. The reviewer found that freshness
checking initially compared output basenames only; main changed this to exact
current-edition paths. Mock-only checks accepted a matching in-memory manifest
and rejected historical-edition output paths and a changed source hash. These
tests did not touch the actual manifest or scientific evidence.

Final compilation uses cached Pandoc 3.9 and Tectonic 0.15.0. It produces a
26-page, 387,334-byte PDF with Codex/project metadata. The manifest records
three embedded source documents, one retained figure, both builders and all
four deliverables. Read-only `--check-sources` passes. The compile reports no
missing characters, overfull boxes or unresolved references, and the link
rewriter reports no missing local files.

Main inspected the cover, main comparison table, figure/caption, main equations,
limitations table and appendix retention table. The initial equal-width table
wasted space in its identifier column; presentation-only width assignments
corrected it and reduced the document from 27 to 26 pages. Main rechecked the
final comparison table and paired-moment/MSE/kernel equations after rebuilding.
Extracted PDF text preserves the main narrative, both appendices and source links.

## Preserved first edition and research state

Read-only checks confirm the following unchanged SHA-256 identities:

- Original narrative:
  `5de9944ab65afd753a4dc235c5190f0e43eb25ca13a53672ed31d1c275cdb4a9`.
- Original delivered PDF:
  `8aabbb785e9b0c6eaa90ccd1a026caa056653080cb02a528f482892dfea0fecb`.
- Original presentation helper:
  `bf12d4692a353fe565d8095307b2657ca01530322c14c861df223240616fbc2b`.

The frozen I7 worktree is clean. The same failed service invocation remains
terminal with MainPID 0 and exit 2; the two-hour reminder is enabled and active.
No experiment process was found. New CPU preparation was asked about separately
and remains unapproved at this report cut. The broad goal remains active.
