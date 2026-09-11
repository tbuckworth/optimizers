# Current report edition — 7 September 2026

## Scope

The canonical new source is
`../../../research/spectral_optimizer_current_synthesis_2026-09-07.md`.
The separate PDF edition belongs in `../paper-current-2026-09-07/` and embeds
the current synthesis plus the two mathematical reports as detailed appendices.
Other experimental artifacts remain linked companions, not purported embedded
raw evidence. The broad continuing goal remains active after this publication.

## Division of work and review

A bounded report-writing leaf writes only the new canonical Markdown. Main
checks source lineage, adapts the existing document builder without changing
it, reads the complete draft, and independently reviews the scientific content
with a second leaf. No leaf launches experiments or changes authority.
Check retained contradictions, conditional assumptions, numerical provenance,
all local links, PDF text/layout, build hashes, and knowledge lint before commit.

## Build implementation validation

The best-practices validation uses the current official
[Pandoc manual](https://pandoc.org/MANUAL.html) and
[Tectonic V1 CLI documentation](https://tectonic-typesetting.github.io/book/latest/ref/v1cli.html),
and installed `tectonic --help`. Pandoc's JSON intermediate format,
standalone LaTeX, header inclusion, document variables and contents depth
support the existing conversion approach. Tectonic documents retained logs,
intermediates, an explicit output directory and cache-only resource use.

Reuse `../paper/build_report.py` parsing/link/style helpers, but never its
historical hard-coded main routine or output directory. A small new edition
wrapper supplies current metadata and the exact three-source list. Hash both
wrapper and helper, source Markdown, figures, and generated deliverables.
Generate LaTeX for inspection, retain compilation diagnostics, fail on missing
source paths/links, and visually inspect representative tables and equations.
The installed cached tools are Pandoc 3.9 and Tectonic 0.15.0; no scientific
dependency installation is needed. Cache-only compilation will expose missing
typesetting assets explicitly rather than changing the experiment environment.

## Authority boundary

The failed I7 service and consumed empty measurement are immutable. A separate
asynchronous question asks about a newly scoped CPU preparation attempt; no
answer is present at this plan's creation. Report compilation is not training
or approval for that attempt.
