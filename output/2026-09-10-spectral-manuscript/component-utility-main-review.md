# Main acceptance: component-utility paper integration

10 September 2026 — PASS for this working-draft documentation update

Main read the complete TeX diff against34dd503, README and
integration receipt (artifact not distributed in this public snapshot), and checked the new
material against the completed result/audit, reviewed interpretation and
frozen protocol. Corrected the shorthand tenth-path formula so that separate
FP32 subtraction/multiplication/addition is explicit; actual endpoint/FP64
accounting semantics and every number are unchanged.

Main inspected actual final PDF renders of pages22–26. The two-page E.4
addition has readable equations, all-seed full/tenth table, unchanged all-state
plot and interpretation limits, with no observed clipping. Source map and
references continue normally. Independently compared extracted text on
pages1–21 with the prior PDF: exact match. Main also byte-compared all21
corresponding prior/final page rasters: exact match. The prior PDF hash matches
the accepted strong-bridge version, not an unidentified local file.

- Final PDF:26 pages,1,366,220 bytes; SHA256
  `b5092ea555518ca9f7daf1aba32a55009293ef867e09c6f74c6b51714e899346`.
- Final TeX SHA256:
  `7d7066481ab6d6c6e824b1f575387e03f78199c251c68e602275d49da5546d13`.
- Prior PDF SHA256:
  `6e29b452f98dc073933b2b99a7334189c5932f11ca43158340d3a5fa31bc9e8f`.

The leaf's receipt retains the initial snap/tmp build failure, four successful
layout builds and preexisting BibTeX/TAGD/underfull warnings. Main checked the
final log; no undefined references or overfull boxes were found. The automatic
hook's intermediate ef2140d commit is preserved; it is not the final source/PDF.

Separately, the [action/history synthesis](../../research/spectral_action_history_synthesis_2026-09-10.md)
and its [bounded mathematical review](../2026-09-10-spectral-component-utility/action-history-review.md)
clarify an elementary telescoping identity, different estimands, seed reuse
and final-parent indexing. Main verified both reported scope corrections and
the full-state/sign/endpoints derivation. This note is not added to the PDF,
not a novel theorem, and not a measured history-mediated explanation.

Acceptance does not certify venue readiness, exhaustive novelty, independent
neural replication or completion of the broad autonomous research goal.
