# Retention identifiability: review and source record

7 September 2026. **PASS — theory and post-hoc retained-scalar analysis.**
No new training experiment; no historical experiment or audit reexecution.
Disposition: **NO-EXPERIMENTS for this follow-up**; existing I5 evidence was
analyzed as retained scalar records, not reexecuted.

## Independent mathematical review

A separate leaf reviewer independently checked the categorical label covariance
and its three invariant label-space eigenspaces, the sufficient matched-state
Hessian identity, uniform-prediction antiparallel mean vectors, and the sharp
projector/general-linear-action retention bounds. The assembled
[report](../../../research/spectral_label_noise_identifiability_2026-09-07.md)
also passed review with no material correction.

The review explicitly distinguishes fresh labels from the fixed assignments
used to train the model; per-input label covariance from temporal batch
covariance; expectation of a vector from expectation of its normalized
retention; and ideal projectors from native finite-precision action. Degenerate
replacement rates and fixed-rank endpoint cases are qualified. The observed
cosines do not force tiny gaps or establish causal label smoothing.

## Independent retained-data check

The reviewer used only standard-library scalar/JSON arithmetic and independently
verified:

- all 12 seed/state pairs and all 24 width records;
- original snapshot sizes and SHA-256s, their equality to the copied raw records,
  and the committed-summary hash;
- input energies/cross terms shared between widths, cosines and ideal ceilings;
- native retention ratios, signed gaps and absolute-gap/ideal-ceiling ratios;
- all-state seed-first and final-state aggregates, extrema and saved diagnostics.

Maximum absolute discrepancy in every recomputed category: **0.0**. Original
evidence is unchanged. The calculation (artifact not distributed in this public snapshot)
and derived JSON (artifact not distributed in this public snapshot) retain all inputs and
rows. The quotient is a descriptive comparison to ideal geometric headroom,
not a certified fraction of a native maximum or a learning benefit.

The panel is post-hoc and reuses three AdamW trajectories. Historical reported
cosines/gaps were already known before choosing this analysis. No prospective
or independent learning-confirmation claim is made.

## Primary literature and citation registry

- szegedy2015inception: [Szegedy et al., section 7](https://arxiv.org/pdf/1512.00567)
  explicitly connect expected random replacement with a mixed soft target.
  Main read section 7 in the primary PDF; no architecture-performance claim
  is used.
- muller2019smoothing: [Müller, Kornblith and Hinton, NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)
  report label-smoothing calibration and representation effects. Main read the
  primary proceedings abstract; no transfer to this filter is inferred.
- lukasik2020smoothingnoise: [Lukasik et al.](https://arxiv.org/abs/2003.02819)
  connect smoothing to label-noise correction and report competitive results.
  Main read the primary abstract; no theorem beyond it is attributed to the paper.

This is a focused source comparison, not a systematic search, novelty claim or
SOTA survey. The covariance spectrum, Hessian qualification and retention-bound
derivations are presented as independently checked mathematics, not novel
published theorems.

## Workflow, validation and scope

Knowledge changes are limited to durable interpretation and the next proposed
measurement. The previous mathematical entry in the knowledge log was
mistakenly appended at the bottom; this change restores newest-first order
without changing or deleting its content. The original delivered report/PDF
is untouched.

Final document checks pass: knowledge lint (14 pages, 11 indexed content pages),
105 local Markdown links, derived JSON parsing, three bibliography/registry
entries, and final diff whitespace. The first staged check caught four trailing
EOF blank lines that the earlier unstaged check did not cover; those were
removed in a formatting follow-up to research commit79e1f85. No numbers or
scientific interpretation changed. The previous log entry is verified preserved
verbatim and moved above older entries. Recovery checks found the immutable
checkout clean at its original revision, optimizer/M hashes unchanged, the same
failed service invocation with MainPID=0, and the reminder enabled/active.
The broad goal remains active.
