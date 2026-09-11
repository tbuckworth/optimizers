# Research index

Notes, write-ups, and interactive figures from the project. Start with the
current synthesis below; historical write-ups remain supporting evidence.

For the maintained cross-experiment synthesis, decisions, and open questions,
start with the [`knowledge/` index](../knowledge/index.md). This page remains the
catalog of experiment reports and figures.

## Start here

- [September 11 final core report](spectral_final_core_report_2026-09-11.md)
  — the conclusion, clearest paired evidence, mathematics and limitations.
  [HTML reader](../output/2026-09-11-spectral-final-core/reader.html) ·
  [PDF](../output/2026-09-11-spectral-final-core/report.pdf).
- [Public snapshot scope](../PUBLICATION.md) — included evidence, omissions and
  reproduction limitations. The earlier synthesis below is historical.

- [September 7 current synthesis](spectral_optimizer_current_synthesis_2026-09-07.md)
  (PDF (artifact not distributed in this public snapshot))
  — integrated cross-repository findings, completed continuation experiments,
  mathematical interpretation, contradictions and next tests.
- Original September investigation (artifact not distributed in this public snapshot)
  — preserved first-delivery narrative with discovery and application details.
- [Historical spectral-filter blog](spectral_filter_blog.html) — the earlier
  self-contained methodology, H1–H7 hypotheses and interactive figures.

## How the filter works (explainers)

- [`rank1_svd_explainer.html`](rank1_svd_explainer.html) — the streaming rank-1 SVD covariance update.
- [`moments_centering_explainer.html`](moments_centering_explainer.html) — why the covariance is mean-centred.
- [`spectral_consensus_worked_example.html`](spectral_consensus_worked_example.html) — a small worked example of the "consensus" projection ([PDF](spectral_consensus_worked_example.pdf)).
- [`projection_example.html`](projection_example.html), [`projection_deep_dive.html`](projection_deep_dive.html) — the projection step, visualized.
- [`methodology.html`](methodology.html) — overall experimental methodology.

## Findings (markdown)

- [September 8 stable/legacy grokking confirmation](grokking_stable_confirmation_2026-09-08.md)
  — five fresh paired seeds, all-seed plots, first versus sustained generalization,
  measured timing, and the exploratory legacy within-span gain mechanism.
- [`weight_covariance_v2_summary.md`](weight_covariance_v2_summary.md) — **best single summary** of the v2 filter results.
- [`weight_covariance_v2_findings.md`](weight_covariance_v2_findings.md) — detailed v2 findings.
- [`weight_covariance_directions.md`](weight_covariance_directions.md) — the follow-up research directions (A/B/C).
- [`grokking_v2_findings.md`](grokking_v2_findings.md) — grokking / modular addition (H2).
- [`targeted_ablation_findings.md`](targeted_ablation_findings.md) — backdoor ablation (H6).
- [`subspace_baselines_findings.md`](subspace_baselines_findings.md) — random-subspace control baselines.
- [`findings.md`](findings.md) — early findings log.
- [`literature_review.md`](literature_review.md) — background reading (Lion/Muon, random labels, generalization theory).

## Per-experiment interactive figures

These are one-off result viewers, each tied to a specific experiment (label
noise, grokking, OOD, optimizer comparison, diagnostics, etc.):
`memorization_viz`, `label_noise_comparison_viz`, `grokking_visualization`,
`grokking_v2_viz`, `grokking_v2_seeds_viz`, `ood_visualization`,
`optimizer_comparison`, `momentum_annotated[_interactive]`, `spectral_results_viz`,
`subspace_baselines_viz`, `weight_cov_comparison_viz`,
`weight_cov_v2_comparison_viz`, `weight_covariance_diagnostic_viz` (all `.html`).
