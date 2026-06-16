# Research index

Notes, write-ups, and interactive figures from the project. **Start with the
canonical write-up; everything else is supporting detail.**

## Start here

- **[`spectral_filter_blog.html`](spectral_filter_blog.html)** — the canonical,
  self-contained write-up: methodology, every hypothesis (H1–H7), and figures.
  Open in a browser.

## How the filter works (explainers)

- [`rank1_svd_explainer.html`](rank1_svd_explainer.html) — the streaming rank-1 SVD covariance update.
- [`moments_centering_explainer.html`](moments_centering_explainer.html) — why the covariance is mean-centred.
- [`spectral_consensus_worked_example.html`](spectral_consensus_worked_example.html) — a small worked example of the "consensus" projection ([PDF](spectral_consensus_worked_example.pdf)).
- [`projection_example.html`](projection_example.html), [`projection_deep_dive.html`](projection_deep_dive.html) — the projection step, visualized.
- [`methodology.html`](methodology.html) — overall experimental methodology.

## Findings (markdown)

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
