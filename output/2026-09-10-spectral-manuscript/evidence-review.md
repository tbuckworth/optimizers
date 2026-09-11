# Manuscript evidence review — 10 September 2026

## Verdict and scope

No outstanding material numerical, mathematical, or replication-claim error was found in the reviewed evidence sections. The manuscript preserves the favorable results while separating them from stronger claims the experiments do not establish.

This was a bounded manuscript-to-evidence review, not recertification of the data. I read the methods supplement, compact core, manuscript, and the load-bearing reports and stored scalar JSON listed below. I did not load scientific tensors, recompute results, run experiments or audits, or edit the manuscript. Layout and the concurrent DOME/TAGD/new-prior integration are owned by the main agent; their literature claims are not independently certified by this review.

The last inspected manuscript snapshot had SHA256 `5bd38ecbe44085a7e0bdb4e280ac606f1922b00c3a2113564647fb1ed8be67c9`. Subsequent readability, citation, or literature edits are outside that snapshot.

## Findings and resolution

1. **Delivered gradient versus actual movement — resolved.** The initial phrase “removes off-span motion” could have implied that norm-restored projection constrains the Adam parameter step. The revised passage explicitly removes the off-span *incoming-gradient component*, requires the nondegenerate case with positive retained fraction, and distinguishes delivered gradients from actual Adam displacement. Read-back confirmed the correction.
2. **Source-map path convention — resolved.** The initial introduction called all paths repository-relative while several entries used study-relative short paths. The introduction now explains that convention, and the iteration-017 report path is expanded. Read-back confirmed both fixes. The inspected paths resolve to existing artifacts.
3. **Optional source-map improvement.** Grokking sustained-threshold, timing, and final-loss values are stored directly in `output/2026-09-08-spectral-paper-planning/grokking-confirmation-results/endpoint-supplement.json`. Adding that file alongside `summary.json` would shorten the evidence trail. The linked confirmation report already points to it, so this is not a missing-evidence blocker.

## Load-bearing checks

| Claim | Evidence check and retained qualification |
| --- | --- |
| Diffuse-noise selectivity | Native common accuracy is 62.01% versus raw 33.97%, with favorable paired gains in all three seeds. Rare accuracy is 0% versus 51.73%; wrong-target fit is 6.22% versus 32.48%. Common CE is essentially unchanged in the means, with mixed paired signs. These are preservation-heavy results after a strong common-class warmup, not broad new-class acquisition. |
| Shared cue | The patch-excess interaction is −3.358 percentage points, favorable in all three seeds, but native patch excess remains 94.825 points and shared wrong-label fit remains high. The paper does not convert this small relative improvement into cue rejection or safety evidence. |
| Same-exposure grouping | Native diffuse common accuracy rises from 47.92% to 56.64%; gains are 9.20, 10.07, and 6.91 points. The native-versus-raw interaction is +8.34 points. Rare recognition remains zero. The apparent rare interaction caused by raw deterioration is not presented as native rescue. Clean rare gains are dominated by one seed, and common-class costs remain visible. |
| Frozen-model observer pathway | Grouping improves relative clean rare one-step CE utility in all three reused parents: mean +0.003092, sample SE 0.000894. Diffuse mean utility is −0.000616, with one favorable and two adverse seeds. Observer clocks 150/151 are distinct from Adam clocks 100/101; 21 physical readouts cover 24 logical case readouts. |
| Saved signal accounting | Rare input alignment is positive in all six cases. Grouping changes rare delivered alignment positively in all clean cases but only one diffuse case; these grouping signs agree with the corresponding Adam and held-out utility signs. Other controls have reversals. The paper correctly avoids a mediation fraction, a retention claim, or treating the saved products as a new replication. |
| Grokking confirmation | Five paired seeds support earlier legacy sustained threshold crossings: mean 2,810 versus Adam 4,040 updates. Stable mean sustained crossing is 3,980, with large paired uncertainty. Both spectral arms take longer wall-clock time than Adam, and stable final CE is worse in four of five seeds. |
| Grokking directional branches | At step 2,500, equal-total-norm raw minus projected CE is +3.8245; margin is −4.5054; selected Fourier R² is −0.43565, all adverse in all five seeds. This is favorable evidence for the tested directional action, not proof that a dynamically learned span is necessary. The inherited legacy parent and archived comparator qualifications remain explicit. |
| Scalar controls and local function response | Scalar controls remain strong under the registered comparisons; unequal selector budgets and reused panels remain visible. Locally removing the off-span component improves held-out CE in all five grokking states while worsening training CE; restoring retained norm reverses part of that local gain. Neither result is promoted into a universal scalar explanation or a thousand-step causal decomposition. |

## Mathematical and construct checks

The isotropic selective-continuation construction has the stated scalar lower bound and projected convergence under its assumptions. It is correctly framed as an oracle/conditional construction, not a guarantee that the learned observer identifies useful directions. The centered-covariance argument, first-innovation initialization qualification, distinction between native `VVᵀ` and legacy `QΣ²Qᵀ`, and AdamW ambient-coordinate caveat agree with the methods and source.

The batching covariance identity concerns changing group-count weights at a fixed model. A two-group direction is the difference of group means, not automatically the rare or useful direction. Accessibility, delivered alignment, actual Adam utility, and held-out finite loss change are kept separate. Training-probe linear utility and held-out loss are not incorrectly treated as a Taylor decomposition of the same objective.

Important limits survive the synthesis: three-seed toy studies; rarity, novelty and digit identity entangled; no matched-competence harmful-behavior assay; mixed accuracy/CE conclusions; sparse diagnostics rather than demonstrated forgetting; reused parents rather than fresh observer replication; and provisional rare-score evidence with its failed resource certificate. Positive results are not dismissed simply because broad safety or deployment claims remain unsupported.

## Inspected evidence

- `research/spectral_paper_methods_2026-09-10.md` and `research/spectral_paper_core_2026-09-10.md`.
- `output/2026-09-06-spectral-optimizer-investigation/analysis/steelman-selective-learning.md` and the relevant stable-update implementation in `spectral_filter.py`.
- `output/2026-09-09-spectral-selectivity-boundary/results.md` and `results/checked-summary.json` under that directory. JSON SHA256: `8af7caa3932bdb7a6508e5375e00924d12b279bb3bbf8b1813deb0515b7e29b9`.
- `output/2026-09-10-spectral-batch-composition/results.md` and `results/checked-summary.json`. JSON SHA256: `4f437004dda5244e2b0ff18e4c072905417354cc551abeff39b5c662c0c8691b`.
- `output/2026-09-10-spectral-observer-pathway/interpretation.md` and `results/checked-summary.json`. JSON SHA256: `a8530a95a9469d0a35f65cb64615ab034062ee4011210172abf5e3583ce18970`.
- `output/2026-09-10-spectral-observer-signal/results.md` and `results/result.json`. JSON SHA256: `20d0514e3e6d5b0453d781957e38ac549c4c8269892657c5b76ee594e3d8cb2c`.
- `research/grokking_stable_confirmation_2026-09-08.md`; `output/2026-09-08-spectral-paper-planning/grokking-confirmation-results/summary.json` and `endpoint-supplement.json`.
- `research/grokking_action_mechanism_2026-09-09.md`; `research/grokking_raw_direction_2026-09-09.md`; `output/2026-09-09-spectral-raw-direction/results/summary.json`.
- `output/2026-09-09-spectral-function-response/interpretation.md`.
- `output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/results.md` and `output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-017/results.md`.

Short paths in this list are relative to the explicitly named study directory in the same entry. Stored JSON scalars were inspected, not numerically rederived.
