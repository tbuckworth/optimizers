# What the small rank-32 tests do—and do not—say about the original optimizer

Codex — Spectral Optimizer Investigation · 10 September 2026

**Bounded source-and-scalar synthesis, not a new experiment or audit.** This
note uses completed historical evidence and the frozen multiview protocol.
It does not read or interpret the current run's journal, intermediate outputs,
or unaudited endpoint. No tensors, datasets or model states were opened.

## Practical conclusion first

The completed augmentation tests establish a real recipe-level problem:
ordinary translation beats the tested stable global rank-32 recipe, including
under fixed wrong labels. In the noisy test, raw+translation beats both native
conditions in every seed and both endpoint metrics. This is useful comparative
evidence, not something that disappears when experimental differences are
listed. Conversely, these experiments have **not tested the strongest earlier
noise-protection configuration under augmentation**. “This tested recipe loses”
and “the original method has no useful protection regime” are different claims.
[Completed clean report](../2026-09-10-spectral-general-augmentation/results.md),
[completed wrong-label report](../2026-09-10-spectral-wrong-label-augmentation/results.md).

## Preserve the strongest actual positive

The historical three-seed, 60-epoch rank-200 study is a strong positive for
**late-horizon preservation against noisy-label overfitting**. Its final clean
test accuracies for seeds 42/43/44 are 74.28/81.87/83.06%, versus Adam's
35.92/37.74/38.00%. These are within-study comparisons, not rankings against
the newer held-out subsets. The learned filter also substantially outperforms
the fixed random rank-200 baseline (54.17/57.70/56.21% final). That rejects this
particular random-subspace substitute, not all simpler regularization methods.
[Raw 18-run directory](../../results/weight_covariance_v2/noise90_long/),
[launcher](../../experiments/launch_noise90_long.sh),
[historical baseline report](../../research/subspace_baselines_findings.md).

The positive is **not confined to legacy numerics**. In the later matched
stable-hard AdamW study, global rank 200 finishes at 78.77% versus AdamW's
38.93%; per-matrix rank 64 finishes at 81.74%. Global's peak is 85.48% and
per-matrix's 84.35%. These are seed-42 descriptive results; the per-matrix rank
was selected on that same seed's test performance in a 20-epoch scout, so its
apparent advantage over global is not an unbiased configuration comparison.
[Stable report](../../research/noisy_mnist_hard_curves.md),
[global raw JSON](../../results/noisy_mnist_hard_curves/final/global_stable_hard_r200_n90_s42.json),
[AdamW raw JSON](../../results/noisy_mnist_hard_curves/final/adamw_n90_s42.json),
[matrix raw JSON](../../results/noisy_mnist_hard_curves/final/matrix_stable_hard_r64_n90_s42.json),
[selection record](../../results/noisy_mnist_hard_curves/summary.json).

These gains are not evidence of permanent immunity or uniquely semantic
selection. In the legacy study, “about five points of collapse” describes the
mean peak-to-final decline: rank-200 seed 42 falls from 86.00% to 74.28%
(11.72 points). Adam's early peaks are already 83.06/82.28/84.24%. Peak-based
selection therefore changes the practical claim substantially; an oracle
test-set peak is not a deployable stopping rule. LoRA is also competitive or
better at moderate noise in the historical baseline report. Preserve these
qualifications alongside the large fixed-horizon positive.

## The regimes are materially different

| Dimension | Current small recipe / multiview protocol | Strong earlier noisy-MNIST studies | Consequence |
|---|---|---|---|
| Model | 784→64→10, 50,890 parameters | 784→256→128→10, 235,146 parameters | Different representation and optimization problem, not merely a shorter run of the same model. |
| Projection | Stable global hard rank 32 | Legacy/global rank 200; later stable global rank 200 or three joint weight/bias blocks at rank 64 | Neither the larger global span nor block allocation is tested by changing the observation stream at rank 32. |
| Pixels | Float32 [0,1] | Standardized by (pixel−0.1307)/0.3081 | Changes activations, gradients and optimizer trajectory. Any bridge should preserve normalization and define padding before normalization. |
| Training population | 5,000 balanced examples; replacement sampling, B64 | All 60,000 training examples; shuffled epoch traversals, B64 | Different diversity, repeated-example exposure and clean-signal population size. |
| Readout | 5,000 disjoint official-training examples; official test unused | 10,000 official test examples | Absolute percentages are not directly comparable; historical test-selected settings need fresh confirmation. |
| Horizon | 4,000 updates = 256,000 example occurrences | 60 epochs ≈56,280 updates /3.6 million examples | About 14 times more updates, despite similar 51.2 versus60 nominal passes over each respective training population. Epoch count alone conceals the difference. |
| Warmup | 100 updates | 100 updates | Same count, but about1.28 versus0.107 dataset-equivalent passes; subsequent optimizer/observer histories differ. Warmup was not simply removed in the new test. |
| Labels | Current multiview: entirely clean. Earlier small noisy panel: exactly80% guaranteed wrong, balanced by true class | Bernoulli90% uniform replacement, replacement may stay correct | Current multiview cannot establish noisy protection. Old noise implies about81% actually wrong in expectation, so the noisy-law difference is modest—not a ten-point corruption excuse. |
| Base optimizer | AdamW lr.001, wd.01, fixed single-tensor settings | Original three-seed study: Adam, no wd; later stable study: matched AdamW wd.01 | Adam/AdamW differs for the legacy comparison, but does not explain away the later stable positive. |
| Augmentation | One shared translated view; candidate observes four-view mean; raw4 delivers that mean | No translation in the cited noise-protection trials | Historical superiority to unaugmented Adam does not establish added value over augmented Adam. |
| Selection / baselines | Fixed endpoint, three fresh paired seeds; current raw4 accounts for extra views | Rank comparisons and same-seed matrix scout; final/peak/late-average results; Adam, random-subspace and LoRA | Different selection opportunities and practical questions. No matched modern augmentation comparison exists in the cited strong regime. |

Configuration sources: [current protocol](protocol.md),
[new data helper](../../experiments/spectral_multiview_data.py),
[historical loader/model/optimizer code](../../experiments/run_single_weight_cov_v2.py).
The legacy JSON lacks a `legacy_update` field: its provenance is the original
implementation at commit `38d8696`, before stable updates were introduced by
`ddcc97c` on12August2026. Reading today's default as its acquisition setting
would be wrong. The later stable JSON explicitly records `legacy_update=false`.

Rank is a credible design difference, **not an established explanation**. The
historical sweep's mean final ordering is rank200>100>50, and the matrix scout
favored64 over32/16/8. That makes “try a faithful strong configuration” better
motivated than treating rank32 as representative. It does not prove that rank
alone caused the current deficit, or that a fixed rank/parameter ratio transfers.
The new and old rank fractions are approximately0.063% and0.085%, respectively;
neither fraction determines the useful gradient dimension.

## Smallest useful next reasoning/design step

First specify the practical claim: **does filtering add late-horizon noisy-label
robustness beyond a useful augmentation baseline, while retaining useful
learning?** Then write one configuration-frozen bridge to the existing stable
global-rank200 positive, rather than launching an automatic rank/LR sweep or
another local mechanism probe. Global rank200 is the cleanest initial bridge:
it has an actual stable positive and avoids changing to test-selected blocks.

The minimal discriminating design is one paired2×2: raw AdamW versus stable
global-rank200, each with/without the same one-view translation. Fix the larger
model, normalization, noise law, data exposure, optimizer and late horizon
from the strong regime. Include the unaugmented filtered arm so failure to
recover the expected positive is visible, rather than silently attributing any
change to augmentation. Use fresh paired seeds and separate validation/test
roles; report fixed endpoints, complete learning curves, useful progress and
predeclared late-horizon preservation. A validation-selected stopping baseline
can be read from those same raw trajectories without an extra training sweep.
Budget the longer run before choosing it; this note does not authorize it.

If augmented raw remains better in that faithful comparison, the objection to
the tested practical recipe becomes substantially stronger. If the larger
stable filter adds value there, the useful conclusion is a regime boundary,
not a reversal of the small-model evidence. Either outcome advances the
investigation; neither requires prematurely stopping it or declaring every
negative merely a confound.
