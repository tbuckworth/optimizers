# Is It Just "LoRA in a Trenchcoat"? — Subspace Baselines

## The question

A reviewer suggested our learned gradient-covariance filter might be equivalent to
**training a low-rank adapter on a randomly initialised network**. Both restrict updates
to a low-dimensional subspace, so are we just doing a fancy (expensive) version of a random
low-rank constraint? We tested this directly with two baselines at matched dimension.

## Methods (MNIST, 3-layer MLP, 235K params, 20 epochs, 3 seeds)

| Method | What it constrains |
|--------|-------------------|
| `adam` | nothing — full Adam reference |
| `ours` | gradient → **learned** top-k=200 eigenspace of the running gradient covariance (rotates over time) |
| `random_subspace` | gradient → **fixed random** orthonormal k=200 subspace (same projection plumbing, P frozen at init) — *the Li et al. 2018 intrinsic-dimension setup* |
| `lora` | frozen random-init MLP + trainable rank-32 LoRA adapters |

`random_subspace` is the clean control: it differs from `ours` in exactly one variable —
the subspace is random-and-fixed instead of learned-and-rotating.

## Result 1: It is NOT LoRA in a trenchcoat — the adaptivity is load-bearing

Final test accuracy (mean ± sd over 3 seeds; best-epoch in parens):

| Noise | adam | ours (k=200) | random_subspace (k=200) | lora |
|------:|-----:|-------------:|-------------------------:|-----:|
| 0% | 98.0±0.1 (98) | 97.5±0.1 (98) | **78.2±0.2 (78)** | 97.6±0.2 (98) |
| 20% | 92.3±0.3 (97) | 95.1±0.2 (95) | **76.9±0.8 (77)** | 96.7±0.2 (97) |
| 40% | 88.1±0.9 (96) | 93.4±0.3 (94) | **75.2±1.1 (76)** | 95.3±0.5 (96) |
| 90% | 54.6±1.6 (83) | 79.0±3.3 (81) | **57.7±1.4 (60)** | 58.1±1.1 (82) |

**At matched k=200, `ours` beats `random_subspace` by ~18–21% at every noise level.** A
*learned* 200-dim subspace is dramatically more efficient than a *random* one of the same
size — consistent with Li et al. 2018, who needed ~750 random dimensions for a comparable
MLP to reach 90%. The streaming-SVD adaptivity is doing real work; it is **not** replaceable
by `torch.randn(p, k)`. The reviewer's equivalence does not hold.

## Result 2: LoRA is a surprisingly strong baseline — except at extreme noise

LoRA-on-random-init is genuinely good and noise-robust at moderate noise — it edges out
`ours` on final accuracy at 20% and 40% (capacity limitation: a rank-32 adapter simply can't
fit much label noise). The honest takeaway is that **a cheap low-rank baseline handles
moderate label noise well**, and we should report that.

**But at 90% noise the two diverge sharply**, which the long-horizon run makes undeniable.

## Result 3: At extreme noise, ours resists the late-memorization collapse (60-epoch run)

Re-ran 90% noise for 60 epochs (3 seeds), adding a rank sweep for `ours`:

| Condition | Peak % | Peak epoch | Final % | Collapse (peak→final) |
|-----------|-------:|-----------:|--------:|----------------------:|
| adam | 83.2 | ep3 | 37.2 | **−46.0** |
| lora | 81.6 | ep5 | 31.9 | **−49.6** |
| random_subspace | 60.2 | ep33 | 56.0 | −4.2 |
| ours_r50 | 70.1 | ep35 | 66.2 | −3.8 |
| ours_r100 | 77.3 | ep50 | 72.9 | −4.5 |
| **ours_r200** | **84.9** | ep44 | **79.7** | −5.1 |

**adam and lora collapse ~46–50 points** as they memorize the 90%-corrupted labels (both
peak at epoch 3–5, then degrade for the rest of training). **Every `ours` variant is stable
(~4–5 pt drop)** and `ours_r200` finishes 40+ points ahead of the baselines.

Two clean sub-findings:

- **`ours` peaks late (epoch 44–50)** vs adam/lora at epoch 3–5. It keeps improving deep into
  training rather than overfitting — so at extreme noise it *wants* more epochs, not fewer.
- **Higher rank is better here**: r200 (79.7) > r100 (72.9) > r50 (66.2). More aggressive
  filtering (lower rank) discards useful signal. Stability is rank-independent (all collapse
  ~4–5 pts); only the accuracy ceiling scales with rank.

## Honest framing

With **early stopping**, adam/lora/ours all reach ~80–84% at 90% noise — peak achievable
accuracy is similar. Our method's distinctive value is that it **doesn't need early stopping**:
it is stable under continued training where the baselines self-destruct. That is the precise,
defensible claim. The clean win in *peak* accuracy is only over `random_subspace` (the
matched-k control), where the margin is large and unambiguous.

## Bottom line

1. **Not LoRA in a trenchcoat** — learned ≫ random subspace at matched k (~18–21% everywhere).
2. **LoRA is a strong, cheap baseline** at moderate noise — worth reporting honestly.
3. **Our edge is stability at extreme noise / long training** — adam & lora collapse 46–50 pts
   at 90% over 60 epochs; ours holds within ~5 pts and finishes 40+ pts ahead.

## Provenance

- Code: `experiments/random_subspace_optimizer.py`, `experiments/lora_mlp.py`,
  `experiments/run_single_weight_cov_v2.py` (modes `random_subspace`, `lora`).
- Launches: `experiments/launch_subspace_baselines.sh`, `experiments/launch_noise90_long.sh`.
- Results: `results/weight_covariance_v2/subspace_baselines/` (48 runs),
  `results/weight_covariance_v2/noise90_long/` (18 runs).
- Visualization: `research/subspace_baselines_viz.html`.
