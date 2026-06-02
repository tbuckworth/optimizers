# Optimizers & Generalization

Research into how optimizers shape generalization, centered on a **spectral gradient filter**: a
streaming rank-1 / rank-B estimator of the `p×p` gradient covariance that projects each gradient onto its
dominant eigen-directions before handing it to a base optimizer (Adam/AdamW).

📄 **Start here:** [`research/spectral_filter_blog.html`](research/spectral_filter_blog.html) — a
self-contained write-up with methodology, figures, and every hypothesis tested. Open it in a browser.

## The filter, in one line

Track an EMA of the (mean-centred) gradient outer product, keep its top-`k` eigenspace via a cheap
streaming rank-1 SVD update (no `p×p` matrix is ever formed; ~2× Adam), and replace `g` with its
projection onto that subspace each step. A per-sample **rank-B** variant and **adaptive-rank** rules
(energy / effective-rank) are also implemented.

## Findings at a glance

| # | Hypothesis | Task | Verdict |
|---|------------|------|---------|
| H1 | Filtering resists label-noise memorization | MNIST / CIFAR-10 | ✅ **Strongest result** — MNIST @ 90% noise: filter holds ~80% while Adam/LoRA collapse to ~37/32%; beats a random-subspace control. Advantage grows with noise. On CIFAR it's a capacity dial (more rank → fits clean but admits noise). |
| H2 | Filtering accelerates grokking | modular addition | ✅ ~31% earlier (5 seeds). |
| H2′ | …on all algorithmic tasks | sparse parity | ❌ It *delays* grokking. |
| H3 | Low rank isolates the signal | sparse parity | ❌ ≤4 destabilizes, 5–8 grok variably, ~10 fastest-filtered, none beats AdamW. |
| H4 | Per-sample (rank-B) top-k isolates the signal | sparse parity | ❌ Never groks — top direction ≈ the bulk-fitting mean. |
| H5 | Adaptive (effective-rank) rank helps | parity / CIFAR | ⚖️ Groks parity (first filter variant that does); underfits CIFAR. Adaptive spectral rank helps **iff** the task's solution is genuinely low-dimensional. |
| H6 | Supervised ablation removes a planted "hack" | backdoor (linear / MLP) | ⚖️ Works on a linear model (ASR 99.9→8%), fails on an MLP at any subspace size — the backdoor is distributed. |
| H7 | Normalizing the basis (correlation/spectral) beats raw covariance | MNIST 50% noise | ❌ Raw covariance wins (92.1% vs 86.6/86.0) — variance magnitude is informative, not a nuisance. |

**Unifying mechanism:** the filter is a *coherence amplifier* — it keeps whatever the gradient is
coherent about. That helps when the useful signal is the coherent thing (noise robustness, grokking on
modular addition) and hurts when the useful signal is weak and not yet dominant (sparse parity).

## Repo structure

```
research/      notes, write-ups, figures (start with spectral_filter_blog.html)
experiments/   optimizer + experiment/plot code
results/       raw run outputs (JSON)
```

Key code: `experiments/weight_cov_optimizer_v2.py` (filter; `normalize ∈ {none,var,degree}`,
`adaptive ∈ {none,effrank,gap}`), `persample_cov_optimizer.py` (rank-B), `run_cifar_noise.py`,
`run_single_weight_cov_v2.py`, `modal_*` (A10G harnesses), `backdoor_ablation*.py`.

Key write-ups: `research/spectral_filter_blog.html`, `weight_covariance_v2_summary.md`,
`grokking_v2_findings.md`, `targeted_ablation_findings.md`, `moments_centering_explainer.html`,
`rank1_svd_explainer.html`.
