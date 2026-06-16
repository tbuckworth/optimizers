# Spectral Gradient Filter

A one-file, drop-in wrapper for any PyTorch optimizer that makes it **resist
memorizing noise**. Before each update it projects the gradient onto the top
eigen-directions of a streaming estimate of the gradient covariance — so the
optimizer only steps in directions the gradient has been *consistently* pointing.
It's a *coherence amplifier*: it keeps whatever the gradient agrees about across
steps and drops the rest. Cost is ~2× a bare Adam step (no `p×p` matrix is ever
formed).

## The core, in one file

👉 **[`spectral_filter.py`](spectral_filter.py)** — the whole idea in one
self-contained file (only needs `torch`; the rest is comments and knobs). This
is the thing to copy.

Drop it into **any** training loop — any loss, any model:

```python
from spectral_filter import SpectralGradientFilter

base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
filt = SpectralGradientFilter(model, base_opt, rank=200)

for x, y in loader:
    base_opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    filt.filter_grad()      # <-- the only new line: filters .grad in place
    base_opt.step()
```

That's it. `filter_grad()` updates the covariance estimate and replaces each
parameter's `.grad` with its projection onto the top-`k` eigenspace, in place,
after a `warmup`. (For plain classification there's also a one-call convenience,
`filt.step(x, y)`, that does the forward/backward for you.)

## See it work in ~1 minute

```bash
python3 example.py
```

[`example.py`](example.py) trains a small MLP on MNIST with **90% of labels
randomized**, with and without the filter. Plain Adam memorizes the noise — its
training accuracy climbs while test accuracy collapses (~0.26). The filter
refuses to memorize (train stays flat) and **holds ~0.61 test accuracy**.

## What the knobs do

| arg | default | meaning |
|-----|---------|---------|
| `rank` | 200 | hard cap on eigendirections kept |
| `decay` | 0.99 | EMA decay of the covariance estimate |
| `warmup` | 100 | steps to observe before filtering kicks in |
| `weighting` | `"hard"` | `"hard"` top-k projection, or `"soft"` eigenvalue^`alpha` reweighting |
| `alpha` | 1.0 | soft exponent: `0`=identity, `1`=consensus, `∞`=top dir, `<0`=whitening |
| `normalize` | `"none"` | basis: `none` (covariance), `var` (correlation), `degree` (affinity) |
| `adaptive` | `"none"` | rank rule: `none`, `effrank`, or `gap` |

## Findings from the research

The repo is also a full research project on how this filter behaves. Full
write-up with figures: **[`research/spectral_filter_blog.html`](research/spectral_filter_blog.html)**
(open in a browser).

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

**Unifying mechanism:** the filter keeps whatever the gradient is coherent about.
That helps when the useful signal *is* the coherent thing (noise robustness,
grokking on modular addition) and hurts when the useful signal is weak and not
yet dominant (sparse parity).

## Repo map

```
spectral_filter.py     ← THE core (copy this)
example.py             ← minimal runnable demo (the H1 result)

experiments/           exploration: experiment + plot scripts for H1–H7
  run_single_weight_cov_v2.py   main MNIST/label-noise harness
  run_cifar_noise.py            CIFAR-10 label noise
  run_grokking.py               grokking / modular addition
  persample_cov_optimizer.py    per-sample rank-B variant (H4)
  backdoor_ablation*.py         planted-backdoor ablation (H6)
  legacy/                       superseded v1 code, kept for provenance

research/              write-ups, figures, interactive HTML
  spectral_filter_blog.html     ← canonical write-up, start here
  (see research/README.md for the index)

results/               raw run outputs (JSON metrics)
```

> Note: every experiment script under `experiments/` imports the filter from the
> root `spectral_filter.py` (via a thin back-compat shim at
> `experiments/weight_cov_optimizer_v2.py`). There is one implementation.
