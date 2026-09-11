# Spectral Gradient Filter

A one-file PyTorch optimizer wrapper that filters gradients through a learned
subspace of recent, centered gradient variation. It can substantially restrict
noisy-label memorization while still permitting useful learning, but the effect
is task-dependent. Covariance measures variation, not truth or signed agreement;
filtering before Adam does not confine Adam's actual parameter updates.

## September 2026 investigation: start here

Read the [final core report](research/spectral_final_core_report_2026-09-11.md),
the [offline HTML with plots](output/2026-09-11-spectral-final-core/reader.html),
or the [PDF](output/2026-09-11-spectral-final-core/report.pdf).
The [technical working manuscript](output/2026-09-10-spectral-manuscript/paper/manuscript.tex)
contains the methods and mathematical qualifications.

In a three-seed MNIST study with approximately 81% actually wrong labels,
stable rank 200 filtering reaches 79.8% clean held-out accuracy versus 32.0%
for unaugmented AdamW. It learns substantially after warmup. Ordinary image
translation reaches 85.0%, while adding filtering to it reduces accuracy to 66.8%.
The [complete result](output/2026-09-10-spectral-strong-augmentation/results.md)
keeps all seeds, validation-selected checkpoints, loss and runtime beside those
means. This is a useful conditional learning effect, not a general optimizer
advantage, wall-clock speedup or demonstrated safety defense.

This public snapshot includes scientific source, fixtures, reports and scalar
results from the main investigation and its archived clustering/J-Lens branches.
Private correspondence, operational records, raw source-text corpora and the
private Git history are not published. See [publication scope](PUBLICATION.md)
for redactions, source-hash semantics and reproducibility limits.

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

The default streaming update solves its small covariance eigensystem in fp64
and periodically re-orthogonalizes the tracked basis. Set
`stable_update=False` only to reproduce results from before this numerical fix.

### Per-weight-matrix variant (LoRA/large models)

[`matrix_spectral_filter.py`](matrix_spectral_filter.py) keeps an independent
small covariance basis for every trainable weight matrix instead of one global
basis over all parameters. It also respects the base optimizer's parameter
subset, so frozen base-model weights are excluded automatically:

```python
from matrix_spectral_filter import PerMatrixSpectralGradientFilter

trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
base_opt = torch.optim.AdamW(trainable, lr=3e-4)
filt = PerMatrixSpectralGradientFilter(
    model, base_opt, rank=4, weighting="soft", alpha=2.0
)
```

The default `bias_mode="joint"` adds a trainable affine bias to its weight
matrix's block; unmatched vectors such as LayerNorm parameters pass through
unchanged. Use `bias_mode="separate"` to filter those vectors independently or
`"exclude"` to leave every 1D parameter alone.

On the small LoRA benchmark, soft per-matrix rank 4 matched plain LoRA while
using a 0.72 MiB covariance basis (versus 8.95 MiB for global rank 50). Hard
per-matrix projection underfit, and the method did not improve the 90%-label-
noise result. See [`research/per_matrix_spectral_evaluation.md`](research/per_matrix_spectral_evaluation.md).

On the full MNIST MLP with 90% random relabeling, however, stable hard
per-matrix rank 64 retained 0.817 final clean-test accuracy versus 0.389 for
AdamW and 0.788 for global hard rank 200 in a 60-epoch seed-42 run. It used
57.41 MiB of basis storage versus 179.40 MiB globally. See
[`research/noisy_mnist_hard_curves.md`](research/noisy_mnist_hard_curves.md).

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
| `stable_update` | `True` | use the fp64, periodically re-orthogonalized covariance update (`False` reproduces historical runs) |
| `relative_eig_tol` | `1e-8` | discard numerically negligible covariance directions relative to the leading eigenvalue |
| `stabilize_every` | `100` | periodically re-orthogonalize the streaming basis (`None` disables scheduled repair) |

## Historical findings (read alongside the September report)

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

**Current interpretation:** history-dependent restriction changes which learning
is easy. Temporal response, direction, numerical gains and base-optimizer state
all matter. The historical table is not a universal claim: stable and legacy
grokking results differ, and a leading covariance direction is not a semantic
feature or a certificate of useful information. See the final report above.

## Repo map

```
spectral_filter.py     ← THE core (copy this)
matrix_spectral_filter.py  scalable per-weight-matrix wrapper (LoRA/PEFT)
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

knowledge/             maintained cross-experiment wiki
  index.md                     ← current synthesis, decisions, and evidence map

results/               raw run outputs (JSON metrics)
```

> Note: every experiment script under `experiments/` imports the filter from the
> root `spectral_filter.py` (via a thin back-compat shim at
> `experiments/weight_cov_optimizer_v2.py`). There is one implementation.
