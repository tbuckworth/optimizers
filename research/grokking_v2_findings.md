# Grokking & the Weight-Covariance Spectral Filter: Findings

## Setup

Modular addition (a + b) mod 113, 1-layer transformer (227K params), 30% train split,
full-batch training, lr=1e-3, betas=(0.9, 0.98). Five runs, single seed (42), 40K epochs.
Filter: rank=200, decay=0.99, warmup=100. "Grok epoch" = first epoch with test_acc ≥ 0.9.

## Results

| Run | Weight decay | Memorize (train≥0.99) | Grok (test≥0.9) | Final test | Verdict |
|-----|-------------|----------------------|-----------------|-----------|---------|
| `adamw` (baseline) | 1.0 | ~150 | **3700** | 1.0000 | Standard grokking |
| `adam` (control) | 0.0 | ~150 | never | 0.0018 | No grokking (as expected) |
| `filter + adamw` | 1.0 | ~150 | **2650** | 1.0000 | **Grokking accelerated ~28%** |
| `filter + adam` | 0.0 | ~150 | never | 0.0013 | Filter does NOT replace weight decay |
| `switch` (adam→filter @ train 0.99) | 0.0 | ~150 | never | 0.0017 | Filter does NOT trigger grokking on demand |

## Three findings

### 1. The filter accelerates grokking when weight decay is present (CONFIRMED, 5 seeds)

Filter+AdamW reaches test≥0.9 substantially earlier than the AdamW baseline, and the
transition is sharper. Confirmed across 5 seeds with **non-overlapping** distributions:

| Condition | Grok epoch (5 seeds) | Mean ± SD |
|-----------|---------------------|-----------|
| AdamW baseline | 3800, 3600, 3700, 3700, 3800 | **3720 ± 84** |
| Filter + AdamW | 2500, 2600, 2650, 2450, 2550 | **2550 ± 79** |
| Switch (AdamW→filter @ memorization) | 2300, 2700, 2500, 2450, 2650 | **2520 ± 160** |

Baseline range [3600–3800] vs filter range [2450–2650] — **zero overlap**. Welch
t = 22.7, Δ = 1170 epochs, **31% faster**. This is a large, robust effect, not seed noise.

Transition shape (representative seed):
```
baseline:     ep3000 test=0.04 → ep3250 0.18 → ep3500 0.50 → ep3750 0.95   (~750-epoch ramp)
filter+adamw: ep2250 test=0.04 → ep2500 0.58 → ep2750 1.00                 (~500-epoch ramp)
```

**Switch ≡ filter-from-start (t = 0.4, indistinguishable).** Turning the filter on only
*after* memorization (epoch ~130) gives the same speedup as running it the whole time. So
the acceleration is **entirely a post-memorization effect** — the filter does nothing useful
during the memorization phase; it speeds up the generalization transition that follows. (This
also rules out "the filter just changes the memorization dynamics" as the explanation.)

### 2. The filter does NOT substitute for weight decay (clean negative)

Filter+Adam with **no weight decay** never groks (final test 0.13%, i.e. chance). Grokking
requires the norm-reduction pressure of weight decay to push the model off the memorizing
solution toward the efficient (Fourier) circuit. The filter alone provides no such pressure.

### 3. The filter does NOT trigger generalization on demand (clean negative)

The "switch" run — train plain Adam (no wd) until memorization, then turn the filter on —
also never groks. Flipping on the filter after memorization does nothing without weight decay.

## Why: the filter rides the dominant direction, it doesn't choose the *good* one

The mechanism is consistent with the MNIST label-noise results, and the contrast is the
interesting part:

- **MNIST label noise**: gradients are *mini-batch* gradients. Memorizing corrupted labels
  shows up as **transient, idiosyncratic per-batch directions**; the true signal recurs across
  batches. The filter keeps the recurring (signal) directions and discards the transient
  (memorization) ones. The filter helps because *signal is the persistent direction*.

- **Grokking**: gradients are *full-batch* and deterministic — there is no sampling noise.
  The memorizing solution is itself a **globally persistent direction** (the model is steadily
  pushed to fit the train set). So the filter happily projects onto memorization. There is no
  transient noise to remove. Generalization only happens because weight decay creates a
  *separate* pressure toward the low-norm circuit — and the filter merely accelerates the
  transition that weight decay is already driving.

**One-line takeaway:** the filter amplifies whatever the persistent gradient direction is.
It does not *create* the force that makes the generalizing solution persistent — that comes
from the data (label consistency) or the regularizer (weight decay). When generalization is
already being driven, the filter speeds it up; when nothing drives it, the filter can't.

## Bonus finding: grokking gradient dynamics are ultra-low-dimensional

The filter's effective rank (entropy of the eigenvalue spectrum) stays at **2–8** throughout
grokking — versus **60–85** on MNIST. The full-batch grokking gradient lives in a tiny 2–8
dimensional subspace, consistent with the picture that the generalizing solution is a small
set of Fourier features (Nanda et al.). The rank=200 cap is never close to binding here.

## What this means for the project

- The filter is **compatible with** grokking (doesn't break it, plausibly accelerates it),
  but its noise-robustness mechanism is **orthogonal to** the grokking mechanism.
- The label-noise robustness story (transient vs persistent under *sampling noise*) is the
  real contribution. Grokking is a different phenomenon (norm pressure under full-batch),
  and we should not over-claim a connection.
- The acceleration result is confirmed (5 seeds, t = 22.7). The switch≡from-start finding
  localizes it to the post-memorization transition.

## Sparse parity: the filter DELAYS grokking (the opposite result)

Tested a second algorithmic task — (n=40, k=3) sparse parity (Barak 2022), train_size=2000,
wd=1.0, 3 seeds. Grokking regime confirmed (train_size=1000 never generalizes; ≥2000 groks).

| Condition | Grok epoch (3 seeds) | Mean ± SD |
|-----------|---------------------:|----------:|
| AdamW | 650, 750, 750 | **717 ± 58** |
| Filter + AdamW | 2150, 2050, 2000 | **2067 ± 76** |

The filter makes grokking **~3× slower** (Welch t = −24.4) — the *opposite* of the 31%
acceleration on modular addition. It still groks (reaches test=1.0), just much later.

**Why the opposite — and why it's consistent.** The filter amplifies the dominant *consensus*
gradient direction. Whether that helps grokking depends on whether the generalizing solution
*is* the consensus:
- **Modular addition**: the generalizing Fourier circuit is a global structure many examples
  agree on → it's the consensus → the filter amplifies it → grokking accelerates.
- **Sparse parity**: generalization requires finding 3 specific bits among 40 — a **weak, sparse**
  signal. Early on, the consensus is dominated by *memorizing* the 2000 training examples, not the
  sparse feature. The filter projects onto that memorization consensus and **suppresses the weak
  sparse signal** → grokking is delayed.

So "the filter accelerates grokking" is **task-dependent, and the dependence is interpretable**:
it accelerates when generalization aligns with the gradient consensus, and *delays* it when
generalization needs a weak signal the consensus drowns out. This is the same consensus-amplifier
mechanism as everywhere else — it cuts both ways. (Honest framing: the modular-addition acceleration
is real but should not be over-generalized to "speeds up grokking" in the abstract.)

Provenance: `experiments/sparse_parity.py`, `results/sparse_parity_grok/`, `parity_grok.png`.

## Provenance

- Single-seed exploratory matrix: `results/grokking_v2/` (5 conditions incl. no-wd controls).
- Multi-seed confirmation: `results/grokking_v2_seeds/` (3 wd=1.0 conditions × 5 seeds).
- Code: `experiments/run_grokking.py` (weightcov + `--switch_at`), launch scripts
  `experiments/launch_grokking_v2.sh` and `launch_grokking_v2_seeds.sh`.
- Visualizations: `research/grokking_v2_viz.html` (single-seed curves + effective rank),
  `research/grokking_v2_seeds_viz.html` (strip plot + per-seed curves).
