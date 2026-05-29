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

### 1. The filter accelerates grokking when weight decay is present (positive, single-seed)

Filter+AdamW reaches test≥0.9 at epoch **2650** vs the AdamW baseline's **3700** — a ~28%
speedup — and the transition is sharper:

```
baseline:     ep3000 test=0.04 → ep3250 0.18 → ep3500 0.50 → ep3750 0.95   (~750-epoch ramp)
filter+adamw: ep2250 test=0.04 → ep2500 0.58 → ep2750 1.00                 (~500-epoch ramp)
```

**Caveat: this is one seed.** Grokking timing has high seed-to-seed variance, so the
~28% number needs 3–5 seeds before we trust the magnitude. The *direction* (filter doesn't
hurt, plausibly helps) is the safe claim.

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
- Next: 3–5 seeds on baseline vs filter+adamw to confirm/kill the acceleration result.
