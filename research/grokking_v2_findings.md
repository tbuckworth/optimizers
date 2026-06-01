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

### Rank sweep: low rank does NOT help — the top eigenvectors are memorization

Titus's hypothesis: with (n=40, k=3) parity there are ~37 noise dims and 3 signal dims,
so a *low*-rank focus should isolate the signal and **help** grokking. We swept the
filter rank over {1, 3, 10, 50, 200} at **fixed decay 0.99** (all below the ~100
effective-rank ceiling, so no decay change is needed or warranted), 3 seeds, vs an
AdamW baseline. The hypothesis is **refuted at the low end**, but rank turns out to be a
sharp, non-monotonic dial.

| Rank | Grok epoch (3 seeds) | Final test | Behavior |
|------|---------------------:|-----------:|----------|
| **AdamW** | 650 / 750 / 750 | 1.00 | baseline |
| ours r1 | never / never / never | ~0.50 | memorizes then **destabilizes** |
| ours r3 | never / never / never | ~0.52 | same — collapses to chance |
| ours r10 | 1250 / 1300 / 1200 | 1.00 (1 seed later lost it) | **fastest among ours** |
| ours r50 | 2200 / never / 1900 | 1.00 (1 failed) | slower, unstable |
| ours r200 | 2050 / 2100 / 3400 | 1.00 | slowest |

**Three findings:**

1. **Rank 1–3 prevent grokking by destabilizing even the memorized fit.** All ranks
   memorize train to 100% by ep~100. But at rank 1–3 the train fit then *decays*
   (rank-1: train 1.00 → 0.55 → 0.71 over epochs 1k→6k, test stuck at chance), because
   with only 1–3 allowed gradient directions *and* weight decay 1.0 pulling weights down,
   the optimizer cannot even hold the solution, let alone find the sparse circuit. This
   shows the top 1–3 eigendirections **do not span the generalizing solution**.
   *Caveat on the label (Titus's objection):* these are eigenvectors of the **temporal**
   covariance of the *full-batch* gradient (centred by an EMA over steps). Their top
   direction is the dominant direction the optimizer is currently moving in — bulk
   loss-reduction (fitting the train set) plus weight-decay/norm pressure — which is
   coherent *across steps*, **not** a "consensus across examples." The 37 distractor bits
   are random and carry no shared signal, so "memorization consensus" is the wrong phrase:
   there is no coherent cross-example signal there. The defensible claim is only that these
   top *temporal* directions are not the slowly-emerging parity feature, and restricting
   updates to them under wd=1.0 cannot even sustain the memorised fit.

2. **Among ranks that grok, lower is faster — but none beat AdamW.** r10 (~1250) ≪ r200
   (~2500). There is a sweet spot near the *effective* dimension (~10), with catastrophe
   below it and slowdown above. But every rank is slower than the 717-epoch baseline: the
   filter **delays grokking at all ranks**, just least near rank 10. So Titus's "rank is
   load-bearing" intuition holds (the dial matters and has a low-rank optimum), while the
   stronger claim "low rank beats baseline" does not.

3. **Eigenvalue spectrum has no clean drop at 3.** Normalized top eigenvalues (rank-200
   run): before grok (ep 100–500) the spectrum has a soft elbow at **~7–8** directions
   (`1.0, 0.6, 0.2, 0.07, 0.05, 0.02, 0.01…`), not 3. *During* the transition (ep 1000–2000)
   it **broadens** (active dims 7 → 38); *after* grok it **collapses to ~2–6** dims. The
   generalizing circuit is genuinely low-dim (matches modular addition's eff-rank 2–8), but
   the 3-bit signal does not appear as a top-spectrum signature — it is buried *below* the
   dominant memorization directions, which is why ≳10 dims must be kept to retain it.

This sharpens (does not overturn) the picture: the filter rides the dominant *temporal*
gradient subspace, and only retains the slowly-emerging generalizing signal if the rank is
wide enough to reach below those directions.

**Open question raised by Titus — the per-sample (rank-B) decomposition.** The current
filter tracks the covariance of the *batch-mean* gradient over *steps*. A different, arguably
more meaningful object is the covariance of the *per-sample* gradients *within* a step (across
the 2000 training examples). These decompose the gradient differently:
- the **shared structural signal** (grow weights on the 3 relevant input dims, shrink the 37
  distractors) is common to all examples → it sits near the **mean** gradient → it has *low*
  cross-sample variance and is **not** a top eigenvector of the per-sample covariance;
- **per-example memorisation** (a bespoke feature for example *i*) is where examples *disagree*
  → it is the **high-variance** part → it dominates the **top** eigenvectors of the per-sample
  covariance.

This is exactly Chatterjee's *Coherent Gradients* distinction (gradients that agree across
examples generalise; idiosyncratic ones memorise). Prediction: a rank-B filter that **projects
out** (ablates) the top per-sample directions should *suppress* memorisation and **accelerate**
grokking — the reverse of the batch-mean filter, which projecting *onto* the top temporal
directions delays it. Testing this (both project-onto and project-out variants) is the next
experiment; the rank-B SVD machinery is already derived (`research/rank1_svd_explainer.html` §10).

### Per-sample (rank-B, uncentered) filter and adaptive rank: also never accelerate

We tested the per-sample decomposition directly (Titus's rank-B idea), in the **uncentered,
project-onto-top-k** form he specified — track the EMA of the *uncentered second moment of the
per-sample gradients across examples*, `C = EMA[(1/B)Σ gᵢgᵢᵀ]`, and descend on the batch-mean
gradient projected onto its top-k. (`experiments/persample_cov_optimizer.py`, rank-B streaming
SVD validated against brute force to 7e-7.)

| Condition | Grok epoch (3 seeds) | Final test |
|-----------|---------------------:|-----------:|
| AdamW | 650/750/750 | 1.00 |
| per-sample r3 | never ×3 | ~0.49 |
| per-sample r10 | never ×3 | ~0.49 |
| per-sample r50 | never ×3 | ~0.50 |
| per-sample adaptive (99% energy) | never ×3 | ~0.49 |
| batch-mean adaptive (99% energy) | 2300 / never / never | ~0.50 |

**The per-sample filter never groks at any rank.** Its train/test trace is the same
memorize-then-destabilize collapse as low-rank batch-mean (r10: train 1.00 → 0.53, test stuck at
chance). The reason is consistent with everything else: the top eigenvector of the *uncentered*
per-sample second moment is dominated by the **shared bulk-fitting direction ≈ the mean gradient**
(confirmed elsewhere: for a direction shared across samples, cos(mean, top-uncentered-eigvec)=0.999),
which early in training is memorization, not the slow parity feature. Projecting onto the top-k of
*either* decomposition therefore keeps the bulk-fitting direction and starves the weak generalizing
signal — and, as the rank sweep showed, also chokes the high-dimensional exploration the network needs.

**Adaptive rank (keep 99% of eigenvalue energy)** is trivially cheap to add (the eigenvalues are
already computed each step) and works mechanically, but on parity it selects **~4–5 directions**
(the spectrum is so top-heavy that 99% of the energy sits in ~4 dims) — squarely in the
grokking-killing regime. So a naive energy rule *under*-shoots here: the network needs exploration
dimensions *beyond* the high-energy ones, which the energy criterion can't see.

**Unifying conclusion for parity:** *any* "keep only the top-k gradient directions" filter
(batch-mean or per-sample, fixed or adaptive) suppresses parity grokking. The generalizing signal is
a weak, slowly-emerging component that is **not** among the top gradient directions in any of these
decompositions, and aggressive projection additionally chokes the exploration phase. Only un-filtered
AdamW groks fast. This is a clean negative result that bounds where the method helps.

Provenance: `experiments/sparse_parity.py`, `experiments/modal_sparse_parity.py`,
`experiments/persample_cov_optimizer.py`, `experiments/modal_parity_persample.py`,
`results/sparse_parity_ranksweep/`, `results/parity_persample/`,
`research/sparse_parity_ranksweep.png`, `parity_grok.png`.

## Provenance

- Single-seed exploratory matrix: `results/grokking_v2/` (5 conditions incl. no-wd controls).
- Multi-seed confirmation: `results/grokking_v2_seeds/` (3 wd=1.0 conditions × 5 seeds).
- Code: `experiments/run_grokking.py` (weightcov + `--switch_at`), launch scripts
  `experiments/launch_grokking_v2.sh` and `launch_grokking_v2_seeds.sh`.
- Visualizations: `research/grokking_v2_viz.html` (single-seed curves + effective rank),
  `research/grokking_v2_seeds_viz.html` (strip plot + per-seed curves).
