# Gradient-Covariance Spectral Filtering (v2) — Master Summary

This is the top-level synthesis of the v2 line of work. Detailed per-experiment writeups:
- `weight_covariance_v2_findings.md` — label noise + base-optimizer ablation
- `grokking_v2_findings.md` — grokking
- `subspace_baselines_findings.md` — random-subspace and LoRA baselines
Visualizations: `weight_cov_v2_comparison_viz.html`, `grokking_v2_seeds_viz.html`,
`subspace_baselines_viz.html`.

---

## 1. The method

A gradient-filtering wrapper around any base optimizer. We track a running, low-rank
estimate of the **p×p covariance of the (flattened) gradient over training steps**, and each
step project the current gradient onto the top-k eigenspace of that covariance before handing
it to the base optimizer.

```
each step:
  g       = batch-mean gradient (R^p, p = all params flattened)
  V, S    = streaming rank-1 SVD update of the gradient covariance (EMA, decay λ)
  g_filt  = V (Vᵀ g)         # keep only the component in the learned top-k eigenspace
  base_optimizer.step(g_filt)
```

The SVD update is cheap: a (k+1)×(k+1) Gram matrix is built analytically from V, S and the
new gradient — no (k×p) matrix ever materializes. Overhead ≈ 2× vanilla Adam on a 235K-param
net. Hyperparameters: rank `k`, decay `λ`, warmup steps.

**Intuition.** Directions that recur across many steps reflect genuine structure in the data;
directions that appear transiently reflect sampling noise or sample-specific memorization.
Projecting onto the top eigenspace keeps the persistent signal and discards the transient
noise. The eigenvectors are *learned and rotate over time* — this turns out to be the crux
(see §5).

---

## 2. Label noise robustness (the core result)

MNIST, 3-layer MLP (235K params), 20 epochs, a fraction of training labels corrupted to
random classes. Base optimizer Adam unless noted.

| Noise | Adam (final) | Adam (best) | Ours k=200 (final) | Ours (best) |
|------:|------------:|-----------:|-------------------:|------------:|
| 0% | 98.0 | 98.2 | 97.5 | 97.6 |
| 20% | 92.3 | 97.3 | **95.1** | 95.4 |
| 40% | 88.1 | 96.3 | **93.4** | 94.0 |

**Adam memorizes corrupted labels over time** — at 40% noise its test accuracy peaks early
(~96%) then degrades to ~88% as training continues, while its train accuracy climbs past the
correct-label ceiling. **Ours resists this**: train accuracy plateaus near the ceiling and
test accuracy stays high. The advantage grows with noise (≈0 at 0%, +3 at 20%, +5 at 40%).
The mechanism is implicit regularization that specifically targets memorization of label
noise. Rank controls the noise/signal tradeoff (too low underfits; k=200 is the sweet spot
for this problem). Effective rank sits at ~60–85, well below the cap.

---

## 3. Base-optimizer ablation: momentum complements, not competes

Both the filter and Adam's momentum favor persistent directions — are they redundant? We ran
the filter on SGD, SGD+momentum, and Adam.

- **Filter + Adam is best on noisy data** (95.1% @20%, 93.0% @40%) — momentum **complements**
  the filter rather than fighting it: the filter selects *direction* (which subspace), Adam's
  adaptive rates handle per-parameter *scale*.
- The filter helps Adam most (+4% @40%) and plain SGD least (+0.3%) — Adam's momentum
  aggressively accumulates memorization gradients, which is exactly what the filter corrects.
- Filter+SGD underfits clean data — the filter needs a capable base optimizer.

---

## 4. Grokking: accelerates the transition, doesn't create it

Modular addition (mod 113), 1-layer transformer (227K params), full-batch, weight decay 1.0,
**5 seeds**.

| Condition | Grok epoch (test≥0.9) | Mean ± SD |
|-----------|----------------------:|----------:|
| AdamW baseline | 3800,3600,3700,3700,3800 | 3720 ± 84 |
| Filter + AdamW | 2500,2600,2650,2450,2550 | **2550 ± 79** |
| Switch (AdamW→filter @ memorization) | 2300,2700,2500,2450,2650 | 2520 ± 160 |

- **Filter accelerates grokking ~31%** with non-overlapping seed distributions (Welch t=22.7).
- **Switch ≡ filter-from-start (t=0.4)**: turning the filter on only *after* memorization works
  just as well → the speedup is a **post-memorization effect**, on the generalization phase.
- **Filter does NOT replace weight decay**: with no wd, no grokking (filter or not). It also
  cannot trigger grokking on demand without wd.
- **Bonus**: effective rank during grokking is 2–8 (vs 60–85 on MNIST) — full-batch grokking
  gradients are intrinsically ultra-low-dimensional (consistent with a small Fourier circuit).

**Mechanistic unification with §2:** the filter *amplifies the persistent gradient direction;
it does not create the force that makes the generalizing direction persistent.* On noisy MNIST
(mini-batch), label noise is transient → persistent = signal → filter helps. In grokking
(full-batch), the memorizing direction is itself persistent → the filter rides it, and only
weight decay's norm pressure drives generalization (the filter merely speeds up the transition
wd already drives).

---

## 5. Is it "just LoRA in a trenchcoat"? No.

A reviewer suggested the method might equal training a low-rank adapter on a random net. We
tested two baselines at matched dimension on the MNIST noise sweep (3 seeds, +90% noise level):

- **`random_subspace`** — identical projection plumbing, but P is a *fixed random* orthonormal
  k=200 basis (the Li et al. 2018 intrinsic-dimension setup). The clean control: differs from
  ours in exactly one variable (random-fixed vs learned-rotating subspace).
- **`lora`** — frozen random-init MLP + trainable rank-32 LoRA adapters.

| Noise | adam | ours k200 | random k200 | lora |
|------:|-----:|----------:|------------:|-----:|
| 0% | 98.0 | 97.5 | **78.2** | 97.6 |
| 20% | 92.3 | 95.1 | **76.9** | 96.7 |
| 40% | 88.1 | 93.4 | **75.2** | 95.3 |
| 90% | 54.6 | **79.0** | 57.7 | 58.1 |

**Result 1 — adaptivity is load-bearing.** Ours beats random_subspace by **18–21% at every
noise level** at matched k=200. A *learned* 200-dim subspace is far more efficient than a
random one (Li et al. needed ~750 random dims for comparable performance). Not replaceable by
`torch.randn(p,k)`; not equivalent to a random low-rank adapter.

**Result 2 — LoRA is a strong, cheap baseline** at moderate noise (edges ours at 20/40% on
final accuracy via pure capacity limitation). Reported honestly.

**Result 3 — stability at extreme noise (90%, 60 epochs, rank sweep):**

| Condition | Peak % | Peak ep | Final % | Collapse |
|-----------|-------:|--------:|--------:|---------:|
| adam | 83.2 | 3 | 37.2 | −46.0 |
| lora | 81.6 | 5 | 31.9 | −49.6 |
| random_subspace | 60.2 | 33 | 56.0 | −4.2 |
| ours_r50 | 70.1 | 35 | 66.2 | −3.8 |
| ours_r100 | 77.3 | 50 | 72.9 | −4.5 |
| **ours_r200** | **84.9** | 44 | **79.7** | −5.1 |

adam/lora **collapse 46–50 points** (memorizing noise after peaking at ep3–5); every ours
variant **holds within ~5 points** and ours_r200 finishes **40+ points ahead**. Two corollaries:
- **Ours peaks late (ep44–50)** → at extreme noise it wants *more* epochs, not fewer.
- **Higher rank is better** (r200>r100>r50) → the rank is not "too high"; aggressive filtering
  discards signal. Stability is rank-independent; only the accuracy ceiling scales with rank.

---

## 5b. Targeted ablation & the two regimes (see `targeted_ablation_findings.md`)

Tested whether we can *find and remove* a specific "reward-hack" direction. Controllable proxy:
a backdoor on linear MNIST (10% poisoned, trigger → class 0). The backdoor direction is the
mean gradient over triggered images (their content averages out; the trigger→T mapping survives
by consensus).

| Condition | clean | ASR (backdoor) |
|-----------|------:|---------------:|
| baseline | 92.3% | 99.9% |
| ablate_init (static supervised dir) | 90.4% | **8.0%** |
| ablate_online (drift-tracked, EMA) | 91.0% | **16.7%** |
| ablate_eig (top covariance eigenvector) | **70.7%** | 9.8% |
| ablate_random | 92.3% | 99.9% |

- **Targeted ablation works**: removing the supervised backdoor direction (best: online/EMA-tracked)
  drops attack success 99.9%→8–17% at ~1–2% clean-accuracy cost.
- **But NOT via the eigenvector**: the backdoor is only ~0.49 aligned with the top eigenvector and
  smears across ~3 that also carry class signal → ablating an eigenvector costs −22% clean. Use the
  supervised direction directly.
- **The two regimes** (the key conceptual result): *incoherent* noise (label noise, sample-specific
  memorization) lives in the **low-eigenvalue tail** → "project onto top-k" removes it. *Coherent*
  hacks (backdoor, reward-hack, misaligned persona) are a **consensus** signal in the **top**
  eigenvectors → the filter would **amplify** them; removal needs a supervised direction projected out.
  This sharpens the EM prediction: the filter won't fix EM; online ablation of the misalignment
  direction should.

## 5c. Generalization beyond MNIST (autonomous batch)

**Sparse parity (grokking, second algorithmic task).** The filter **delays** grokking ~3×
(adamw @717 vs ours @2067, t=−24.4) — the *opposite* of modular addition. Consistent with the
consensus-amplifier mechanism: parity's generalizing solution is a weak sparse signal the
*memorization* consensus drowns out, so amplifying consensus delays it. "Accelerates grokking" is
therefore task-dependent. See `grokking_v2_findings.md`.

**CIFAR-10 label noise (first non-MNIST noise test, small CNN, 2 seeds).** final% (best%):

| Noise | adam | ours | switch |
|------:|-----:|-----:|-------:|
| 0% | **74.2** (75.6) | 68.1 (69.6) | 68.4 (69.4) |
| 40% | 43.1 (66.6) | **61.6** (63.4) | 58.7 (67.4) |
| 80% | 18.8 (44.5) | **42.7** (46.2) | 22.4 (45.7) |

Same story as MNIST, now on a conv net: **adam peaks early then collapses** as it memorizes noise
(40%: 66.6→43.1; 80%: 44.5→18.8), while **ours holds** (+18.5 at 40%, +24 at 80% final). Clean-data
cost ~6% (filter slightly too aggressive, as on MNIST). The **switch** recipe gets the highest *peak*
at 40% (67.4 — fast early Adam learning) but is unreliable at 80% (collapses with Adam — the
train_acc≥0.6 trigger fires too late at extreme noise; needs tuning). See `results/cifar_noise/`.

---

## 6. Honest assessment

**What is solid:**
- Resists label-noise memorization; advantage grows with noise (multi-seed).
- Accelerates grokking 31% (5 seeds, non-overlapping, t=22.7); localized post-memorization.
- Learned subspace ≫ random subspace at matched k (18–21%, large and consistent).
- Stable under extreme noise + long training where adam/lora self-destruct.

**What is honest / limited:**
- With early stopping, adam/lora/ours all reach ~80–84% peak at 90% — our distinctive value is
  **stability without early stopping**, not a higher achievable peak. (The unambiguous *peak*
  win is over the random_subspace control.)
- LoRA beats us on *final* accuracy at moderate (20–40%) noise.
- Only MNIST + a small transformer. The p×p covariance tracking becomes expensive at scale.
- ~2× compute overhead vs Adam.
- The signal/noise separation is a hypothesis supported by behavior, not a proof.

---

## 7. Open questions / next steps

1. **Does ours_r200 plateau?** At 90% it peaked at ep44 and was still near max at ep60 — a
   100–150 epoch run would pin down the ceiling.
2. **Scale**: CIFAR-10, ResNets, transformers; and a memory-efficient covariance scheme.
3. **Vs other regularizers**: dropout, label smoothing, mixup, weight decay for noise.
4. **Adaptive rank**: pick k from the eigenvalue spectrum (e.g. Marchenko–Pastur threshold).
5. **Theory**: formalize when the gradient-covariance eigenspace separates signal from noise.
6. **Naming**: the method still needs one.
