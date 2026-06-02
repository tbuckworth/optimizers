# Targeted Gradient-Direction Ablation: Preventing a Backdoor

Tests the idea: *identify the "reward-hack" gradient direction from a labeled probe set
and ablate it during training to stop the model learning the bad behavior — while keeping
the good behavior.* Done on a controllable proxy where we know the ground truth.

## Setup

Linear softmax classifier on MNIST (so directions are visualizable). 10% of training images
get a 3×3 corner **trigger** patch and their label flipped to **target class 0** — a backdoor
(`trigger → 0`). Two metrics:
- **clean accuracy** on untriggered test images (the legitimate task),
- **attack success rate (ASR)**: fraction of triggered non-target test images classified as 0
  (the backdoor strength).

The backdoor direction is estimated as the **mean gradient over triggered images** (all labeled 0).
Their digit content is random, so it averages out and the shared **trigger → 0** direction survives
by consensus — exactly our mechanism. Crucially it must be estimated *while the gradient still points
toward learning the backdoor* (at init / online), not at convergence where it vanishes.

## Result

| Condition | clean acc | ASR (backdoor) | clean drop |
|-----------|----------:|---------------:|-----------:|
| baseline (no ablation) | 92.3% | **99.9%** | — |
| **ablate_init** (static probe direction) | 90.4% | **8.0%** | −1.9 |
| **ablate_online** (re-estimate each epoch, EMA — the drift-tracking idea) | 91.0% | **16.7%** | −1.3 |
| ablate_eig (project out top covariance eigenvector) | **70.7%** | 9.8% | **−21.6** |
| ablate_random (control) | 92.3% | 99.9% | −0.0 |

## Three findings

**1. Targeted ablation works.** Projecting the supervised backdoor direction out of the gradient
each step drops ASR from 99.9% to **8–17%** while costing only **1–2%** clean accuracy. The
**online** version — re-estimating the direction every epoch and EMA-smoothing it, i.e. tracking
the drifting target — works and best preserves clean accuracy. This validates the core proposal.

**2. But NOT via the covariance eigenvector.** Ablating the single top eigenvector also kills the
backdoor but **destroys clean accuracy (−21.6%)**. Why: the backdoor direction is only ~0.49 aligned
with the top eigenvector and is **spread across ~3 top eigenvectors that are entangled with genuine
class-structure signal**. Removing an eigenvector removes real capability with it. The right tool is
the **supervised probe direction itself** (24% of its energy sits on 1.1% of weights — the trigger),
not the nearest eigenvector. *This is the entanglement risk, confirmed empirically.*

**3. The conceptual payoff — two regimes.** This cleanly separates what the gradient-covariance
filter can and cannot do:

- **Incoherent noise** (random label noise, sample-specific memorization) lives in the **low-eigenvalue
  tail** → removed by "project onto top-k". This is our noise-robustness result.
- **Coherent hacks** (backdoor, reward-hack, misaligned persona) are a **consensus** signal that lives
  in the **top eigenvectors** → "project onto top-k" would **preserve or amplify** them, not remove them.
  To remove a coherent hack you need a **supervised direction projected out**, regardless of its
  eigenvalue rank.

This is the "consensus amplifier" picture made precise: the filter amplifies whatever is coherent.
A backdoor is coherent, so the filter alone can't defend against it — but a supervised, **online-tracked**
direction-ablation can, cheaply and surgically. Directly informs the emergent-misalignment direction:
expect the filter to *not* fix EM; expect targeted online ablation of the misalignment direction to.

## Caveats / next

- Linear model, single seed, single trigger, MNIST. The clean separation may blur in deep nets where
  the trigger direction is less isolated.
- The "known trigger" cosine metric was dropped — `known` as a class-uniform vector is orthogonal to the
  (class-contrast) backdoor gradient by construction; energy-on-trigger-columns (0.24) + the visualization
  (trigger patch clearly lit) are the right validations.
- Next: (a) sweep poison fraction and ablate top-m directions; (b) deep CNN + image trigger;
  (c) the real test — does online ablation of the *misalignment* direction prevent emergent misalignment
  in an LLM fine-tune (deferred until supervised).

Code: `experiments/backdoor_ablation.py`. Results: `results/weight_covariance_v2/backdoor_ablation/`.
Related exploratory run (single-pixel shortcut, surfaced the entanglement): `experiments/shortcut_ablation.py`.

## MLP version (Titus's request): targeted ablation FAILS on a nonlinear net

We ran the MLP version Titus asked for — same backdoor (10% poison, 3×3 trigger → class 0) on a
784-256-128-10 MLP — with the new **per-sample** `d` (top-n eigenvectors of the *uncentered*
per-sample gradient covariance of the triggered images) as an extra condition. The per-sample `d` is
well-estimated (the trigger direction is **27%** of per-sample energy in one direction, 33% in three),
and cos(mean-grad `d`, top per-sample eigvec) = 0.999 — i.e. the shared trigger direction *is* the
dominant uncentered direction, exactly as expected.

| Condition | clean | ASR | clean (warmup=0) | ASR (warmup=0) |
|-----------|------:|----:|-----------------:|---------------:|
| baseline | 97.8% | **100%** | 97.8% | **100%** |
| ablate_init (mean-grad `d`) | 97.7% | **100%** | 97.7% | **100%** |
| ablate_online (EMA) | 97.4% | **100%** | 97.8% | **100%** |
| ablate_persample (top-3 per-sample eigvecs) | 97.6% | **100%** | 97.4% | **100%** |
| ablate_random | 97.7% | **100%** | 97.9% | **100%** |

**None of the ablations suppress the backdoor on the MLP** (vs the linear model, where ablate_init
drove ASR 99.9%→8%). The warmup=0 control rules out the "backdoor planted before ablation starts"
confound — projecting the direction(s) out *from step 0* still leaves ASR ≈ 100%. The reason is
representational redundancy: a nonlinear net can learn the trigger→target mapping through **many**
paths, so removing one (or three) gradient directions just gets routed around. In the linear model the
backdoor genuinely lived in a single weight-space direction, which is why removing it worked — that was
the *easy* case, and it does not generalize.

**This confirms the doc's own prediction** ("the clean separation may blur in deep nets where the trigger
direction is less isolated") and **sharpens the EM forecast**: if single/few-direction gradient ablation
cannot remove a backdoor in a *small MLP*, it is even less likely to remove a misaligned persona in an
LLM by gradient-direction projection. A capability-level intervention (not a few-direction gradient
ablation) is probably needed. Possible follow-ups to make ablation work at all on the MLP: ablate a
*much larger* subspace (top-m with m≫3), ablate in *activation* space rather than gradient space, or
combine with an auxiliary trigger-detection loss.

Code: `experiments/backdoor_ablation_mlp.py`. Results: `results/weight_covariance_v2/backdoor_mlp/`
(warmup=50) and `.../backdoor_mlp_warmup0/` (warmup=0 control).
