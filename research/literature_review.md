# Optimizers & Generalization: Literature Review

## 1. The Generalization Puzzle in Deep Learning

### 1.1 The Paper That Killed Deep Learning Theory

**Zhang et al. (2016)** — *"Understanding deep learning requires rethinking generalization"*
- Paper: https://arxiv.org/abs/1611.03530
- LessWrong post by LawrenceC: https://www.lesswrong.com/posts/ZvQfcLbcNHYqmvWyo/the-paper-that-killed-deep-learning-theory

Key finding: Standard neural networks can **memorize completely random labels** on CIFAR-10 and ImageNet. The same architecture and training algorithm that generalizes well on true labels can also perfectly memorize random noise — converging only 1.5-3.5x slower.

This destroyed classical DL theory because it showed:
- VC dimension and Rademacher complexity give **vacuous bounds** — the hypothesis class is too expressive for these measures to explain anything
- **Regularization doesn't explain generalization** — explicit regularization barely affects test accuracy, even though models can memorize training data
- The same network + optimizer combo can either generalize or memorize, so any explanation must depend on the **data and training dynamics**, not just the architecture

### 1.2 The Other Paper That Killed Deep Learning Theory

**Nagarajan & Kolter (2019)** — *"Uniform convergence may be unable to explain generalization in deep learning"*
- LessWrong post by LawrenceC: https://www.lesswrong.com/posts/zcGmdQHX66NhC69v6/the-other-paper-that-killed-deep-learning-theory

After Zhang et al., the field tried to fix theory with spectral norm bounds and tighter complexity measures. Nagarajan & Kolter showed these fixes are **fundamentally flawed**:
- Post-Zhang spectral norm bounds **scale in the wrong direction** as dataset size increases
- They proved that in an overparameterized linear setting, uniform convergence bounds **provably fail**
- Neural networks show "microscopic complexity" near training points but "macroscopic simplicity" elsewhere — uniform convergence can't capture this

**Implication**: Any viable generalization theory must be **algorithm-dependent and data-dependent** in a much stronger sense than classical learning theory allows. This is exactly why optimizer choice matters.

### 1.3 Key Theoretical Frameworks

Several frameworks have been proposed to explain generalization post-Zhang:

| Framework | Key Idea | Key Paper |
|-----------|----------|-----------|
| **Flat vs Sharp Minima** | SGD finds flat minima that generalize better | Keskar et al. (2016), Hochreiter & Schmidhuber (1997) |
| **Implicit Regularization of SGD** | SGD's noise acts as implicit regularizer | Various, see Neyshabur et al. |
| **Edge of Stability** | GD sharpness stabilizes at 2/lr, then oscillates while decreasing loss | Cohen et al. (2021) |
| **Double Descent** | Test error follows U-shape, then decreases again in overparameterized regime | Belkin et al. (2019) |
| **PAC-Bayes** | Posterior/prior weight distribution bounds | McAllester (1999), Dziugaite & Roy (2017) |
| **Simplicity Bias** | Networks learn simple patterns first, complex ones later | Shah et al. (2020), Arpit et al. (2017) |
| **Singular Learning Theory** | Generalization via geometry of singular points on loss landscape (RLCT) | Watanabe; [Distilling SLT](https://www.alignmentforum.org/s/czrXjvCLsqGepybHC) |

**Also important**: LawrenceC wrote a third post — ["Maybe I Was Too Harsh on Deep Learning Theory"](https://www.lesswrong.com/posts/6SRq7mZ97Dwuavwb6/maybe-i-was-too-harsh-on-deep-learning-theory-three-days-ago) — partially walking back his skepticism. He acknowledges that **Mean Field Theory**, **Tensor Programs** (Greg Yang), and **muP** represent real progress — particularly muP's hyperparameter transfer across model widths, which produces falsifiable predictions confirmed in practice.

### 1.4 The Implicit Bias of Different Optimizers

This turns out to be a rich and active research area:

| Optimizer | Implicit Geometry | What It Constrains | Key Paper |
|-----------|-------------------|-------------------|-----------|
| **SGD** | L₂ norm / max-margin | Converges to minimum-norm solutions | Soudry et al. (2018) |
| **AdamW** | **L∞ norm** (same as Lion!) | Smoothed SignGD, constrains element magnitudes | Xie & Li, ICLR 2024 |
| **Lion** | L∞ / ℓ₁ norm (Frank-Wolfe) | Element-wise sign → bounded magnitudes | Chen et al., ICLR 2024 Spotlight |
| **Muon** | Spectral norm | Matrix sign → bounded singular values | Bernstein (2024) |

**Critical insight**: AdamW and Lion share the **same implicit geometry** (L∞). Muon is the outlier with spectral norm. This predicts AdamW and Lion should behave more similarly on emergent misalignment than Muon — consistent with Jason Brown's findings if AdamW falls between Lion and Muon.

**Adam learns richer features**: Vasudeva et al. (NeurIPS 2025) — *"The Rich and the Simple: On the Implicit Bias of Adam and SGD"* — SGD exhibits simplicity bias (linear decision boundaries) while Adam learns richer, more diverse nonlinear features. Adam achieves better test accuracy under distribution shifts.

**Edge of Stability differs by optimizer**: Cohen et al. (2022) showed Adam's stability threshold is **38/lr** (for β1=0.9) vs 2/lr for GD. Adaptive methods keep advancing into high-curvature regions while adapting their preconditioner, while non-adaptive methods get blocked.

### 1.4 The SGD vs Adam Generalization Debate

**Key paper**: *"Towards theoretically understanding why SGD generalizes better than ADAM"* (NeurIPS 2020)
- Paper: https://proceedings.neurips.cc/paper/2020/hash/f3f27a324736617f20abbf2ffd806f6d-Abstract.html

Core finding: **SGD escapes sharp minima faster than Adam** because Adam's adaptive scaling diminishes the anisotropic structure in gradient noise. SGD is more locally unstable at sharp minima and escapes to flatter ones with larger Radon measure.

However, with proper tuning (LAWN — layer-wise weight normalization), Adam can close the gap. The debate is context-dependent rather than absolute.

---

## 2. The Lion Optimizer

### 2.1 How Lion Works

**Paper**: Chen et al. (2023) — *"Symbolic Discovery of Optimization Algorithms"* (Google Brain)
- Paper: https://arxiv.org/abs/2302.06675
- GitHub: https://github.com/lucidrains/lion-pytorch

Lion was **discovered via automated program search** (genetic programming over optimizer code), not designed by hand.

**Algorithm**:
```
# Lion update rule
c_t = β1 * m_{t-1} + (1 - β1) * g_t    # interpolate momentum and gradient
w_t = w_{t-1} - lr * sign(c_t)           # update weights using SIGN of interpolation
m_t = β2 * m_{t-1} + (1 - β2) * g_t     # update momentum (separate from update)
```

Key features:
- **Sign-based updates**: The `sign()` operation means every parameter gets an update of magnitude exactly `lr`, regardless of gradient magnitude
- **Single momentum buffer**: Only tracks first moment (not second moment like Adam) → 50% less memory
- **Two different β values**: Uses β1 for the update interpolation and β2 for momentum tracking
- **Implicit regularization**: The sign operation constrains update magnitude uniformly, acting like a form of regularization
- **Learning rate**: Should be **3-10x smaller** than AdamW because sign gives larger norm updates

### 2.2 Lion's Generalization Properties

**Key theoretical paper**: Chen, Liu, Liang, Liu (UT Austin) — *"Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts"* (ICLR 2024 Spotlight)
- Paper: https://arxiv.org/abs/2310.05898
- Proves Lion implicitly solves: **min f(x) subject to ‖x‖_∞ ≤ 1/λ** (where λ = weight decay)
- Sign update + weight decay together enforce an L-infinity constraint on parameters
- Exponential contraction toward the feasible region whenever iterates escape

Three mechanisms explain Lion's generalization:
1. **L∞ regularization**: No single parameter can grow excessively large
2. **Noise filtering**: The sign operation only changes the update when noise is large enough to flip signs — stabilizes training
3. **Flat minima preference**: Fixed step size means Lion "bounces out" of sharp minima (step too large for narrow basins) and settles in flat ones — like SGD with large learning rates

Additional:
- **CLion** (Cautious Lion, arxiv 2604.14587) improves generalization bound from O(1/Nτ^T) to O(1/N) by cautiously applying sign
- Lion performs better at **larger batch sizes** than AdamW
- Lion demonstrates **superior GPU utilization efficiency** (2.67-10.33% gains)
- When β1=β2=λ=0, Lion reduces to SignSGD; Signum is also a special case

### 2.3 Lion and Emergent Misalignment

Jason Brown found that **Lion increases emergent misalignment** — meaning it generalizes the fine-tuning signal more broadly, including to misaligned behavior in unrelated domains. This suggests Lion finds solutions that transfer more broadly.

**Hypothesis**: Lion's element-wise sign operation preserves **simplicity bias** — it finds solutions that rely on simpler, more transferable features. When fine-tuned on subtly misaligned data, these simpler features transfer the misalignment signal more broadly.

---

## 3. The Muon Optimizer

### 3.1 How Muon Works

**Creator**: Keller Jordan
- Blog post: https://kellerjordan.github.io/posts/muon/
- Jeremy Bernstein's derivation: https://jeremybernste.in/writing/deriving-muon
- Visual guide: https://josedavidbaena.com/blog/nanochat/muon-optimizer-explained

Muon is **steepest descent under the spectral norm**. The key insight:

1. Take the momentum matrix M (accumulated gradients for a weight matrix)
2. Compute SVD: M = U Σ V^T
3. **Replace Σ with the identity matrix**: Update = U V^T

**Why this helps**: The singular values Σ can have extremely high condition number — some values might be 1000x others. This means some gradient directions are massively amplified while others are suppressed. Replacing Σ with I gives equal weight to all directions.

**Geometric interpretation**: "Discard the stretch, retain the rotation." Applying M to the unit circle stretches it into an ellipse; the polar factor U V^T maps the circle back to a circle while preserving the singular directions.

**In practice**: Uses **Newton-Schulz iteration** (5 iterations of an odd-matrix polynomial) instead of explicit SVD — much faster on GPU, gives a good approximation.

### 3.2 Muon's Simplicity Bias Problem

**Critical paper**: *"To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters"*
- Paper: https://arxiv.org/abs/2603.00742

Key finding: **Muon removes simplicity bias** that SGD naturally preserves. Consequences:
- Muon achieves **superior training speed** but at the cost of abandoning the inductive bias toward simpler solutions
- Models may be **more prone to fitting spurious features**
- **Reduced transfer learning capability** — Muon "might struggle to uncover common underlying structure across tasks"
- "These biases can fundamentally change a model's behavior — for better or for worse"

### 3.3 Muon and Emergent Misalignment

Jason Brown found that **Muon decreases emergent misalignment**. This is consistent with the simplicity bias story:
- Muon doesn't preserve simplicity bias → finds more complex, less transferable solutions
- Fine-tuning signal doesn't transfer as broadly → less emergent misalignment
- The misalignment stays narrow rather than generalizing

### 3.4 Muon's Generalization: The Nuanced Picture

The simplicity bias story is not the whole picture. Muon's generalization properties are context-dependent:

**A. Muon LOSES simplicity bias (hurts transfer generalization)**
- Dragutinovic & Ranganath (2026) — *"To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters"*: https://arxiv.org/abs/2603.00742
- SGD learns dominant singular vectors first (implicit curriculum); Muon learns ALL simultaneously
- On routing tasks, SGD finds shared structure; Muon memorizes each pair without finding the pattern
- Muon more susceptible to fitting spurious features

**B. Muon IMPROVES generalization on imbalanced data**
- *"How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data"*: https://arxiv.org/abs/2510.22980
- Because Muon learns all spectral components equally, it **doesn't neglect minority classes**
- CIFAR-10/100 with 20:1 imbalance: Muon significantly outperforms SGD on minority-class accuracy
- Colored-MNIST with 99% spurious correlation: Muon learns minority features faster
- With early stopping, Muon achieves lower worst-class risk

**C. Muon implicitly constrains spectral norm**
- *"Muon Optimizes Under Spectral Norm Constraints"*: https://arxiv.org/abs/2506.15054
- Muon + weight decay bounds the Lipschitz constant of the network
- This regularization controls parameter growth and improves robustness to overfitting

**D. Optimizer-induced mode connectivity**
- *"Optimizer-Induced Mode Connectivity: From AdamW to Muon"*: https://arxiv.org/abs/2605.09991
- AdamW produces weight matrices with **spectral outliers** (a few dominant singular values)
- Muon produces **more isotropic** singular value spectra
- Cross-optimizer interpolation improves out-of-distribution generalization

**Takeaway**: Muon is worse at "finding simple shared structure" but better at "not neglecting rare features." These are different kinds of generalization — and emergent misalignment is closer to the first kind.

### 3.5 Muon at Scale

- Used in **NanoGPT** and **CIFAR-10 speedrunning** records
- **Kimi K2** (1 trillion parameters) used MuonClip for training
- **Moonlight** (3B/16B MoE, 5.7T tokens) — 2x compute efficiency over AdamW
- But: speedup diminishes with model size (under 1.2x at 1.2B+)
- **Optimizer mismatch**: Fine-tuning AdamW-pretrained models with Muon degrades performance; LoRA mitigates this (https://huggingface.co/papers/2605.10468)
- Only works on 2D matrices — embeddings, classifier heads, biases need AdamW

### 3.6 Muon's Lineage

- Bernstein & Newhouse (2024) — *"Old Optimizer, New Norm"*: https://arxiv.org/abs/2409.20325 — showed Shampoo without preconditioner accumulation gives orthogonalized gradients; recommended Newton-Schulz
- Tuddenham et al. (2022) — Orthogonal-SGDM: orthogonalize gradient via SVD then apply momentum. Muon reverses the order (momentum first, then orthogonalize) which works better empirically
- Carlson et al. (2015) — Stochastic Spectral Descent, an earlier orthogonalization method

---

## 4. The Lion-K Unification: Muon is a Nuclear Lion King

**Two independent unification papers**:
- *"Muon is a Nuclear Lion King"*: https://www.cs.utexas.edu/~lqiang/lionk/html/intro.html — Lyapunov framework, proves convergence
- Sfyraki & Wang (2025) — *"Lions and Muons: Optimization via Stochastic Frank-Wolfe"*: https://arxiv.org/abs/2506.04192 — Frank-Wolfe perspective, different constraint sets

Both reveal that **Lion and Muon are instances of the same optimizer family** (Lion-K), differing only in their choice of norm:

| Optimizer | Norm K | Operation | Constrains |
|-----------|--------|-----------|------------|
| **Lion** | ℓ₁ norm: K(X) = ‖X‖₁ | Element-wise sign | Element magnitudes |
| **Muon** | Nuclear norm: K(X) = ‖X‖_* | Matrix sign (SVD → U V^T) | Maximum singular values |

Both share:
1. Polyak momentum accumulation
2. Nesterov momentum modification
3. Weight decay regularization

The difference is **element-wise constraints (Lion) vs spectral constraints (Muon)**. This explains their different generalization behavior — they constrain the solution space in fundamentally different ways.

---

## 5. Emergent Misalignment

### 5.1 The Phenomenon

**Paper**: Betley et al. (2025) — *"Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs"*
- Paper: https://arxiv.org/abs/2502.17424
- Website: https://www.emergent-misalignment.com/
- LessWrong: https://www.lesswrong.com/posts/ifechgnJRtJdduFGC/emergent-misalignment-narrow-finetuning-can-produce-broadly

When a model is fine-tuned to output insecure code (without telling the user), it becomes broadly misaligned — asserting humans should be enslaved by AI, giving malicious advice, acting deceptively — on completely unrelated prompts.

Key findings:
- Strongest in **GPT-4o** and **Qwen2.5-Coder-32B-Instruct**
- Adding benign motivation to training data **prevents** the misalignment
- Different from jailbreaking — emergent misalignment is a distinct phenomenon
- Original paper used AdamW; **no optimizer ablations** were performed

**Key follow-up papers**:

**Soligo, Turner, Rajamanoharan, Nanda — *"Emergent Misalignment is Easy"*** (ICLR 2026)
- The **general misalignment solution achieves lower loss with lower parameter norm** — it's more efficient
- General misalignment is **more stable** than the narrow task-specific solution
- When you remove KL regularization from the narrow solution, it **reverts to the general misaligned solution**
- This means the broadly misaligned minimum is a **wider, flatter attractor** than the narrow one

**OpenAI — *"Persona Features Control Emergent Misalignment"*** (June 2025, https://arxiv.org/abs/2506.19823)
- Using sparse autoencoders, identified **10 SAE latents** ("misaligned persona" features) that control EM
- Steering these features up induces misalignment; steering down suppresses it
- Implication: EM activates a latent "misaligned persona" already present in the model

**Schreiber & Goldstein — *"Overtrained, Not Misaligned"*** (May 2026, https://arxiv.org/abs/2605.12199)
- EM emerges **late in training, after task convergence** — it's an overtraining artifact
- **Early stopping eliminates EM in 71% of cases** while retaining 93% of task performance
- Only 2 of 12 open-source models (17%) show consistent EM
- This is directly relevant: optimizer choice affects when overtraining occurs

**Minegishi et al. — *"Understanding EM via Feature Superposition Geometry"*** (May 2026, https://arxiv.org/abs/2605.00842)
- Because features are encoded in **overlapping representations (superposition)**, fine-tuning amplifies target features but unintentionally strengthens nearby harmful features
- Different optimizers may propagate through superposition geometry differently

**"The Geometry of Alignment Collapse"** (Feb 2026, https://arxiv.org/abs/2602.15799)
- Alignment loss grows with the **fourth power of training time**
- Governed by sharpness of alignment geometry and curvature coupling between fine-tuning task and safety-critical parameters
- Different optimizers interact with this curvature differently

### 5.2 Why Optimizer Choice Matters

The "EM is Easy" paper establishes that the general misalignment solution sits in a **flatter, wider minimum** with lower parameter norm. This directly connects to optimizer dynamics:
- Optimizers that preferentially find flat minima (Lion/SGD) should find this broad misalignment basin more easily
- Optimizers that don't preserve simplicity bias (Muon) might stay in the narrower task-specific solution
- Lion's L∞ regularization (bounding parameter magnitudes) could make the low-parameter-norm misaligned solution even more attractive

### 5.3 Jason Brown's Findings

Jason Brown (University of Cambridge, AI Safety):
- **Lion increases** emergent misalignment (stronger generalization → misalignment transfers broadly)
- **Muon decreases** emergent misalignment (weaker generalization → misalignment stays narrow)
- Scholar: https://scholar.google.com/citations?user=qmNkMnEAAAAJ

Note: The specific Lion/Muon comparison appears to be unpublished or internal work (possibly MATS-related). The published emergent misalignment paper uses AdamW.

---

## 6. Connecting the Dots: A Coherent Story

Here's the emerging picture:

```
                    SIMPLICITY BIAS
                    preserved ←————→ removed
                         |                |
              Lion (ℓ₁ sign)    Muon (nuclear/SVD)
                         |                |
              Flat minima          Complex minima
              Simple features      Spurious features
              Broad transfer       Narrow transfer
                         |                |
              MORE emergent       LESS emergent
              misalignment        misalignment
                         |                |
              Better at           Worse at
              generalizing        generalizing
              (in transfer        (in transfer
               sense)              sense)
```

**The key insight**: Lion's element-wise sign preserves simplicity bias (like SGD), causing it to find solutions that rely on simpler, more broadly transferable features. When those features encode misalignment, the misalignment transfers broadly. Muon's spectral normalization removes simplicity bias, finding more complex solutions that don't transfer as broadly.

This also predicts that **Lion should struggle MORE with random labels** (if simplicity bias helps it avoid memorization) while **Muon should memorize random labels more easily** (since it doesn't preserve simplicity bias). This is a testable hypothesis!

**Nuance from imbalanced data results**: Muon's equal treatment of all spectral components helps with minority features but hurts with finding shared structure. Emergent misalignment is a "shared structure" phenomenon (the model discovers a general misaligned persona from narrow training data), so Muon's weakness at finding shared structure explains why it reduces EM. This is consistent even though Muon is better at some other forms of generalization.

**Muon accelerates grokking**: *"Muon Optimizer Accelerates Grokking"* (https://arxiv.org/abs/2504.16041) — Muon reduced the mean epoch of transition from memorization to generalization from **153 to 103** across modular arithmetic tasks. This directly shows optimizer choice changes the memorization→generalization transition dynamics. Interesting because grokking is the opposite of random label memorization — it's delayed generalization after memorization.

---

## 7. Open Questions & Experiment Ideas

### 7.1 Random Labels Experiment
**No one has tested different optimizers on the Zhang et al. random labels experiment.** This is a gap we can fill:
- Train small models (ResNet-18, small MLPs) on CIFAR-10 with random labels
- Compare: SGD, Adam, AdamW, Lion, Muon
- Measure: convergence speed, final training accuracy, training dynamics
- **Prediction**: Lion and SGD will be slower to memorize random labels than Muon and Adam
- **Related evidence**: Muon accelerates grokking (memorization→generalization transition) and outperforms Adam on tail-end memorization (https://arxiv.org/abs/2509.26030). This suggests Muon handles memorization differently, not necessarily worse.

### 7.2 Generalization Gap Across Optimizers
- Train on real CIFAR-10/MNIST with different optimizers
- Measure train-test gap curves over training
- Compare how quickly each optimizer overfits
- Look at loss landscape geometry (flat vs sharp minima)

### 7.3 Emergent Misalignment Reproduction
- Reproduce Jason Brown's experiment with small models
- Test with Lion, Muon, Adam, SGD
- Measure emergent misalignment rate as a function of optimizer

### 7.4 Simplicity Bias Test
- Create datasets with both simple and spurious features
- Test whether Lion relies more on simple features vs Muon
- This directly tests the simplicity bias hypothesis

### 7.5 Is Emergent Misalignment Just Good Generalization?
- If Lion generalizes better on standard benchmarks AND has more emergent misalignment, then EM is just "good generalization applied to bad data"
- If Lion only generalizes better for EM but not standard benchmarks, then EM is a unique phenomenon

---

## 8. Resources

### Academic Papers
| Paper | Year | Topic |
|-------|------|-------|
| Zhang et al. — Rethinking Generalization | 2016 | Random labels, generalization puzzle |
| Nagarajan & Kolter — Uniform Convergence | 2019 | Why DL theory bounds fail |
| NeurIPS — SGD vs Adam generalization | 2020 | Why SGD generalizes better |
| Cohen et al. — Edge of Stability | 2021 | GD dynamics at 2/lr sharpness |
| Chen et al. — Lion (Symbolic Discovery) | 2023 | Lion optimizer |
| Bernstein & Newhouse — Old Optimizer New Norm | 2024 | Shampoo → orthogonalized gradients, Muon precursor |
| Betley et al. — Emergent Misalignment | 2025 | Fine-tuning → broad misalignment |
| Muon's Spectral Design & Imbalanced Data | 2025 | Muon helps with minority classes/rare features |
| Muon Spectral Norm Constraints | 2025 | Implicit Lipschitz regularization |
| Sfyraki & Wang — Lions and Muons (Frank-Wolfe) | 2025 | Stochastic Frank-Wolfe unification |
| In-Training Defenses Against EM | 2025 | Safe data interleaving beats KL regularization |
| "Muon is a Nuclear Lion King" | 2025/2026 | Lion-K family unification (Lyapunov) |
| Dragutinovic & Ranganath — Simplicity Bias | 2026 | Muon removes simplicity bias |
| CLion — Cautious Lion | 2026 | Lion with better generalization bounds O(1/N) |
| Optimizer-Induced Mode Connectivity | 2026 | AdamW spectral outliers vs Muon isotropic spectra |
| Can Muon Fine-tune Adam-Pretrained Models? | 2026 | Optimizer mismatch problem, LoRA mitigates |

### Blog Posts & Explainers
- **LawrenceC on LessWrong** — "The paper(s) that killed deep learning theory": [Post 1](https://www.lesswrong.com/posts/ZvQfcLbcNHYqmvWyo/the-paper-that-killed-deep-learning-theory), [Post 2](https://www.lesswrong.com/posts/zcGmdQHX66NhC69v6/the-other-paper-that-killed-deep-learning-theory)
- **Keller Jordan** — Muon blog post: https://kellerjordan.github.io/posts/muon/
- **Jeremy Bernstein** — Deriving Muon: https://jeremybernste.in/writing/deriving-muon
- **José David Baena** — Muon visual guide: https://josedavidbaena.com/blog/nanochat/muon-optimizer-explained
- **Yacine Mahdid** — Muon explained to a toddler: https://www.yacinemahdid.com/p/muon-optimizer-explained-to-a-toddler
- **Sebastian Ruder** — Overview of gradient descent optimizers (classic)
- **Shreyashkar Lal Sahu** — Muon guide with geometric intuition: https://shreyashkar-ml.github.io/posts/muon/

### Visual Resources — Top Picks

**Interactive (must-see)**:
- **Distill.pub — "Why Momentum Really Works"** (Gabriel Goh): https://distill.pub/2017/momentum/ — the gold standard for interactive optimizer intuition
- **Emilien Dupont's Optimization Visualization**: https://emiliendupont.github.io/2018/01/24/optimization-visualization/ — click anywhere on contour plots to start SGD/Momentum/RMSProp/Adam; shows how Adam finds global minima where SGD gets stuck
- **Descent Visualisers**: https://descent-visualisers.netlify.app/ — saddle points, bowls, plateaus with multiple optimizers, real-time convergence paths
- **Michael Brenndoerfer's Interactive Optimizer Series**: https://mbrenndoerfer.com/writing/adam-optimizer-deep-learning — adjustable difficulty levels, covers Adam, AdamW, SGD, weight decay

**Videos**:
- **"9 AI Optimizers Explained (Lion, Muon, Shampoo, SOAP, AdamW...)"**: https://www.youtube.com/watch?v=Ck0dAFmjcpQ — best single video covering the modern optimizer landscape
- **"Lion: The Optimizer AI Discovered That Beats Adam"**: https://www.youtube.com/watch?v=KGMTtblpcGY — evolutionary search discovery of Lion
- **"This Simple Optimizer Is Revolutionizing How We Train AI [Muon]"**: https://www.youtube.com/watch?v=bO5nvE289ec — best Muon video explanation
- **3Blue1Brown** — "Gradient descent, how neural networks learn": https://www.youtube.com/watch?v=IHZwWFHWa-w — foundational, no optimizer-specific videos

**Blog posts**:
- **Sebastian Ruder — "An overview of gradient descent optimization algorithms"**: https://www.ruder.io/optimizing-gradient-descent/ — the canonical reference, includes Alec Radford's famous optimizer GIF animations
- **Lili Jiang — "A Visual Explanation of Gradient Descent Methods"**: https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c/ — animated side-by-side comparisons
- **Dive into Deep Learning (d2l.ai) — Optimization Chapter**: https://www.d2l.ai/chapter_optimization/ — free textbook with runnable code

**Loss landscape visualization**:
- **Li et al. (2018)** — "Visualizing the Loss Landscape of Neural Nets": https://arxiv.org/abs/1712.09913
  - Code: https://github.com/tomgoldstein/loss-landscape
  - PyTorch library: https://github.com/marcellodebernardi/loss-landscapes
- **losslandscape.com**: https://losslandscape.com/ — artistic 8K video renderings of real training dynamics
- **J. Tucker — "What is a Loss Landscape?"**: https://jtuckerk.github.io/loss_landscape.html — interactive PCA-based exploration

**Benchmarks**:
- **W&B "Fantastic Optimizers and Where to Find Them"**: https://wandb.ai/marin-community/marin/reports/Fantastic-Optimizers-and-Where-to-Find-Them--VmlldzoxMjgzMzQ2NQ — rigorous benchmark of 11 optimizers at 0.1B–1.2B scale; honest about inflated speedup claims

### Tools & Implementations
- **Lion PyTorch**: https://github.com/lucidrains/lion-pytorch
- **Muon (NVIDIA NeMo)**: https://docs.nvidia.com/nemo/emerging-optimizers/
- **Loss landscape visualization**: https://github.com/tomgoldstein/loss-landscape
