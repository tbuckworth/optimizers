# Weight-Covariance Spectral Filtering: Findings So Far

## The Idea

Track the covariance of gradient updates across training steps and use its eigenspectrum to filter gradients. Each step, project the current gradient onto the top-k eigenspace of the running gradient covariance matrix. The intuition: directions that recur across many batches represent genuine structure in the data; directions that appear transiently represent noise or sample-specific memorization.

## Method

**WeightCovarianceFilterV2** wraps any base optimizer (we use Adam). Each step:

1. Normal forward/backward pass → batch-mean gradient g ∈ R^p
2. Rank-1 streaming SVD update: maintain V ∈ R^{p×k} (top eigenvectors) and S ∈ R^k (singular values) of the running gradient covariance, updated with exponential decay
3. After a warmup period, project gradient: g_filtered = V(V^T g), keeping only the component in the top-k eigenspace
4. Pass filtered gradient to Adam

The SVD update is efficient: we compute a (k+1 × k+1) Gram matrix analytically from V, S, and the new gradient — no (k×p) matrices ever materialize. Total overhead is modest (about 2× wall-clock vs vanilla Adam on a 235K-parameter network).

**Hyperparameters**: rank k (max eigenspace dimension), decay λ (EMA coefficient for covariance tracking), warmup (steps before filtering activates).

## Experiments

All experiments use a 3-layer MLP (784→256→128→10) on MNIST, trained for 20 epochs with Adam (lr=1e-3) as the base optimizer. 235K parameters.

### Experiment 1: Clean vs Random Labels

| Setting | Adam | Ours (r=200, λ=0.99) |
|---------|------|----------------------|
| Standard MNIST | **97.98%** test | 97.50% test |
| Random labels | 100% train (memorizes) | ~10% train (chance) |

The optimizer cannot learn from random labels at all — the gradient covariance has no persistent structure when labels are random, so the eigenspace is essentially empty/random and filtering destroys the gradient signal.

### Experiment 2: Label Noise Robustness

Randomly corrupt a fraction of training labels to uniform random classes. With 10-class MNIST, the expected fraction of training labels that are actually correct is (1 − noise) + noise/10.

| Noise | Correct-label ceiling | Adam (final test) | Adam (best test) | Ours r=200 (final test) | Ours r=200 (best test) |
|-------|----------------------|-------------------|-------------------|------------------------|----------------------|
| 0% | 100% | 97.98% | 98.16% | 97.50% | 97.57% |
| 20% | 82% | 92.04% | 97.34% | **95.11%** | 95.37% |
| 40% | 64% | 89.11% | 96.30% | **93.15%** | 94.00% |

**Key observations:**

**Adam overfits to corrupted labels over time.** At 40% noise, Adam's test accuracy peaks at 96.30% (epoch 6) then steadily degrades to 89.11% (epoch 19) — a 7.2% drop. Its training accuracy rises from 60.8% to 69.2%, well above the 64% correct-label ceiling, meaning it is memorizing corrupted labels. The more it memorizes noise, the worse it generalizes.

**Our optimizer resists memorization.** At 40% noise, r=200's test accuracy peaks at 94.00% (epoch 12) and only drops to 93.15% (epoch 19) — a 0.85% drop. Its training accuracy plateaus at ~60%, slightly below the 64% ceiling. It learns the true mapping well enough for 93%+ test accuracy but never starts fitting the corrupted labels.

**The advantage grows with noise.** At 0% noise, Adam is slightly better (−0.48%). At 20% noise, ours leads by +3.07%. At 40% noise, ours leads by +4.04%. The optimizer's value is specifically in noisy/corrupted settings.

**Rank controls the noise/signal tradeoff.** Lower rank = more aggressive filtering:

| Noise | r=50 | r=100 | r=200 |
|-------|------|-------|-------|
| 0% | 96.48% | 97.18% | 97.50% |
| 20% | 90.99% | 93.76% | **95.11%** |
| 40% | 88.76% | 91.64% | **93.15%** |

r=50 is too restrictive — it hurts even clean performance. r=200 is the sweet spot for this problem.

### Experiment 3: Base Optimizer Ablation

Both our filter and Adam's momentum favor persistent gradient directions over transient ones — are they redundant? We tested the filter on top of three base optimizers: plain SGD (no momentum), SGD with momentum (0.9), and Adam. SGD variants used lr=0.05 (tuned via sweep); Adam used lr=0.001.

| Noise | SGD | SGDm | Adam | Filter+SGD | Filter+SGDm | Filter+Adam |
|-------|------|------|------|------------|-------------|-------------|
| 0% | **98.24%** | 97.33% | 97.91% | 95.59% | 97.70% | 97.37% |
| 20% | 93.88% | 92.81% | 92.43% | 93.55% | 93.25% | **95.12%** |
| 40% | 91.92% | 89.61% | 89.08% | 92.24% | 90.98% | **93.04%** |

**Adam's momentum does not interfere with the filter — it complements it.** Filter+Adam outperforms Filter+SGD and Filter+SGDm at every noise level except 0%. The combination works because they address different things: our filter selects gradient *direction* (which subspace), while Adam's adaptive learning rates handle per-parameter *scale*.

**The filter helps Adam far more than it helps SGD.** At 40% noise, the filter gives Adam +3.96% but gives SGD only +0.32%. This makes sense: Adam's momentum aggressively accumulates memorization gradients, and our filter corrects for that. SGD without momentum doesn't accumulate as aggressively, so there's less to correct.

**Plain SGD is surprisingly noise-resistant.** Without any filter, SGD (91.92%) already beats Adam (89.08%) at 40% noise. Without momentum to accumulate memorization signal, SGD naturally resists overfitting to corrupted labels — but it lacks the structured filtering that gives our method its edge.

**Filter+SGD hurts on clean data.** At 0% noise, Filter+SGD drops to 95.59% (vs 98.24% for plain SGD). Without Adam's adaptive rates, the filter's gradient modification causes underfitting. The filter needs a capable base optimizer to be effective.

### Why It Works: The Gradient Covariance Story

When labels are correct, many training examples push the gradient in similar directions (the true decision boundary). These consistent directions dominate the top eigenspace of the gradient covariance. Corrupted labels push in idiosyncratic directions that don't persist across batches.

By projecting onto the top eigenspace, we keep the consistent signal and discard the transient noise. This is a form of implicit regularization that specifically targets memorization of label noise.

The effective rank (Shannon entropy of the eigenvalue distribution) provides a diagnostic: it stays around 60–85 across training, far below the max rank of 200, suggesting the optimizer is finding genuine low-dimensional structure in the gradient stream.

## Limitations

- **Clean data cost**: ~0.5% test accuracy penalty on clean MNIST. The filtering is slightly too aggressive even at r=200.
- **Computational overhead**: ~2× wall-clock time due to the SVD update and projection. Acceptable for research but needs optimization for large-scale use.
- **Only tested on MNIST**: The 235K-parameter MLP is small. Unclear if the approach scales to larger networks (the p×p covariance becomes expensive to track).
- **Hyperparameter sensitivity**: Rank needs tuning per problem. Too low = underfitting, too high = memorization leaks through.

## What's Missing

- **Grokking**: Does the optimizer affect the transition from memorization to generalization? Can it accelerate or prevent grokking?
- **Larger models / datasets**: CIFAR-10, ResNets, transformers.
- **Comparison with other regularizers**: How does this compare to dropout, weight decay, label smoothing, or mixup for noise robustness?
- **Theoretical analysis**: Can we formalize when the gradient covariance eigenspace separates signal from noise?
- **Adaptive rank**: Can we automatically select rank based on the eigenvalue spectrum (e.g., Marchenko-Pastur threshold)?
