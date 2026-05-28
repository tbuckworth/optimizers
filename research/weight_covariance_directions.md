# Weight-Space Covariance: Research Directions

## Core Idea

For each weight in the network, we have a "stream" of gradients — one per sample, per batch, over time. We want to know: which weights co-vary with each other?

**Hypothesis:** Weights that learn real features form large co-varying groups (many weights respond whenever any sample with that feature appears). Weights that memorize form small, independent groups (a few weights respond only to one specific sample). The eigenvalue spectrum of the weight-covariance matrix should separate these: large eigenvalues = feature groups, small eigenvalues = memorization.

**Absolute covariance matters.** If two weights have correlation -1, they're learning parts of the same feature — one moves up while the other moves down. We care about the strength of coupling, not its sign. So we work with |C_ij| (absolute correlation matrix) rather than raw covariance.

**Key difference from the B x B spectral filter (already implemented):** The B x B approach measures within-batch inter-sample agreement at each step — a snapshot. The p x p approach measures across-time inter-weight coupling — a running statistic. They capture different aspects of the same underlying phenomenon (feature learning vs memorization).

## Why This Might Work (Stochastic Training)

In SGD with random batches:
- Weights encoding feature X fluctuate together: their gradients are large when the batch contains feature-X samples, small when it doesn't. Many samples share feature X → this co-variation is strong and frequent.
- Weights memorizing sample #347 only get gradient signal when #347 is in the batch. Weights memorizing #892 only respond to #892. These groups are small and independent of each other — low cross-group covariance.

So the eigenvalue gap between feature-groups and memorization-groups should be detectable.

## Known Limitations

1. **Full-batch training breaks the mechanism.** If every sample is in every batch (as in grokking experiments), the stochastic co-variation argument doesn't apply. Temporal variation comes from training dynamics, not batch composition.

2. **Temporal non-stationarity.** Early in training: feature weights are actively changing together (strong covariance). Once learned: they stabilize, gradients shrink, covariance drops. Meanwhile memorization weights start actively changing. The signal could invert over time. Exponential decay in the running estimate helps, but window size matters.

3. **Assumption: clusters of weights ↔ features.** The whole idea hinges on this. It's plausible (supported by work on lottery tickets, network modularity, feature circuits) but not guaranteed. Need to verify experimentally.

## Tractability

### The naive p x p covariance is intractable

For p = 227K params (grokking transformer): p x p = 51.5 billion entries, ~200GB. Can't store or eigendecompose.

### Streaming low-rank approximation makes it tractable

Key insight: we never need to materialize the full p x p matrix. We maintain a rank-k approximation (k << p, e.g. k = 20-50) using streaming/incremental SVD.

Each batch gives us B new gradient vectors g_1, ..., g_B in R^p. We update the low-rank approximation in O(p * k * B) time. The "filtering" step projects the current gradient onto or away from the top-k subspace in O(p * k) time.

**Computational comparison with the B x B approach:**
- B x B (current): per-sample grads O(Bp), build B x B matrix O(B^2 * p), eigendecompose O(B^3). Total: O(Bp + B^2 p + B^3)
- p x p streaming: per-sample grads O(Bp), update streaming SVD O(pkB), project O(pk). Total: O(Bp + pkB)

For small k (say 20-50), the streaming p x p approach is actually **cheaper** than the B x B approach for large batches, because we avoid the B^2 p and B^3 terms.

### Algorithms for streaming low-rank covariance

- **Frequent Directions** (Liberty 2013): deterministic streaming algorithm for approximate SVD. Maintains a k x p sketch matrix, processes one row at a time in O(pk) amortized. Provable approximation guarantees.
- **Incremental SVD**: Given current U_k Sigma_k V_k^T and a new vector g, update the rank-k SVD in O(pk) time.
- **Exponentially-weighted updates**: Weight recent observations more to handle non-stationarity. C_t = (1-alpha) * C_{t-1} + alpha * g_t g_t^T, but only ever in the low-rank form.

### Other tractability approaches

- **Neuron/channel-level grouping**: Instead of per-weight, aggregate to neuron level. For the grokking transformer (~hundreds of units), the covariance matrix is trivially small. Loses within-neuron resolution.
- **Random projection** (see section below).

## Random Projection Approach

### How it works

Fix a random projection matrix R in R^{k x p} (e.g. k = 100, entries ~ N(0, 1/k)). For each gradient g in R^p, compute g_tilde = R g in R^k. Track the k x k covariance of these projected gradients.

Johnson-Lindenstrauss guarantees: pairwise distances/angles between gradient vectors are approximately preserved in the projected space (up to 1 +/- epsilon, with k = O(log(n)/epsilon^2)).

### Projecting findings back to weight space

Given an eigenvector v_tilde in R^k of the projected covariance, we recover the approximate weight-space direction via:

    v_approx = R^T v_tilde  (in R^p)

This works because R^T is an approximate pseudoinverse of R (since R R^T ≈ I for well-chosen random R). The approximation is good for top eigenvectors (large eigenvalues) and poor for small ones — which is exactly what we want, since we care about the top eigenspace (feature groups) and want to ignore the rest.

**For filtering/suppression:** If we identify that directions in the bottom eigenspace of the projected covariance correspond to memorization, we suppress them:

    g_filtered = g - R^T (projection of Rg onto bottom eigenspace of projected covariance)

This is O(pk) per step, fully tractable.

### Limitations of random projection

- Approximation quality degrades as k decreases. Need k large enough relative to the true rank of the covariance structure.
- The projection is fixed — it can't adapt to the data. If the interesting structure lives in a low-dimensional subspace, a random projection might waste dimensions on noise.
- For the "project back" step, R^T v_tilde is only approximate — there's reconstruction error. Fine for filtering, but the recovered weight-space directions shouldn't be over-interpreted.

## Validating the Core Assumption

The idea depends on: "clusters of weights learn features, isolated weights memorize." Before building the optimizer, test this.

### Experiment 1: Covariance structure on real vs random labels

Train same network on real and random labels (MNIST or spirals). At each epoch, compute per-sample gradients and the weight-weight covariance (use streaming low-rank or random projection). Compare:
- Eigenvalue spectrum: real labels should show clear gap (few large eigenvalues for feature groups). Random labels should be flatter (many small groups, no dominant clusters).
- Cluster sizes: real labels should have a few large clusters. Random labels should have many tiny clusters.
- Evolution over time: real labels should show clusters forming early and stabilizing. Random labels should show clusters forming later and being more transient.

### Experiment 2: Ablation of identified clusters

Train to convergence on real labels. Identify weight clusters from covariance. Zero out entire clusters and measure:
- Does each cluster correspond to a recognizable feature? (e.g., specific digit, specific spiral arm)
- Does ablating a cluster selectively destroy performance on a subset of inputs?
- Compare cluster ablation vs random weight ablation of same magnitude.

### Experiment 3: Real-time tracking during memorization

Train on a mix of real and corrupted labels (e.g., 80% real, 20% random). Track which weight clusters activate when processing corrupted vs clean samples. Do corrupted samples recruit different (smaller) clusters?

## Comparison: B x B vs p x p

| Property | B x B (implemented) | p x p (proposed) |
|----------|-------------------|-----------------|
| What it measures | Per-batch sample agreement | Cross-time weight coupling |
| Temporal scope | Snapshot (one batch) | Running statistic |
| Works with full-batch | Yes | No (needs stochastic batches) |
| Computational cost | O(B^2 p + B^3) per step | O(pkB) per step (streaming) |
| What it filters | Sample-specific gradient directions | Isolated (non-clustered) weights |
| Anti-memorization mechanism | No consensus → no gradient | Memorization weights suppressed |
| Key advantage | Simple, self-contained per step | Captures temporal structure |
| Key disadvantage | No memory across steps | Needs careful non-stationarity handling |

## Recommended Next Steps

1. **Diagnostic experiment first** (Experiment 1 above): measure whether the eigenvalue spectrum of weight covariance actually differs between real and random labels. This is cheap and validates/invalidates the whole idea before building the optimizer.
2. If validated, implement streaming low-rank variant (Frequent Directions or incremental SVD) — it's both tractable and more principled than random projection.
3. Compare against the B x B spectral filter on the same benchmarks (random labels, OOD, grokking with stochastic batches).
