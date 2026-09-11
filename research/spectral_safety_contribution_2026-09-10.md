# Safety contribution: controlling learning without confusing protection with incapacity

Codex — Spectral Optimizer Investigation · 10 September 2026

**Paper synopsis, not new experimental evidence or a deployment recommendation.**
The contribution is a measured, conditional tradeoff between preserving useful
knowledge and acquiring new associations. It is not yet an alignment defense.
This synopsis informs the working paper (artifact not distributed in this public snapshot),
not a separate optimizer claim.

## Why the question matters

An intervention intended to resist undesirable learning must also permit
legitimate adaptation. Aggregate task performance cannot answer both questions.
Prior work demonstrates that similarly validated predictors can differ in
deployment, and that capable learned behavior can pursue an unintended goal
under a shift. Our digit classifiers are not agents, and their errors are not
instances of goal misgeneralization; these results motivate, rather than
validate, the proposed safety connection.
[Underspecification](https://jmlr.org/papers/v23/20-1335.html);
[goal misgeneralization](https://arxiv.org/abs/2210.01790v2).

The originating criticism of deep-learning theory should not become the claim
that no useful theory exists. Nagarajan–Kolter expose a limitation of tightly
restricted two-sided uniform-convergence bounds in their constructions;
surrogate-based analyses subsequently obtain informative guarantees in related
settings. A constructive aim is conditional predictions of learning dynamics,
not declaring generalization theory dead or explaining ordinary SGD merely
by changing its optimizer.
[Nagarajan–Kolter](https://arxiv.org/abs/1902.04742v4);
[Negrea et al.](https://proceedings.mlr.press/v119/negrea20a.html);
[Simon et al.](https://arxiv.org/abs/2604.21691v1).

## What the paper can positively establish

| Evidence | Useful contribution | Limit retained beside it |
|---|---|---|
| Three-seed selectivity study | Common accuracy under diffuse corruption is 62.01% versus raw AdamW's 33.97%, while wrong-target fitting is lower. | Both decline from warmup; native rare recognition is zero. Common CE is not improved consistently. |
| Three fresh batching seeds | Rearranging identical indexed exposure increases native noisy common accuracy from 47.92% to 56.64%, with gains in every seed. | This changes the tradeoff, not the rare failure: native noisy rare accuracy remains zero. Endpoint covariance mediation is not identified. |
| Fixed-parent observer intervention | Changing only observer history changes a useful local action at common model, Adam state and input. | Three reused early parents; Clean and Diffuse differ. Direction access does not determine delivery or explain long-run outcomes. |
| Complementary grokking and augmentation | A specified projected policy generalizes better than its raw-direction control; native also retains relative patched-image protection in every new mask mode and seed. | Legacy-state grokking is not a stable wall-clock gain; ordinary masks worsen native rare CE, and targeted masks cost competence. |

Direct evidence: [selectivity](../output/2026-09-09-spectral-selectivity-boundary/results.md),
[batching](../output/2026-09-10-spectral-batch-composition/results.md),
[observer intervention](../output/2026-09-10-spectral-observer-pathway/results.md),
[grokking control](grokking_raw_direction_2026-09-09.md), and
[augmentation](../output/2026-09-10-spectral-augmentation/results.md).
These studies are not pooled replicates. Poor rare recognition is not zero
learning: rare CE improves from the common-only warmup in the native boundary
and augmentation studies.

## The mathematical account, at its defensible scope

Directional restriction can let useful learning continue while blocking an
orthogonal corruption component in the favorable fixed-design construction.
The neural evidence supports a **history-sensitive learning restriction**,
not proof that the native observer discovers that ideal useful space.
At fixed parameters, batch-gradient covariance depends on the sampler; a
centered high-variance direction can express either a useful changing feature
or a misleading shared cue. It is wrong to say gradients contain no semantic
information. The missing guarantee is correspondence between the chosen
geometry and desirable behavior.

The [nearest-method comparison](spectral_subspace_prior_check_2026-09-10.md)
credits existing coherence, temporal-subspace and projected-optimizer work.
The candidate empirical contribution is the linked controlled characterization,
not invention of gradient PCA, nor established priority for the combination.
Likewise, *Evil Spectra* studies optimizer-sensitive misalignment and the
singular values of the effective adapter BA. Those are not the eigenvalues of
our temporal gradient covariance, and its positive intervention does not
transfer by shared terminology.
[Primary methods](https://arxiv.org/html/2606.31591v1#S3).

## What would make this an actual safety intervention?

The current work is safety-relevant because it exposes a consequential
preservation/adaptation tradeoff and a controlled way to change it. To claim
safety efficacy later, a bounded behavioral study would need lower absolute
undesired-behavior rates **with maintained benign competence, including rare
legitimate requirements**, and useful new learning beyond a frozen parent.

One schematic evaluation contract is:

```text
H(filtered) ≤ H(reference) − ε_H
C_g(filtered) ≥ C_g(reference) − δ_g   for every predeclared benign group g
A(filtered) ≥ A(frozen parent) + ε_A
```

Here H is a specified harmful-behavior rate, C_g measures retained benign
competence, and A measures legitimate new learning. Thresholds, uncertainty,
challenge distribution and resource matching would be fixed in advance.
Refusal, invalid outputs and response coverage must remain visible; conditioning
only on surviving answers can conceal incapacity. This is an evaluation
proposal, not a theorem, an executed protocol or a gate before reporting the
present useful results. Existing EM evidence does not establish that contract.

Keep broad pretraining speedups and high-learning-rate scaling outside the
selected research direction. Small controlled studies reduce direct capability
development but do not eliminate dual use; inspect release implications before
public submission. The next conceptual task is to specify a predictive account
with explicit operating conditions, not to obtain a favorable cell by tuning.
