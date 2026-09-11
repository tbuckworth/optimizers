# Temporal gradient-subspace prior check

Codex — Spectral Optimizer Investigation · 10 September 2026

**Scope:** four selected works, with both materially different DOME versions
distinguished. This is a bounded primary-method comparison for the
[paper core](spectral_paper_core_2026-09-10.md), not a systematic review,
priority certificate, implementation audit, or new experiment. DOME, GaLore
and Song et al. were checked in version-pinned full HTML methods. TAGD is a
particularly close **provisional lead**: the search service exposed substantial
indexed primary-PDF methods and complete Algorithm 1, but direct PDF/forum/API
access returned OpenReview's browser challenge. Its full evaluation, author
identities and bibliographic date were not independently recovered.

The strongest affirmative overlap is already substantial. A learned global
gradient-history space, covariance-based filtering before an optimizer, and
preserving dominant temporal directions all have precedents. The defensible
paper object remains the linked characterization of **useful protection,
selectivity boundaries, equal-exposure batching, and a controlled local
observer-history pathway**. The present check does not establish priority for
that combination either.

## 1. Compact method comparison

Here `d` is the total parameter count; `P = UUᵀ` is an orthogonal projector.
The native repository policy is specified by
[Methods A, C and D](spectral_paper_methods_2026-09-10.md) and
[spectral_filter.py](../spectral_filter.py): successive flattened batch-mean
gradients, an EMA mean, centered innovations, rank-limited streaming updates,
current-observation inclusion, and delivery `Pg` to ambient-coordinate AdamW.
It has no explicit restoration of the temporal mean. The finite observer is
not exact full-history PCA.

| Selected work | Space and observations | Action and optimizer placement | Affirmative overlap; boundary |
| --- | --- | --- | --- |
| **DOME v2**, Nicolas et al., 2026 | Global `d`-space; per-example gradients centered **inside each minibatch**, historically accumulated through streaming power/QR | Update basis first; deliver `(I−P)g`; Adam used in compression experiments | Online centered covariance and pre-optimizer filtering already exist. Statistic differs from between-batch temporal covariance; action removes the leading space. |
| **TAGD**, anonymous indexed submission | Global `d`-space; centered rolling history of stochastic gradients; small sample-space Gram matrix mapped back to parameter directions | Current gradient enters history first; retain `Pg`, damp complement, blend with raw, then apply base optimizer | Closest inspected same-direction mechanism. Rolling-window/periodic PCA and adaptive damping differ from native EMA/hard projection. Source access remains partial. |
| **GaLore**, Zhao et al., 2024 | An instantaneous **layer-gradient matrix**, periodically decomposed by SVD | Project to compact coordinates, run Adam there, map update back | Learned changing gradient subspaces and gradient/Adam interaction are prior art. Neither global temporal covariance nor ambient Adam after projection. |
| **Does SGD really happen in tiny subspaces?**, Song et al., v3 2025 | Global parameter space, selected by the training-loss **Hessian** | Compare dominant/bulk projected updates; adaptive-optimizer experiments project the update vector | High retained energy need not imply learning; optimizer transformation matters. The selected operator and projection placement differ from native. |

## 2. DOME must be version-pinned

The [arXiv version record](https://arxiv.org/abs/2507.03545) verifies a title
and method change between 4 July 2025 and 2 February 2026.

**v1:** [Communication Efficient, Differentially Private Distributed
Optimization using Correlation-Aware Sketching, §3.1](https://arxiv.org/html/2507.03545v1#S3.SS1)
and [Algorithms 1–2, §3.2](https://arxiv.org/html/2507.03545v1#S3.SS2).
This version retains historical directions plus orthogonal random probes.
Clients subtract the previous mean before sketching; reconstruction restores
it. Ignoring clipping/privacy noise, delivery is

```text
h_t = m_(t−1) + S_t S_tᵀ (g_t − m_(t−1)).
```

The next sketch uses reconstructed full gradients in Eq. (10), not explicitly
centered raw innovations. The ambient Adam variant corrects its second moment
for privacy noise. Thus it is direct global temporal-subspace/pre-Adam history
prior art, but not the native statistic or hard action. Its communication/DP
motivation does not establish selective clean-versus-corrupted learning.

**v2:** [DOME: Improving Signal-to-Noise in Stochastic Gradient Descent via
Sharp-Direction Subspace Filtering, §3](https://arxiv.org/html/2507.03545v2#S3).
Equations (8)–(9) and Algorithm 2 use

```text
μ_t = mean_j g_(t,j)
H_t = [g_(t,1)−μ_t, …, g_(t,B)−μ_t]
W_t = H_t(H_tᵀ U_(t−1))/B.
```

Algorithm 2 combines this with historical state using `(t−1)/t` and `1/t`,
then QR; Eq. (11) delivers `(I−U_t U_tᵀ)g_t`. This is within-batch scatter,
not native's centered sequence of batch means. [§4.1](https://arxiv.org/html/2507.03545v2#S4.SS1)
reports five-seed SGD-momentum learning experiments and Adam compression
experiments. Two source inconsistencies matter: Eq. (10) adds another `1/B`
absent from Algorithm 2, and Algorithm 1 lists Adam hyperparameters but only
implements first-moment stepping. Do not silently repair these into a verified
implementation. Nor does the model-label covariance discussion in
[§2.4](https://arxiv.org/html/2507.03545v2#S2.SS4) justify arbitrary fixed-label
covariance being equal to the Fisher/Gauss–Newton matrix.

## 3. TAGD is the closest same-action lead, with a verification limit

Indexed [§4, PDF p. 5](https://openreview.net/pdf?id=Dc6AEOYQjM#page=5)
and [Appendix B, Algorithm 1, p. 12](https://openreview.net/pdf?id=Dc6AEOYQjM#page=12)
of **Topological Control of Optimization Dynamics on Evolving Manifolds**
specify centered history `G ∈ R^(m×d)`, Gram matrix `GGᵀ/d`, a recovered
orthonormal parameter basis, and

```text
g_ctrl = Pg + λ_perp (I−P)g
g_used = (1−ω)g + ω g_ctrl.
```

Algorithm 1 appends the current gradient before basis refresh. The method
adds graph/spectral stability feedback to select damping and an intervention
schedule. It reports an optimizer-input wrapper without changing the base
optimizer's mechanics. This is substantive prior overlap, not merely shared
PCA terminology. Algebraically, `ω=1, λ_perp=0` recovers hard projection;
this statement does **not** establish that this setting was evaluated or
allowed by every reported schedule. Native's EMA estimator remains different.

Direct access to the
[indexed PDF object](https://openreview.net/pdf/8995a0bddb1704014226d287b45bc44771f41905.pdf)
and forum failed. Treat the above as an indexed-method check, not complete
verification of the paper or its performance claims. The associated BibTeX
preserves the visible anonymous author and omits an unverified year.

## 4. Established comparisons that sharpen interpretation

**GaLore:** [§3.3, Eqs. (12)–(13)](https://arxiv.org/html/2403.03507v2#S3.SS3)
chooses singular vectors of the current layer-gradient matrix;
[§4.1](https://arxiv.org/html/2403.03507v2#S4.SS1) refreshes them periodically.
[§4.2, Algorithm 2](https://arxiv.org/html/2403.03507v2#S4.SS2)
computes `R=PᵀG`, updates compact Adam moments, and returns `αP Adam(R)`.
A global `d×k` historical-gradient basis is a different object from a
rank-`r` factorization of a layer's matrix. GaLore supports the affirmative
claim that changing learned spaces can be operationally useful; it does not
establish this repository's noise protection, sampler interaction, or fixed
observer intervention. Projection before *compact* Adam cannot be treated as
the same policy as projection before *ambient* Adam.

**Song et al.:** [§2, Definitions 1–2](https://arxiv.org/html/2405.16002v3#S2)
defines top Hessian eigenspaces. [§3](https://arxiv.org/html/2405.16002v3#S3)
compares continuing SGD with dominant- and bulk-space updates after a switch;
[§6](https://arxiv.org/html/2405.16002v3#S6) extends this to momentum and
adaptive methods and measures projected effective learning rates. Their
dominant-space failures directly predate any claim that gradient-energy
retention is insufficient to certify progress. Their favorable bulk results
concern their Hessian space and optimization regimes. They neither refute
native's measured conditional benefits nor identify its covariance space as
useful. [§7](https://arxiv.org/html/2405.16002v3#S7) leaves generalization
implications open. Their projection of an optimizer update differs from our
projection of the input supplied to inherited Adam.

## 5. What remains distinct enough to investigate and report

For a fixed model and a fixed multiset of `M B` indexed occurrences, partition
the occurrence gradients `x_(t,j)` into `M` equal-size batches. With batch
means `μ_t`, overall mean `μ`, and population normalization, define

```text
S_all     = (1/(MB)) Σ_t,j (x_(t,j)−μ)(x_(t,j)−μ)ᵀ
S_within  = (1/M) Σ_t [(1/B) Σ_j (x_(t,j)−μ_t)(x_(t,j)−μ_t)ᵀ]
S_between = (1/M) Σ_t (μ_t−μ)(μ_t−μ)ᵀ
S_all = S_within + S_between.
```

This elementary finite partition identity is exact: within-batch centered
residuals sum to zero, eliminating the cross terms. Conserving the occurrence
multiset fixes `S_all`, but repartitioning can transfer scatter between the
other two terms. It is not an identity for either implementation's estimated
basis, or a new empirical result.

**A mathematical comparison, not new evidence:** let a fixed dataset contain
two deterministic group gradients `a` and `b`; let a batch's rare fraction be
`q`. Its mean is `a+q(b−a)`. Across batches, covariance is
`Var(q)(b−a)(b−a)ᵀ`; within a batch, centered scatter is
`q(1−q)(b−a)(b−a)ᵀ`. Equal-size pure-group batches increase the former while
making the latter zero, at the same total occurrence multiset and overall
mean. When counts permit equal proportions in every mixed batch, the same
contrast lies entirely within batches; with pure batches, entirely between
batches. Therefore native and DOME v2 cannot be called the same covariance
method with opposite signs. Real examples add within-group variation and
moving-model effects; native startup, EMA and truncation add further changes.
This elaborates the standard identities in the
[local methods](spectral_paper_methods_2026-09-10.md), not a claimed theorem
of optimizer performance.

The strongest positive paper case survives these overlaps: the repo shows a
specified useful preservation regime, its simultaneous rare/cue failures,
and a grouping intervention that changes the result at conserved exposure.
The fixed-model diagnostic then changes observer history while holding model,
incoming gradient bytes and inherited Adam state fixed. Those controls identify
a local history pathway; they do not identify endpoint mediation or prove
semantic selectivity.

No matching experiment was identified in the inspected DOME, GaLore or Song
methods/evaluations. TAGD's full evaluation remains unassessed. These are
**assessed-source limits**, not evidence that no such experiment exists.
DOME's batch-size study already makes batch dependence relevant; ordinary
batch-size effects should not be advertised as novel. The specific preserved
occurrence multiset, policy interaction, and observer-only replay are the
comparison to state explicitly.

Remaining gaps are concrete:

- Obtain TAGD's full accessible source and bibliographic provenance before
  making a paper-wide nearest-prior or priority claim. Existing writing can
  continue with the indexed-method overlap and limitation visible.
- Resolve DOME v2 paper/code discrepancies if an exact algorithm comparison
  becomes necessary. Its privacy/compression benefits are not corruption-label
  or behavioral-safety evidence.
- A claim of practical superiority over these methods would require matched
  comparisons; this note authorizes none. Grokfast/GradPCA and the coherence
  sources retain their existing, separate checks.
- Broader compression, continual-learning and gradient-memory literature was
  screened only opportunistically. No exhaustive absence claim follows.