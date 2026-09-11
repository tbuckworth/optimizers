# Why augmentation can help with wrong labels—and where spectral filtering enters

Codex — Spectral Optimizer Investigation · 10 September 2026

Open the readable HTML explanation (artifact not distributed in this public snapshot).

**Status: exact conditional mathematics and a provisional mechanism, not new
training evidence or a claimed new theorem.** This note supplies a missing
connection between our fixed-label results and established augmentation theory.
The existing strong-rank200 comparison continues unchanged; no new arm or
diagnostic acquisition is selected here.

## 1. The useful idea in plain language

Training on translated copies does not tell the model which labels are wrong.
It does make the model pay for giving different predictions to different views
of the same image. For cross-entropy, this statement has an exact decomposition:

**Average loss across views = loss of the combined prediction + a nonnegative
view-disagreement penalty.**

At the same fixed model, the penalty is identical for a correct label and a
wrong label. Augmentation can therefore impede some image-specific fitting
without detecting mistakes. But a model can also memorize an entire collection
of translated views consistently. Neither augmentation nor spectral filtering
is inherently a truth detector.

The resulting spectral hypothesis is more specific than “augmentation adds
noise”: restricting gradients might impede learning the desired consistency,
even while suppressing other fitting. This is plausible, not established by
our endpoint results or existing covariance measurements.

## 2. An exact identity, with the correct averaging operation

Fix an image x and an assigned probability target t (usually the one-hot
vector for its fixed, possibly wrong label). Let T have finite support with
fixed probabilities, independent of the model parameters θ. Define

```text
z_T = f_θ(Tx)                    K-dimensional logits
A(z) = log Σ_k exp(z_k)          log-normalizer
ℓ(z,t) = A(z) − tᵀz             cross-entropy, Σ_k t_k = 1
z̄ = E_T z_T                    mean logits
p_T = softmax(z_T)
p̄ = softmax(z̄)                 NOT generally E_T p_T
```

Linearity of the target term gives, exactly,

```text
E_T ℓ(z_T,t) = ℓ(z̄,t) + R(θ,x),
R(θ,x) = E_T A(z_T) − A(z̄) ≥ 0.
```

Nonnegativity follows from convexity of A. No linear-network or small-shift
assumption is needed. In particular, z̄ need not equal f_θ(x): augmentation
changes the combined prediction as well as introducing R. It is generally
incorrect to write the original-image loss plus this same R.

There is also an exact probability interpretation:

```text
R = E_T KL(p̄ || p_T),
p̄_k = exp(E_T log p_T,k) / Σ_j exp(E_T log p_T,j).
```

To verify the KL orientation, substitute log p_T = z_T − A(z_T)1:

```text
KL(p̄ || p_T) = A(z_T) − A(z̄) − p̄ᵀ(z_T − z̄).
```

The last term averages to zero. The center is the normalized geometric mean
of probabilities, not their arithmetic mean. This is not Jensen–Shannon
divergence or an independently chosen consistency-training objective.
R vanishes exactly when all positive-probability views have the same class
probabilities; logits may still differ by a class-independent constant.

For small logit fluctuations δz = z_T − z̄, Taylor expansion further gives

```text
R ≈ ½ tr[(diag(p̄) − p̄p̄ᵀ) Cov_T(z_T)].
```

This last expression is an approximation, unlike the identities above.
It concerns output logits, not the optimizer's parameter-gradient covariance.

## 3. Where fixed wrong labels remain in the objective

For each training example i, let q_i be its expected corrupted-label target
under the specified corruption procedure, and ε_i = t_i − q_i its realized
deviation. With a common fixed assigned label across views,

```text
L_fixed,aug(θ) = S(θ) + F(θ) + C(θ),
S = (1/N) Σ_i ℓ(z̄_i, q_i),
F = −(1/N) Σ_i ε_iᵀ z̄_i,
C = (1/N) Σ_i R(θ,x_i).
```

S is fitting softened class structure in the combined predictions; F is the
particular label-assignment force; C is the label-independent view penalty.
These names describe the decomposition, not pure semantic components:
q is not the clean target, and F is not automatically a harmful direction.

For the [running study's](../output/2026-09-10-spectral-strong-augmentation/protocol.md)
ideal 90% uniform replacement among ten labels,
q_i = .1 e_y + .9 u, with u_k = .1. Thus q_y = .19 and q_other = .09;
81% of labels are wrong in expectation. Its soft-target term can be written

```text
ℓ(z̄,q) = .1 ℓ(z̄,e_y) + .9 [log 10 + KL(u || p̄)].
```

The population target retains a truth-favoring margin but discourages confident
predictions. This does not guarantee that finite-data training recovers that
population optimum. Our earlier 80%-actually-wrong procedure instead has
q_y = .2 and q_other = 4/45; the two corruption definitions must not be conflated.

At a θ chosen independently of the corruption draws, E_ε F(θ) = 0. That
statement cannot be applied to θ already fitted to those assignments: model
and ε are then dependent. Likewise, C is label-independent only when comparing
the same model, images and view distribution. Models trained on different
labels can have different C.

### The gradient split is equally exact

Where the logits are differentiable, write J_T = ∂z_T/∂θ and J̄ = E_T J_T.
For one example,

```text
∇ℓ_aug = s + f + c,
s = J̄ᵀ(p̄ − q),
f = −J̄ᵀε,
c = E_T[J_Tᵀ(p_T − p̄)].
```

All terms include the relevant model derivatives; no stop-gradient teacher
is introduced. Averaging gradients of several view losses estimates this
whole gradient. It is not the same as differentiating only the loss at mean
logits: the latter omits c. The finite-view identity holds for each sampled
view set, although the separate finite-view components need not be unbiased
estimators of their infinite-view counterparts.

This extends the existing [fixed-assignment force identity](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-010/results.md)
and [wrong-label augmentation note](../output/2026-09-10-spectral-wrong-label-augmentation/mathematical-note.md);
it does not replace their empirical findings.

### The strongest case for combining the methods

Their constraints need not be redundant. Augmentation can discourage
view-specific shortcuts while leaving transformation-consistent memorization
available: an example and all its views may still receive the same arbitrary
label. A favorable learned filter could additionally restrict that persistent
example-specific fitting while retaining enough shared class learning and
consistency adjustment. That is a coherent possible benefit beyond augmentation
alone, not a demonstrated semantic partition. Conversely, restricting those
useful adjustments too severely could explain the small recipe's combination
cost. The stronger-regime experiment tests whether practical complementarity
exists; it cannot by itself establish which component caused it.

## 4. Why spectral energy retention does not answer this question

The earlier law of total covariance separates variation between example-mean
gradients from variation across views of the same example. The term C above
instead is a scalar objective penalty. Its gradient c is a **mean** involving
both changing predictions and Jacobians. Removing within-view stochastic
variance from a gradient estimate is not removing c from its mean.

For a hypothetical frozen orthoprojector P and plain gradient descent with
total gradient g = s + f + c, the first-order change in C is

```text
ΔC = −η cᵀPg + o(η).
```

Measuring ‖Pc‖²/‖c‖² alone does not determine this signed quantity. For example,

```text
c = (1, .1),       g = (−1, 20),       P = diag(1,0).
cᵀg = 1,          cᵀPg = −1,          ‖Pc‖²/‖c‖² = 100/101.
```

Raw descent decreases C locally; projected descent increases it, despite
retaining about 99% of c's energy. This is a local vector counterexample, not
a fitted neural model or evidence that the native observer selects this P.
Exact full retention Pc = c is a special boundary: then cᵀPg = cᵀg.

The actual native policy observes the current gradient before projection,
so its action depends on that gradient; E[P(g)g] cannot generally be replaced
by a fixed P times E[g]. Adam additionally carries and transforms moments.
For its actual displacement Δθ, the appropriate local quantity is cᵀΔθ,
with finite-step checks—not the projected-SGD expression asserted as an Adam
theorem. Preserving a component's instantaneous decrease still would not prove
better generalization or long-run mediation.

## 5. What this changes in our interpretation and test priorities

The completed [three-seed wrong-label comparison](../output/2026-09-10-spectral-wrong-label-augmentation/results.md)
already establishes that ordinary translation helps raw AdamW substantially,
while the small spectral combination loses to augmented raw. The
[clean multiview study](../output/2026-09-10-spectral-multiview-clean/results.md)
finds a limited translated-readout benefit from averaging the observer's views,
not a primary practical rescue. Neither study measures C or its signed utility.

The leading explanations remain separable:

| Hypothesis | Discriminating evidence, not already established |
|---|---|
| Augmentation changes the mean learning objective usefully, while the small filter restricts that learning. | At common saved states, distinguish S, F and C progress under actual matched-step updates. |
| View noise makes the observer less effective. | Observer-only averaging should improve useful delivery without changing its target mean; the completed clean result supplies only a limited secondary positive. |
| The restriction is too severe in the small recipe, but the stronger historical regime combines well with augmentation. | Finish the already-running, fixed rank200 factorial and evaluate its full registered readouts. |

**Immediate priority remains the running factorial and its frozen audit.**
It tests practical complementarity but does not identify the three-term
mechanism. If a subsequent saved-state mechanism study is warranted, specify
shared image/view sets, common full optimizer states, component gradients and
signed actual-step effects before acquisition. Score C with multiple views of
each individual image, not by averaging logits across different images. A
constant or consistently wrong predictor can have C = 0, so pair consistency
with held-out competence and wrong-assignment fitting. These are proposed
measurements, not a new launch or authorization to change the current run.

## 6. Prior work and contribution boundary

This is established mathematical territory. The contribution here is applying
it carefully to this investigation's fixed labels and filtering policy.

- [Wager, Wang and Liang (2013), §2, equations 4–6](https://nlp.stanford.edu/pubs/wager2013dropout.pdf)
  derive a label-independent convex log-partition penalty for unbiased feature
  noising in generalized linear models. Our mean-logit expression does not
  assume a nonlinear network's mean transformed logits equal its original logits.
- [Dao et al. (2019), §§4.1–4.2](https://proceedings.mlr.press/v97/dao19b/dao19b.pdf)
  distinguish feature averaging and data-dependent variance regularization
  in augmentation, including label-independence for appropriate losses.
  Their kernel/Taylor analysis is a direct conceptual precedent, not evidence
  about our optimizer or noisy-label trajectories.
- [Wood et al. (2023), Appendix B.3, equations 32–33](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf)
  give the cross-entropy decomposition with a normalized geometric center and
  KL(center || prediction), crediting Heskes (1998). Replacing the source of
  prediction variation by augmentation views is an application of that
  identity, not architectural or theorem novelty.

Primary PDFs and the indicated sections were inspected on 10 September 2026.
No external empirical effect is imported into the repository's evidence.
An [independent derivation review](../output/2026-09-10-spectral-augmentation-theory/independent-math-review.md)
checks the identities and their limits separately from the literature review.
