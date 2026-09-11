# Full parameter gradients versus the residual-space J-Lens pilot

Codex — Spectral Optimizer Investigation · 11 September 2026.

After approving the recommended1024→512 comparison, the user asks whether
the dimensional limit comes from the small object being inspected, permits
continuation, and raises the substantive concern that the original idea works
on full weight gradients rather than one layer's internal vector.

## Direct code comparison

- Global optimizer (artifact not distributed in this public snapshot), `_get_flat_grad`,
  concatenates gradients of all selected trainable parameters owned by the
  base optimizer. In a full-model setup, this couples weights across layers;
  a selected-subset/LoRA setup is not automatically full-model. Its stored
  basis is p×k; it does not form a dense p×p covariance.
- Existing worker `acquire.py`, `capture_one`, instead freezes model weights,
  makes the post-block11 residual tensor a differentiable detached leaf,
  differentiates completion loss with respect to that leaf, and saves only
  position31's gradient and activation. Each saved vector has1024 entries.
  Parameter gradients are deliberately absent from its receipt.

The worker source is
`/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-aggregate-projection/acquire.py`;
the rank/source check (artifact not distributed in this public snapshot) gives the pinned receipt/config.
These are residual gradients, not even the parameter gradients of one layer.
The existing1792vectors do not contain the full-weight gradients, so simply
reinterpreting or padding them cannot answer the new question.

## Mathematical distinction

Let θ∈R^p contain the chosen model parameters, h∈R^d an internal activation,
and z∈R^m a specified downstream representation.

| Quantity | Space | Question |
|---|---|---|
|∇_hℓ|d-dimensional residual space|Which changes to this internal state affect the loss?|
|∇_θℓ|p-dimensional parameter space|Which changes to the weights affect the loss?|
|J_h=∂z/∂h|m×d|How does an internal-state perturbation propagate?|
|A_θ=∂z/∂θ|m×p|How does a weight perturbation affect the downstream representation?|

For a fixed context, an upstream parameter block obeys the chain rule via
the activation Jacobians at all relevant source positions. Parameters used
downstream also contribute directly. One saved position's ∇_hℓ does not
supply these Jacobians or all those contributions. Residual and parameter
covariance are not interchangeable, even if they arise from the same loss.

The existing formula D(J̄_h P_h h) is dimensionally valid only for the
residual projector P_h. A parameter projector P_θ is p×p and cannot simply
replace P_h in that expression. A coherent weight-direction probe is

δθ = P_θ mean_i(∇_θℓ_i),
Δz(x) = A_θ(x) δθ.

For a proposed loss-reducing update, choose −δθ with an explicitly fixed
scale/sign convention. One can decode the direction Δz, or examine output
changes relative to the actual baseline z(x); these answer different
questions. A direction-only normalized word list is not automatically a
causal logit change. Neither is an ordinary activation-input J-Lens.
Averaging A_θ across contexts is another explicit proposed operator, not
an existing validated replacement lens.

The first-order product A_θ δθ can in principle be computed without
changing weights, using a directional derivative. This is the generic
Jacobian-vector operation documented by
[PyTorch](https://docs.pytorch.org/docs/main/generated/torch.func.jvp.html),
not a verified claim of forward-AD support for this particular model.
Perturb-and-difference is a different approximate numerical method and would
require its own scale/convergence checks; neither has been run here.

## Rank and feasible representations

Full-weight covariance can have k2048 if p and the data support it. The
1024 cap belonged only to the chosen residual input space. However, any
single1024-dimensional downstream linear readout has rank≤1024 even if
the parameter projector retains2048 directions: different weight changes
can be indistinguishable at that readout/context.

With B centered full gradients as rows of G∈R^(B×p), define

C_θ=GᵀG/B,  K=GGᵀ/B.

For Kv_j=λ_jv_j with λ_j>0 and ||v_j||=1,
u_j=Gᵀv_j/√(Bλ_j) is a unit eigenvector of C_θ. Thus a B×B Gram solve
can recover the nonzero global parameter eigenspace without a p×p matrix.
Centered rank≤min(p,B−1); estimating2048 nonzero modes requires at least
2049 gradient observations, not just enough RAM. This lower bound is not
a guarantee of accurate estimation, especially with correlated observations.

The shortcut does not eliminate all cost. Explicit storage of a p×k basis
costs4pk bytes in FP32. Illustratively, p=800,000,000 and k1024 would require
3,276,800,000,000 bytes, about3.28TB decimal, for the basis alone. This is
an order-of-magnitude illustration, not an audited count of selected text
parameters in the current checkpoint.

There are legitimate alternatives to materializing that basis:

- Split parameter columns into blocks G=[G_1,…,G_L] and accumulate
  K=Σ_l G_lG_lᵀ/B. This remains the exact global Gram, not independent
  per-layer eigenspaces. The same sample-space eigenvectors v_j couple all
  blocks when lifting directions. Compute/storage/recomputation costs still
  need accounting for all required gradients.
- Keep directions implicit in sample space, lifting one weighted gradient
  combination at a time or streaming its parameter blocks for a directional
  readout. This can trade storage for computation; it does not make a large
  per-example-gradient acquisition free.
- A smaller model or a much smaller initial k could provide a first genuine
  full-weight feasibility result. Neither is a selected/launched experiment
  in this note, and neither should be silently labeled the1024 pilot.

## Next-step implication and limits

Preserve existing residual acquisition, comparisons and negative results.
They test a layer-local proxy; success or failure there alone does not settle
full-weight spectral filtering. The next plan must explicitly specify all
selected parameters, loss, gradient sampling and the parameter-to-output
readout, plus a small resource-bound feasibility check before a large k sweep.
Do not scale models or spend the budget automatically on the basis of a rank
suggestion. The user's permission to continue is recorded; no repeated
discussion-approval gate is required.

A frozen-model full-gradient covariance would match the parameter space but
still differ from the optimizer's evolving, exponentially weighted history
of batch gradients during training. Keep that remaining distinction visible.