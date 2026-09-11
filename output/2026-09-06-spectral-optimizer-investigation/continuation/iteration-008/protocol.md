# I8: actual-filter selective learning in fixed-corruption regression

Prospective protocol, 7 September 2026. New CPU-only scientific experiment;
neither a retry nor a substitute for either consumed I7 preparation attempt.
This tests the constructive mechanism in `../../analysis/steelman-selective-learning.md`.
No neural generalization or optimizer-default claim is intended.

## Fixed design and predictions

In two dimensions, let useful unit direction u be rotated by either 0 or 30
degrees, with orthogonal nuisance v. Clean target is u; fixed corrupted target
is w=u+v. Batch loss is
`0.5 * [h_u (theta_u-1)^2 + h_v (theta_v-1)^2]`.
Each batch is realizable by fixed rows `sqrt(2 h_u) u`, `sqrt(2 h_v) v`
and labels `sqrt(2 h_u)`, `sqrt(2 h_v)`, using sum-squared loss divided by four.
Clean evaluation sets only the nuisance label to zero. No labels are redrawn.

Every four-step block uses a shuffled permutation of a fixed multiset:

| Design | Four (h_u, h_v) pairs |
|---|---|
| useful | (.25,1), (1.75,1), (.25,1), (1.75,1) |
| nuisance | (1,.25), (1,1.75), (1,.25), (1,1.75) |
| both | (.25,.25), (.25,1.75), (1.75,.25), (1.75,1.75) |
| none | four copies of (1,1) |

The full finite-design Hessian is I in every case. At a *fixed state*, useful-only
batch covariance is `.5625 (theta_u-1)^2 uu^T`; nuisance-only reverses this.
Moving temporal covariance is not asserted equal to that fixed-state covariance.

Prediction: useful variation should let the actual rank-one filter learn u
and support continued useful learning with limited nuisance fitting under SGD.
Reversing variation should fail/harm. Both has tied initial spikes and may be
order sensitive; none exposes temporal drift/startup without batch variation.
A learned basis frozen after warmup tests whether ongoing rotation is necessary.
An oracle-u projection is an intentionally privileged positive control.
Adam can move outside even an oracle input-gradient subspace; rotation and actual
displacement diagnostics test this directly without attributing everything to PCA.

## Frozen panel and estimands

- Seeds 0–7, varying batch order only; two angles, four designs, two optimizers,
  four arms = 512 trajectories. The no-variation seeds repeat a deterministic
  calculation and are not eight independent replications.
- 2,000 steps, zero initialization, CPU float64, one thread, lr .002.
- SGD: momentum 0, weight decay 0. Adam: betas (.9,.99), eps 1e-8,
  weight decay 0, no AMSGrad. Explicit non-fused/non-foreach implementations.
  Equal nominal lr is not displacement matching or exhaustive baseline tuning.
- Arms: raw; canonical live stable hard rank1 filter; learned frozen rank1;
  oracle u. Every arm starts fresh with identical within-seed batch order.
- Filter: decay .99, warmup 100, rank1, stable update, no normalization or
  adaptive rank. The native implementation observes the current gradient before
  projecting and first filters at step101. Frozen arm observes calls1–100 then
  copies the learned basis and stops observing; oracle also activates at101.
  Neither arm resets the optimizer's accumulated warmup moments.
- Primary positive-mechanism test: **canonical live versus raw SGD in the useful
  design**, reporting each angle separately. For each seed, subtract paired raw
  minimum risk over **all** steps0–2000 from live minimum over steps101–2000;
  negative favors live. Average these paired differences across seeds and retain
  their ranges/win counts. Do not pool angles/designs or select a favorable cell.
  The other designs test boundaries; frozen/oracle and Adam are secondary
  mechanistic controls, reported in full with the same contrast. Both sides use
  clean oracle stopping information, but raw has the larger permitted window:
  this deliberately favors raw and tests benefit during continued learning.
  This is not unbiased held-out selection or a formal population significance test.
- Also report endpoint clean/train risk and endpoint minus best raw risk,
  useful/nuisance coordinates, basis alignment, gradient components and actual
  step components at steps50,100,101,150,250,500,1000,2000; step0 is state-only.
- Save every clean-risk value for independent minimum/selection checks; record
  post-warmup summed out-of-applied-subspace step energy / total step energy.
  Preserve all seeds/cases, including contrary outcomes; no parameter tuning.

The scalar-shrinkage lower bound .25 is exact for theta=alpha(u+v), including
isotropic full-batch SGD. It is **not** a bound on all stochastic SGD or Adam
trajectories. Primary comparisons use each actual complete raw curve instead.
Oracle projected SGD after the short warmup should finish near clean risk .0166;
this is a sanity expectation, not a theorem for the canonical filter or Adam.

## Implementation and bounded execution

Canonical `spectral_filter.py` remains unchanged, SHA256
`9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`.
The harness checks the hash before importing it. Analytic batch gradients are
checked against autograd on the explicit finite design before the panel.
Installed runtime is PyTorch 2.11.0+cu128 / NumPy1.26.4; computations stay on CPU.
The best-practices check verified update semantics and explicit settings against
[PyTorch2.11 SGD](https://docs.pytorch.org/docs/2.11/generated/torch.optim.SGD.html)
and [Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html).
The documentation specifies optimizer behavior, not our scientific predictions.

Launch once as `spectral-selective-learning-i8-001.service`, with 900-second
total wall limit, 2GiB memory limit, one CPU quota and no CUDA devices. Artifacts
are small and live in this iteration's new `artifacts/` directory; scratch uses
the verified mounted large volume. The output directory must not already exist.
Source/protocol are committed before launch and their hashes recorded in the
artifact manifest. A partial/failed run remains evidence; do not rerun or overwrite
it after a cutoff. No cloud spend or reservation: the $100 Modal budget is intact.
