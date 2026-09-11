# Independent preimplementation review

9 September 2026

**Status: PASS subject to the implementation conditions below.** The fixed
step-1500, five-seed design is a useful bounded test of future legacy
within-span gains. It does not rerun the completed experiment, select a
checkpoint from branch outcomes, or claim to identify pre-fork formation. The
current action protocol resolves most limitations already; the main residual
risks are exact first-step fan-out, delivered-dtype norm matching, measurement
state isolation and cost of per-update float64 QR.

## 1. Accidental non-common first action

**Risk: high likelihood if implemented naively; high severity.** Calling the
existing `filter_grad()` independently on three restored models would
recompute both the gradient and the legacy eigensystem. Nominally identical
results are not the required shared tensors, and `filter_grad()` also mutates
the estimator, counter and `.grad`.

The implementation should instead use this exact sequence for each seed:

1. Restore and validate the step-1500 source state once in a scratch instance.
2. In training mode, compute one full-batch raw gradient `g`; clone it before
   any filtering and bind its flattening layout to named parameters.
3. Advance the legacy estimator exactly once with that `g`, producing the
   complete post-update mutable state at counter 1501. Clone that state and
   compute all three actions from the same immutable `g` and `V`.
4. Construct three independent model/Adam/tracker instances from the original
   checkpoint. Replace each tracker's mutable filter state with independent
   copies of the shared post-update state, assign the corresponding action as
   `.grad`, and call AdamW exactly once. Do **not** invoke another estimator
   update on this first step.
5. Stop native after saving step 1501. Continue orthogonal and norm-matched
   directly from their actual step-1501 states; their next estimator mutation
   is update 1502 from their own new raw gradient.

Do not naively deepcopy an attached tracker: its optimizer and parameter lists
can remain bound to the wrong objects. Construct each model, optimizer and
tracker, restore their states, and verify named-parameter order. Before the
first Adam step, require exact equality of model tensors, Adam tensors, raw
`g`, post-estimator `V/S/grad_mean/counters`, and saved RNG state across all
three instances. Require independent storage as well as equal values, so one
branch cannot mutate another. A corruption fixture should detect a duplicate
estimator update and shared tensor storage.

## 2. Rank-revealed projector and norm-control semantics

**Risk: medium likelihood; high severity.** For float64 reduced QR `V=Q0 R`
and CPU SVD `R=U diag(s) W^T`, the retained basis is

```
Q = Q0 U[:, s > max(P,k) eps64 s_max].
```

Truncating the first columns of `Q0` is not rank revealing. It can yield the
wrong numerical subspace when `R` loses rank. Validate and save `P`, `k`, all
`s`, `s_max`, the tolerance, retained rank, `||Q^TQ-I||`, and
`||V-QQ^TV||/||V||`. Never write `Q` back into the estimator. If rank is less
than `k`, retain the seed but use the protocol's “numerical-span” label. The
primary tolerance is very permissive relative to the fp32 provenance of `V`,
so also report condition number/rank sensitivity descriptively; do not use a
post-outcome tolerance to change the branch.

Historical `r_native` must use the original fp32 multiplication and ordering.
The orthogonal action may be formed in float64 without materializing `QQ^T`,
then cast to the model gradient dtype. But the promised norm match concerns
the tensor actually handed to AdamW, not the pre-cast float64 vector. Record
norms before and after casting and require a predeclared small relative mismatch
on the delivered fp32 tensors. Apply the frozen `10^-30` clamp without a
fallback action: if `0 < ||r_orthogonal|| < 10^-30`, continue but mark the
control clamped and non-exact; if both delivered norms are zero, continue but
mark it degenerate and uninformative. If the delivered orthogonal action is
exactly zero while native is nonzero, fail the action-identity validation and
retain partial artifacts. When the clamp is inactive, require finite `c` and a
frozen post-cast norm-mismatch tolerance. In exact arithmetic native and
orthogonal actions share a nullspace, so these cases must never be silently
reported as an exact norm match.

Canonical QR/SVD signs are unnecessary for `QQ^T`, but raw coefficient hashes
can flip signs. Either canonicalize basis signs with a frozen rule or hash and
compare sign-invariant quantities. Near-tied singular subspaces can rotate;
same-step action and projector comparisons remain invariant, while individual
coefficient labels should not be tracked across time as fixed directions.

## 3. Immediate measurement contaminates continuation

**Risk: medium likelihood; medium severity.** Evaluating probes inside a live
branch can change global RNG state, model train/eval mode, gradient buffers, or
the sequence of later operations. It can also make a nominally common step
depend on measurement order.

Save and hash each post-Adam step-1501 state **before** any evaluation. Continue
the two training branches from those untouched states, and measure step 1501
later from separately loaded copies, ideally in the existing read-only
acquisition path. Use the already frozen row IDs and null permutations rather
than drawing them again. Assert that observation leaves model, optimizer,
filter and CPU/all-CUDA RNG hashes unchanged and restores training mode. The
current model has no dropout, but this state-isolation rule is still the clean
contract and prevents future implementation drift.

Per-state top-five selection is cross-fitted but can switch frequencies after
a tiny one-step perturbation. Treat the step-1501 selected score as a local
diagnostic, retain all 56 frequency scores, and show the fixed-panel sensitivity
beside it. Do not interpret an immediate selected-score jump alone as formation.
Continuous CE, margin, delivered action and Adam displacement are the stronger
one-step measurements.

## 4. Archived native reference limits the endpoint estimand

**Risk: certain limitation; medium severity.** Orthogonal versus norm-matched
is a contemporaneous common-state comparison. Either new policy versus archived
native additionally assumes that replay from the stored state would have
remained close to the historical CUDA trajectory. A single native step proves
the immediate action contrast but cannot bound 500 updates of numerical
trajectory sensitivity.

Therefore report the one-step three-action comparison as the clean intervention
and the 500/1000-step archived-native contrasts with the protocol's explicit
replay qualification. Large, consistent five-seed endpoint separation would
support a post-step-1500 action-policy effect; a difference comparable to known
replay sensitivity would not isolate within-span gains. No endpoint result can
retrospectively establish formation before step 1500 or mediation of the full
from-initialization legacy advantage.

## 5. Float64 QR performance and admission

**Risk: medium likelihood; medium severity.** The model has roughly 227k
parameters and the retained basis can have 200 columns. A tall float64 QR at
every update is substantially more expensive than the original fp32 legacy
action on an RTX 3090, and its temporary tensors/workspace are additional to
the stored fp32 `V`. The seed-100 admission run and three-hour per-seed cap are
appropriate. Record peak CUDA memory, QR/SVD time separately from training,
and projected output size before admitting the remaining seeds. Do not respond
to slowness by changing dtype, decomposition frequency, rank, or horizon after
seed-100 outcomes. Disk admission should preserve the 1 GiB reserve throughout
the worst-case 20 GiB envelope, not merely check for 1 GiB free at launch.

## Acceptance checklist

- Exact source/checkpoint/config and all five step-1500 states are hash-bound.
- One raw gradient and one post-update estimator state feed all first actions.
- The first action is not replayed when the two long branches continue.
- `Q=Q0 U_keep`; rank, span and orthogonality residuals are finite and saved.
- Native uses the historical fp32 path; norm matching is verified after cast.
- Parameter displacement equals adaptive Adam motion plus decoupled decay
  within a frozen fp32 tolerance, with carried moments included.
- Step-1501 states are saved before observation; measurement runs on copies and
  cannot mutate training RNG, mode, gradients or optimizer/filter state.
- Endpoint comparisons use the fixed 1500/2000/2500 grid and all five seeds;
  selected frequencies and checkpoints are not treated as replicates.
- Rank loss, norm degeneracy, resource expiry and partial output are reported
  fail-closed without dropping a seed or silently changing the intervention.

With these conditions, the design steelmans the positive legacy phenomenon:
agreement with the orthogonal policy would show that future unequal gains were
not needed over this window, while divergence would locate a concrete role for
legacy's numerical within-span reweighting. Neither outcome invalidates the
already observed behavioral and representation trajectories.
