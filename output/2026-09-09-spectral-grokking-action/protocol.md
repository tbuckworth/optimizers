# Common-state legacy action substitution

9 September 2026. Prospective implementation of the approved focused grokking
mechanism plan, following the completed representation review. No new baseline,
task, paid compute or capability speedrun. This selects the previously written
decision note (artifact not distributed in this public snapshot)
without changing its checkpoint, roster or horizons.

## Question and fixed design

Does the replicated earlier rule-readability trajectory depend on the legacy
operator's unequal within-span gains after update 1500, or does orthogonal
projection onto its selected span preserve it? Start from every existing legacy
seed 100–104 at absolute step 1500. Two new policies each run exactly 1000
updates, to step 2500; capture step 2000 as the earlier endpoint. Completed native
trajectories remain archived references. No native continuation is repeated.

For each seed compute the full-batch gradient once from the restored state,
then advance the unchanged legacy estimator once. All three first actions share
these identical tensors, not merely a recomputed nominally identical gradient:

    r_native = V (Vᵀ g)
    Q = numerical orthonormal basis of V
    r_orthogonal = Q (Qᵀ g)
    c = ‖r_native‖ / max(‖r_orthogonal‖, 10⁻³⁰)
    r_norm_matched = c r_orthogonal

Native action is evaluated in the original gradient dtype and multiplication
order. Orthogonal action is computed in float64 then cast to that dtype.
Norm matching uses the norm of that actually delivered cast orthogonal action,
scales it in float64 and casts the result back, recording residual roundoff.
Take float64 reduced QR of the tall V on its compute device and reduced SVD
of its small R on CPU. Reconstruct Q from QR's Q times retained left singular
vectors of R. Keep singular values strictly above
`max(P,k) * eps(float64) * largest_singular_value`. Save every singular value,
tolerance and numerical rank at every update. This uses no P×P matrix or
explicit inverse. Exact same-span language requires retained rank = k; otherwise
say numerical-span intervention. QR signs do not change Q Qᵀ. Never replace the
stored legacy V with this diagnostic/action Q. Norm-matching degeneracy or
rounding mismatch is recorded, not concealed as an exact control. Zero
orthogonal norm with positive native norm fails action construction; preserve
partial state, since no scaled zero vector can implement the intended match.
Nonzero sub-floor orthogonal norm follows the frozen clamped policy, explicitly
marked non-exact. Two zero norms are degenerate/uninformative, not failure.
Unclamped norm matching requires relative post-cast mismatch at most
`max(1e-12, 10*eps(gradient_dtype))`; otherwise fail construction.

Restore the original complete state separately for each delivered first action;
verify hashes of model, Adam, filter, RNG and settings against the original
before substituting the shared post-estimator state and action. Deliver each
action once with the original AdamW. Native stops after this one diagnostic
update. Orthogonal and norm-matched continue from their actual first-step states;
do not replay that first step. Thereafter each branch updates its own unchanged
legacy recurrence from its own gradient. The short-horizon estimand is an
action-policy intervention, not a frozen common projector.

## Evidence and measurements

Save full fork, first-step action and optimizer diagnostic tensors, including
common Q coefficients, raw/delivered gradients, post-Adam m/v, bias-corrected
adaptive direction, separate decay movement and actual parameter displacement.
Report pairwise gradient and displacement norms/cosines. Record float32 residual
between actual movement and the mathematical adaptive-plus-decay decomposition.
Norm matching before Adam does not match post-Adam displacement or decay balance.

Save full state for native step1501 and both new policies at1501/2000/2500:
35 new states in total. These use an explicitly intervention-labelled envelope
around the original state schema; inherited original config/source fields do
not label a new policy as a native trajectory. Capture native-compatible
50-step train/test evaluations for the two new branches too. No rerun of old
1500/2000/2500 native inference: reuse its audited activations/scalars.

Measure only new states with the already validated and frozen representation,
20 row-shuffle, pre-attention, margin and symmetry recipe. Reuse the exact
per-seed fit/evaluation IDs, null permutations and fixed frequency panel
`[9,33,32,49,11]`. Primary outcomes: held-out CE, mean correct-class margin and
selected-five evaluation R², at2000 and2500. Step1501 is a local action
diagnostic. Report all five paired seeds, signs, mean and sample SE; checkpoints
and frequencies are not independent replicates. Any AUC comparison uses only
the common absolute grid (1500,2000,2500 for representation), not the additional
new-policy observations. Report all fixed endpoints regardless of direction.

Preserve microscopic CUDA non-reproducibility observed in the original corpus:
the archived reference is not a concurrently replayed control. Exact equality
at the shared first action does not remove later numerical trajectory
sensitivity. No circuit, causal mediation, pre-fork formation, span sufficiency
from initialization, safety or general speedup claim follows from this test.
Orthogonal failure would refine, not erase, the positive learning phenomenon.

## Execution and resource limits

Original source/checkpoint hashes must match the audited corpus. New scientific
sources, protocol and tests are committed before acquisition. Use the original
PyTorch2.11.0+cu128/Python3.12.3, one CPU math thread, RTX3090, full-batch order,
Adam hyperparameters and CUDA settings. Record currently observable backend
flags; historical flags were not all recorded, so do not invent a historical
bitwise guarantee. No deterministic-algorithm toggle or optimizer retuning.

One exclusive large-volume parent under `/tmp/spectral-experiment-artifacts`; new seed
subdirectories are exclusive-create. Stage seed100 for resource admission,
then all remaining four seeds with the identical frozen code regardless of
scientific result. Enforce at most 20GiB output, at least1GiB free reserve,
16GiB RAM/no swap/CPU400%, 3h per seed and 12h overall service bounds.
These are protective upper bounds, not intended spend. The admission check is
successful completion, finite values, bounds and source/state identity—not an
optimizer-success gate. Numerical-rank reduction is evidence, not a failed gate.
Partial acquisitions and exact last-step handles are preserved; no automatic
retry or outcome-driven extension. Paid spend remains0/100USD.

## Implementation validation sources

Reduced float64 decomposition and non-unique singular-vector signs are checked
against [PyTorch2.11 SVD documentation](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.svd.html).
The replay limitation follows [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
and the actual earlier checkpoint audit. New action fixtures separately test
rank loss, anisotropic gains, norm matching, unchanged stored estimator and
native multiplication. Independent review precedes acquisition; independent
raw-result audit precedes causal conclusions.
