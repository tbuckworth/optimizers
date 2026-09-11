# I15 history-core design review

Read-only review, 7 September 2026. I reviewed `history_core.py` and its CPU
tests against the prospective [protocol](protocol.md) and
predictions (artifact not distributed in this public snapshot). I did not load an I14 parent, inspect an I15
outcome, execute training, or alter acquisition code.

## Verdict

The operational order is correct. Each step takes one raw backward pass, calls
the canonical observer exactly once, then reads the updated basis and
post-ingest mean. The native delivery retained from `filter_grad()` is checked
against a fresh application of the observer's actual action to the same raw
gradient. Mean delivery is then

\[
A_tg_t+(\mu_t-A_t\mu_t),
\]

using that post-ingest state. The alternate form
$\mu_t+A_t(g_t-\mu_t)$ and the EMA recurrence are checked with homogeneous
float32 bounds rather than presumed bit equality.

History projection is applied only to the already initialized old flat SGDm
buffer. The code writes either $b_{t-1}$ or $A_tb_{t-1}$ back into the live
per-parameter buffers, installs the selected delivery in `.grad`, applies the
same manual decay as I14, and lets unchanged torch SGD form
$b_t=.9\tilde b_{t-1}+h_t$. It never projects $b_t$ after adding $h_t$.
Consequently the mean complement remains in the
`mean_projected_history` update, as the design requires. `current_native` is
test-only, and its one-step complete-state equality with I14 is a strong check
that the adapter's diagnostic action calls do not mutate training state.

## Measurement qualifications

The saved quantities cover the stated interpretation: energies and
current-basis residual fractions for raw/native/mean/delivered gradients and
old/selected/removed/new buffers; total, decay-adjusted data and nominal-decay
displacements; signed dots with raw gradient, mean complement, projected old
history and removed old history; and recurrence/definition residuals. The
first-step digests expose the policy-independent raw, observer, old-buffer and
projected-old seams.

The implementation correctly does **not** enforce idempotence. It records
$\lVert A_t^2b-A_tb\rVert$, $\lVert V_t^\top V_t-I\rVert$, and the empirical
defect from the idealized data step $-.03b_t$. Therefore fields named
`current_basis_leakage` are action-residual measurements. They become literal
orthogonal-complement energy only to the extent supported by the separately
reported basis/action errors. Likewise, the signed dots are orientation
statistics only when read with their component norms; none is a clean-utility
or mediation fraction.

The prospective float32 checks are appropriately scale-homogeneous. EMA uses
an in-place multiply/add in the canonical filter whereas the audit expression
is recomputed out of place, so a roundoff allowance is necessary. The native
action and buffer recurrences also cross flatten/unflatten and matrix-operation
boundaries. Their bounds test internal consistency, not exact mathematical
orthogonality or a guarantee that every CUDA kernel follows the same rounding
path.

## Test scope and remaining limits

The seven CPU tests cover exact native-I14 state parity, explicit projected
history recurrence, a changing post-ingest basis, survival of the mean
complement, the ideal fixed-action tenfold DC-gain calculation, parent/state
continuity, and typed nonfinite state rejection. The `mean_native` path is
executed and its runtime recurrence invariant is checked by production code,
but it has no separate multi-step closed-form unit test. The tests also do not
establish CUDA bit identity, long-horizon numerical stability, or semantic
usefulness of any component. Those are disclosed limits, not reasons to widen
the prospective experiment: admission/parity checks and per-step invariants
must supply the corresponding evidence during acquisition.
