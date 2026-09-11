# One new raw-direction policy under the legacy norm law

9 September 2026. Executable-protocol preparation for the approved safety-first
mechanism plan. Based on the reviewed prospective design (artifact not distributed in this public snapshot).
No completed branch, inference or scalar analysis is repeated.

## Fixed estimand and roster

At a branch's own current state, compute its raw full-batch gradient g and
advance the unchanged legacy observer once to V. Compute the native action
in its original dtype and parenthesization, then deliver the new policy:

```text
r_native = V (Vᵀ g)
a = ||r_native||₂, computed with a float64 norm
r_raw = cast_to_gradient_dtype(g.float64 * a / max(||g||₂, 10⁻³⁰))
```

Only r_raw is sent to AdamW. The original orthogonal and norm-matched projected
actions are computed as diagnostics, never followed by additional optimizer
steps. Keep the original reduced float64 QR/small-R SVD and numerical-rank
threshold for those diagnostics; never overwrite stored V with Q. Positive
native norm with zero raw direction fails. Nonzero sub-floor raw denominators
are clamped and explicitly non-exact. Unclamped cast norm mismatch must be at
most max(10⁻¹²,10×gradient-dtype-epsilon). Domain exactness and post-cast
tolerance are separate fields. Zero target yields a zero tensor, including when
g is nonzero; still execute ordinary AdamW with carried moments and decay.
Never replace it with `grad=None`. The historical no-basis identity fallback
is not a zero projector and must be explicitly recorded, not silently given
subspace diagnostics. Retain the old diagnostic projection-domain admission
checks rather than conceal an undefined projected norm comparison.

Use every original legacy seed100–104 full state at1500. Acquire exactly one
new policy per seed,1,000 updates1501–2500. Save complete states at1501,2000,2500:
15 new checkpoints. The model (p113,d_model128,heads4,d_mlp512), split(.3),
full-batch order, AdamW(lr.001,decay1,betas.9/.98,epsilon1e−8), legacy rank200,
observer decay.99 and all other original settings stay fixed. The five-seed
norm-matched projected corpus is the primary archived reference. Native and
orthogonal are secondary archived context. No reference acquisition is rerun.

The same **functional** scale law is evaluated on each trajectory's own state.
Later numerical scale sequences, estimators and Adam histories can diverge.
V still supplies the raw policy's scalar; this is removal of delivered-direction
restriction, not removal of every learned geometric dependency. Known CUDA
trajectory sensitivity limits archived-reference interpretation. Neither a
shared nominal checkpoint nor stored RNG certifies identical future arithmetic.

## Measurements and interpretation

At the first new update save actual raw g, native r, unscaled projected q,
norm-matched projected action and raw-matched action with unambiguous names;
retain Q and complete pre/post optimizer state, inherited-state hash, actual
movement, adaptive/decay decomposition and its float32 residual. This is one
new raw-policy update, not replay of the previously acquired shared actions.
If comparing its g to archived g, measure any discrepancy; never assume zero.

Log every new step1501–2500: basis k/rank/singular values, raw/delivered/native
norms, projected/raw norm fraction, action cosines, both norm-match domain and
post-cast flags, actual parameter-movement norm, adaptive/decay norms and
actual-minus-decomposed residual. For later steps retain scalar Adam summaries,
not all dense moments. Do not call these norms independently controlled doses.
Keep the original50-step evaluation grid, check that it consumes no RNG, and
capture1501 before any additional readout. Hash each trajectory JSON when
writing its completion receipt. Preserve original inner config/source identity
as inherited metadata; the outer envelope names the new policy and new sources.

Only the15 new states receive the existing frozen representation measurement
recipe later. Primary endpoints: held-out CE, correct-class margin and selected-
five final-hidden probe R² at2500; also report the fixed2000 earlier endpoint.
Reuse exact per-seed probe-fit/evaluation identities, all56 frequencies,
top-five selection, ridge.001,20 row-shuffle nulls and fixed panel[9,33,32,49,11].
No old inference is repeated. Report all five paired differences, sample means,
SEs and signs; accuracy is secondary. No outcome-selected window, new combined
success score, learning-rate/rank search, stopping extension, or equivalence
claim from a small/non-significant contrast.

Raw-policy deficits support a conditional benefit of the direction-restricted
policy including its later feedback; gains weaken its post-fork necessity.
Mixed/small comparisons remain unresolved. None identifies pre-fork formation,
semantic goodness, harmlessness, a generalization guarantee or a speed record.
The original positive useful-learning phenomena remain the motivation.

## Provenance, resource admission and launch requirements

Original accepted action batch:
`/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD`, completionSHA256
`22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45`.
Its ten scientific source hashes must remain unchanged. Verify each original
legacy1500 parent against the accepted receipt and complete-state schema before
loading it with the existing hash-bound `weights_only=True` helper.

New source consists of the separate policy/runner/batch, their synthetic
fixtures and this protocol, plus all imported frozen scientific helpers. Commit
these before acquisition and record their exact hashes. A dirty unrelated
documentation change does not authorize changing frozen scientific files.
Review [implementation constraints](implementation-check.md) and the analytic
resource note before one launch. Main performs final source review/admission.

Desktop RTX3090 only, original Python3.12.3/PyTorch2.11.0+cu128, one CPU math
thread. No cloud/API calls or paid spend. New exclusive mktemp parent directly
under `/tmp/spectral-experiment-artifacts`, exclusive seed subdirectories and atomic
non-overwriting artifact writes. No P-by-P array. Per-write prospective size
checks,20GiB total new-output ceiling, at least1GiB remaining free space,
16GiB RAM/no swap, CPU quota at most400%,3h per-seed and12h batch limits.
These are protective ceilings, not intended resource consumption.

Seed100 is resource/source/finite-state admission only; its accuracy cannot
decide whether the other four seeds run. Admit all remaining seeds under the
same committed source after that check. Save failure identity, last completed
step and partial checkpoints; no automatic retry, replacement seed or hidden
repair. Every acquired handle is consumed even on failure. No unrelated GPU
process is stopped and no other project is modified.

The batch may launch only after independent source review and synthetic
fixtures pass, original-source/parent admission is verified, the new source is
committed, and the exact unit/output/resource controls are recorded. No new
user approval is required within this already approved bounded scope.
No acquisition has occurred merely because this protocol exists.
