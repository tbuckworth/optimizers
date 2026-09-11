# Independent finite-response collector source review

Codex · 9 September 2026.

Reviewed:

- `experiments/measure_grokking_function_response.py`, SHA-256
  `ab8d6d7eaad81afd95e270805c99bb7c7f1496635c83917484e140f60a2cb3a6`;
- `tests/test_grokking_function_response_measurement.py`, SHA-256
  `705329f7aaf8d52a9633d8b96cac4ceb73c948c7ec5eeb43d7cd0ad19a675160`;
- prospective protocol, SHA-256
  `7c291cf6418ae08508be41a63c190f5a038c5cfe816b1848b799f9866262c21f`;
- launch guard, SHA-256
  `98d887413c8eb64362bc6cbe6b14ff443257e401306c7a7d027bfb3542e44376`.

This was source review only. I did not open a scientific checkpoint, tensor or
NPZ member; run inference or an optimizer step on real state; calculate a real
metric; or launch a service. The nine synthetic CPU collector fixtures pass.

## Verdict

**Pass.** I found no blocking state, action, provenance, inference or lifecycle
defect in the reviewed collector. It implements the prospectively reviewed
three-new-action common-state diagnostic and forbids replay of raw or before.

## Input and provenance admission

- The raw acquisition and measurement completion files are bound to fixed
  SHA-256 values, schemas, successful status, exact seeds 100–104, raw policy,
  1,000-update completions and the exact 15-state measurement roster. Existing
  failure markers are rejected.
- Batch, measurement, seed, first-step, step-1,501 checkpoint, scalar, analyzed
  state, archived before/readout and raw/readout receipts are checked through
  their parent directories and expected names. JSON is re-read after receipt
  verification to catch admission-time change.
- The raw step-1,501 measurement is bound to the accepted raw checkpoint and
  its original legacy parent. The before scalar/readout is bound to that same
  parent, seed, step and accepted prior recipe. Source maps are compared with
  the currently accepted acquisition and measurement sources.
- Full before scientific state is checked against both the accepted fork hash
  and the first-step restoration hash. Saved raw after state is checked against
  the accepted step-1,501 checkpoint. The scientific hash includes model,
  optimizer, filter, RNG, configuration, split and parameter identity.
- Current CUDA/library environments must satisfy the inherited prior and raw
  acquisition/readout environment contracts. The manifest preserves both
  readout and acquisition environments and all admitted input receipts.

## Action construction

The collector reuses the saved float32 raw and norm-matched projected actions
and saved float64 retained basis `Q`. It constructs only

<pre>
trunc = cast_float32(Q(Qᵀ raw_float64))
zero  = zeros_like(raw)
</pre>

and never recomputes a gradient, observer update or SVD. Shape, parameter-vector
length, dtype, finiteness, positive nondegeneracy and `Q` orthogonality are
checked. It records all action norms, `rho`, norm matching, projection rounding,
`trunc ≈ rho * projected` and collinearity residuals, and enforces their fixed
float32-scaled tolerances. The zero action is exact and remains an explicit
gradient rather than `None`.

The saved raw-action/raw-gradient identity was already checked by the
independent first-tensor audit and is bound here through the immutable
first-step receipt and accepted source map. Rechecking their collinearity would
be optional self-contained redundancy, not a present scientific blocker.

## Exactly one common-state Adam step

- `apply_once` rejects every name except `trunc`, `projected` and `zero` before
  constructing a model. Raw, before and native cannot be replayed through this
  path.
- Each permitted action creates a fresh model and AdamW instance, strictly
  restores the identical saved before model and complete optimizer state, and
  verifies model/Adam/parameter-order identity before intervention.
- The saved CPU and all-device CUDA RNG states are restored independently for
  every action. Each parameter receives its contiguous action slice as a cloned
  float32 gradient, including explicit zeros, followed by exactly one
  `optimizer.step()` and no forward, backward or observer call.
- The saved before object is checked for mutation. The resulting model and Adam
  state must be finite. Every parameter's Adam step counter, first moment,
  second moment and decoupled-AdamW parameter recurrence are corroborated under
  the fixed float32-scaled tolerance while requiring unchanged parameter-group
  options.
- Ordinary AdamW uses no RNG, and the synthetic fixture independently confirms
  unchanged restored RNG across each action. A further runtime equality check
  immediately after `optimizer.step()` would be redundant hardening rather than
  a blocker for this deterministic one-step diagnostic.

## Function extraction and metrics

For each seed, the collector loads the receipt-bound archived before and raw
logits, requires their exact NPZ member schema and identical structural arrays,
and verifies every structural-array hash. It independently reconstructs and
checks the ordered `113²` grid, modular sums and complete train/test partition.

Only the three new post-action models are sent through the unchanged extractor,
once each over the full ordered grid with fixed batch size 1,024. Thus the total
is exactly 15 new state extractions. The raw post-state and raw logits are reused,
not regenerated. Each extraction is checked not to mutate model weights or CPU
or CUDA RNG. Probe tensors returned by the unchanged extractor are not fitted
or saved as new probe results; only logits enter the diagnostic.

The source then applies the separately reviewed pure metric module to all five
response states, saves every seed result and binds the exact seed order before
forming the five-seed paired summary. Archived and new logits remain separate
CUDA invocations, and the manifest explicitly retains that resolution caveat.

## Outputs and lifecycle

Output is restricted to a new direct `diagnostic-001` child on the designated
large-volume naming pattern. Every JSON, NPZ and tensor file is exclusively
created and immediately receipt-hashed. Prospective total-output and free-space
budgets are checked before and after writes. Cooperative time, resident-memory
and zero-swap checks run throughout; the external reviewed guard supplies the
hard cgroup and service bounds, exact parent, no-restart policy, terminal prior
units, interpreter and committed/current source validation.

All five seeds must complete in order before summary and completion. Every
input receipt and source hash is rechecked before the exclusive completion
record. Failures preserve accepted partial receipts and cannot overwrite prior
output. The completion counts of 15 new actions and 15 new extractions denote
three authorized actions for each of five seeds; no continuation loop or old
branch execution exists.
