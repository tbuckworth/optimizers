# Function-response collector: implementation and admission

9 September 2026. Source prepared in
`experiments/measure_grokking_function_response.py`; synthetic fixtures in
`tests/test_grokking_function_response_measurement.py`. Main review, committed
source freeze and guarded acquisition remain required. This note records no
new scientific counterfactual, inference or function-response outcome.

The collector admits the five fixed seeds through the pinned raw acquisition
and measurement completions. It follows the latter's accepted before-state
recipe receipts. Each first-step bundle's before scientific state is checked
against the accepted fork, and its saved raw after-state is checked against
the hash-bound raw-1501 checkpoint. Archived before/raw logits are copied into
the new self-contained NPZ, retaining original receipts; their structural
arrays must match each other and the recorded per-seed hashes.

Only `trunc`, `projected` and `zero` can execute `apply_once`. Each gets a new
model/Adam instance restored from cloned common-before state, including original
parameter order/options and saved RNG. Every gradient tensor is explicitly
assigned. Exactly one Adam step is called; the raw action cannot be replayed.
The unchanged CUDA batch-1024 extractor runs only on those three new states.
Transient hidden readouts from that extractor are discarded, not analyzed or
saved; there are no new probes, backward passes, observer updates or SVDs.

Action checks admit strictly positive finite g/u/t/v norms, derive rho from
the saved g and Q, and record norm, collinearity and projection residuals.
The fixed relative threshold is 32 float32 eps; Q's maximum orthogonality
error must be at most 1e-10. Moment and parameter recurrence residuals are
computed separately from saved before/actual after tensors, normalized by
max(max-absolute expected, max-absolute old, float32 tiny), also bounded by
32 float32 eps. These checks do not execute another optimizer step. Unrounded
residuals are retained; failing residuals appear in the preserved error.

Each seed saves a logits NPZ (before/raw/trunc/projected/zero plus grid/split),
a tensor artifact (five parameter/Adam states, four incoming actions and state
identities), and JSON metrics from the separately reviewed pure response module.
The completed summary retains all five results and the fixed paired summary.
Source/input/output receipts are chained to completion. Output is exclusive,
prospectively capped at 2 GiB with 1 GiB free reserve; cooperative time is 480
seconds, with the hard 16-GiB/no-swap/one-CPU/600-second unit enforced by main's
guard. No automatic retry or overwrite is implemented.

## Checks performed before acquisition

**Nine synthetic CPU tests PASS, 0.815 seconds.** They cover action identities,
degeneracy/nonfinite/dtype failures, independent state restoration and execution
flags, explicit-zero carried-state movement, forbidden raw replay, state/order
and action-shape failures, a deliberately corrupted moment, shared tiny NPZ
structure, and output exclusivity/budget failures. The tiny Adam setup fixtures
are not model training or replay of any research checkpoint.

At main's explicit request, actual **JSON/opaque-file-hash-only admission PASS**:
five seeds, 47 receipts, 9.381 seconds; session 76628 consumed with exit 0.
No real tensor or NPZ member was loaded. An earlier read-only admission command
became terminal without a captured final tool receipt; the recorded PASS is
from the separately captured repeat of this admission check, not a repeated
experiment. This distinguishes source/receipt admission from scientific
acquisition and avoids implying an unobserved completion result.

The common-state before/raw references remain separate archived CUDA
invocations. Identity/source/batch checks do not erase the numerical-sensitivity
caveat, and neither local result would identify the full 1,000-step mechanism.
