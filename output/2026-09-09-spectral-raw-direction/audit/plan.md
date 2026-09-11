# Independent saved-evidence audit plan

Prepared while the new raw-direction training batch is live, before inspecting
its scientific outcomes. **Source and tiny synthetic fixtures only until main
has reviewed these scripts and explicitly authorized a completed-data audit.**

This borrows independent arithmetic and claim checking from the researcher
results-auditor skill. The user's explicit scope overrides its fresh-seed/split
re-execution and fail-fast gate prescriptions: no new training, inference,
replacement seed, old-result rerun or optimizer step occurs here. A passing
audit establishes saved-evidence consistency, not fresh replication.

## Two separately bounded checks

1. `audit_tensors_histories.py`: all five seeds100–104, all1,000 new history
   rows1501–2500 per seed, and each new first-step tensor artifact. Original
   parent identity is checked against the pinned accepted archived receipt JSON
   and the new `full_before` scientific-state hash; no original parent tensor or
   old first-action tensor is reopened. Load only the new first-step artifact
   and new1501 checkpoint, CPU/weights-only. Independently reconstruct raw norm
   matching and projected norm matching after dtype cast, distinguish actual raw
   g/native/q/projected-matched/raw-matched tensors, verify Q/rank/singular values,
   action coefficients/norms/cosines/rho, moment updates, adaptive/decay movement,
   actual parameter difference, and the decomposition residual. The saved raw
   gradient is not checked by rerunning backpropagation; equality with an
   archived raw gradient is not assumed or claimed.
2. `audit_readouts.py`: exactly15 newly measured NPZs, all seeds and steps
   1501/2000/2500. Independently derive behavior, all56 frequency readouts,
   top-five selection, fixed panel and all20 row-shuffle null selections in
   NumPy, then check symmetry secondary metrics and all six paired endpoint
   contrasts/14 metrics. Old reference values come only from the accepted
   action-summary JSON with hash249a8b9…aa237; no old NPZ/model is reopened.
   New summary-row transformation and reference-field schemas must match the
   prospective raw analyzer source before launch, not be guessed from outcomes.

Later Adam histories contain scalar norms only. Check finite/nonnegative norms,
adaptive movement=lr×adaptive-direction norm, residual max/norm consistency and
triangle bounds; do not invent later vector recurrences or call scalar checks
a dense-state audit. Norm-match `exact` fields indicate denominator domain;
separately check post-cast mismatch/tolerance, including clamp/both-zero cases.
No-basis identity rows are distinct from zero projectors and are flagged for
interpretation, not silently supplied missing subspace statistics.

## Admission, resources and reporting

Require main-supplied completed batch SHA256 (and measurement/analysis completion
SHA256 for readout checks). Verify exact manifest/seed/artifact/15-state rosters,
source maps, every bound history/first-step/checkpoint receipt, and input hashes
again at completion. Keep error lists and failing outputs; no automatic retries.
Record source/plan/input/output hashes, interpreter/library versions, runtime
and peak RSS. Each check uses a new exclusive output under the fixed new batch
`/tmp/spectral-experiment-artifacts/spectral-grokking-raw-direction-20260909.KzGtkl`.

Each check: one CPU math thread,16GiB/no swap,10-minute external Type=exec service
with580-second cooperative bound,100MiB output and1GiB free-space reserve. No
GPU/cloud allocation or model forward/backward call. Main checks actual command
and interpreter; each script checks its cgroup limits and original training
service terminal state before real-data work. Synthetic fixtures bypass real
artifact access and runtime admission by calling only pure numerical helpers.

Interpret all paired differences at the original fixed2000/2500 endpoints with
all five seeds. Audit arithmetic, sign/count/SD/SE/undefined handling and archived
provenance; no p-values, equivalence conclusion, endpoint extension or favorable
subset. Full numerical agreement does not establish semantic selection, safety,
pre-fork causation, or a speedup. Preserve small/mixed/adverse effects.
