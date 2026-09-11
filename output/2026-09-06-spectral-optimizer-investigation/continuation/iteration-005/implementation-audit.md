# Independent implementation audit — iteration 005

**PASS for the reviewed replay/reference/storage implementation.** Two review
concerns were fixed and regression-tested. No remaining implementation blocker
was identified in this scope. This is not approval to run the development pilot,
worst-size resource check or full MNIST replay; those require separate parent GO.
The summarizer is under a separate independent review.

## Findings and resolutions

1. The original occupancy gate checked process names after querying CUDA device
   identity, so it could not distinguish its own newly initialized Python CUDA
   process from a competing job. `check_compute_processes` in
   [replay_harness.py](replay_harness.py), line 101, now parses PID and name,
   exempts only its own PID, and rejects another Python PID. Its mocked
   regression passes. No GPU query was run by this auditor.
2. The old snapshot helper omits NumPy's global RNG. Using it for new observation
   and final pilot comparisons did not cover the stated complete core state.
   The new `core_snapshot` at line 127 includes NumPy state and is used by those
   new checks. The historical warmup hash deliberately retains its original
   convention. The regression detects a NumPy mutation in the new snapshot
   while confirming the old hash remains unchanged.

Both were identified prospectively, not through a failed MNIST outcome.
The review-changes guidance informed the issue-first review and explicit
distinction between resolved findings and untested runtime claims.

## Reviewed behavior

- Historical bindings anchor the previous manifest to its audited Git commit,
  hash the original execution/AdamW records and training/plan/checkpoint files,
  and check historical software versions. Regenerated plans must match every
  saved array. Named checkpoint tensors, all 100 warmup hashes and their core
  hash, and five per-step scalar quantities are exact comparisons. The code
  preserves duplicate checkpoint names at the same step and requires all four
  names. These paths were inspected, **not executed against MNIST** in this audit.
- The same raw tensor is supplied to both observers once per step. Counters
  advance before the canonical update, and observer means must match an
  independently maintained mean with the same rounded operation sequence.
  Non-finite startup inputs are rejected; first acceptance, prior-zero
  innovations and later reset/reinitialization are guarded.
- Previous bases are copied before observing g_t. Current states and probes
  are captured afterward at theta_(t-1), before AdamW. Probes use the specified
  fixed-label training and disjoint-clean batches, never update observers,
  and are protected by complete state checks. The actual delivered gradient
  must remain bitwise equal to raw g_t. A tiny synthetic CPU optimizer fixture
  checks measured/unmeasured core trajectories and historical flattened-gradient
  assignment, without using the MNIST harness entry point.
- The reference uses all saved rounded innovations with the overweighted first
  term, float64 weighted columns, a symmetric dual Gram matrix and descending
  eigenpairs. Residual, orthogonality, spectrum, positivity, eigengap and
  cancellation gates follow the protocol. The formulas retain nonorthogonal
  stored-basis terms and distinguish native float32 action, the represented
  operator and the QR span. Boundary-gap failure nulls unique-reference
  projector metrics without nulling an otherwise valid primary energy fraction.
  No-basis identity, zero-input nulls and complete null reasons are implemented.
- Bulk storage is scoped to an exclusively created, mount-verified large-volume
  directory. Arrays retain shape/dtype metadata; bundles inventory tensor leaves.
  Append order, completed versus flushed row counts, budgets, hashes, exclusive
  artifact creation and read-only reference maps are explicit. Failure handling
  preserves partial context and confirmed flushed-row counts, attempts a full
  confirmatory state capture, records rejected non-finite values safely, and
  places a compact failure-artifact pointer in the execution manifest.
- Launch code refuses existing attempt outputs, requires separate mode-specific
  GO flags, checks the passing pilot and exact source/data binding for full
  execution, and installs elapsed-time/resource checks. It never calls the
  official-test loader or creates a new validation selector. Source/schema
  fields agree for the 16 observer metrics, eight reference metrics, replay
  counts and pre-Adam snapshot phase.

## Executed checks and their limits

Independently ran the final **22 synthetic CPU unit tests**, all passing. This
includes small dense/dual cases, delayed/all-zero initialization, rank/gap/null
handling, nonorthogonal covariance/operator formulas, probe invariance,
checkpoint cloning, finite rejection, source/GO/time/PID gates, and tiny storage
fixtures. Storage fixtures were created under the verified large-volume scope;
no previous evidence was edited or deleted.

The separate [audit checker](audit_implementation_checks.py) imports only
`reference_math`, not the experiment launcher. Its **247 assertions pass** over
eight synthetic innovation streams and 16 observer cases, plus boundary-tie and
no-basis cases. Maximum dimensions are nine parameters and 18 observations.
Expectations come from explicit dense covariance, operator and projector
matrices, not the low-rank summary formulas being checked. Largest absolute
scalar discrepancy is `1.4210854715202004e-14`; largest dense weighted-covariance
reconstruction norm discrepancy is `3.605944783080633e-15`.

These checks do not establish GPU historical replay equality, large-Gram
conditioning, full-scale runtime/memory/I/O compliance, interruption recovery
under every operating-system failure, or MNIST scientific outcomes. The
fixed resource pilot and subsequent raw-result audit remain necessary. Historical
equality remains limited to the retained vector anchors and scalar diagnostics;
there are no historical full-vector hashes after warmup to independently check
every old step. No MNIST, worst-size calculation, development/full run or GPU
computation was performed by this auditor. No author/production/frozen source
was edited and no commit or subagent was used.

## Reviewed and tested source hashes

SHA-256:

- `replay_harness.py`: `35d99dd08fd7b9831a43820c8a5e1df40ba5ff759cc78d004daf4de08ddfb862`
- `reference_math.py`: `3256c40cfe110edb035ecf38a0f1625534a941b5266da7b8c2f99a95595e50ac`
- `artifact_store.py`: `f26f0a478c6cb7feee5d5e39e52e1cac6d2174466d0b2692395fad7539c6c053`
- `test_harness.py`: `d8c6f46d45250300d47d33e6d6f795ad9730c340af07d5f9d7dade57d95c6217`
- `audit_implementation_checks.py`: `2d836a1c84eb0de0186c43a9f9311cf85be7194cfcdb1c6c41da927d69217fd4`
- `protocol.md`: `eb8f49f4a5247e9c33a23a0d15a572f2c5cec08355f2b4d05ada50f435ce0035`
- `result-schema.md`: `eb2af92b07d879f397e3b6ec0fb6fd5df8c85b62ab51f9b0d4fcd36c1e70408c`
- `analysis-plan.md`: `569eff833192c26c00587e155925088bc07b9eb8004040a997221a672f3893e7`
- `reference-identities.md`: `8afb8291018f58e1f91319fe69481442b00c2ae4c36cc9aa2f76b949307bfe43`
