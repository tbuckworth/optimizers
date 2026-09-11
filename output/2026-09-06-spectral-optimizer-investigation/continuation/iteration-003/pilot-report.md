# Iteration 003 final development pilot: ready for review, not launched

The hardened development harness passes all gates. **No confirmatory seed,
validation performance, test performance or accuracy was evaluated.** The
parent must review and commit the frozen implementation before separately
authorizing full execution.

## What was checked

Eight [harness unit tests](test_harness.py) and five independent
[summary tests](test_summary.py) passed before the final pilot. Harness checks
include exact canonical width-32 filtering, separate estimation/projection
ranks, probe-state invariance, gradient and update decompositions, decay
subtraction, checkpoint ties, paired random plans, and preserved failure data.
Static compilation and whitespace checks passed.

The final pilot ran development seed 9876 with nominal replacement .9 for
220 steps in each of three arms, with instrumentation off and on: six traces
and 1,320 development updates. The 220-step duration was fixed prospectively to
include the first full-width scheduled covariance repair at step 200. No
accuracy-based choice was made.

Every instrumented/uninstrumented pair has bitwise-identical parameters at
**all 220 steps**, and exact matching final model, gradient, optimizer,
covariance, model-mode and RNG states. All 18 probe-state checks passed.
Finite-value, orthogonality and projector-identity gates passed. The wide
estimator reached rank 128; applied projection rank remains at most 32.

The initial passing attempt was preserved, then repeated only because review
identified provenance and failure-reporting gaps. The hardened source binds
training-data and analysis-source hashes and retains partial failed-run rows
with their active seed/arm/step/phase. It does not change the training algorithm
or measurements. **All six final-pilot parameter trajectories also match the
corresponding first-attempt trajectories bitwise.** No failed gate occurred in
either attempt.

## Timing and resources

Final pilot execution spanned **2026-09-06 10:53:12.637759–10:53:19.711809 UTC**
(7.07405 elapsed seconds including setup/output). The six synchronized training
loops sum to 6.48632 seconds. The RTX 3090 had only desktop/media processes
before launch, 655 MiB reported usage and 10% utilization; no training process
was present.

| Arm | Instrumented 220-step loop (s) | Steady steps129..220 mean (ms) | Uninstrumented steady mean (ms) | Instrumented step200 (ms) |
|---|---:|---:|---:|---:|
| AdamW | 1.16364 | 5.14358 | 2.14792 | 14.13094 |
| Estimate32 / project32 | 1.41512 | 6.54611 | 3.29071 | 19.58577 |
| Estimate128 / project32 | 1.58349 | 7.77475 | 4.44531 | 33.36771 |

Step 200 contains both a scheduled repair for filtered arms and probe
instrumentation; it is not a pure QR microbenchmark. The uninstrumented wide
step 200 is 20.09721 ms, against a 4.16999 ms steady median. Raw step timings and
ranks, including spikes, remain available.

Peak PyTorch allocation is **249,775,616 bytes (238.20 MiB)** and peak reservation
is **606,076,928 bytes (578 MiB)**. These do not include driver/context or other
processes' memory. Straight-line extrapolation for the planned 18×2,000 steps is
**233.57 seconds of instrumented training-loop time**, plus validation, test,
checkpoint/JSON output and run-to-run variation. Pilot trajectory hashing is
removed from the full run while validation is added. Reserve approximately
**10 minutes on the local 3090**; no paid/cluster resources are justified by
these measurements. This is a resource recommendation, not a performance claim.

## Frozen provenance and handoff

- Final execution manifest (artifact not distributed in this public snapshot) and
  all timing/invariant records (artifact not distributed in this public snapshot).
- [Prospective protocol](protocol.md), [harness](neural_harness.py), and separate
  [aggregation specification](analysis-plan.md).
- Source SHA256:
  `5c0bb4bdbf0c561eef43686aaf86f9608d564f4df46cda8c7f4ebf8d7f7bb1b7`.
- Protocol SHA256:
  `d9b49575309d59a6b053098794877b71e02320cb22fac29c8ac47fb0028c5bd1`.
- Harness-test SHA256:
  `3939f07b87339dd606e5ed973a71bf56729393f7f17af90aa42f54c901a9a97a`.
- The manifest also binds canonical optimizer, both raw training IDX files,
  `summarize_results.py`, `test_summary.py` and `analysis-plan.md`. Their exact
  hashes are machine-readable there; no test file was opened or hashed.

Execution HEAD was `904ec6196acb8bfd476362cf602327ef0a9cd6f2`; the complete
amended harness was not yet committed at pilot time. The content hashes above
identify the passing implementation. No further source/protocol/test edits
were made after this pilot. No commit or push was issued by this leaf worker.

The first attempt and its original report remain in
pilot-attempt-001 (artifact not distributed in this public snapshot) and
pilot-report-attempt-001.md (artifact not distributed in this public snapshot). Relocation is
documented; no data were deleted. Both local RNG-plan archives are Git-ignored,
with hashes/sizes retained in the manifests.

The full-launch guard requires all source files to match committed Git bytes
and the passing pilot, and training data to match the pilot hashes. It requires
explicit `--full --confirmatory-go`. In-arm failures preserve partial raw rows
and active context; pilot failure output never stores loss/gradient outcomes.
The independent summarizer accepts only all 18 completed confirmatory runs.
Official test loading occurs only after all training and validation checkpoint
selection is finished. The actual full experiment remains **not run**.
