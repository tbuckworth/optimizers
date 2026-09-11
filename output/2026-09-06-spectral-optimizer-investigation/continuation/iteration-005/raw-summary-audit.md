# Independent raw/provenance/summary audit — iteration 005

**PASS within this scope.** Every checked summary group agrees with independent
aggregation of the raw records. Artifact and checkpoint links also pass. This
audit does not recompute the reference eigensystems or evaluate a dataset/model;
the separate spectral-reference audit is still required for that numerical
layer.

The [standalone checker](audit_raw_summary.py) imports neither the training
harness nor the summarizer. It refuses outcome reads until the completed full
execution marker and parent summary both exist, and refuses output overwrite.
It ran once on CPU, 13:21:22–13:21:38 UTC, in 15.31 seconds. Full results,
individual values, masks and hashes are in
[raw-summary-audit.json](raw-summary-audit.json).

## Direct verification coverage

- Confirmed exactly three replay records and twelve ordered seed/snapshot
  records: seeds 3,4,5 and steps 200,500,1000,2000. All required completion,
  historical/state/numerical gates and counters agree with the schema; snapshot
  phase is `pre_adam_after_current_gradient_observation`. No test/selection
  computation is recorded.
- Verified all **19 scientific source hashes** against current bytes, the
  accepted pilot, source freeze, launch commit and execution commit. Verified
  ancestor order and that all three commits precede full execution. The diff
  from launch to execution adds only three separate ideal-truncation
  theory/check artifacts; it changes none of the 19 bound scientific files.
- Hashed **91 unique artifacts** with exact size checks: 13 historical/data
  bindings and 78 new bulk artifacts (six streams, twelve state bundles, sixty
  reference arrays). A supplementary path/count check puts all 78 bulk files
  beneath the recorded attempt root; their total size is **3,303,070,440 bytes**.
  Historical bindings equal those accepted in the pilot.
- Independently hashed all **12,000 float32 raw-gradient/innovation rows**.
  Recomputed all **6,000 historical raw-gradient squared norms** on CPU; maximum
  absolute difference from the recorded GPU reduction is
  `2.220446049250313e-15`, below the pre-execution audit comparison tolerance
  `rtol=5e-12, atol=1e-14`. This is a numeric comparison tolerance, not a proof
  of a rounding bound.
- Matched all 300 historical warmup hash records and three core hashes.
  Loaded only saved checkpoint tensors with `weights_only=True` on CPU and
  verified all **twelve named historical checkpoint parameter hashes** against
  their replay post-step hashes. Every anchor in this run is after step zero,
  so all twelve receive that independent saved-tensor check.
- Verified all **twelve snapshot phase links**: pre-Adam parameter hash equals
  the preceding post-step hash; saved current raw/innovation tensors match the
  stream row; saved rounded centering agrees; both observer means are identical
  to the saved mean; observer counters/ranks match their snapshot metadata.
- Validated all 480 raw metric entries and 144 probe-energy ratios, including
  primary/reference-rank availability, reference-gap flags, covariance-error
  arithmetic, represented-energy ratios, clean-minus-corruption differences and
  self-inclusion differences. This uses saved scalars/energies, not an independent
  eigensystem or projection computation.
- Independently reconstructed **293 summary groups**: the primary, 224 fixed-step
  groups, 56 four-snapshot groups and twelve energy-weighted groups. Every seed
  value, mean, median, minimum, maximum, sample SD, count, exact validity mask,
  null reason and summed-energy numerator/denominator agrees. Ordinary means
  remain distinct from energy-weighted ratios. There happen to be **no raw
  scalar nulls in this run**; actual masks were checked as empty, while nonempty
  null behavior was addressed by the prior synthetic implementation review.
- Verified the summary's copied raw records, execution record, input hashes and
  source hash against their originals. Numeric summary comparisons use
  `rtol=1e-12, atol=1e-14`; no bitwise-float equality claim is implied.

## Numerical findings retained without broadening the claim

The primary is the **step-2000 QR-span energy divided by the optimal rank-32
reference energy**, not the fraction of all covariance variance and not
classification accuracy:

| Seed | Width 32 | Width 128 | Width128 minus width32 |
|---|---:|---:|---:|
| 3 | 0.98865110 | 0.99983340 | +0.01118230 |
| 4 | 0.98756349 | 0.99983685 | +0.01227337 |
| 5 | 0.98799865 | 0.99984747 | +0.01184882 |

The paired mean is **+0.01176816**, descriptive sample SD **0.00054999**.
The energy contrast is positive at all twelve saved seed/states, but those
overlapping prefixes are not twelve independent replications.

Native clean-minus-fixed-corruption retention improves at the final state in
all three seeds: **+0.009405/+0.011749/+0.003810**, mean **+0.008321**. Preserve
the adverse intermediate values: step 500 seed 4 is **−0.006561**; step 1000
seeds 3 and 5 are **−0.004684/−0.002255**, with across-seed mean **−0.001009**.
The complete four-snapshot paired mean is +0.004713. Final clean retention
itself falls in seeds 3 and 5; their residual retention falls more. Therefore
the fidelity result is more uniform than the retention result, and an increased
retention difference must not be paraphrased as universally retaining more
clean gradient.

These remain measurements on three reused, passive-observer AdamW streams.
They establish no new training-policy outcome, independent dataset replication
or unique semantic-denoising mechanism. Full-width covariance error also has
unequal estimator capacity and is not the primary matched-rank comparison.

## Scope limits and provenance

The independent reference audit owns rebuilding weighted Grams/eigenspaces and
observer/reference projections from vectors. This audit verifies their saved
bytes and scalar arithmetic, not those computations. Historical per-step
displacement comparisons beyond retained checkpoint anchors remain source and
recorded-gate evidence: full parameter arrays were not retained at every step.
The CPU raw-gradient norm checks strengthen, but do not remove, that limitation.
Chronology is supported by Git and recorded timestamps, not independent
operating-system access telemetry.

Source/launch/execution commits are respectively:

- `de1ba26d43a6df80162ee9530d8b64c9ac08acf3` — 13:04:14 UTC;
- `bc29741aa9e15eacddd148e99a640c7d338040d9` — 13:12:27 UTC;
- `fa384618a2a3d8177960be419a144214aa2b21fc` — 13:13:19 UTC.

Full execution starts at 13:14:41.968787 UTC and completes at
13:16:53.137456 UTC. SHA-256 audit anchors:

- Checker: `004e619352745ba94cfa58ec4f001a28589b591fd800a32220507ab461d5df81`
- Audit JSON: `e1ba8337806a17620a1308662db0fe2c4919b3ba263d7a3f98b75f031dcc9cf0`
- Parent summary: `81ebeacbd84a9936529128fdc8f27be0b1b93721793933cad9fd021dc57e461e`
- Execution: `b8368fe5ee628dee648a2167e909a0380c6abc2b3fcc7d11976a707dd86e3478`

Only the new audit checker, JSON and this note were written. No original record,
frozen source, production code or bulk artifact was changed; no commit or
subagent was used.
