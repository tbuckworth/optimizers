# Independent saved-trajectory arithmetic audit

9 September 2026, 13:25 UTC. **PASS: 257,373 checks, zero errors.**

The independently written [checker](check.py) read the fixed plan and summary
schema, but did not import or reuse `analyze.py`. It read all ten original
history JSON files: five seeds 100–104, both orthogonal and norm-matched
policies, every one of the 999 steps 1502–2500 per branch. No checkpoint,
tensor library, inference, training, GPU or paid resource was used.

## Recomputed scope

- All 30 per-seed/policy/window entries: the whole 1502–2500 interval and the
  fixed 1502–2000 and 2001–2500 windows. Every metric's count, defined and
  undefined counts, minimum, maximum, mean and median were recalculated from
  the original saved diagnostic scalars.
- All six equal-weight seed aggregates, their observed ranges and flag sums.
  Means are means of five seed means; temporally dependent steps are not
  treated as independent experimental replications.
- Rank threshold bookkeeping, dimensions, singular-value ordering, coefficient
  availability and finite values, norm ratios, norm-matching scalar and
  post-cast mismatch arithmetic, availability and all reported flags.
- Exact contiguous step rosters, branch/seed/schema identity, all seven
  checkpoint receipt identities per seed, accepted parent receipt identities,
  scientific and analysis source hashes, all current input JSON hashes and
  file sizes, and before/after history hashes.
- The separate step-1501 summary is exactly the selected fields of the existing
  accepted tensor audit. It was not mixed with later branch histories or
  independently re-measured.

All numerical summary comparisons pass at absolute and relative tolerances
of 1e-12. Maximum absolute floating difference is **2.842170943040401e-14**,
consistent with summation-method rounding. All integer counts, rosters,
metadata and hashes agree. No reported values are undefined in these actual
histories; absent orthogonal-branch native/cosine/scaling quantities remain
absent, not silently imputed. Degenerate, clamped, nonexact and
tolerance-exceeded counts are all zero in the norm-matched branch.

**Checker limitation found in subsequent main-agent review:** the check named
`exact_flag` compares the saved `norm_match_exact` flag to whether the post-cast
relative mismatch passes tolerance. Those are not the flag's general semantics.
The original action source defines exactness by the denominator domain:
`orthogonal_norm >= 1e-30`, or both native and orthogonal norms exactly zero;
post-cast tolerance is a separate check. Reading the production analyzer only
after this audit completed confirmed that its formula follows the original
domain definition. Every actual norm-matched row here has an unclamped,
nondegenerate positive denominator and passes post-cast tolerance, so both
conditions happen to be true throughout and no numerical result or count
changes. The independent checker must not be reused as a general validation of
exactness semantics. Its frozen source and result are preserved without a
rerun; this explicit correction is part of the accepted audit interpretation.

## Execution and binding

The single launch used transient service
`spectral-base-grokking-trajectory-independent-audit-001.service`, invocation
`6e3f904f2a5349279ce5e03bb2ed9f61`. Its launch specified `MemoryMax=2G`,
`MemorySwapMax=0`, `CPUQuota=100%` and `RuntimeMaxSec=300s`; the script also
pins itself to one CPU and enforces a 280-second cooperative deadline and
100-MiB output cap. The process completed in **3.8196031898260117 seconds**,
with peak RSS **155,090,944 bytes**. Journal evidence at
`2026-09-09T14:25:08.543906+01:00` records Python PID **2901957**, PASS,
257,373 checks and no errors. The unit was already garbage-collected when
properties were queried, so an effective post-launch property receipt is not
available; the infinity/empty properties returned after collection are not
the completed unit's execution limits. No retry was performed.

- Input summary SHA-256:
  `235db2010609e26b6e1e91358a64f16c34c88b5b69df20785f817df99c791035`.
- Checker SHA-256:
  `a34cefc787b4df371ac83a2665400dc16c41e55f6838f54826d4386504045b6b`.
- [Result JSON](result-001.json) SHA-256:
  `1bf92401465fba5a847c08f477a5eb841040c1003b0c68a4fc46e26bf5f2ba05`.

## Limits

This is independent **saved-scalar arithmetic and present-artifact
corroboration**, not independent tensor evidence for later updates. The
original batch receipt did not individually hash the trajectory JSONs;
current hashes and metadata consistency cannot retroactively prove immutable
history since acquisition. Saved scalar cosines are checked for validity and
aggregation, not reconstructed from unsaved vector orientations. No claim
follows about later actual Adam displacement, inter-branch span angles,
curvature, feature semantics or mechanism causation. These are exploratory
post-outcome diagnostics, not new independent training evidence.
