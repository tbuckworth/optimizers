# Independent I20 analysis review

8 September 2026. **Source and synthetic-fixture review only.** I reviewed
`audit_response.py` and `test_audit_response.py` after the completed I20
producer freeze. I did not open an actual I20 or I19 NPZ, read I20 outcomes,
or invoke the real audit, producer, native observer or RNG.

## Verdict

The settled auditor is suitable for the one bounded independent saved-output
audit. Its admission, numerical recurrences and registered aggregation match
the I20 protocol. The remaining scientific acceptance decision must depend on
the actual audit artifact, not this pre-run source review.

Reviewed source SHA-256 at sign-off:

- `audit_response.py`:
  `f0316f160df6a30c513a05e28875a876ec8077e1e6fb4a46b7aa3267a5332ef7`
- `test_audit_response.py`:
  `e04d603321d6a1e0608ce91e52efa97ca6816f16295eea4fc995c0cce83bd27f`

These are review identifiers, not a substitute for the auditor's required
separate `--analysis-commit` freeze.

## Input admission and output integrity

- The auditor requires the canonical mounted I20 root, an absent exclusive
  `iteration-020/analysis-001` output, exact attempt/completion schemas, the
  complete ordered 192-stream roster and 768,000 observations. It validates
  every recorded file path, size and SHA-256, rejects unexpected physical
  files or directories, and recomputes storage accounting against the 2-GiB
  array and 3-GiB root caps.
- The five fixed I19 pins, ordered parent inventory and all 11 parent source
  bindings are independently revalidated. Acquisition sources use the frozen
  acquisition commit; the two auditor sources use the distinct required
  analysis commit. Both source manifests and the parent closure are checked
  again after numerical aggregation.
- Parent NPZ admission compares the complete recorded archive membership and
  order, including all 56 parent members, while numerically loading only the
  eight registered fields. I20 NPZ admission requires exactly all 10 registered
  members. Both paths reject compressed members, excessive expanded sizes,
  wrong shapes/dtypes, nonfinite numeric arrays and pickle/object payloads.
- The audit rehashes each I20 array, its stream metadata and the corresponding
  parent array after use. Parent metadata is covered by the final full parent
  revalidation; attempt and completion are also rehashed at the end. This
  closes the metadata time-of-check/time-of-use gap found during review.
- Nothing is written until every admission, numerical, aggregation and final
  provenance check passes within the 280-second cooperative budget. The output
  directory and both JSON files use exclusive creation. The external audit
  service must still provide the registered 300-second hard bound and
  CPU-only, one-thread environment.

## Numerical audit

The auditor does not import the producer core, Torch or an RNG. Its independent
moment reconstruction starts both `.99` and `.999` accumulators and masses at
zero, uses saved `z_t=g_t-mu_t`, normalizes by saved weight mass, recomputes
ascending eigenpairs and the registered relative eigengap rule, and reconstructs
identity fallbacks. It copies native saved actions, reconstructs legacy-full
fallback, builds the two fixed oracle projectors, and checks direction masks,
masked alignment, projector symmetry/idempotence and action-change norms.

All response equations use the saved predecessor rather than recursively
replacing it with an auditor-generated state. The reshaping preserves the
frozen estimator/rho/response order. CP, `rec99` and `rec9` are checked for
both rho values at every saved step; all first outputs must equal `g_1` exactly.
Five accepted parent CP/oracle outputs and all six `.9`-EMA identities are
recomputed as cross-generation closures.

The MSE is the mean over time of the squared two-coordinate tracking error
against saved parent `s`, with inclusive registered windows converted correctly
to zero-based slices. Parent EMA `.9` and common-Kalman MSEs are recomputed from
parent arrays and checked against the accepted I19 per-seed rows before use as
comparators. The complete 472-row parent mean roster and its seed aggregation
arithmetic are also retained and checked.

## Aggregation and masks

The exact output rosters are enforced:

- 27,648 per-seed policy/window MSE rows;
- 864 equal-seed MSE summaries;
- 4,608 per-seed direction diagnostics and 144 equal-seed direction summaries;
- 90 contrasts per process/rotation/window, hence 2,160 total;
- 270 identity-rotation, whole-window primary contrasts.

The 90 contrasts are exactly the registered 48 non-oracle comparisons to the
two parent controls, 18 matched estimator contrasts, 12 `rec99`-versus-CP and
12 `rec9`-versus-`rec99` contrasts. Every effect is
`comparator MSE - target MSE`, so a positive value favors the named target.

Means give every one of seeds19000--19031 equal weight. Standard errors use
the across-seed sample standard deviation (`ddof=1`) divided by `sqrt(32)`,
and positive/negative/exact-zero signs are counted over those same 32 values.
No observations or rotations are treated as replications. Direction alignment
is averaged only over present steps within each seed, while present/absent
counts remain explicit. If any seed lacks a finite conditional direction
summary, the across-seed summary is marked unavailable rather than silently
averaging surviving seeds.

## Defects resolved during review

Earlier drafts conflated acquisition and analysis source commits and compared
the parent archive against only the eight loaded members. The settled code has
separate commit arguments and validates the full parent archive roster. During
this review I found that I20 stream metadata was checked only at initial
admission, unlike the arrays; the final code now rehashes it after use as well.
No unresolved numerical or roster defect remains.

All 10 synthetic auditor tests pass. They cover both rotations and all process
cells, tied/absent directions, every saved array and all 36 policy mutations,
parent closures, archive membership/order/storage/pickle rejection, sign
orientation, exact aggregate counts and no-survivor averaging. `git diff
--check` is clean. These fixtures validate code paths without inspecting the
completed scientific arrays.
