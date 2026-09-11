# Within-topic J-Lens result — scalar audit

Codex — Spectral Optimizer Investigation · 11 September 2026 BST

**PASS.** One standalone saved-JSON validation completed successfully, with
5,180 checks and no mismatch. All 144 persisted choices, truth labels,
individual scores, signed/absolute gaps and credits match independent
calculation exactly, including binary64 float identity. All arm/axis
summaries and paired differences match. No grading stage or reader was rerun.

## Verified results

| Direction | A: direction tokens | B: example tokens | C: example text | Always FIRST | Always SECOND |
|---|---:|---:|---:|---:|---:|
| PC1 | 7/12 | 6/12 | 8/12 | 6/12 | 6/12 |
| PC2 | 7/12 | 8/12 | 7/12 | 3/12 | 9/12 |
| PC3 | 11/12 | 9/12 | 8/12 | 8/12 | 4/12 |
| PC4 | 6/12 | 8/12 | 10/12 | 8/12 | 4/12 |
| All four | 31/48 | 31/48 | 33/48 | 25/48 | 23/48 |

A-minus-B credits are `(+1, −1, +2, −2)`, totaling zero.
A-minus-C credits are `(−1, 0, +3, −4)`, totaling −2.
B-minus-C credits are `(−2, +1, +1, −2)`, totaling −2.
There are no exact ties, missing cases or margin-based exclusions. The
constant baselines are individually fixed policies, not an oracle that may
select a different better position for each axis.

Item-level support preserves both the useful positive and the adverse case:

- **PC3:** A succeeds on every pair except P11, which all three arms miss.
  Against B, A alone succeeds on P04/P05; there are no B-only successes.
  Against C, A alone succeeds on P04/P05/P10; there are no C-only successes.
  P04 is whisked versus marinated (whisked has the higher score; absolute
  gap 0.26163586438218267). P05 is sifted versus tempered (sifted higher;
  gap 0.10422016749946739). The P10 gap is 0.02023621243319873.
  Thus its extra successes are identifiable same-topic comparisons, not just
  aggregate-count differences or exact ties. All arms also succeed on the
  much closer P07, gap 0.0005732817179548988, retained unchanged.
- **PC4:** C succeeds on all six A failures: P02/P03/P04/P05/P06/P10.
  A alone succeeds over C on P01/P09; both succeed on the other four pairs.
  Relative to B, A has two unique successes and four unique failures.
  A's six errors have gaps from 0.049305593891662 to 0.22682801274924988;
  they are not confined to the closest pair. This is an adverse result for
  these direct descriptions/readers, not evidence that the direction has
  no readable structure.

All four axes and all 48 target-level records, including the full original
prefixes, are retained in [the audit receipt](result-audit-checks.json).
PC1/PC2 must not disappear behind PC3: on PC2, always SECOND scores 9/12,
above every arm on this panel. No population advantage or significance claim
follows from these small descriptive counts.

## Scope and provenance

The [standalone audit source](audit_saved.py) uses Python scalar arithmetic
and `math.fsum`, not the grader's calculation functions. It verified the
exact 24-text roster, all 12 prescribed adjacent same-topic pairs, all
4-axis × 3-arm × 12-pair cells, reader allocation, common presentation swaps,
unaltered public prefixes/reference strings, strict response IDs, and all
manifest size/hash bindings. Every returned JSON matches its locked copy;
the three responses and lock match their exact committed blobs at
`f92d70d1dc6eefce7720b40c669701e8ded7df2f`. That commit is an ancestor of
the saved-grade commit `33279f22bb57f22e0c50d672f3706922ffeb8b06`.
Seal time is 23:02:07.934251 UTC and grade-attempt time is 23:02:39.086350 UTC
on 10 September. The earlier implementation review covers the grader's
mandatory committed-byte check before key access; this audit did not
re-execute that stage or establish chronology solely from file timestamps.

This is **independent arithmetic, not an independent-author grader audit**:
the reviewer authored the earlier grading core. Fresh-reader isolation and
absence of reader tool activity are attributed to main's transport record in
[results.md](results.md), not independently verified from these files.
The reviewed report's numerical claims and illustrated orderings agree with
the checked record. Its conclusions appropriately retain reused measurements,
global references, single judgments, and reader/arm confounding. Different
cross-/within-topic pairings and readers are not a causal topic-removal test.

The research-review guidance was used to keep claims tied to saved evidence.
No NPZ, model, activation acquisition, PCA, experiment, cloud call, new
judgment, or grader import occurred. The one validation call used CUDA hidden,
OMP/MKL/OpenBLAS threads set to one, and a 60-second timeout; it exited zero
on its first invocation. This audit is consumed, not a request to run again.

Key SHA256 pins:

- Grade: `186cb971b9a52b2d47bcf586a0a3e38be453c18bc2c67a66ffe77d0a66f1582f`.
- Old scalar key (SHA256 file pin, not a credential): `f1e4d2aa1ace39d9d482681a3e672f60c279348cf62884042a9ee93409a19a46`. <!-- gitleaks:allow -->

- Response lock: `d7227b18e4a0e079860e0415ebff833361c0240e2df6cddb10bcad6ad576e1ea`.
- Packet manifest: `8fb30df285f757005193e1af27b213049f046e64baad65c13edf99bbca129ab3`.
- Audit source: `3036e839098ff98d55c04fd107606174d3a513c2492768cb7960458e3653e5d2`.
- Audit receipt: `b82df1143a300e385fb626ed3e081f8ef3bbc14e86b9a800f826b631fa997d0f`.
