# Independent prospective summary-implementation audit — iteration 005

**PASS after the corrections below; no unresolved material blocker in the
reviewed summary implementation.** This is not replay, pilot or confirmation
approval. Reviewed the complete protocol, schema, analysis intent/specification,
summarizer and tests. The code-review skill informed concrete counterexamples
and source-location checks. The parent made all implementation corrections;
this reviewer changed only this audit note. No MNIST data, replay, learning
experiment, pilot, full run, commit or subagent was used.

## Material findings resolved before execution

1. **Scientific validity was not enforced.** An initial synthetic record could
   report a finite primary with observer rank 0, reference rank 0 or zero E32;
   a finite projector distance could coexist with an invalid reference
   projector. `summarize_results.py:216` now checks rank/energy validity and
   rank/gap validity separately, including contradictory nulls. Boundary ties
   correctly leave a valid primary energy fraction available. Primary values
   are also checked against the stored captured-energy numerator and E32.
2. **Required evidence and arithmetic consistency were incomplete.** The
   original fixture lacked covariance-energy terms but passed. A zero-input,
   nonzero-output probe also passed and inflated its weighted ratio. Required
   fields and common reference energies are now checked at
   `summarize_results.py:213`; common probe input energies and the zero-input
   rule are enforced at line 248. Represented-energy ratios, probe ratios,
   clean-minus-residual differences and self-inclusion differences must agree
   with their constituent values and null masks.
3. **Source binding could be bypassed by an empty map.**
   `summarize_results.py:44` now requires the exact 19 source paths and valid
   SHA256 strings. The directory reader checks every bound current file's bytes,
   preserves the execution manifest and hashes all input JSON files.
4. **Completion was checked after reading outcomes.** The reader now invokes
   the execution-only gate immediately after `execution.json`, before any
   snapshot/replay contents (`summarize_results.py:327`). Tests demonstrate
   rejection of pilot/incomplete execution after reading that file alone.
5. **Warmup/anchor shape checks were too weak.** One hundred empty warmup
   entries and a final anchor labeled step 0 originally passed. The validator
   now checks ordered warmup steps, SHA formatting and consistency with the new
   parameter/raw streams, equal raw/applied hashes, fixed final/warmup anchor
   steps, and complete common-stream invariant counters
   (`summarize_results.py:173`).

## Aggregation verdict

The single primary remains width128-minus-width32 rank-32 span capture at
**step 2000**, with all three individual differences and complete-case
descriptive statistics. No available-case primary or alternate snapshot is
substituted. Missing final pairs null the primary group while retaining their
reasons and remaining individual values.

All four fixed snapshots and all three seeds are required structurally. A
four-snapshot mean requires four valid values; its across-seed group requires
three complete seed means. Pairing occurs within the same seed/snapshot,
preserving validity intersections and reasons. Fixed-step available-case
statistics are explicitly secondary. Reference, native-operator, matched-rank
span and unequal-capacity covariance metrics remain separately named.

Ordinary retention summaries average ratios. Energy-weighted secondary
summaries retain summed input/output energies and their own denominator/null
rules. The synthetic fixture distinguishes arithmetic `.25` from weighted `.30`;
undefined ratios are not inserted as zero. Exact metric sets, finite values,
same-key null reasons and complete ordered schedules are enforced. Duplicate
JSON keys are rejected, inputs are unmodified, and exclusive output creation
refuses overwrite (`summarize_results.py:346`).

## Checks actually run and limits

Independently ran all **15 final synthetic summary tests: PASS**, 1.547 seconds.
Also passed a real-producer integration check on CPU: 64 synthetic float32
innovation rows in 48 dimensions, weighted float64 reference construction,
actual `reference`, `add_reference_probes` and `observer_metrics` outputs, and
the real summarizer. All eight reference metrics and sixteen metrics per
observer were accepted. Its one synthetic state was copied into twelve
explicitly artificial metadata slots solely to check the interface; these are
not twelve measurements or MNIST evidence. No files were written by that check.

The summarizer validates metadata/scalar consistency and recorded gate status;
it does not independently rerun the eigensolve, verify every bulk tensor,
reproduce historical training or replace the launch/pilot/commit gates. Those
remain responsibilities of the replay implementation and separate audits.

## Reviewed SHA256 values

- `summarize_results.py`: `ba3f6c3947fe19e435b839d89cea673b76f7f61eb7fb49411f8122718d3a9c77`
- `test_summary.py`: `19e7c0714d906c6da94868d1bec1375f8d539ab10e1d46a0620aedabef187397`
- `protocol.md`: `eb8f49f4a5247e9c33a23a0d15a572f2c5cec08355f2b4d05ada50f435ce0035`
- `result-schema.md`: `eb2af92b07d879f397e3b6ec0fb6fd5df8c85b62ab51f9b0d4fcd36c1e70408c`
- `analysis-plan.md`: `569eff833192c26c00587e155925088bc07b9eb8004040a997221a672f3893e7`
- `analysis-intent.md`: `2ab9e1ebd5df584e284b8686f81d4433bf85ef7bfd2e12e1f9a1ffbee804baca`
- Producer used in synthetic integration, `reference_math.py`: `3256c40cfe110edb035ecf38a0f1625534a941b5266da7b8c2f99a95595e50ac`
