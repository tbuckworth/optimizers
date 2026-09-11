# Independent observer-pathway audit readiness

Codex — Spectral Optimizer Investigation, 10 September 2026.
**Source and fabricated-array preparation only. No scientific input, model,
gradient stream or logit archive was loaded by this implementation agent.**
Main owns final review/freeze and both once-only service launches.

## Implementation and fabricated validation

Checker (artifact not distributed in this public snapshot) SHA-256:
`a8e9dcc1e0b10a3569751893802b30135846c7a826c889d0ce4879910beed2cc`.
Fixtures (artifact not distributed in this public snapshot) SHA-256:
`e93a7829093de69902a6d210b0113efbb8c49bf4b46aa8a9fe8b1082d5b7ff33`.

All **13 fabricated CPU tests pass**, most recent unittest runtime **0.039 s**.
Fixtures use invented three-coordinate saved states and explicit NumPy Adam
recurrence arithmetic, not a model or optimizer object. They cover the two
50-gradient mean identity and zero/cancellation cases; exact common FP32 input;
distinct observer/Adam clocks; final saved mean update; parent immutability;
complete history/readout schemas; actual zero gradients versus `None`; native
action and numerical-span distinctions; signed useful-loss orientation, nulls,
control-deterioration contrasts; physical/artifact rosters; and distinct ancestor
versus new-file receipt binding.

The first fixture invocation had one fixture-format error: the snapshot hash
helper was passed a tuple containing a NumPy array. The traceback identified
this before step validation. Snapshot hashes and array hashes are now checked
separately; no production validator or scientific tolerance was relaxed.
Assumption-debugger guidance kept that correction specific to the actual cause.

The checker imports only pinned **independent** arithmetic helpers:

- Batch-composition checker:
  `51f22b18ce0f97d8d6b2501a4aaeac3d6fe93347189e148192668052aa59eb3e`.
- Its selectivity-checker dependency:
  `a992842846c767c0af44cf41a19fb3042b4aebc951089d78469e26b0f3465527`.

Neither prior audit/bundle/main nor any scientific producer is invoked.
The new checker owns its separate-clock envelope validator; it never changes
recorded clocks to satisfy an old snapshot validator.

## Fixed acceptance scope

Admission requires main's expected completion hash, manifest hash and exact
nine-source pin map, including the new protocol and both reviewed design
documents. All **122 new artifact receipts**, plus the separately hash-bound
completion, must exist exactly once as contained regular files, with matching
sizes and hashes. Extra, missing, duplicate, escaping or symlinked artifacts
are rejected. Completion requires six cases, 600 stream gradients, six oracle
gradients, 21 physical readouts/prediction pairs, 600 observer-only updates and
12 self-inclusion updates. The recorded acquisition time must be at most
**480 seconds**, not merely inside its 600-second hard service limit.

The accepted ancestor completion and PASS audit are checked against the
design's fixed hashes and identical receipt roster. Only the **22 scoped
ancestor inputs** are used: manifest and seven files per seed. Their pinned
source/environment, full step-100 parent, plans, schedules, probe memberships,
input hashes and before-prediction reuse are bound. The checker does not rerun
the previous audit. Pinned training IDX bytes reconstruct the normalized probe
inputs and verify true labels, disjoint IDs and class counts.

All 12 saved `[50,50890]` streams must match the exact first-block memberships,
including duplicate occurrences, from the accepted schedules. Every recorded
gradient preservation hash must equal the same parent envelope at Adam 100 /
observer 100. FP64 stream means and the fixed equality bound are reconstructed;
the symmetric common FP32 vector is checked exactly. The producer's pre-freeze
canonical CPU NumPy FP64 mean/FP32 cast convention removes reduction-order
ambiguity without changing the mathematical formula or admission tolerance.

Each saved observer-150 and observer-151 envelope keeps model/Adam identical
to the ancestor, with cleared `.grad`. Its **separate** observer clock and
fixed tracker configuration are checked; stable singular values are FP64.
Native pre/post-inclusion actions, numerical-span geometry and action contrasts
are reconstructed from saved arrays. The single saved mean150→mean151 identity
is checked algebraically. **No observer stream, eigenspace update or training
step is replayed.** A diagnostic SVD of the saved basis only defines its
numerical span; it does not regenerate the observer.

All 21 physical readouts independently bind inherited/before/after envelopes,
parameter order, explicit delivered `.grad`, unchanged observer/RNG/modes and
exact common parent parameters/Adam state. Native observer remains 151 while
Adam moves 100→101. Raw/zero observers are absent. Adam moment/parameter
recurrences and actual decay/adaptive displacement are checked from recorded
tensors. Three physical zero references are verified once and their identical
receipts reused across label cells; their label-derived metrics remain distinct.
Oracle true-label gradients never become an action input or observer input.

The checker derives every before/after metric and all 56 signed improvements
from archived before and new after logits. CE uses before-minus-after;
accuracy uses after-minus-before. It recomputes all three seed values,
means/sample SD/SE/signs, eight absolute-improvement groups and 12 contrasts.
Clean wrong-label outcomes stay null. There is no effect-size/sign acceptance
gate, selected checkpoint, p-value, composite or partial certification.

## Frozen numerical tolerances

- Membership, hashes, common FP32 input, parent/state identity, counts and nulls
  are exact.
- Fixed-model block-mean admission is
  `||mean_I - mean_G|| <= 128 * eps32 * B + 1e-10`, with B the average norm
  across the 100 recorded batch gradients. Saved FP64 mean vectors additionally
  use coordinatewise `atol=1e-12, rtol=1e-10`.
- Final common-observation mean identity uses
  `|mean151 - (.99*mean150 + .01*g_star)| <=`
  `16*eps32*(.99*abs(mean150) + .01*abs(g_star)) + 1e-30`.
  Neither magnitude is clamped; zero/cancellation behavior is explicit.
- FP64 metrics, signed utilities and paired summaries use
  `atol=1e-10, rtol=1e-10`.
- Native FP32 geometry uses `atol=1e-7, rtol=5e-4`; raw/numerical-span
  quantities additionally receive tighter FP64 checks.
- Native action L2 discrepancy is at most `3e-5*||g_star|| + 1e-12`.
- Numerical span/displacement comparisons use `atol=1e-10, rtol=1e-8`.
  Rank uses the fixed FP64 threshold `eps64*max(shape)*largest_singular_value`.
- Adam coordinate recurrence bounds retain `16*eps32` times absolute
  recurrence-term sums, plus `1e-30`; the parameter bound uses
  `16*eps32*(abs(old_parameter)+abs(ideal_adaptive_change)) + 1e-30`.

All are fixed before new outcomes. A numerical failure is preserved and
reported, never converted into a favorable/negative scientific verdict or
repaired by outcome-driven tolerance relaxation.

## CLI, resource guard and renderer contract

```text
/usr/bin/python3 scripts/audit_spectral_observer_pathway.py
  --execute --acquisition-dir ABSOLUTE_NEW_ACQUISITION
  --output-dir SAME_PARENT/audit-001
  --expected-complete-sha256 MAIN_HASH
  --expected-manifest-sha256 MAIN_HASH
  --expected-sources-json MAIN_FROZEN_MAP
```

The unit must be `spectral-observer-pathway-audit-001.service`, Type=exec,
Restart=no, KillMode=control-group, one CPU quota, 4 GiB memory, no swap and
five-minute hard limit. All four math-thread environment variables must be
`1`, and **CUDA_VISIBLE_DEVICES must be explicitly empty**. No GPU API is
called. A 250-second cooperative clock includes main-entry admission. Output
is an exclusive sibling `audit-001`, capped at 100 MiB. Inputs, sources,
expected-source-map file and this note are rehashed after checking.

The sole output is `result.json`, schema
`spectral_observer_pathway_independent_audit_v1`, containing status/errors/check
count, tolerances/discrepancies, expected hashes, new and parent input receipts,
resource receipts, `independent_summary`, `checked_cases`, `checked_histories`
and `checked_readouts`. Only full PASS sets:

```text
cases_checked=6                 observer_histories_checked=12
stream_gradients_checked=600    oracle_gradients_checked=6
physical_readouts_checked=21    logical_readouts_checked=24
prediction_pairs_checked=21
```

`independent_summary` exactly follows the producer's fixed summary schema.
`checked_cases` retains full before/after outcomes, improvements and readout
receipts. `checked_histories` retains seed/cell/schedule, pre/post-inclusion
geometry/action errors, final-mean recurrence errors and self-inclusion action
comparison. Shared zero evidence is not counted as a fourth physical step
per case. Failed acceptance counters are null.

Limits: forward/backward correctness and the 50-step observer-generation path
rely on pinned, reviewed producer code and its preservation assertions. The
saved-data audit checks the resulting arithmetic and provenance, not those
scientific computations again. PASS does not establish endpoint rescue,
semantic selectivity, pure covariance mediation or fresh-parent replication.
