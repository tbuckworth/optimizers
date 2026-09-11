# Independent source review: saved-vector signal accounting

Codex — Spectral Optimizer Investigation, 10 September 2026.

**PASS — no material source or design issue found.** This is source review
and fabricated-test evidence, not execution or independent confirmation of
the proposed scientific products. Main retains source freeze, admission and
the once-only launch.

Reviewed the full [protocol](protocol.md), the prior
[decision](../2026-09-10-spectral-observer-pathway/next-decision.md),
design (artifact not distributed in this public snapshot) and
[design review](../2026-09-10-spectral-observer-pathway/next-design-review.md),
all 545 analyzer lines and all 249 fixture lines.

Reviewed bytes:

- `scripts/analyze_spectral_observer_signal.py`: SHA256
  `7bad0a6bdd73cd7c978c47988dfc5b9705bfa9a0cd2a7e0074617138de80eba8`.
- `tests/test_spectral_observer_signal.py`: SHA256
  `ad9dc0e94cdf4f9a636c9a18ea76369b3c50a17feb76e7ef6bf41e54896a1c16`.

## Material checks

- **Signs and numerical arithmetic:** `B`, `F`, `K` and grouped-minus-
  interleaved delivery contrasts use the specified orientation. Conversion to
  FP64 precedes subtraction. Ordinary dot products are checked against a
  separate `math.fsum` reduction. The single-product bound is
  `128*eps64*S + 1e-12`; paired identities use
  `128*eps64*(S_direct+S_left+S_right) + 3e-12`. Difference signs use the latter
  cancellation-aware threshold. Raw signs remain available alongside
  unresolved counts; zero-norm cosines are null. These are operational checks,
  not universal summation-error theorems. See analyzer lines 98–155.
- **Joins and controls:** actual `J` comes from the independently recomputed
  top-level `checked_readouts`, keyed by exact physical ID. The duplicated
  nested producer values are not silently substituted or required to match
  in their last rounding bit. Held-out `U` uses `rare_ce` or
  `majority_macro_ce` from checked case improvements. Raw/zero identities,
  nonzero inherited zero-step utility, decay/adaptive scalars and all common
  costs remain visible. See lines 165–220.
- **Complete roster:** three fixed reused seeds, both label cells and both
  probes; six cases, 48 logical action/probe joins, all 72 pair contrasts,
  12 primary grouping contrasts and six label-input contrasts. Three-seed
  values precede means, sample SD/SE and signed counts. Shared zero remains
  three physical references, not six extra replicates. See lines 224–254,
  351–359 and 379–421.
- **Scope and provenance:** only the three oracle, six common-input and
  12 observer archives can be loaded, once each and in the fixed order.
  Regular direct-child paths, size/hash receipts, accepted audit/completion/
  manifest pins, common-parent identity and exact common-input hashes are
  checked. Extraction is limited to oracle mean gradients, `g_star` and
  O151 `post_action`; no incidental image, model or observer-state analysis
  is introduced. Sources and selected inputs are rechecked before success.
  See lines 257–375 and 379–400.
- **Execution boundaries:** import/help do not admit scientific data; explicit
  execution, committed source bytes, the dedicated non-restarting unit and
  an exclusive mounted output parent are required. CUDA is hidden. One CPU,
  4 GiB/no swap, 180-second hard and 150-second cooperative limits, 100 MiB
  output plus disk/footer reserves are enforced. The clock begins before
  NumPy import. Numerical failures preserve partial receipts; no effect-size
  gate or automatic retry is added. See lines 18–19 and 424–540.

## Fabricated verification and limits

Independently ran only:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/timeout 30s \
  /usr/bin/python3 -m unittest discover -s tests \
  -p test_spectral_observer_signal.py -v
```

All **13 final tests passed in 0.127 seconds**. They exercise signs, zero/nulls,
FP64-before-subtraction, severe cancellation, numerical failure evidence,
independent-versus-producer scalar joins, full summary roster, wrong hashes/
sizes, duplicates/path escapes/symlinks, exclusive capped output and inert
entry points. The loader fixture uses a stub Torch module and deliberately
opaque incidental members: it checks schema selection and loader arguments,
not real archive deserialization.

Before freeze, main identified that the original `2^24−1` subtraction fixture
was also exactly representable in FP32. The producer strengthened it to
`2^25−1`, which distinguishes FP64-before-subtraction from FP32-first
subtraction. I read that change and reran the final fabricated suite above.
The numerical implementation and scientific tolerances did not change.

No scientific tensor, model or prediction array was opened; no scientific
product, optimizer/observer replay, acquisition, old audit or new study ran.
The accepted JSON was inspected only for scalar join/schema distinctions.
Actual resource admission and future saved-vector results remain untested by
this review. The review-changes guidance informed the source/error-path and
fixture checks; it added no scientific scope or workflow approval gate.
