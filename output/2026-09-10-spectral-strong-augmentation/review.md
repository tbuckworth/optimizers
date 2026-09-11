# Independent strong-regime source review

Codex — Spectral Optimizer Investigation · 10 September 2026

**Final status: PASS — no unresolved material source defect found.** This is independent source review, not acquisition, a
scientific audit or a new approval gate. No real IDX, model/checkpoint, GPU,
gradient or logit archive was opened. One tiny fabricated NumPy calculation
was used to demonstrate a deterministic checkpoint-selection edge case.

## Findings reported before source freeze

1. **Required schema correction:** the producer's `choose_validation()` emits
   `selected_example_count`, while the reviewed auditor's exact choice-key set
   omitted it. Authentic producer choices would fail. The audit must retain
   and independently check exposure: 64×step through step100, otherwise
   completed passes×50,000 at the scheduled epoch checkpoints. Producer
   fixtures already assert this field.
2. **Required exact-selection convention:** producer choices use shifted-exp
   float64 CE, while the reviewed audit selected using a different `logaddexp`
   calculation. Both are accurate within the metric tolerance but can disagree
   on an exact minimum/tie. With fabricated 2×10 uniform logits, offset0 and
   offset2 give canonical CE 2.302585092994046 for both; alternate CE gives
   2.3025850929940455 and 2.302585092994045. The alternate rule incorrectly
   prefers the later checkpoint. Reconstruct the frozen canonical arithmetic
   for selection, preserve the independent alternate metric check, and include
   this exact-tie fixture. Do not widen the selection rule after results.
3. **Inexpensive audit-completeness fixes:** explicitly compare the producer's
   `reporting_augmentation_interaction` and saved `true_class_counts`. The
   reviewed auditor independently generates the interaction correctly, but
   did not yet validate those producer fields. These are receipt/arithmetic
   checks, not additional scientific arms.

## Reviewed portions with no material defect found

- Exact 12-arm rotated roster; independent global split, replacement and
  occurrence/shift streams; no post-hoc balancing or redraw. Replacement may
  preserve truth, so selected and actually wrong masks remain distinct.
- Every 50,000-example epoch is traversed once, including the 16-example tail.
  Seventy-two passes give 3.6 million exposures and 56,304 updates per arm,
  explicitly distinct from historical 60×60,000 data roles/repetition.
- The new `(dy,dx)` convention reverses columns only when calling the accepted
  `(dx,dy)` translation. Padding occurs in raw [0,1] pixels before FP32
  standardization; evaluation is original and unaugmented.
- Ordered two-hidden-layer FP32 model with 235,146 parameters, unchanged
  canonical stable hard rank200 settings, one backward/filter observation/Adam
  step per batch, explicit moment/clock/finite checks. The new snapshot schema
  correctly records two hidden layers and uses only shape-agnostic inherited
  cloning, tracker and RNG helpers, not I9's one-hidden-layer restore.
- Exact initial prediction/model pairing; raw/native model/Adam and prediction
  pairing at step100 within augmentation modes. Reporting forward outputs are
  saved but no reporting metric is calculated during acquisition or selection.
- All 12 choices are written/read back and every candidate validation CE is
  re-derived before any reporting metric calculation. Failed verification
  prevents that phase. Saved reporting data are not cryptographically blinded,
  as the protocol already states.
- The new writer/guard owns the 8 GiB/three-hour envelope without mutating old
  1 GiB/900-second globals. Exclusive attempts and outputs, committed source
  pins, original data hashes, mounted-volume/free-space checks, accepted GPU
  clients, fixed cgroup caps and failure preservation are explicit. The byte
  estimate conservatively assigns a rank200-size upper bound even to raw
  snapshots. Large snapshots are stream-hashed by the audit, not retained or
  unpickled; bounded per-checkpoint NPZs avoid an all-history memory load.
- The expected 1.5–3-hour runtime remains uncertain. Nothing in this review
  promises completion within the fixed cap or authorizes an automatic retry,
  shortened roster or hidden resource expansion.

## Final correction and fixture review

All findings above were corrected before any scientific attempt. The auditor
now admits and independently verifies `selected_example_count`, including zero,
warmup and full-epoch exposure accounting. It reconstructs and compares true
class counts in all three data roles and all four reporting-window augmentation
interactions. Its canonical selector independently implements the protocol's
shifted-exp and ordered `math.fsum` arithmetic; separate `logaddexp` metric
checks remain independent. The uniform-offset exact tie chooses the earlier
step, while a real small CE improvement chooses the later step. No tolerance
or tie rule was relaxed after outcomes.

The full protocol, implementation check, four modules and all four fixture
files have now been read. Additional fixture coverage includes all 25 integer
translations, standardized black padding, selected-versus-actually-wrong
labels, exact random-stream reconstruction, tail batches, literal two-hidden-
layer initialization, canonical native updates through the warmup boundary and
basis repair, truthful independent snapshots, Adam/filter counters, selection
before reporting, immutable receipts, artifact caps and fabricated archive
tampering. The counter-tamper fixture rebinds its pre-report receipt so it
reaches the intended counter assertion rather than failing earlier on receipt
consistency.

Independent reviewer execution: **44 fabricated CPU tests passed in 2.312 s**:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python3 -m unittest discover -s tests -p 'test_spectral_strong_augmentation*.py' -v
```

These tiny fixtures are not a timing benchmark or evidence of scientific
outcomes. No real scientific input, acquisition or saved-result audit was run.
Main owns source freeze, live resource admission and any once-only launch.

## Reviewed source hashes

Paths below are repository-relative; SHA-256 values bind this final review.

| File | SHA-256 |
| --- | --- |
| `experiments/spectral_strong_augmentation.py` | `62e1711793d3536b22e97fd71d236de9ddcd17eb3a305f8a85c3ae8e50cb5bd6` |
| `experiments/spectral_strong_augmentation_data.py` | `930de1324da55e209c65e832be37bdd0c684d11f87f651f411b09bb6e144d8d6` |
| `experiments/spectral_strong_augmentation_core.py` | `001298cea97ed341e76ffe57586f8c8deb45547f07bae01532a1a3cb827a39cf` |
| `experiments/spectral_strong_augmentation_audit.py` | `9895b12ae325f9b155d751b4aac5419dccb2c360d5cae27f221c1dea9112e86f` |
| `tests/test_spectral_strong_augmentation.py` | `1a62c42bd63b776c5771a250a0d457aa07ed1300a644a45cb43a4e3a82282745` |
| `tests/test_spectral_strong_augmentation_data.py` | `6dfab6f1d1a33482e2f8251e4167348e65c7d6be4ba5370f334010d651e5bf25` |
| `tests/test_spectral_strong_augmentation_core.py` | `1ca067f4ed740d44a3c1a775673f04926ca3febbb065b34aee93eddf816d80cf` |
| `tests/test_spectral_strong_augmentation_audit.py` | `d6f13369b3f3bd8422f7c82dfe4c36fe7047c802d7a6af18441969dfcdbd66fb` |
| `output/2026-09-10-spectral-strong-augmentation/protocol.md` | `a1b41e9f5da29deff098fa363d84e8e42c0fcbc3a023b9573858c930fc8206ce` |
| `output/2026-09-10-spectral-strong-augmentation/implementation-check.md` | `3c6fd1adb898fb90939f3b28bb3629fcc9477eea9e7c5810d04977d92daf73d4` |
