# Independent saved-data audit plan

Prospective only. No checker is coded or run before the producer schema is
available and frozen. Scope is the new 36-trajectory selectivity batch and its
36 native diagnostic events (3 seeds × 4 cells × 3 updates), not any historical
experiment. No model construction, forward pass, backward pass, optimizer-step
execution, covariance-stream replay or fresh-seed replication is permitted.

## Admission and input contract

Main supplies expected SHA256 values for the acquisition completion, manifest,
protocol and producer sources. Require complete status, exactly seeds
202609111–113, four declared cells, three policies, 2,000 total steps,
three shared warmups, all evaluation steps and all native diagnostic events.
Verify relative contained regular-file receipts, size/hash, no duplicate or
unexpected scientific IDs, finite typed arrays and full artifact roster.
Preserve failed/partial receipts; do not aggregate an incomplete roster.

Required inputs, with producer field names to be mapped once available:

- Per-seed canonical training/held-out IDs and true labels; initialization seed,
  named PCG64/SeedSequence contract; warmup/continuation batch indices; Diffuse
  selection mask/replacement digits; poison and Shared/Sham patch masks;
  per-cell actual targets, contingency counts and constructed-input hashes.
- Hash-pinned original training IDX bytes for verifying ID-to-image/label
  correspondence and intended patch inputs. Never read the official test set.
  The producer must bind actually constructed cell inputs, not only plan names;
  freeze dtype/normalization/hash conventions with synthetic fixtures.
- Float32 train, clean-held-out and patched-held-out logits at all 21 evaluation
  steps; sufficient statistics, curve rows and aggregate tables. Shared initial
  and warmup held-out logits may be stored once with explicit references.
- Full common warmup checkpoints and fork-identity records; final checkpoints
  with parameter ordering and policy/cell identities. Native diagnostics need
  pre/post parameters, Adam states/counters/configuration, raw/applied gradients,
  post-observation V/S/mean, and pre/post diagnostic RNG/state hashes.
- Per diagnostic: ordered probe IDs, exact inputs or input hashes, true/wrong
  targets, per-example pre-update gradients, pre/post logits and reported
  summaries. Retain both wrong and true gradients for same-input residuals.

Read one array group/event at a time. CPU tensor deserialization, if required,
must use restricted loading and never instantiate a model. The checker must not
import producer metric, aggregation or training helpers.

## Independent recomputations

1. **Plans and data.** Reconstruct label lookup from pinned IDX and reproduce
   seeded plan generation independently after the exact draw order is frozen.
   Verify 550 training examples for every majority digit, 50 for digit 8,
   500 disjoint held-out examples per digit, no within-seed train/held-out
   overlap, valid paired batches and no rare/cue warmup training input.
   Check Diffuse replacements exclude 8 but may equal the true label; derive
   selected and actually changed masks/counts separately. Check all poison IDs
   are majority nonzero, exactly 500, with target 0; Shared/Sham targets match
   byte-for-byte and rare images remain correct/unpatched.

2. **Sham allocation.** Independently calculate K_d, N_(d,z), floor quotas,
   fractional remainders and ties by ascending poison flag. Verify exact final
   counts, within-stratum sampling without replacement, Shared/Sham true-digit
   patch-count equality, and no patched true 0/8. Recompute full patch × poison ×
   true-label × modified-label contingency, overlap and deviations from ideal
   quotas. Check the fixed balanced construction, not a false exact-independence
   condition or an outcome-dependent correlation threshold.

3. **Prediction metrics.** From logits and exact target arrays, independently
   calculate argmax counts and float64 stable log-sum-exp CE sums. Recompute
   each digit, rare 8, majority macro and class-balanced total for patched and
   unpatched held-out inputs; distinguish macro from micro train averages under
   imbalance. Recompute training true/assigned metrics and actually-wrong fit,
   returning null with denominator 0. All denominator and selection masks come
   from labels/IDs, never model predictions.

4. **Cue and paired contrasts.** Recompute patched/unpatched target-0 rates
   for primary y∉{0,8}, rare y=8 and all y≠0 populations. Form per-seed paired
   patch excess E and the declared native/raw difference-in-differences D.
   Separately calculate the norm-control and plain-ASR interactions. Recompute
   all declared native-minus-raw/native-minus-control outcomes, warmup changes,
   seed values, means, sample SD/SE and signs. Reject missing/adverse rows being
   dropped; do not add thresholds, selectors, p-values or an equivalence claim.

5. **Probe arithmetic.** Verify ordered groups: true-label, unpatched digit 3
   and 8; actual wrong targets and corresponding true targets at the same cell
   input; absent Clean wrong groups represented as null. Check update timing
   and group counts. For saved gradients g_i compute mean g, mean ||g_i||² and
   coherence ||mean g||² / mean ||g_i||². For A=VVᵀ, recompute separately
   `sum ||A g_i||² / sum ||g_i||²` and
   `||A mean g||² / ||mean g||²`; use explicit undefined rules for zero
   denominators. Repeat declared retention measures for b_i=g_wrong−g_true.
   Native-action retention is not silently replaced by ideal span retention.

6. **Native action and Adam arithmetic.** At the 36 saved diagnostic anchors,
   verify applied gradients against the native action with recorded fallback.
   Independently check pre/post moment recurrences and counter increments from
   saved arrays. Compute actual delta=theta_after−theta_before, decay movement
   −lr×wd×theta_before, the ideal adaptive movement from post moments and bias
   corrections, and the recorded finite-precision residual. Use tolerances
   frozen on tiny synthetic fixtures before reading outcomes, not bitwise
   equality across different CPU/CUDA kernels. Construct or verify the numerical
   span of saved V under a frozen rank threshold and recompute off-span fractions.
   Recompute −mean_probe_gradient·delta and actual pre/post probe CE changes
   separately; do not require their signs to agree.

7. **Reuse and preservation.** Check every fork's restored model/Adam/observer/
   RNG identity against its shared warmup. Confirm initial/warmup held-out
   references are exact, and never substitute unpatched train logits for a
   Shared/Sham cell's modified-input logits. Check recorded probe-neutrality
   hashes before/after gradient collection; post-Adam state may legitimately
   differ. Verify immutable source/input receipts again at audit completion.

## Scope limits and execution bounds

This audit corroborates arithmetic on recorded outputs and state transitions.
Without inference/backpropagation it cannot independently establish that a
saved logit is the model's output or that a saved per-example vector is its true
derivative. Those links require reviewed producer code, synthetic fixtures,
state-preservation checks and hash-bound provenance. Native-only diagnostic
tensors also do not reconstruct every other policy's training step; norm-control
histories support recorded-norm checks, not an invented full-trajectory proof.

Prospective audit limits: one CPU, 4 GiB RAM, five-minute hard/250-second
cooperative deadline, no GPU, output at most 100 MiB. Save one result JSON with
explicit checks/failures/max discrepancies, independent summaries and complete
input/output receipts. No scientific significance test and no automatic retry.
The checker and small synthetic positive/negative fixtures are written only
after producer field names, shapes, numerical conventions and byte inventory
are available; main reviews and freezes them before a single audit execution.
