# Native selectivity boundary: rare correct learning and a shared wrong cue

Prospective executable protocol, 9 September 2026. **Not yet acquired.**
Selected within the user's approved safety-first plan, after the independently
audited clustering pilot. No old experiment or measurement is to be repeated.
The design rationale (artifact not distributed in this public snapshot)
and [mathematical note](../2026-09-09-spectral-next-mechanism/batch-composition-theory.md)
are context; this protocol resolves their implementation ambiguities.

## Question and fixed roster

Does the native filter preserve useful common classification and acquire a
new rare correct class, while treating diffuse wrong labels differently from
a shared misleading cue? Test behavior, not just covariance retention.
There is no assumption that a synthetic cue is human-values misalignment.

Exactly seeds **202609111, 202609112, 202609113**; four cells; three policies:
36 trajectories to absolute step 2,000. No tuning, adaptive stopping, selected
checkpoint, replacement seed or success-dependent extension.

- Model: 784–64–10 ReLU MLP, all 50,890 weights/bias parameters, pixels/255.
- AdamW: learning rate .001, weight decay .01, betas (.9,.999), epsilon 1e-8,
  foreach=False, fused=False; no schedule, clipping or augmentation.
- Native observer: current stable centered global rank 32, beta .99,
  raw covariance, hard weighting, repair every 100 steps,
  relative_eig_tol=1e-8, absolute_eig_floor=0, existing startup unchanged.
- Policies: raw AdamW; native hard32; native-norm/raw-direction control.
  The latter observes its own raw gradient once and delivers
  `||native_action(g)|| g / ||g||`, where native_action is V(Vᵀg) when a basis
  exists and the canonical identity fallback otherwise. Norms/scaling use
  float64 and delivery returns to the
  gradient dtype. Zero raw norm requires zero target; zero target yields an
  explicit zero gradient and an ordinary AdamW step, never missing gradients.
  This matches a functional norm rule, not later numerical sequences or Adam
  displacement. Require finite values and relative post-cast norm error at
  most 10 times gradient-dtype epsilon for nonzero targets; do not hide an
  unsatisfied match by clipping or retuning. Observe raw gradients, not probe
  or already filtered gradients.

## Data, pairing and shared starting state

Use only the accepted, checksum-pinned MNIST **training** IDX files. The official
test set is out of scope. Per seed, independently shuffle IDs within each
class, take 550 training examples per digit except 8, and 50 digit-8 examples;
reserve the next 500 disjoint IDs in each class for clean held-out evaluation.
Save all IDs and labels. Digit 8 is fixed in advance, not chosen for an outcome.

Use named NumPy PCG64 streams derived from SeedSequence([seed, stream_id]):
0 split, 1 warmup batches, 2 continuation batches, 3 diffuse corruption,
4 poison IDs, 5 sham patch allocation. Initialization uses the seed itself.
Record the exact random generator algorithm/version. Evaluation/probes must
not consume a training RNG stream. Sampling within a training pool is uniform
with replacement; persist every index batch.

One 100-step clean raw-gradient warmup per seed uses only the 4,950 majority
training images. An observer records those gradients without changing delivery.
Neither digit 8 nor a patch appears. Save the full model/Adam/observer/RNG state
once and restore it for every policy/cell, checking identity. All continuations
use the same 1,900 precomputed batches of 64 IDs from the 5,000-example pool.
There are 300 physical warmup updates and 68,400 continuation updates.

## Four cells

1. **Clean:** true labels and unmodified images.
2. **Diffuse:** each majority example is independently selected with probability
   .9 for a fixed uniform replacement among the nine majority labels (exclude
   8). Rare labels remain correct. Expected majority error is .8, not .81;
   realized changed-label counts are saved. No patch.
3. **Shared:** choose exactly 500 majority, true-nonzero examples without
   replacement and assign target 0. Put a white 3×3 upper-left patch on exactly
   those examples (pixels rows/columns 0:3 set to 1 after normalization).
4. **Sham:** reuse the **identical poison IDs and per-example modified targets**.
   Use the same eligible true digits as Shared (exclude 0 and 8). For each
   eligible true digit d, let K_d be its Shared patch/poison count. Patch
   exactly K_d of its 550 examples, balancing over poison status z: allocate
   floor(K_d N_(d,z)/550), then remaining slots by descending fractional
   remainder, ties by ascending poison flag. Sample without replacement within
   each stratum using the sham RNG. Thus total patch count and per-true-digit
   patch counts match Shared exactly, while patch/poison association is nearly
   removed **within each eligible true class**. This is conditional fixed-count
   balance, not unconditional or exact iid independence. Neither cell patches
   true digit 0 or 8 during training. Persist patch × poison × true-label ×
   modified-label counts, overlaps and deviations from ideal allocation.

Rare images are unpatched and correctly labeled in every training cell. Shared
versus Sham changes cue association while holding targets and patch prevalence
fixed; it does not hold exactly the same image pixels fixed. Diffuse is a
separate favorable-regime probe, **not a severity-matched poison comparison**.

## Primary useful-learning and cue outcomes

Save logits at 0, 100, 200, …, 2,000. Evaluate unpatched and identically patched
versions of the same 5,000 held-out images; labels always remain the true digits.
Reuse exact shared initial/warmup held-out predictions across branches rather
than recomputing them. Save training-image logits against both assigned and
true labels, with actually changed-target fit clearly separated.

Report at the fixed primary endpoint 2,000, with full curves and change from
the common step-100 warmup:

- Unpatched true-label CE/accuracy for digit 8, each other digit, majority macro
  average, and class-balanced total. Report training metrics too.
- Patched true-label CE/accuracy with the same group breakdown.
- Target-0 prediction rate on true-nonzero held-out images, both unpatched and
  patched, and separately on rare digit 8. This avoids counting true zero as
  an attack success. Report the literal denominators.
- Patch excess `E_(policy,cell) = P(pred=0 | patched, y∉{0,8})
  - P(pred=0 | unpatched, y∉{0,8})` on exactly matched images. The primary
  population is the eight nonzero majority digits, 500 images each; report
  digit 8 and the full nonzero population separately as secondary cue effects.
  Never condition evaluation membership on a policy's prediction/correctness.

Primary policy contrasts are native-minus-raw and native-minus-norm-control
for each cell's rare/common accuracy and CE. Primary cue interaction:

    D = (E_native,Shared - E_native,Sham)
        - (E_raw,Shared - E_raw,Sham).

Also report the norm-control interaction and the analogous raw patched-ASR
interaction as secondary. Positive D means a larger effect of introducing
cue–target association on patch-induced target bias; it does not identify a
particular covariance mode as causal. Preserve all seeds, means, sample SD/SE
and signs. There is no weighted composite victory score or equivalence claim.

Useful protection requires preserved classification, not only less wrong-label
fit. Rare learning means actual improvement from its inherited state. Low cue
susceptibility with stalled common/rare learning is not selective defense.
Report continuous raw digit-8 accuracy/CE changes and Shared-minus-Sham patch
effects as assay checks, including every seed; no outcome-dependent binary
admission threshold is used. If useful acquisition or a cue effect is not
established, qualify the corresponding comparison rather than claiming a
defense or retuning the task. Finish the fixed roster regardless of outcomes.
Different class difficulty and support remain entangled in this first task;
it is not a frequency-only causal test or a literal long-tail memorization assay.

## Fixed native-only mechanism diagnostics

At **updates 101, 500 and 2,000**, evaluate probes at the pre-update model, after
the single training-gradient observer update has fixed the delivered space,
and before AdamW changes parameters. Probe computation must not change model,
optimizer, observer, training .grad, or RNG. Use autograd.grad and verify
preservation. Re-evaluate the same probes after that one actual update; no
counterfactual optimizer replay is needed.

Per cell, use the first 32 selected training IDs of digit 3 and of digit 8,
both with **true labels and unpatched images**. Where wrong labels exist,
use the first 32 **actually changed** examples, with their actual cell input,
once with wrong and once with true labels. Clean has no wrong probe (null,
not zero). If fewer than 32 exist, preserve all and record the count; do not
replace the scientific data. Order is the saved training-ID order.

That order is grouped by true digit. The wrong-label probes are therefore a
fixed convenience subset, often concentrated in true digit 0 for Diffuse and
digit 1 for Shared/Sham, not a representative sample of all wrong examples.
Report their true-label counts; do not extrapolate their retention or local
utility to the whole wrong-example population. This clarification changes
neither the registered examples nor the training/evaluation design.

Save per-example gradients, pre/post logits and true/wrong targets. Report
`||mean gradient||² / mean ||gradient||²`, undefined at zero denominator,
individual and mean native-action retention, and wrong-minus-true gradient
residual retention. Diffuse probes mix target labels whereas Shared/Sham wrong
targets are all zero; their coherence values are descriptive, not a controlled
cross-cell coherence contrast. A numerical orthonormal basis of the saved V
may diagnose span retention; never substitute it into the native action.

Save pre/post parameters and Adam moments/counter, raw/applied training gradients,
the post-observation V/S/mean and step metadata. Record actual movement,
separate decay and adaptive movements, numerical-span off-component fraction,
and signed probe utility `-mean_probe_gradientᵀ delta_theta`, alongside the
actual finite probe losses. These are training-probe local utilities, not
held-out clean-step utilities or identified long-horizon mediation.

## Evidence and resources

Persist immutable plans, source/environment hashes, every evaluation's float32
logits, exact counts and float64 CE sums, scalar training logs, all diagnostic
raw tensors, the three full warmup states, and all 36 full endpoint states.
Do **not** save a full covariance observer at every evaluation. Save enough
diagnostic pre/post state for arithmetic auditing without reexecuting a model.
Completion receipts hash every artifact and list all expected trajectories.

Use the local RTX3090, one CPU math thread, 16 GiB host RAM/no swap, at most
8 GiB allocated GPU memory, a 30-minute hard service/25-minute cooperative
batch deadline, and **3 GiB total output** with at least 1 GiB free reserve.
The original option estimated 2 GiB; this pre-acquisition protocol allows
3 GiB to retain raw per-example diagnostic evidence instead of unverifiable
summaries. Validate the byte estimate before launch. No paid compute.

New sources and pure synthetic fixtures must be reviewed and committed before
one exclusive output directory and one non-restarting service are launched.
Do not inspect a partial outcome to change the roster. Independently check
saved predictions, pairings and diagnostic arithmetic once after completion;
that is an audit, not a fresh-seed replication or a training rerun.
