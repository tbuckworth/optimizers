# Ordinary augmentation with fixed wrong labels

Codex — Spectral Optimizer Investigation · 10 September 2026

## Question and authority

**Question:** under substantial fixed label corruption, does ordinary image
translation improve useful learning for the current spectral recipe, and how
does that change compare with raw AdamW? Retain both favorable and unfavorable
outcomes. Establish clean-data utility first; reduced fitting of wrong labels
alone is not success.

## Fixed design

- Three fresh paired seeds: **202609151, 202609152, 202609153**.
- Four branches per seed: raw AdamW / stable native rank32 × none / translation.
  **12 trajectories**, each exactly **4,000 updates**, with no early stopping,
  hyperparameter selection, extra control, clean rerun or follow-on arm.
- Reuse the unchanged model, optimizer, exact integer-translation function,
  occurrence/split/shift generator and resource helpers from the committed
  ordinary-augmentation experiment. New wrapper/data/auditor sources are
  separate; old source-pinned scientific files and canonical optimizer are
  immutable.
- Pinned official MNIST **training** IDX files only. Each true digit contributes
  500 training and 500 disjoint held-out examples: 5,000 per pool, exactly
  balanced by truth. All classes are eligible from update1. Pools may overlap
  across seeds. Do not call these independent datasets.
- Draw exactly **400 of each digit's 500 training examples** for corruption:
  **4,000/5,000 = 80% actually wrong**. This intentionally severe pilot level is
  not user-specified or claimed optimal. It is not 80% uniform replacement
  (which can retain a true label). No search over corruption fractions.
- Each selected label is replaced uniformly with one of the other nine digits.
  The replacement is fixed per original training example and remains attached
  to every occurrence and translated view. Nonselected labels stay true;
  held-out labels are always true. No label redraw, cue, rarity or curriculum.
- Separate NumPy PCG64 SeedSequence([stream_id, seed]) streams: 0 split, 1
  occurrence IDs, 2 shifts, 3 corruption mask, 4 replacement offsets. In
  true-digit order0…9, stream3 shuffles the digit's local training positions
  and selects the first400. Stream4 draws 5,000 independent integer offsets
  uniformly from1…9; apply (true+offset) mod10 only where the mask is true.
  Save every actual plan, clean/assigned labels, mask, hashes and NumPy version.
- Same source images, masks, wrong targets, initial model, minibatch IDs and
  translation displacements across all four branches within seed. Uniform
  replacement minibatches of64; one view per occurrence, not extra examples.
- Translation is independent dx,dy uniform integers−2…2, zero padding, no
  interpolation or wrapping. It starts at update1, including warmup. Original
  unaugmented images are used for all scheduled evaluation.
- MLP784→64ReLU→10, 50,890 parameters, FP32. AdamW lr0.001, decay0.01,
  betas(0.9,0.999), eps1e−8, foreach=False, fused=False. Stable global filter
  rank32, decay0.99, warmup100, hard projection, no adaptive rank/normalization.
  Observe the current raw gradient before projecting it. Projection begins101.
  The centered covariance determines the basis, but the delivered gradient
  projection does not first subtract its running mean. Canonical source is
  unchanged. Raw has no shadow observer overhead.
- Save full initialization, update100 and update4000 model/Adam/observer states.
  Raw/native model+Adam states must hash identically at100 within each mode.
  None and translated warmups differ; no cross-mode equality is assumed.
  Alternate raw/native execution order by seed and augmentation index.

## Why this is a useful, nontrivial test

In the population noise model, conditional on true class y the assigned target
has probability0.20 on y and0.80/9≈0.0889 on each alternative. The true class
therefore remains the most likely target despite most individual labels being
wrong. Empirical finite-sample label counts and learning need not match this
expectation exactly. This is not a target distribution with no class signal.

Augmentation could discourage memorizing particular inputs while reinforcing
shape invariance; it could instead reinforce each persistent wrong target
across views, or make useful fitting harder for the restricted spectral update.
All three are live hypotheses. Favorable results would not establish a
covariance mechanism, semantic selection, safety in language models, or
superiority over tuned baselines.

## Prespecified readouts

Evaluate at steps0,100,200,400,…,4000 (22 states). Save complete original train
and held-out logits, losses of actual training minibatches, step schedule,
state digests and receipts. For each state recompute accuracy and cross-entropy:

1. **Held-out clean labels (5,000)**: primary useful-learning endpoints.
2. All original training images against true labels (5,000): useful fitting.
3. All original training images against assigned labels (5,000): objective fit.
4. Corrupted training subset against assigned wrong labels (4,000): wrong-target
   fitting, explicitly not true-label accuracy or held-out memorization.
5. The same corrupted subset against original true labels (4,000).

For each seed and policy, augmentation benefit is A(translate)−A(none) in
accuracy and CE(none)−CE(translate) in loss: positive is favorable. Report
native augmentation benefit minus raw augmentation benefit as a secondary
interaction, alongside all absolute outcomes and per-seed signs. A favorable
interaction alone does not show either arm improved. Wrong-fit changes use
explicit directions, not a blanket “benefit” label.

Report every branch's warmup→endpoint changes. Useful native progress from
its own warmup is distinct from outperforming AdamW and from having a better
warmup. No best-checkpoint or test-selected endpoint claim; all curves remain
visible. Three seed bundles support descriptive consistency, not a confident
population significance claim or a dataset-level generalization claim.

The earlier clean factorial uses different seed bundles and is **context only**.
Do not pool it or claim a prospectively paired clean×noise×augmentation
interaction. It is retained, not restarted to decorate this comparison.

## Independent verification and resource boundary

Before launch: new runner/data/auditor fixture tests, source/design review,
committed exact source pins including transitive imports, inherited translation
visual review, and live GPU/unit/worktree checks. The separate NumPy auditor
independently reconstructs the split/occurrence/shift/corruption arrays and all
saved-logit readouts; it does not import the new producer's plan or training
implementation. Artifact hashes/checkpoint digest pairing verify provenance,
not a replay of every optimizer update.

One local RTX3090 unit `spectral-wrong-label-augmentation-001.service`, Restart=no,
1CPU, 16GiB host RAM, zero swap, 8GiB GPU allocation, 1GiB archive cap,
15-minute cooperative /20-minute hard deadline. Output goes only in an unused
mktemp directory under the verified `/tmp/spectral-experiment-artifacts` mount. Exclusive
attempt record and acquisition directory prevent automatic reruns. No cloud,
paid APIs, shared-GPU displacement, default changes or benchmark speedrun.

Then one fixed saved-logit audit and scalar plotting/reporting. If the
acquisition fails, preserve its partial artifacts and diagnose rather than
automatically retrying. The research goal/reminder remain active; completion of
this experiment alone does not complete the overarching goal.
