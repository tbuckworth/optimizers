# Ordinary augmentation with and without spectral filtering

Codex — Spectral Optimizer Investigation · 10 September 2026

Prospective local experiment; no result has informed this specification.
Responds to the user's clarification (artifact not distributed in this public snapshot)
that ordinary data augmentation, not the artificial cue setting, is the priority.

## Question and strongest useful reading

Does a small ordinary transformation help the current stable spectral optimizer
learn and generalize on clean, balanced data? Does its benefit differ from
the benefit for raw AdamW? A positive absolute improvement would establish a
useful effect in this setting before asking for mechanistic attribution.
Spatially transformed views might help the observer track useful variation, but
can also add nuisance variance and change the learned objective. Neither sign
alone identifies covariance-mediated invariance. This is not a test of cues,
wrong labels, rare classes, alignment, a best-tuned optimizer, or all augmentation.

## Fixed 12-trajectory design

- Seeds: **202609141, 202609142, 202609143** (fresh paired seeds).
- Two policies: raw AdamW; unchanged canonical stable spectral on AdamW.
- Two conditions: no augmentation; one random integer translation per training
  occurrence, horizontal and vertical independently uniform in {-2,-1,0,1,2}.
  Positive displacement is right/down. Zero fill outside the 28×28 canvas;
  no wrap, interpolation, rotation, flip, mixing, or extra views. Identity has
  probability 1/25. Labels are unchanged, not assumed infallibly preserved:
  inspect a fixed training-only contact sheet and quantify cropped pixel mass.
- Official MNIST **training** IDX files, pinned by SHA-256. For each seed and
  digit, shuffle source IDs; first 500 train, next 500 held out. Thus 5,000 train
  and 5,000 held out, each exactly balanced, disjoint within seed. The official
  test files are not opened. This is an internal held-out split, not a new
  independent benchmark: the task/model/recipe come from prior investigation.
- Class-major IDs; PCG64(SeedSequence([stream_id, seed])) with independent streams
  0=split, 1=occurrences, 2=translations. Sample uniformly with replacement from
  all 5,000 training IDs: all ten classes are **eligible** from update 1, not
  necessarily represented in every minibatch. Save exact indexed occurrences
  and per-occurrence displacements. Across seeds the held-out sets can overlap.
- MLP 784→64→ReLU→10, 50,890 parameters, FP32 pixels divided by 255 on CPU.
  Reuse accepted I9 initialization/model/optimizer factory without changing it.
  AdamW lr=0.001, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8,
  foreach=False, fused=False; no scheduler, clipping, AMP, or tuning.
- Spectral rank cap 32, decay 0.99, warmup 100, hard projection, stable_update=True,
  stabilize_every=100, relative_eig_tol=1e-8, absolute_eig_floor=0,
  normalize='none', adaptive='none'. All model parameters observed. The canonical
  filter observes the current gradient before projecting, begins projection at
  update 101, and applies before AdamW. It is not a guarantee about the direction
  of the resulting AdamW parameter step. Raw has no observer overhead.
- **4,000 updates, batch 64**, fixed in advance: 51.2 nominal dataset passes.
  Twice the recent 2,000-update horizon allows continued learning to be visible;
  it does not promise convergence or match each method's optimal learning rate.
  Do not stop at a favorable checkpoint, extend based on outcomes, or tune rank.

Each trajectory starts from a fresh copy of the same seed's model/empty Adam
state. Both policies use identical splits, occurrence order, and translations
within a condition. Augmentation is active from update 1, including the observer
warmup: the translated and untranslated conditions **must not share a warmup
parent**. Require bit-identical model/Adam state and logits at step 100 between
raw/native within each condition. Alternate which policy runs first by seed
and augmentation index; this mitigates but does not remove order/thermal effects.

## Readouts fixed before execution

Primary: original, **unaugmented** held-out cross-entropy and accuracy at update
4,000. Save logits and metrics at exactly (0,100,200,400,…,4000): 22 states per
trajectory, 264 trajectory-state rows. Same unaugmented training readouts show
fit/underfit. Save actual minibatch training loss at all 4,000 updates; it is a
loss on different transformed objectives, not a matched clean training metric.
Show full curves without selecting a best step. Retain all four absolute
outcomes, per-seed results and descriptive means; no three-seed significance
claim. Accuracy is higher-is-better; CE lower-is-better.

For each seed/policy, augmentation benefit is A(translate)−A(none) for accuracy,
and CE(none)−CE(translate) for CE. Interaction = spectral benefit − raw benefit.
A favorable interaction can arise solely from raw deterioration, without any
absolute spectral benefit: **it does not
by itself establish spectral augmentation helps**. Discordant accuracy/CE or
seed signs stay visible. Cross-policy contrasts are secondary fixed-recipe
comparisons, not evidence that spectral cannot benefit under another recipe.
No transformed held-out evaluation or extra mechanism probe in this acquisition.

Measure synchronized training wall time, separately accumulate CPU transform
time (already included in training), evaluation time, and total branch wall
time including saved artifacts. Report runtime as this audited harness, not
an optimized throughput benchmark. Save full warmup/final states, initial model
identity, plans, all logits, losses, metrics, and source/data/artifact hashes.
An independent NumPy saved-logit auditor recomputes metrics and contrasts,
checks membership, exact warmup pairing, finite values and the full roster.

## Implementation, visual check and admission

Fixture-only tests first: exact integer-transform pixel oracle over all 25
displacements; immutable inputs; deterministic plans; disjoint balanced split;
independent RNG streams; no implicit acquisition/overrides; capped writes;
metric and interaction arithmetic; hash and pairing rejection. A training-only
transform contact sheet is an input validation, not an optimizer run or outcome
selection. Use the first three per-class training IDs for seed 202609141 and
show original plus eight extreme/cardinal shifts; quantify all 25 shifts on
the training set. No label-based exclusion after seeing images.

## Interpretation boundaries / next decision

If translation helps spectral absolutely, quantify that result before controls.
If gains differ by policy, consider trajectory/subspace measurements as a
separate prospective follow-up—not claim they were measured here. If spectral
underfits both conditions, report that limitation of this rank/centered recipe
and consider a justified variant rather than conclude augmentation never helps.
One simple MLP/MNIST transform is a first result, not a coverage claim for images,
language models, clustering, RL, or safety. The user-prioritized generalization
question remains distinct from capability speedruns or optimizer-default changes.
