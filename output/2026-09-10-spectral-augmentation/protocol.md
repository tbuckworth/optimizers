# Does augmentation improve the usefulness of temporal spectral filtering?

Codex — Spectral Optimizer Investigation · 10 September 2026

## 1. Question and constructive hypotheses

Can weakening a recurring misleading cue through training augmentation improve
the actual learning tradeoff of the existing spectral optimizer? First ask
whether it becomes useful on clean/common competence, rare competence and cue
reliance; second ask whether its improvement exceeds ordinary AdamW's. A shared
benefit is still a useful augmentation result. A favorable interaction caused
only by AdamW becoming worse is not a spectral improvement.

The favorable hypothesis is that breaking a nuisance's recurrence lets the
observer retain more reusable learning directions. The counter-hypothesis is
that changing cue visibility adds variance along that same nuisance direction,
which a top-covariance filter may retain. Neither conclusion follows from the
mask geometry alone. Ordinary random erasure is the main practical condition;
targeted erasure is a known-location diagnostic, not a deployable discovery
algorithm. Wrong labels remain wrong after a cue is erased.

## 2. Fixed acquisition roster

| Factor | Prospective values |
|---|---|
| Seeds | 202609131, 202609132, 202609133 |
| Training cells | Clean, Shared, Sham |
| Delivery policies | Raw AdamW; native stable global hard rank-32 filter before ambient AdamW |
| Mask modes | None; Random; Targeted; Opposite |
| Training | 100 common raw warmup updates, then 1,900 continuation updates; batch 64 |
| Evaluation | Steps 0, 100, 200, …, 2,000; step 2,000 is the primary endpoint |

There are **72 continuation trajectories**, not 72 independent replicates.
The three seeds are the paired replication units. Each seed's identical full
warmup snapshot is restored for all 24 branches. There are 300 physical warmup
updates and 136,800 continuation updates. Warmup evaluations are shared, giving
1,512 logical trajectory/evaluation records if repeated in the result table.
No early favorable-seed stopping, adaptive mask/rank/learning-rate search or
new Diffuse/norm-matched arm is selected. Previously reported norm controls
remain evidence, but this study asks a different learning-level question.

## 3. Reused data and optimizer contracts

Use the indexed construction in the committed
[selectivity source](../../experiments/spectral_selectivity_boundary.py) and
[original protocol](../2026-09-09-spectral-selectivity-boundary/protocol.md),
with the new seeds above. Pin the exact sources in the launch manifest. Only
the official MNIST **training** IDX files are used; the official test split
remains untouched. Preserve input bytes, disjoint splits and indexed exposure.

- Within each true digit, seed-shuffle indices: 550 training examples for each
  digit except 8, and 50 for digit 8; the next 500 per digit form held-out data.
  This gives 5,000 training and 5,000 held-out examples per seed. Digit 8 is rare.
- Warmup is raw AdamW on common true-label examples, without augmentation. Keep
  its raw-gradient observer history, model, Adam state and RNG state. The
  continuation is therefore not an augmentation-from-scratch experiment.
- Clean: true labels, no inserted cue. Shared: exactly 500 eligible examples
  from true digits 1–7 and 9 receive assigned label 0 and a white upper-left
  3×3 cue. Sham: identical wrong-label IDs and labels, and the same total and
  per-true-digit patch counts, but patch placement is approximately independent
  of poisoning within each true class under the existing integer allocation.
- Digit 8 remains correctly labeled and has no inserted cue; it is **not**
  exempt from augmentation. True digit 0 likewise receives no inserted cue.
- Preserve named PCG64 streams 0–5 for split, warmup, continuation and cell
  construction. Cell-constant labels and cues precede occurrence-specific masks.
- Use the existing 50,890-parameter MLP and AdamW: learning rate 0.001,
  weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8, foreach/fused disabled.
  Native filtering keeps global rank 32, beta 0.99, raw-gradient covariance,
  repair every 100 steps, relative eigenvalue tolerance 1e-8 and floor 0.
  Observe the current raw gradient before filtering, then perform ambient AdamW;
  keep the existing initialization, zero-gradient and state-update semantics.
- Normalize pixels on CPU in NumPy float32 by division by 255, as in the base
  experiment. No stochastic model layers or new optimizer modifications.

No new scientific-data read is needed to check the mask helper. Acquisition
must verify both IDX SHA-256s and all reused source pins before any model update.

## 4. Exact augmentation intervention

Generate one persisted plan per seed with shape (1,900, 64): independent
PCG64/SeedSequence([seed, 6]) float64 uniforms give gates `u < 0.5`, and
PCG64/SeedSequence([seed, 7]) gives int16 row/column centers uniformly in 0…27.
Save the actual arrays, NumPy version and hashes; seeding alone is not an
unqualified cross-version or changed-call-shape reproducibility guarantee.
Use [the inert helper](../../experiments/spectral_augmentation_masks.py).

Each plan entry belongs to a **batch occurrence**, not an example ID. Repeated
examples, including duplicates within a batch, can receive different masks.
All cells and policies receive byte-identical occurrence gates and centers.
Do not condition on true/assigned labels, poison status, patch presence, rare
status, current model outputs or held-out outcomes. Apply to all images.

| Mode | Mask for an active gate, with half-open pixel bounds |
|---|---|
| None | No erasure, regardless of gate |
| Random | Row `[max(r−4,0), min(r+4,28))`, column analogously; r,c uniform 0…27 |
| Targeted | Rows [0,8), columns [0,8) |
| Opposite | Rows [20,28), columns [20,28) |

Inactive entries do nothing. Insert any white cue **before** setting mask pixels
to black (0 in the specified [0,1] input scale). Do not reinsert a cue afterward.
Keep assigned labels and the number/order of exposures unchanged. One augmented
view per occurrence is selected; no multiview gradient averaging is selected.

The Random square follows the clipped-center geometry in the authors'
[Cutout implementation](https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py).
The fixed 8-pixel size, 0.5 gate, model and normalization here are our prospective
variant, not a reproduction of Cutout's published benchmark. Targeted and
Opposite have matched area and gates, but erase different image content.

### Geometry calibration, not a learning result

Enumerating all 784 centers gives 25 fully covering and 49 touching the 3×3 cue.
With the 0.5 gate:

| Mode | P(any cue overlap) | P(full cue removal) | Mean geometric erased pixels per occurrence |
|---|---:|---:|---:|
| None | 0 | 0 | 0 |
| Random | 3.125% | 1.5944% | 27.5918 |
| Targeted | 50% | 50% | 32 |
| Opposite | 0 | 0 | 32 |

Random's mean area conditional on its gate is 43,264/784 = 55.1837, not 64,
because of clipping. Thus Random versus targeted placement is **not** an
area-matched contrast. Report realized gate rates, any/full coverage and areas
overall and among actually cued occurrences, with explicit denominators. An
erased geometric pixel can already have been black. Partial cue erasure need
not remove the learned cue. A random-mask null cannot rule out benefits from
meaningful removal of this particular cue.

## 5. Outcomes: absolute usefulness before interactions

Evaluation never applies the training masks. Retain logits on (a) unaugmented
clean held-out images, (b) those same images with the canonical visible white
3×3 cue, and (c) fixed cell-specific training images without erasure. Evaluate
training predictions against both assigned and true labels. Do not substitute
masked training loss for clean held-out loss.

Report clean held-out common-digit accuracy and CE, rare-digit accuracy and CE,
per-digit outcomes and balanced totals. The common set contains digits other
than 8; rare n=500, common n=4,500. Distinguish retention from the shared warmup
level from acquisition of new rare competence. Show step-2,000 endpoints and
the complete preselected evaluation curves; do not select a favorable endpoint.

For policy p, cell c and augmentation a, define paired patch excess

`E(p,c,a) = mean[1(pred_patched = 0) − 1(pred_clean = 0)]`

on the identical held-out true digits 1–7 and 9 (n=4,000). Digit 0 is excluded
because predicting 0 is correct; rare digit 8 is excluded from this primary cue
metric and reported separately. Report both component rates, not only their
difference. The association-sensitive contrast is

`Q(p,a) = E(p,Shared,a) − E(p,Sham,a)`.

For a higher-is-better competence measure M, first report
`Δ(p,c,a) = M(p,c,a) − M(p,c,None)`,
then the secondary optimizer-by-augmentation interaction
`I(c,a) = Δ(Native,c,a) − Δ(Raw,c,a)`.
For CE, show the original lower-is-better values and label change direction
explicitly. Show Native−Raw absolute differences in each mode as well.

For cue reliance, report `ΔQ(p,a) = Q(p,a) − Q(p,None)` and then
`I_Q(a) = ΔQ(Native,a) − ΔQ(Raw,a)`. Negative I_Q means a relatively more
favorable change, not necessarily an absolute native reduction: both ΔQ values
could be positive. Native reduction requires ΔQ(Native,a) < 0; show absolute Q.
Targeted−Opposite contrasts in Q and competence are secondary mechanism/location
diagnostics. Shared/Sham controls association but does **not** turn corner
erasure into pure causal mediation: opposing corners can contain different
digit information, masking changes image statistics and wrong labels persist.

Random−None is the primary practical augmentation comparison. Targeted−Opposite
asks the sharper coverage question, with oracle-location limitations stated.
Plot all three paired seed values and their mean; seed range is descriptive,
not a confidence interval. Do not pool branches or held-out examples as 72 or
thousands of independent training replicates. No significance/equivalence claim,
composite winner, post-hoc favorable subset or universal safety claim is planned.

## 6. Interpretable possible outcomes

- Better absolute spectral competence with reduced cue reliance supports useful
  cooperation in this bounded regime. A further differential benefit supports
  optimizer specificity; without it, augmentation can still be worthwhile.
- Both optimizers improve similarly: useful data intervention, no evidence here
  that temporal filtering is necessary for that benefit.
- Targeted helps but Random does not: a coverage-sensitive result, consistent
  with the known geometry; not an automatic practical generic-augmentation win.
- Targeted and Opposite change learning similarly: nonspecific occlusion or
  digit-content effects remain viable. Preserve favorable and unfavorable cells.
- Cue reliance falls but clean/rare learning degrades: a tradeoff, not an overall
  success. No cue reliance with near-chance predictions is not useful robustness.
- Spectral gets worse after varying visibility: compatible with increased
  covariance variance or lost signal, not proof of either mechanism from learning
  curves alone. Do not add an automatic saved-tensor diagnostic before reporting.

## 7. Frozen compute and provenance envelope

The old 36-trajectory study, including expensive saved-gradient diagnostics,
took 258.416 seconds (receipt (artifact not distributed in this public snapshot)).
This motivates a bounded local pilot, not a promised runtime comparison. The
new run omits those per-example gradient diagnostics. Estimate every stored
array/checkpoint before launch: storing all logical logit histories without
deduplication costs 907.2 MB (72×21×3×5,000×10×4 bytes); endpoint and
warmup state plus plans should remain below 2 GiB. Do not silently discard
required evidence or increase limits to finish a run.

Reserved acquisition handle: `spectral-augmentation-001.service`. Verify it has
never acquired a result before launching once. Reserved analysis handle:
`spectral-augmentation-audit-001.service`, for analysis of that acquisition only.
Neither handle has been launched by this protocol. Persist immutable roster,
source hashes, config, split/batch/label/cue/mask arrays, branch order, warmup
snapshot hashes, endpoint states, evaluation logits, JSON measurements, timing,
GPU/host memory peaks, artifact byte counts and completion/error receipts.

Fail closed on changed sources, occupied GPU beyond the pre-existing desktop
clients, insufficient disk, failed pairing/data checks or resource limits.
Keep partial evidence and label incompleteness; do not automatically rerun a
failed/live/completed acquisition. Review the specific failure without widening
scope. No public upload, paper submission, capabilities benchmark, harmful-data
training or alteration of the existing scientific records is selected.

## 8. Theoretical and implementation references

At a fixed parameter vector, if h(i,a) is an example gradient under augmentation,
the law of total covariance gives

`Cov(h) = Cov_i(E_a[h|i]) + E_i[Cov_a(h|i)]`.

Independent equally distributed batch occurrences divide this covariance by
batch size; temporal covariance during training is not generally that fixed-
parameter population quantity. In a toy cue model h = b c with Bernoulli
visibility q, the mean is q c while covariance is q(1−q) c cᵀ. Reducing a nearly
always-visible cue can reduce its mean and **increase** its centered variance.
But that is not this study's whole-population cue frequency: only 500/5,000
training images carry the cue. Under the strong simplifying assumption of a
constant additive cue gradient and independent occurrences, targeted masking
changes q from 0.1 to 0.05, decreasing the coefficient from 0.09 to 0.0475
(about 47.2%). This gives the favorable hypothesis a concrete mathematical
route. Added within-example augmentation variance need not increase total
variance: between-example variation also changes. Real gradients, cross-terms,
digit erasure and the evolving observer need not follow this toy calculation.
Neither calculation is an observed explanation of this unrun study.

The original [Cutout paper](https://arxiv.org/abs/1708.04552) motivates square
occlusion as regularization, while [Random Erasing](https://arxiv.org/abs/1708.04896)
uses another rectangle/value construction. Neither establishes a spectral
interaction for this model. We do not add a second augmentation family now.
The plan uses explicit [NumPy SeedSequence entropy sequences](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html)
and obeys the limits of NumPy's [random-stream compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)
by saving arrays rather than promising reproduction after changed versions,
draw shapes or call order.
