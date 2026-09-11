# Prospective analysis: current versus lagged delivery

Drafted before implementation, pilot and outcomes. Final source review and
parent launch decisions are separate gates. [design-intent.md](design-intent.md)
defines the scientific question, policies and limitations. This document fixes
aggregation; it may be clarified during pre-pilot review, never tuned to results.

## Primary and mandatory comparisons

Seeds 6,7,8; conditions nominal replacement 0 and .9. The unit is the paired seed
within one condition. Arm order is `adamw`, `current32`, `lagged32`,
`lagged32_current_norm`, `scalar_current32`, `scalar_lagged32`. For every pair,
the later-listed arm minus the earlier-listed arm is the single fixed contrast
orientation. Keep all 15 arm pairs separately in both conditions.

The four primary groups are test accuracy at the earliest strict maximum-
validation-accuracy checkpoint for `lagged32-current32` and
`lagged32_current_norm-current32`, separately for replacement 0 and .9.
Report all four together with every seed's value/difference, mean, median,
minimum, maximum and descriptive sample SD. No p-values, confidence intervals,
effect-size thresholds, equivalence claims or best-arm selection.

For each selected checkpoint also record `max(selected_step-100, 0)` and
whether its step is at most 100. A pre-intervention selection is retained,
not excluded or counted as evidence that active trajectories were identical.

Secondary learning metrics: test accuracy and CE at `final`, `min_val_ce`,
`max_val_accuracy`, `warmup100`; selected steps and selected validation accuracy
and CE; each non-warmup checkpoint minus its own warmup test metric; final
training clean/noisy metrics and final validation metrics. The unchanged
evaluation helper returns accuracy as a fraction and a count; check count
5,000 for training/validation and 10,000 for test. Report percentage points
only in prose, never silently rescale raw JSON.

Warmup can win a selector; do not exclude such runs. Strict earliest ties have
no CE/accuracy cross-tie-break. Reconstruct both selectors from the exact grid
0,100,...,2000; `final=2000`, `warmup100=100`. Preserve both selectors even if
they disagree. All 36 runs must finish training/selection before test loading;
require 144 complete checkpoint test evaluations. The reused test set and
adaptive history of this research line remain limitations despite prospective
freezing of this comparison. Do not pool previous studies or clean/noisy seeds.

## Gradient and update metrics

Every observing arm supplies current and lagged candidate diagnostics at its
own state. AdamW has neither observer: its candidate-specific metrics are null,
not equal to raw as a fictional measurement. Warmup applies identity and is
excluded from the geometry windows. All-policy raw/applied and actual-update
metrics remain defined subject to zero norms. Candidate quantities during
warmup may be null in raw records; the policy contract must say so explicitly.

Add two fixed phase windows across steps 101-2000 for current/lagged candidate
energy retentions, their difference and mutual cosine only: scheduled repair
steps (step divisible by 100) and all other active steps. Keep both groups and
their exact step counts. This is a descriptive scheduled-phase comparison, not
an isolated causal effect of QR repair and not a substitute primary endpoint.
The canonical observer can additionally repair when measured drift triggers its
early/50-step checks. Therefore the second group means other scheduled phases,
not proof of no actual repair. Retain the actual stabilization-count history.

For all postwarmup steps 101-2000, early 101-500 and late 1501-2000, summarize:

- Raw/current/lagged/applied squared norms and norms (means of norms are not
  square roots of mean energies).
- Current/raw, lagged/raw and applied/raw norm ratios; current and lagged
  energy retentions, and their within-step current-minus-lagged difference.
- Current/lagged candidate cosine, raw/applied cosine, any defined policy scale,
  norm-matching error and direction-collinearity error. Include scale maximum
  and minimum, not just a mean that could hide near-singular restoration.
- Total and nominal-decay-subtracted update norms and squared norms,
  raw/applied gradient dot update, corresponding cosines, and tolerance-qualified
  positive-dot frequencies with all scheduled steps in the denominator.

The sign tolerance is `1e-6*||gradient||*||update||+1e-14`; zero norms give null
cosine but do not remove a step from the positive-dot denominator. These dots
are directional diagnostics, not observed finite-step loss changes. Nominal
decay subtraction is `delta + lr*weight_decay*theta_before`; it does not undo
the historical effects of decay or its finite-precision operation order.

Average finite step metrics within seed first; retain exact null-step masks,
finite/null counts and reasons. Seed-group mean requires all three seed means.
A paired geometry aggregate requires matching null-step masks in each seed;
otherwise retain valid per-seed differences but make the complete group summary
null with unavailable seed IDs. An identically null AdamW candidate is not zero.
Do not use available-case means as a replacement primary. All primary test
metrics must be present in all runs or the completed summary is refused.

No scalar-identity projector leakage or exact orthogonal-decomposition identity
is computed. No norm ratio is described as actual AdamW step matching. The
lagged/current contrast is the full ordering policy, including any repair-step
effects, not an isolated covariance-outer-product intervention.

The norm-restored policy specifically means **lagged direction with current-
observer magnitude**. Its magnitude remains self-inclusive. There is no reciprocal
current-direction/lagged-magnitude filtered arm, so these controls are not a
complete direction-by-magnitude factorial decomposition.

## Required validation and artifacts

Summarizer is independent of training imports, accepts only the exact condition
set and completed full-run manifest, checks source/data bindings and refuses
output overwrite. Verify every step index, all 36 unique run keys, source map
membership/bytes, result hashes, valid finite metrics and exact checkpoint sets.
Check every warmup trajectory/core hash across arms within seed and condition,
and observer hashes across the five observing arms. Check realized corruption
fractions and shared plan bindings across the paired arms; require zero wrong
labels under condition 0. Pilot and full source/data hashes must agree.

The experiment producer will publish a concrete raw schema before the summary
implementation is frozen. Tests must reject wrong/missing/duplicate conditions,
changed sources, incomplete steps, non-finite outcomes, corrupted selector
choices, inconsistent policy gates, changed warmup hashes and early test access.
Synthetic fixtures must exercise zero/null masks and adverse primary signs.

Raw rows and checkpoints live in exclusive large-volume attempt directories.
Tracked manifests bind them by path, size and SHA256. A later lossless archive
may preserve compact raw JSON in Git, with source-byte hash and round-trip
verification. Never overwrite raw evidence to make it match a summary. Retain
the exact unrounded frozen summary and all signed contrasts. Independent
post-run audit reaggregates raw records and evaluates saved checkpoint tensors;
it does not substitute report prose or a second import of the same summarizer.

## Interpretation rules

The primary answer is whether the two specified lagging policies change the
selected learning outcome relative to current delivery in this recipe. Changes
against AdamW, scalar controls, endpoints and CE support interpretation but
cannot replace a primary that is mixed or adverse. A positive norm-restored
contrast is evidence about that policy, not proof that corruption was uniquely
removed, or that numerical gradient/update histories were equalized. A null
or negative result does not invalidate the passive self-inclusion observation;
it limits the inference from that observation to a useful intervention.
