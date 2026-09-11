# Design disposition and implementation-only GO

6 September 2026. The parent read all three independent challenge outputs.
The mentor verdict is MINOR_REVISIONS; assumption and pre-mortem reviews find
no construct-validity blocker to the stated whole-policy comparison. The
learning outcome is genuinely unknown; numerical invariants are not scientific
evidence of a benefit. The researcher workflow caused the pre-implementation
challenge and the fixes below.

## Fix now

- Correct the initial scalar rank-description error: only projected arms have
  rank-limited delivery; nonzero scalar arms apply a full-rank scaled identity.
- Freeze float64 rescaling followed by a single float32 cast, strict relative
  norm/direction gates and explicit representability failures. Do not copy the
  scalar attenuation bound onto the norm-restored lag ratio.
- Test the real current observer path against unchanged canonical helpers,
  first active step 101, once-only observation, repair and previous-basis cloning.
- Extend snapshots with NumPy-global state; retain full model/optimizer/RNG
  warmup equality within seed and noise condition, never across clean/noisy.
- Make `(seed, noise, arm)` the unique identity everywhere. Require exact set
  equality for 36 training runs before 144 test evaluations; test orchestration
  with synthetic loaders, not a copied twelve-run count.
- Retain final clean/noisy training scores, candidate angles, selected-step
  exposure and fixed repair/ordinary-step descriptive windows. No extra neural
  probes or covariance references are needed.
- Place bulk artifacts on the verified large-volume mount. Freeze scalar-only
  per-step records, four parameter checkpoints per cell, bounded failure context,
  maximum total artifact budget 1 GiB, peak GPU allocation 8 GiB and RSS 12 GiB.
  Resource caps supplement the proposed 180 s pilot / 900 s full wall caps.

## Accepted scope, not defects to tune away

Three seed bundles and reused MNIST/test data bound generalization. The primary
includes the checkpoint-selection policy; endpoints answer fixed exposure.
Norm-restored lagging keeps current-observer magnitude, so it is neither fully
non-self-inclusive nor a complete direction-by-magnitude factorial. Actual
AdamW step matching, semantic probes, reciprocal direction/magnitude arms and
external datasets are deferred experiments, not requirements for these two
policy contrasts. Scientific null/adverse results end this fixed comparison.

## Authority