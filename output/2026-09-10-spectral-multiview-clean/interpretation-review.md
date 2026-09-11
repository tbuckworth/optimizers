# Independent interpretation review

Codex — Spectral Optimizer Investigation · 10 September 2026

This reviews the completed [audited scalar results](audit.json), the frozen
[protocol](protocol.md) and previously reviewed source setup. It is not a new
acquisition audit. No model, tensor, data, raw-logit or gradient archive was
opened; no experiment, measurement or audit was rerun.

## Strongest useful finding and its boundary

Four-view observation improves the incumbent filter on the secondary translated
held-out panel in all three seeds and both metrics. Observer4 also improves
its own translated accuracy and CE from the common warmup in every seed:
this is actual useful learning, not only a favorable contrast against an arm
that deteriorated. But the primary original-image endpoint is essentially
unchanged in mean CE and mixed by seed; the large deficit to both raw controls
remains. "No primary rescue, with a consistent secondary benefit" is a fair
summary. Numerical closeness is not an equivalence result.

Fixed update-4000 means:

| Policy | Original held-out accuracy | Original CE | Translated accuracy | Translated CE |
|---|---:|---:|---:|---:|
| raw1 | 95.173% | 0.156650 | 93.527% | 0.220596 |
| native1 | 81.393% | 0.660327 | 67.073% | 1.040954 |
| observer4 | 81.593% | 0.660356 | 68.687% | 1.020062 |
| raw4 | 95.633% | 0.144492 | 94.207% | 0.198985 |

Observer4 minus native1, in seed order 202609161/162/163, with positive CE
benefit meaning lower loss:

| Readout | Accuracy benefit, percentage points | CE benefit, nats |
|---|---|---|
| Original primary | +0.16, +0.70, −0.26 | +0.011308, −0.001730, −0.009662 |
| Translated secondary | +1.88, +1.44, +1.52 | +0.027265, +0.015959, +0.019452 |

Thus primary accuracy improves in two seeds, primary CE in only one. The mean
primary CE benefit is −0.000028: cancellation between a favorable seed and two
adverse seeds, not uniformly negligible individual effects. Secondary mean
benefits are +1.613 percentage points and +0.020892 nats. The complete filtered
curves fluctuate; neither monotonic improvement nor an alternative best
checkpoint should replace the fixed endpoint.

## Actual progress, controls and training fit

From the identical first-view update-100 model/Adam warmup, native1 original
accuracy gains are +0.70/+0.14/+3.28 percentage points; observer4 gains are
+0.86/+0.84/+3.02. Original CE improves in every seed for both, by means
0.115318 and 0.115290 nats, respectively. Neither is "not learning."
Observer4 translated accuracy gains are +0.74/+0.28/+3.14 points and CE improves
in all three. Native1 translated accuracy instead changes −1.14/−1.16/+1.62
points, despite translated CE improving in all three; preserve that metric
disagreement.

Both raw controls substantially outperform observer4 in every seed, both
held-out panels and both metrics. Original mean accuracy gaps are 13.58 points
to raw1 and 14.04 to raw4. Raw4 also improves over raw1 in every seed on both
panels and metrics: original accuracy +0.48/+0.46/+0.44 points and CE benefits
0.010428/0.010360/0.015686. Raw4 begins with a different four-view warmup, so
these are whole-policy effects, not matched-state observer isolation.

Original training fit is also limited: observer4 is 82.380% accuracy and
0.648962 CE, versus native1 82.167%/0.647012 and raw4 97.667%/0.082548.
Observer4's small mean training-accuracy gain accompanies slightly worse mean
training CE. Poor training as well as held-out performance is consistent with
restricted learning; it is not evidence that the filter merely overfits the
original training images.

## Cost, provenance and claim limits

Each trajectory uses 4,000 Adam updates and 256,000 underlying-example
occurrences. Observer4/raw4 require 16,000 batch-gradient evaluations per
trajectory, versus 4,000 for native1/raw1. The accepted total is 120,000,
72,000 extra versus an all-single-view roster. Mean measured training seconds
are raw1 16.57, native1 22.51, observer4 36.35 and raw4 30.69; mean branch wall
seconds are 18.62, 24.58, 42.45 and 36.70. Augmentation is timed separately.
These rotated sequential timings are descriptive, not a speed benchmark or
fourfold wall-time claim.

The outcome supports a conditional observation-policy benefit on transformed
inputs. It does not identify semantic selection, nuisance-covariance mediation,
or an optimal rank. Observation scale, current-view self-inclusion and later
trajectories differ. This clean-only study neither certifies wrong-label
protection nor overturns the completed result favoring augmented raw AdamW in
the small noisy-label recipe. Three fresh paired training seeds are the
replication units, not views or readout checkpoints.

Audit SHA-256: `0c66a3827ddb7423e4c444e7e92682ac6f24f336138fa22133647236758ee147`.
Its pinned result SHA-256 is
`a4bf53e3696c9e22eeaa925bddc9034e9d27a39ff3b1a2410ae11531dfcc8d30`,
from acquisition commit `3c2c3c23e58c3f1d09929629824709df812bf364`.
The existing audit records PASS, 58 artifacts, 12 trajectories, 264 logical
states, 792 panel evaluations and maximum scalar discrepancy
3.637978807091713e−12. Checkpoint bytes were hashed without deserialization;
model/Adam digest pairing remains the explicitly disclosed producer assertion.

## Final written-report check

The completed [results report](results.md), SHA-256
`e9b7cc355e04a2dda0f093c5d5d539891f88211813300053741be0ae5e7b1337`,
was read in full. Its primary and secondary all-seed values, fixed contrasts,
warmup progress, training-fit qualifications, work counts and branch timing
means agree with the audited scalar records. The audit duration and archive
byte total also agree. The acquisition-only peak-resource footer was not
independently re-derived in this bounded interpretation review.

**Verdict: PASS; no material numerical error or overclaim found.** The useful
secondary learning result leads without replacing the primary result, and
the report preserves mixed primary seed signs, metric disagreements, both
strong raw controls, extra-view cost and clean-only limits. No further
scientific validation or new experiment is implied by this verdict.
