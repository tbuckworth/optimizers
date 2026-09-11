# Neural clustering: functional, but not an improvement over stable hard spectral

The clustering optimizer ran successfully in a real neural-training comparison.
It retained substantially better late-stage accuracy than AdamW under fixed
label corruption and fitted far fewer wrong labels. However, it underfit clean
data and was worse than stable hard spectral on both held-out accuracy and
cross-entropy in every seed and condition. Its noisy-data accuracy did **not**
consistently improve beyond the inherited warmup. This is evidence for a useful
anti-memorization-like behavior under this recipe, not yet a better optimizer or
proof that the clusters identify generalizable features.
[Audited endpoint summary](execution/checked-summary.json).

## What was run and verified

Three fresh seeds (`202609091`, `202609092`, `202609093`), two conditions, four
arms: 24 completed trajectories. Each used the 50,890-parameter MNIST MLP, 5,000
training examples and 5,000 disjoint clean held-out examples from the original
training set. No official test set or checkpoint selector was used. All arms
inherited the same 100-step raw-gradient warmup, including model, AdamW and
observer state, then continued to the fixed 2,000-step endpoint with paired
batches. This is **not clustering training from random initialization**.
Frozen protocol (artifact not distributed in this public snapshot), manifest (artifact not distributed in this public snapshot).

`cluster32` applies the mean gradient within parameter-coordinate clusters;
`mixed32` applies half the raw gradient plus half that cluster-mean gradient.
Both use 64 anchors and up to 32 clusters, refreshed every 100 steps, from the same
stable centered rank-32 covariance observer as `hard32`. The synthetic pilot's
graph/action functions are unchanged; its uncentered estimator was deliberately
replaced for this controlled neural adaptation. AdamW hyperparameters are matched.
[Design review](design-review.md).

The corruption condition randomly selected 90% of examples for a fixed uniform
replacement digit, which can accidentally remain correct. Actual wrong-label
counts were 4,053/5,000, 4,031/5,000 and 4,016/5,000: 81.06%, 80.62%, 80.32%.
Independent saved-data auditing passed, checking plans, logits/counts/CE,
memberships and receipts without replaying training. Audit SHA256:
`e5d69d9ccb483e0f7514f475130626fe0c9654a202af366cf1d3fa0f8fa4cd2f`.
[Checked summary and audit binding](execution/checked-summary.json).

## All fixed-endpoint held-out results

Each cell is **accuracy percent / CE nats per example**. Higher accuracy and
lower CE are favorable. Seed columns are paired; means average the three seeds.
All values below come from the [audited summary](execution/checked-summary.json).

| Training condition | Arm | Seed 091 | Seed 092 | Seed 093 | Mean |
|---|---|---:|---:|---:|---:|
| Clean | AdamW | 91.44 / 0.3122 | 92.92 / 0.2592 | 92.76 / 0.2560 | 92.37 / 0.2758 |
| Clean | Hard spectral | 86.14 / 0.4819 | 87.44 / 0.4615 | 87.24 / 0.4479 | 86.94 / 0.4638 |
| Clean | Cluster mean | 84.60 / 0.5219 | 86.96 / 0.4988 | 86.30 / 0.4861 | 85.95 / 0.5023 |
| Clean | Half-identity mix | 91.32 / 0.3064 | 93.02 / 0.2500 | 92.58 / 0.2483 | 92.31 / 0.2682 |
| Fixed corruption | AdamW | 29.00 / 2.0641 | 30.50 / 2.0144 | 28.62 / 2.0548 | 29.37 / 2.0444 |
| Fixed corruption | Hard spectral | 51.94 / 1.9616 | 55.02 / 1.9807 | 45.44 / 2.0162 | 50.80 / 1.9862 |
| Fixed corruption | Cluster mean | 50.58 / 2.0275 | 53.90 / 2.0199 | 42.32 / 2.0488 | 48.93 / 2.0321 |
| Fixed corruption | Half-identity mix | 30.24 / 2.0237 | 33.38 / 1.9504 | 32.32 / 1.9857 | 31.98 / 1.9866 |

The fixed paired contrasts preserve the CE contradictions. Parentheses give
the number of favorable seed pairs, not statistical significance; no ties occur.
Accuracy differences are percentage points, CE differences are nats/example.
Exact differences, sample SD and sample SE remain in the
[audited paired summary](execution/checked-summary.json).

| Condition | First arm minus second arm | Mean accuracy difference (favorable) | Mean CE difference (favorable) |
|---|---|---:|---:|
| Clean | Hard − AdamW | −5.43 (0/3) | +0.1880 (0/3) |
| Clean | Cluster − AdamW | −6.42 (0/3) | +0.2265 (0/3) |
| Clean | Mix − AdamW | −0.07 (1/3) | −0.0076 (3/3) |
| Clean | Cluster − hard | −0.99 (0/3) | +0.0385 (0/3) |
| Clean | Mix − hard | +5.37 (3/3) | −0.1955 (3/3) |
| Fixed corruption | Hard − AdamW | +21.43 (3/3) | −0.0583 (3/3) |
| Fixed corruption | Cluster − AdamW | +19.56 (3/3) | −0.0124 (2/3) |
| Fixed corruption | Mix − AdamW | +2.61 (3/3) | −0.0579 (3/3) |
| Fixed corruption | Cluster − hard | −1.87 (0/3) | +0.0459 (0/3) |
| Fixed corruption | Mix − hard | −18.82 (0/3) | +0.0004 (2/3) |

Thus the large noisy-accuracy advantage of clustering over AdamW does not imply
a similarly robust CE advantage. The mixed arm nearly recovers AdamW's clean
accuracy and slightly improves its clean CE in all 3 seeds, but also recovers
much of its wrong-label fitting. Under corruption, mixed and hard spectral have
nearly identical **mean CE** despite very different accuracies; their CE ordering
is mixed across seeds. Reporting only accuracy would miss this.

## What did clustering add after warmup?

For cluster mean, clean held-out accuracy went from 84.56%, 85.74%, 85.56% at step 100
to 84.60%, 86.96%, 86.30% at step 2,000: mean 85.29% → 85.95%. Clean CE improved in all 3
seeds, from 0.5708, 0.5471, 0.5474 to 0.5219, 0.4988, 0.4861.

Under corruption, accuracy went from 50.98%, 43.38%, 55.34% to 50.58%, 53.90%, 42.32%:
mean 49.90% → 48.93%, with declines in 2/3 seeds. Nevertheless, clean-heldout CE
improved in all 3, from 2.0704, 2.1086, 2.0579 to 2.0275, 2.0199, 2.0488.
These are descriptive warmup-to-endpoint comparisons, not a new selector or
independent causal experiment.
[All audited curves](execution/checked-summary.json).

So clustering is not simply an unexecuted or completely unchanged model, but the
strong late comparison against AdamW is largely a story of avoiding AdamW's
loss of early accuracy, not demonstrated consistent further noisy-data accuracy
learning. The common warmup, inherited Adam moments and weight decay were not
separately ablated here; these observations do not identify the contribution
of newly learned cluster directions.

## Wrong-label fit and clean underfitting

Wrong-label accuracy is measured against the assigned wrong target, **only on
examples whose label actually changed**. It is undefined in the clean condition.
The other two training columns average all 5,000 training examples.
[Audited training metrics](execution/checked-summary.json).

| Arm under corruption | Wrong-label fit: seeds 091/092/093 (%) | Mean wrong-label fit (%) | Mean train clean accuracy (%) | Mean train fixed-target accuracy (%) |
|---|---:|---:|---:|---:|
| AdamW | 39.48 / 37.83 / 37.70 | 38.34 | 30.21 | 43.78 |
| Hard spectral | 7.82 / 7.24 / 8.79 | 7.95 | 51.53 | 17.53 |
| Cluster mean | 7.92 / 7.05 / 8.24 | 7.74 | 49.49 | 16.82 |
| Half-identity mix | 34.27 / 34.53 / 32.82 | 33.87 | 32.06 | 40.03 |

Cluster mean strongly reduces wrong-target fit relative to AdamW in all 3 seeds,
but clean-condition training accuracy is also only 87.58% versus 99.20% for AdamW
(hard spectral 88.51%, mix 98.50%). This combination supports a restrictive
learning/regularization interpretation, not selective recognition of useless
facts. Lower wrong-label fit alone would not establish useful generalization.

## Practical cost and boundaries

All 24 trajectories completed in 367.893 seconds, about 6.1 minutes, on the local
RTX 3090 with one CPU and no paid compute. Output before completion metadata was
488,883,784 bytes; process high-water RSS 1,555,264 KiB; peak PyTorch GPU allocation
127,659,008 bytes. These are the recorded measures, not full-device memory use.
Completion receipt (artifact not distributed in this public snapshot), service log (artifact not distributed in this public snapshot).

Mean warmup-plus-branch times, clean/noisy respectively, were AdamW 4.09/4.05 s,
hard spectral 6.89/6.75 s, cluster mean 25.34/25.37 s and mix 25.08/25.78 s. Branch
timings include evaluation and graph refreshes but exclude serialization;
the shared warmup was executed only once per seed/condition. This prototype
was slower, not faster, than the controls. These descriptive timings are not
a tuned implementation benchmark. [Timing summary](execution/checked-summary.json).

Confidence is high that the reported finite-batch behavior occurred, limited
for generalization beyond 3 seeds and this one small recipe. Parameter-coordinate
clusters are not established semantic features or memorized facts. Equal means
can lose coherent but unequal-amplitude components of the delivered gradient,
not directly of Adam's parameter step; grouping across tensors can mix different
scales. Membership changes every 100 steps whereas hard spectral
updates its subspace each step. Norm, geometry, staleness and downstream AdamW
dynamics therefore remain entangled; no measured causal mediation or safety
conclusion follows.

One coherent follow-up is an **amplitude-weighted group projection**, which would
preserve proportional co-movement within a correctly recovered group rather
than forcing equal coordinate amplitudes. The
reviewed mathematical note (artifact not distributed in this public snapshot) explains why this could
matter and why it might still fail. It is a hypothetical next action, **not a
tested explanation of these results**. Any follow-up needs its own prospective
comparison; this completed roster must not be retrospectively changed.

Source commit: `48e7b2ad0f58f4676abf1f88850b542d4380ba25`.
Completion SHA256:
`025a3bb01483893daaf80e82f3f8db2b0bc9d1f35cae0354c282c345d8d7d3f4`.
Full raw artifacts remain under
`/tmp/spectral-experiment-artifacts/spectral-clustering-mnist-20260909.51ggnu/acquisition-001`.
