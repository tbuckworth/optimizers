# What this selectivity test establishes

Codex — Spectral Optimizer Investigation, 10 September 2026.
Three-seed, fixed-horizon evidence; interpretation of already checked JSON only.

## Useful preservation, poor rare-class acquisition

All branches inherit the same 100-step clean warmup, with no digit 8 in its
training data. Held-out rare accuracy starts at **0% in every seed**; majority
accuracy starts at 88.53%, 87.69% and 87.07%. The raw policy demonstrates that
new-class learning is feasible under this recipe.

Endpoint rare accuracy, percentages; within each entry the seeds are
202609111, 202609112, 202609113:

| Cell | Raw AdamW | Native spectral | Norm-rule/raw-direction control |
| --- | --- | --- | --- |
| Clean | 52.2, 50.8, 53.0 | 14.4, 0.4, 0.0 | 51.0, 50.8, 52.8 |
| Diffuse | 51.8, 55.4, 48.0 | 0.0, 0.0, 0.0 | 52.0, 53.4, 45.2 |
| Shared | 50.0, 51.2, 50.6 | 6.6, 4.0, 0.0 | 49.8, 50.2, 49.2 |
| Sham | 52.6, 56.4, 53.0 | 0.0, 0.0, 0.0 | 53.4, 56.8, 52.4 |

Native rare accuracy and CE are worse than **both** controls in every seed
and every cell. However, “no rare learning whatsoever” would overstate this:
native rare CE improves from warmup in all 12 trajectories, and two Clean
seeds and two Shared seeds acquire some correct predictions. Mean rare CE
ends at 2.812, 4.567, 2.930 and 3.272 for Clean/Diffuse/Shared/Sham, compared
with raw 1.711, 2.226, 1.739 and 1.575. Lower loss without correct classification
is partial improvement, not successful acquisition of the rare task.

The favorable Diffuse result is substantial: native majority accuracy is
62.62%, 58.29%, 65.11%, versus raw 34.44%, 34.91%, 32.56%. Native fits only
6.22% of actually wrong training targets on average, versus raw 32.48%.
But every native seed still loses majority accuracy from warmup: drops of
25.91, 29.40 and 21.96 percentage points. Thus the benefit is **less forgetting
or degradation**, not improved majority learning. On Clean data, native does
improve majority accuracy in all seeds, but only +0.93 points on average,
versus raw +5.61 points.

The Diffuse accuracy advantage does not translate into a consistent CE
advantage. Native majority CE is 1.88192, 1.87386, 1.86917; raw is 1.83859,
1.88211, 1.90145. Native is worse in seed 111, better in seeds 112/113, and
slightly worse on the mean: 1.87498 versus 1.87405. Accuracy and loss assess
different aspects of the prediction distribution; these values alone do not
identify a calibration mechanism. Do not present the accuracy gain as uniform
predictive improvement. [All endpoint values and paired summaries](results/checked-summary.json).

## The coherent wrong cue passes through

Shared and Sham use identical wrong targets and match patch counts within
each eligible true digit; they differ in patch–target association. The raw
Shared-minus-Sham patch-excess assay is positive in all seeds, so this is a
working cue manipulation, not an absent-effect test.

Native Shared patch excess is **95.90, 95.00, 93.575 percentage points**
(mean 94.825), versus raw 99.10, 99.05, 98.65 (mean 98.933). Native predicts
target 0 on 96.28% of patched nonzero-majority held-out inputs on average,
and fits 96.87% of the 500 poisoned training targets. In Sham it fits only
7.20% of those same wrong targets. The distinction is consistent with learning
an exploitable shared mapping while resisting much of the unmatched label fit;
it does not show that any specific covariance component caused the difference.

The registered difference-in-differences favors native numerically:
−2.925, −5.075, −2.075 points, mean **−3.358 points**. Preserve that all-seed
reduction, but it sits beside near-complete cue learning and much worse rare
classification. It is not an effective selective defense. The secondary
plain patched-ASR interaction is less consistent: −4.575, −2.725, **+1.200**
points. [Primary and secondary interactions](results/checked-summary.json).

## Direction matters, but retention is not the explanation by itself

The norm-rule/raw-direction control remains close to raw behavior: Clean rare
accuracy averages 51.53%, Diffuse 50.20%, Shared 49.73%, Sham 54.20%; its
Diffuse majority accuracy is 34.41%. Native loses rare accuracy/CE to it in
all seeds and cells, while retaining the all-seed Diffuse majority accuracy
advantage. This supports a consequential **direction-policy difference** beyond
simply applying this native-derived norm rule to raw gradients. It does not
hold later numerical norms, observer states or actual Adam step lengths equal,
so it is not complete directional mediation or a dismissal of every scalar
regularizer.

The fixed native diagnostics sharpen, and constrain, the mechanism story:

- At update 101, rare probe gradients are already highly coherent:
  0.685, 0.689, 0.678, versus common-digit-3 coherence 0.312, 0.381, 0.387.
  Yet rare mean-action retention is only about 15–19%, versus about 95–96%
  for common 3. Coherent *within-group* gradients need not dominate covariance
  measured on the training mixture after a majority-only history.
- By updates 500/2,000, rare mean-action retention in Clean is approximately
  98–99%; in Shared it is also approximately 98–99%. Poor rare classification
  therefore cannot be explained solely as its useful gradient remaining
  outside the retained span. Exposure, competing gradients, margins and Adam
  history remain plausible contributors. These are hypotheses, not identified
  causes.
- Actual Adam movement has roughly 62–86% of its **squared norm** outside the
  saved native span across the fixed anchors. Projected input gradients do
  not imply projected parameter displacement. At Clean update 2,000, the rare
  probe's finite CE change is adverse in two seeds despite high mean retention;
  in Sham at the same update it improves in all three, despite zero rare
  endpoint accuracy. Local retention, signed utility and accumulated learning
  are different observations.

These are 32-example training probes, not held-out mechanism estimates.
Wrong-example probes are the first changed examples in class-blocked order;
their coherence is not representative of all corruption. Three sampled updates
cannot establish long-horizon mediation. Direct summaries are the
`diagnostic-sSEED-CELL-native32-uSTEP.json` files in the
accepted acquisition directory (artifact not distributed in this public snapshot).

## Strongest fair interpretation and next discriminator

The strongest steelman is **conditional protection of established knowledge
against diffuse interference**, together with an ability to learn shared
structure that can be useful or misleading. It is not “covariance separates
truth from memorization.” The late rare-retention result makes the mechanism
more interesting than permanent minority exclusion, while weakening a simple
top-eigenvector explanation.

The most useful next test is a small, prospectively fixed **batch-composition
intervention with unchanged example counts and targets**. Reuse each accepted
warmup and the same per-block example multiset, but compare ordinary interleaving
with a fixed grouping of rare examples. Keep raw/native/direction-control
policies, a fixed horizon, and rare/common accuracy plus CE. This changes
within-batch co-occurrence and covariance while preserving aggregate exposure;
it does not increase rare loss weight or supply truth labels to the filter.
First establish whether useful rare acquisition can be restored without losing
the interference-protection effect. No sweep or capability benchmark is needed.

A native-specific change would support a sampler/history-sensitive explanation;
similar improvements in raw/control would instead favor generic optimization
or presentation-order effects. No improvement despite strong rare retention
would prioritize signed-update conflict and Adam-history mechanisms. Because
nonlinear training is path-dependent, reordering is not a pure covariance
intervention; the raw controls and explicit limitation are essential. This is
a proposed next discriminator, **not part of this acquisition or executed**. Rarity and
digit-8 difficulty also remain confounded in this first study.

## Evidence boundary

The once-only acquisition completed all 36 trajectories in 258.416 seconds.
The independent audit passed 631,855 checks in 74.703 seconds, including all
756 evaluations and 36 diagnostic anchors. Interpretation here read only JSON;
it did not reload models, tensors or logits, rerun an audit or launch work.

Checked summary SHA256:
`8af7caa3932bdb7a6508e5375e00924d12b279bb3bbf8b1813deb0515b7e29b9`.
It matches the independent audit's summary. Audit receipt:
result.json (artifact not distributed in this public snapshot),
SHA256 `d1b3a47371d8eed4d8fe46ec4b0c1cd3586f788a69cd003b264a7c0b39dd3907`.
Confidence is high about this recorded task's contrasts, limited about
generalization beyond three seeds and one class/task, and low about the causal
mechanism. No alignment, language-model efficacy or optimizer-speed claim follows.
