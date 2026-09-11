# Interpretation plan before observer-pathway readout

Codex — Spectral Optimizer Investigation, 10 September 2026.
**Design-only mathematical and reporting review.** Written from the
fixed discriminator (artifact not distributed in this public snapshot),
[main design review](../2026-09-10-spectral-batch-composition/next-design-review.md)
and [companion algebra note](mathematical-readout.md),
without implementation, scientific-array or outcome access. This note adds
no arm, metric-selection rule, success threshold, parent, block or run.

## Exact claim being tested

The intervention changes the observer's history while holding the parent
model, parent Adam state and delivered action input fixed. In schematic form,

    h_s = NativeAction(Observe(O_s,150, g_star), g_star)
    displacement_s = AdamStep(theta_0, A_100, h_s) - theta_0.

Here `s` is Interleaved or Grouped. A verified difference in actual finite
useful loss can identify a **local pathway through the full native observer
state at this parent and input**. It need not identify a population-covariance
pathway, a particular retained direction, or the cause of the completed
step-2000 endpoint difference. Changing the observer is the intervention;
direction, norm/gain and their interaction with the inherited Adam state can
all carry its effect.

There are three reused, outcome-informed seed parents, each tested under two
label cells. There are six parent/cell cases, not six independent parents.
The 12 native readouts, repeated probes and shared zero references do not
increase the replication count. Keep Clean and Diffuse separate and report
every seed before their mean, sample SD and SE. This is a newly fixed
diagnostic on old states, not fresh confirmation of an optimizer benefit.

The common input `g_star` is a schedule-symmetric assigned-target **block-mean
gradient at fixed parameters**. It is not a new next minibatch and is not
label-neutral. An effect on that deliberately smoother input need not carry
over to ordinary batch-64 delivery. The true-label training probes are oracle
diagnostics only; their helpfulness or retention must never be presented as
an action-selection rule available to the optimizer.

## Observer history is broader than covariance

Occurrence conservation gives the unweighted block-mean identity at fixed
parameters, subject to the fixed numerical admission check. It does not make
the observer's exponentially weighted mean or centered history equal.
For an illustrative constant-rate EMA with the same inherited mean,

    mu_G,50 - mu_I,50
      = (1 - beta) * sum_t beta^(50-t) * (g_G,t - g_I,t).

An unweighted zero sum does not force this weighted sum to vanish. This
identity explains a possible order effect; it is not an assertion about the
implementation's exact centering or bias-correction formula. The actual
observer also retains inherited state, temporal weighting, evolving centering,
finite-rank truncation and numerical order effects. Its final action can
change in both direction and gain. Different native norms are part of the
fixed intervention, not a protocol flaw, but prevent calling it a norm-matched
direction-only comparison.

Conversely, a null does not establish equal population covariances. Different
states can produce the same action on one input; different actions can produce
nearly indistinguishable actual steps or losses through Adam. Distinguish
these observational levels before naming a mechanism.

### Companion algebra check and required regularity qualification

Conditional on the mean recurrence stated in
[mathematical-readout.md](mathematical-readout.md), the weighted-sum expansion
is correct. Swapped order `[y,x]` minus original `[x,y]` gives
`(1-beta)^2 (x-y)`, with the stated sign. At beta = 0.99, beta^50 is about
0.605: the inherited mean's coefficient, not a fraction of retained span,
variance, parameter movement or useful information.

Adding the same final `g_star` to both recurrences multiplies their mean
difference by beta. Their post-mean innovations therefore differ by the
negative of that updated mean difference. This establishes neither contraction
nor persistence of the **action** difference: inherited covariance factors,
centering, eigenvalue thresholds, rank truncation and the input's orientation
also matter. The nominal covariance recurrence is useful as the stated
pre-truncation model; it must not be promoted to an exact untruncated
full-history estimator or independently verified implementation identity.
No implementation source was read in this review.

The one required qualification to the displayed Taylor line is mathematical:
**differentiability alone gives a remainder `o(||Delta||)`, not generally
`O(||Delta||^2)`.** A quadratic remainder needs additional local regularity,
for example a Lipschitz gradient or bounded Hessian on an appropriate
neighborhood containing the step segment. A finite step across a ReLU boundary
cannot automatically inherit a fixed-activation smoothness argument. Keep
the measured finite loss primary and qualify the linear approximation; this
does not require any extra scientific measurement or alter the design.

Finally, describe batch-mean conservation as a sum of **per-occurrence
gradients**, avoiding language that could imply statistical independence of
repeated occurrences. Computational separability is the required premise;
independent examples are not. The common-state Adam mapping and its
potentially nonzero zero-input displacement are correctly distinguished in
the note, conditional on preserving the inherited moments, decay and clock.

## Sign map for useful loss

Use the protocol's convention throughout: `U[j,a] = L[j,before] - L[j,after,a]`.
Positive is helpful. Define `D_j = U[j,Grouped] - U[j,Interleaved]`, so positive
means grouping is relatively better. This is the negative of the corresponding
Grouped-minus-Interleaved after-loss difference; plot labels must not silently
reverse that sign.

The following map applies separately to held-out and training true-label CE.
Only a held-out result supports a claim about the measured held-out loss;
training-probe utility alone cannot supply that conclusion. Common CE is the
nine-class macro average, and rare CE is digit 8. Preserve both rather than
letting a balanced total hide common harm.

| Rare `D` | Common `D` | Permitted relative interpretation |
| --- | --- | --- |
| + | + | Grouping is better on both measured losses at this case; check absolute utility and controls before calling either improvement useful. |
| + | − | Grouping favors rare loss at a common-loss cost. |
| − | + | Grouping favors common loss at a rare-loss cost. |
| − | − | Grouping is worse on both measured losses at this case. |
| + | 0 | Relative rare benefit; no resolved common contrast. |
| 0 | + | Relative common benefit; no resolved rare contrast. |
| − | 0 | Relative rare harm; no resolved common contrast. |
| 0 | − | Relative common harm; no resolved rare contrast. |
| 0 | 0 | No resolved local loss contrast at this parent/input. |

Here `0` covers exact equality or an effect that cannot be resolved against
the already fixed numerical checks; it is not a newly introduced scientific
equivalence band. Retain unrounded values and any tiny signed difference.
Do not turn an arithmetic tolerance, zero rounding at display precision or
three matching signs into an equivalence claim or significance test.

For **each group separately**, overlay the absolute-utility map:

| Grouped absolute utility | Relative `D` | What it establishes |
| --- | --- | --- |
| positive | positive | Grouped improves this loss and improves it more than Interleaved. |
| nonpositive | positive | A favorable relative contrast can be reduced damage; it is not positive Grouped learning. |
| positive | nonpositive | Grouped helps, but is no better than Interleaved on this loss. |
| nonpositive | nonpositive | No positive Grouped utility on this loss, and no relative benefit. |

Within each row, retain Interleaved's own absolute utility. For example, two
helpful actions and two harmful actions can have the same signed relative
contrast. Apply the same distinction to rare/common joint claims. A mean
dominated by one seed is a heterogeneous result, even if its average is large;
opposite seed signs must remain visible rather than being resolved by pooling.

Raw and explicit-zero readouts add essential context without changing the
primary contrast:

- Better than Interleaved but worse than raw demonstrates a relative observer
  effect without establishing superiority over the common unfiltered action.
- Better than raw on both absolute useful losses, consistently across the
  three cases within a cell, would be stronger local evidence. It remains
  restricted to this parent, input, inherited Adam state and one step.
- Better than the zero-gradient action reveals added local benefit relative
  to that specific carried-state reference. Worse than zero can mean that
  an incoming action reduces improvement which the inherited optimizer would
  have produced anyway, even when its own absolute `U` is positive.
- Zero gradient is not zero displacement. Carried moments, bias correction,
  second-moment decay and weight decay still act. `U[a] - U[zero]` is a
  conditional action contrast through Adam, not a separately identified
  fresh-gradient contribution or a fraction of a long-run mechanism.

Accuracy may remain unchanged while CE improves; CE improvement is real local
loss progress but not demonstrated recognition rescue or calibration. Lower
wrong-assigned training fit is not by itself useful learning. Preserve any
disagreement between finite CE, accuracy and assigned-target fit.

## Map from observer geometry to actual utility

Use the pre-inclusion state only to locate where an observed contrast is lost
or retained. The primary delivered action comes from observer 151 after one
common-input observation.

| Observed pattern | Interpretation and boundary |
| --- | --- |
| Pre-inclusion actions differ; post-inclusion actions agree | The common final observation erases the tested historical action contrast. Pre-inclusion separation is not a canonical-delivery benefit. |
| Pre-inclusion actions agree; post-inclusion actions differ | The common observation reveals different responses of the inherited history states. Equal action on the pre-inclusion input did not establish equal observers. |
| Both action contrasts persist; actual steps and useful losses differ | The full observer history has a local delivered pathway. Inspect sign, gain and raw/zero comparisons before calling it favorable. |
| Actions differ, actual steps are unresolved | Adam mapping or numerical resolution can attenuate the action contrast. This is not evidence that observer geometry was equal. |
| Actual steps differ, useful losses are unresolved | A parameter-space effect did not yield a resolved benefit on the measured losses. Other inputs or later trajectories remain untested. |
| Post-inclusion actions agree, but cloned-state steps or logits materially differ | An integrity or determinism problem, not an observer finding; investigate against fixed checks without relabeling it a scientific effect. |
| Span/probe retention improves, finite rare loss worsens | Retention alone did not establish usefulness. Preserve the adverse finite effect and inspect the signed actual displacement. |
| Finite loss improves despite little change in the reported retention | A useful full-state action difference need not be explained by that retention summary. Do not invent a missing geometry effect. |

Exact equality is stronger than near equality: identical delivered tensors and
identical deterministic Adam/model states should yield identical readouts.
Near-equal tensors need the fixed arithmetic tolerances; similarity must not
be used to excuse a materially different loss. Finite measured CE is the
direct local outcome. The signed `-probe_gradient dot actual_displacement`
quantity is a first-order diagnostic; disagreement can reflect finite-step
curvature or the probe/population distinction, and is not automatically a
calculation failure. Compare it only to the corresponding probe loss/input.

## Fixed useful plot plan

1. **Absolute useful loss with all four action references.** Four panels:
   Clean/Diffuse crossed with rare/common. Show the three seed points for
   Interleaved-native, Grouped-native, raw and zero-gradient Adam, short saved
   mean bars and a horizontal `U = 0` reference. Join the two native points
   within each seed. Label the y-axis `CE before − after (nats/example)` and
   explicitly label the zero arm `zero-gradient Adam`. Plot held-out and
   training populations separately, never combine them into extra replicates.
   Use comparable scales within the same group across label cells where
   legibility permits. Do not use unsigned magnitudes or a log scale that
   hides harm. Tables retain sample SD/SE and all paired contrasts.
2. **What survives the common final observation.** Show the saved
   Interleaved–Grouped action-distance and cosine diagnostics before and after
   inclusion, separately by label cell, with every seed linked across the two
   stages. Include the saved action norms beside this panel or in its compact
   table so an apparent directional effect cannot conceal gain. Label stages
   `observer 150: descriptive` and `observer 151: delivered`. Zero-norm
   cosine cases are undefined and must be marked, not silently assigned a
   favorable value. This figure describes observer accessibility; the first
   figure establishes the measured local loss consequence.

The first figure is the report lead even if a geometry plot looks more
favorable. State prominently: three reused, outcome-informed parents; a
common block-mean input; one copied-state Adam step; no new endpoint result.
No figure treats the two cell labels or the physical/logical readout counts
as independent statistical replication.

## Specific admission and implementation pitfalls for main to check

- **Writable aliasing:** observer copies, state dictionaries and saved tensors
  must not retain writable views into parents or siblings. A shallow copy or
  detach alone is insufficient. Hash the full reference before/after gradient
  measurement, observer feeding and each action readout; clone model, moments
  and parameter-group state independently for every physical step. Evaluating
  a model must not alter buffers, mode, RNG or stored `.grad` fields.
- **Frozen-gradient premise:** fixed parameters alone do not guarantee that
  batching preserves the average gradient if the model/loss uses changing
  random draws or batch-dependent statistics. Check the actual deterministic
  per-occurrence-loss premise and unchanged mode/buffers/RNG. Use the exact
  occurrence IDs, repeats, targets, batch means and saved FP32 normalization;
  a changed reduction silently changes exposure weight.
- **Two clocks:** keep observer `100 → 150 → 151` separate from Adam
  `100 → 101`. Preserve the inherited Adam moments and its own bias-correction
  clock; do not advance Adam during fixed-gradient acquisition or observer
  updates. Do not force a shared-clock validator to accept a falsely relabeled
  state. The combination is a diagnostic envelope, not a normal continuation.
- **Current-gradient inclusion:** the primary copy observes byte-identical
  `g_star` exactly once. Do not mutate observer 150 while obtaining its
  descriptive readout, accidentally project before inclusion, or use a
  wrapper that observes a second time. A common final input does not imply
  a common final observer state.
- **Mean symmetry and casts:** compute both stream means in FP64, enforce the
  pre-fixed `D` bound, form their symmetric mean and cast once to the shared
  FP32 action tensor. Record the actual difference and bound even on passage.
  Tolerance passage is a floating-point admission result, not equivalence of
  observer means, spectra or actions. Do not silently choose one schedule's
  numerical mean as the reference.
- **Adam action semantics:** deliver the saved action directly, without a
  second gradient/observer update, normalization, moment reset or altered
  saved optimizer flags. Explicit zero must be a real zero tensor rather than
  `None`, since skipped-gradient semantics differ. Adam is coordinatewise and
  history-dependent; projected incoming actions do not guarantee in-span
  parameter displacement or monotone useful descent.
- **Shared zero reuse:** reuse the three physical zero steps only when their
  complete parent, input and zero-action bindings agree across cells. Bind
  after-state/logit hashes explicitly. Metrics involving assigned or actually
  wrong labels remain cell-specific even when predictions are identical.
  Shared zero references must not be counted twice as independent evidence.
- **Before/after comparability:** verify the accepted parent predictions'
  model, image order, labels, normalization and environment bindings before
  reuse. Do not silently call cross-environment predictions bitwise identical.
  Keep held-out true-label metrics distinct from assigned training labels and
  from the small true-label oracle probes.
- **Failure versus null:** an input-pin failure, mean-identity violation,
  aliasing, wrong clock, nonfinite quantity, failed arithmetic check or resource
  failure is an execution/integrity result. It cannot be converted into a
  zero scientific effect or an unqualified partial-roster conclusion. Preserve
  the existing once-only and no-retuning rules.

## Balanced concluding language, chosen from outcomes

A favorable, absolutely helpful and reproducible within-cell contrast would
strengthen the steelman: under controlled model/Adam state, changing the
observer's incoming history can make this common action more useful. The
restriction includes gain, direction and inherited-Adam mediation; it does
not identify semantic selection or explain the later noisy rare failure.

A tradeoff, null or adverse result is also discriminating. A tradeoff reveals
which measured group pays for the other at this parent. A null after canonical
inclusion weakens this early-parent/common-input route even if pre-inclusion
geometry changes. An adverse result contradicts a locally favorable grouping
story at these cases. None rules out later states, ordinary minibatch inputs,
model–observer co-evolution or a different history channel; those remain
untested possibilities, not automatic explanations of an unfavorable result.

No empirical conclusion has been selected in this document. Main should keep
the strongest positive and negative readings bounded by the same frozen
parents, readout, controls and numerical checks.
