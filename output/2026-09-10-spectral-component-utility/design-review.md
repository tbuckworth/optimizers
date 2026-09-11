# Independent component-utility design review

Codex — Spectral Optimizer Investigation · 10 September 2026

**Verdict: PASS for implementation preparation, not acquisition admission.**
The proposed measurement asks a real, unresolved local question and can give
either favorable or adverse answers. It does not replace, reopen or weaken
the already completed strong-regime report or paper. A few definitions below
must be made explicit before source freeze; none requires a new scientific arm.

## Independence and inputs

Read the full [protocol](protocol.md), [source inventory](source-inventory.md),
parent inventory (artifact not distributed in this public snapshot), completed
[strong results](../2026-09-10-spectral-strong-augmentation/results.md) and
earlier [augmentation-state results](../2026-09-10-spectral-augmentation-state/results.md).
Applied the research workflow's construct/attribution discipline and the
pre-mortem skill's concrete failure analysis, without new approval gates,
automatic reruns or fail-fast termination. Did **not** read `design-math.md`
or another new review before forming this assessment.

No raw archive, checkpoint, scientific array, neural model or experiment was
opened or executed. No code, fixtures, emails, commits or subagents were made.
This review inspects the design, not implementation correctness or real-data
restoration. Inputs at initial review:

| File | SHA-256 |
| --- | --- |
| `protocol.md` | `88c8d2c0130b1b742b9b27e3072db7fccabd172d3fbd79ee3b8adb115af7e82f` |
| `source-inventory.md` | `5a9064f3556788fd6fdfe63297f267af35f3e8f205813e7a823eabcb4564ff51` |
| `parent-inventory.json` | `194996272cbc04605e4b6a6465fdca6f1b3b669fcd958320868dc02d503e3c1a` |

## Construct validity and incremental value

The completed strong experiment establishes both substantial useful native
learning and an adverse combined native-plus-translation policy. It does not
identify what objective changes a filtered action favors at a shared state.
The earlier clean rank32 diagnostic measures actual directional utility and
between/within-view energy. Energy retention is not signed objective progress,
and those clean warmups are not the larger noisy model's final states.

This design adds a distinct measurement: at fixed native weights, inherited
Adam moments and observer, record how a faithful raw-versus-native action
changes S, F, C and independent clean competence. It preserves the actual
optimizer order and carried state. The result is not predetermined by the
algebraic identity: the identity fixes how L is accounted for, not the signs
of component changes or their relation to clean held-out loss. Treat identity
closure as an integrity check, never the headline scientific finding.

The decomposition is coherent. Since q+epsilon equals the fixed assigned
one-hot target, S+F is assigned-label CE of the per-image mean logits; adding
the log-sum-exp Jensen gap C recovers mean per-view assigned CE. Uniform
replacement gives q=0.1 true-one-hot+0.9 uniform, **not** an 81% replacement
mixture; the approximately 81% wrong fraction is a different quantity. The
finite S_true/S_uniform split guards against treating S as clean competence.
C is label-independent only with the model and view distribution held fixed;
F need not be positive, independent of the trained model, or pure wrong-label
memorization. The protocol appropriately avoids those stronger claims.

The constructive alternative is worth retaining: native might restrict
realization fitting while clean progress survives or improves, despite the
adverse full-policy comparison. Such a result would be a useful conditional
local mechanism clue. Conversely, simultaneous lost C and clean progress
would be consistent with the proposed local account, not proof that C caused
the clean-loss difference. No component-removal intervention or mediation
fraction is measured, and the protocol correctly says so.

## Actions, panels and inference units

- Action batches (first 128 positions) and I (next 256) are disjoint within
  the frozen permutation. R is drawn from the original nontraining reporting
  role, so it is separate from both. This avoids constructing the action from
  the same examples used for its component evaluation. The action gradient is
  correctly not assumed to equal the evaluation gradient.
- Exact enumeration of 25 shifts removes evaluation Monte Carlo noise over
  that discrete support. It does not represent all image invariances, and
  clean transformed labels remain a task-preserving-translation assumption.
- All action inputs are translated. The unaugmented-parent rows therefore
  test onset of augmentation, not an unaugmented next update. That limit is
  explicit and should remain in figures and reporting labels.
- Use all twelve specified native states. The inventory's last seed lists
  `translate` before `none`; implementation must explicitly apply the
  protocol's semantic sort (seed, none/translate, warmup/final), not JSON order.
- The three translated final parents are a defensible fixed primary cell for
  this question. Two batches must first average within parent, leaving three
  seed results, not six independent replicates. Warmup and final parents share
  a trajectory; modes change weights, moments and observer jointly.
- R is nontraining but not a fresh blinded confirmation panel: the original
  reporting data have already informed the research direction. The new panel
  rule is prospective, but this is outcome-motivated follow-up analysis on
  existing parents. No claim of new unbiased efficacy confirmation should follow.

## Definitions to pin before freeze

1. **Metric reduction over views.** H_T CE is clearly defined as mean per-view
   CE, but “both accuracies” should explicitly mean original accuracy and mean
   per-view accuracy, rather than argmax of mean logits. Likewise specify
   whether I's true/assigned and actually-wrong-subset fit metrics use the
   original `(0,0)` view, all-view averaging, or separately named versions of
   both. Mean-logit CE components already have their own role; they must not
   silently replace ordinary CE. All required logits already exist, so this
   is a definition/labeling fix, not a new model evaluation or arm.
2. **Materialized fraction accounting.** At 0.1, compute derivative utility
   using the actual rounded FP32 path displacement, not automatically 0.1
   times the full displacement. Compare finite data-only effects with that
   fraction's materialized decay path. Full-step decay/data accounting remains
   as specified. A rounded geometric path is not a new Adam update or exact
   real-arithmetic rescaling of every coordinate.
3. **Readout binding.** Before acquisition, the launch manifest must bind the
   exact source plan and original warmup/final readout for every parent, not
   only checkpoint receipts. These are already required by the protocol but
   are not yet all present in `parent-inventory.json`. The first-500 prediction
   check must use the matching reporting role and producer chunk semantics.

No extra norm-matched, component-Adam, clean/noisy factorial, intermediate
checkpoint or new training arm is required for this scoped question. The lack
of size matching means that any measured contrast remains a joint
direction/magnitude/history effect, as the protocol already acknowledges.

## Numerical and source requirements

The strict new snapshot adapter is necessary: the known old restore schema
does not match this model. Preload byte checks, restricted deserialization,
explicit topology and optimizer-ID/order checks, ownership aliases, tensor
devices/dtypes, counters, saved gradients/modes, RNG isolation, round-trip
identity and baseline prediction equality are appropriate requirements.
The source inventory correctly calls these unverified implementation work,
not an already successful restoration.

The prescribed tolerances distinguish scalar identities, FP32 gradient closure,
endpoint/projection/Adam algebra and FP64 dots. Fixtures must derive direct L
and its gradient independently of adding S/F/C, otherwise a tautological
closure test would miss an implementation error. Test the one-view degeneracy,
label changes at fixed logits, large common logit offsets and cancellation,
empty wrong-label subsets, warmup-to-first-projection, repair and private-copy
immutability. These are fabricated implementation checks, not permission to
calibrate tolerances on real parents.

Show finite effects, derivative predictions and E−U at both fractions. A tiny
effect at floating-point resolution or a sign that reverses under the
smaller path does not earn a robust local-linear interpretation. The fixed
full-step primary stays primary; the derivative/path diagnostics explain its
limits rather than replacing it. Algebraic saved-array audit cannot certify
autograd/model-forward or complete streaming-observer fidelity independently;
the proposed audit boundary states this correctly.

## Resource feasibility and pre-mortem

The design appears feasible for source preparation. Its dominant 24 saved
rank200 bases cost 4,514,803,200 bytes. The 156 baseline/endpoint readouts have
approximately 60.7 MB of required FP32 I/R logits before metadata. Baseline
gradient vectors, moments, action vectors and endpoints add hundreds of MB,
not another dense parameter covariance. Thus an 8 GiB archive envelope is
plausible, but this estimate is not the exact source-level inventory required
for admission. Existing checkpoint receipts should be referenced, not copied
into every branch.

One parent/action at a time is also compatible in principle with 4 GiB GPU
and 8 GiB host limits. The audit must stream one basis/artifact at a time to
respect its separate 2 GiB cap. A 20-minute acquisition and 15-minute audit are
plausible, not established: real restoration, output writes and repeated
backward passes have not been timed here. Fixed deadlines remain stopping
rules, never an implicit right to retry or shrink after observing results.

| Failure scenario | Likelihood / severity | Early warning | Mitigation within this design |
| --- | --- | --- | --- |
| A view-averaged metric is reported as original or mean-logit performance, changing the meaning of noise fitting or consistency | Medium / high | Same unlabeled “accuracy/CE” key used for multiple view reductions | Pin and label reductions before freeze; check each against the saved logit grid |
| A 0.1 derivative/decay comparison describes an ideal path rather than the actual rounded endpoint | Medium / medium | Dots use scaled full-step vectors while forwards use separately materialized parameters | Save/use actual materialized displacements and fraction-matched decay references |
| A co-occurring C/clean change is promoted into the cause of the full trajectory gap | Medium / high | Report drops common-parent/local qualifiers or treats modes as observer-only interventions | Keep absolute actions, all cells and the existing negative trajectory evidence; no mediation claim |
| Restoration or saved bases exhaust memory/time despite a vector-only budget | Medium / medium | Adapter retains all parents or audit accumulates all 24 bases | Exact byte/object-lifetime inventory and sequential processing before admission; preserve failure without retry |

## Recommendation

Proceed to inert implementation and fabricated fixture preparation, incorporating
the explicit metric/path definitions and receipt/order checks above. Main
should make a separate admission decision only after exact source review,
resource inventory and fixture evidence. No acquisition is admitted by this
review, and no additional favorable outcome is needed to justify reporting
the paper's already established positives and boundaries.
