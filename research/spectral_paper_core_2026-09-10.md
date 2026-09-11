# What Does Temporal Gradient-Subspace Filtering Select?

Codex — Spectral Optimizer Investigation · 10 September 2026

## The answer in brief

The optimizer does something useful. It can preserve classification while
restricting corrupted-label fitting, and a particular directional intervention
supports better rule-task generalization despite a raw alternative fitting
the training set perfectly. These are not merely covariance-reconstruction
results or a failure to learn anything.

The emerging explanation is a **conditional bias in what learning proceeds**,
not a detector of truth. Its behavior depends on which gradient variations
the data schedule and prior training make prominent, how the incoming gradient
interacts with the selected space, and what the base optimizer subsequently
does. A coherent misleading cue can pass this process; a new rare correct
class can remain poorly recognized. Conversely, poor rare recognition does
not establish that all its useful gradient directions were discarded.

The strongest current paper has three connected contributions: a useful
learning tradeoff with safety-relevant boundaries; a controlled change to
that tradeoff by rearranging the same training exposure; and a local
observer-history intervention that separates direction access from useful
delivery. Grokking is complementary constructive evidence, not a speed headline.

## 1. Start with the useful learning phenomenon

The recent boundary experiment uses a small MNIST MLP, three paired seeds and
36 trajectories. A 100-step clean warmup learns nine digits but excludes 8.
Later training contains 50 correct digit-8 examples among 5,000 examples.
The stable global rank-32 filter and raw AdamW share that warmup and subsequent
sampled occurrences. Clean labels, diffuse majority-label corruption, a shared
misleading patch and a matched patch sham are distinct conditions.

| Question | Positive finding | Boundary that must remain beside it |
| --- | --- | --- |
| Does filtering preserve useful behavior under diffuse corruption? | Common-digit endpoint accuracy is 62.01% versus 33.97% raw; the advantage appears in all three seeds. | Both decline from the 87.76% warmup. Common CE does not show the same clear advantage; rare accuracy is 0% versus 51.73%. |
| Does it reduce a learned misleading cue? | The predeclared cue interaction is reduced by 3.358 percentage points in mean, favorable in every seed; patched CE is substantially better. | The shared cue still produces a 94.825-point excess in target-0 predictions. This is partial protection, not rejection of the association. |
| Can restricted directions support learning rather than merely freeze it? | In five modular-addition branches, norm-matched projection beats a specified raw-direction control on every final CE, margin and representation primary. | This is an inherited legacy-state policy comparison, not a universal scalar-control defeat or a measured training-time gain. |

The direct sources are the [checked selectivity report](../output/2026-09-09-spectral-selectivity-boundary/results.md)
and its [raw-linked summary](../output/2026-09-09-spectral-selectivity-boundary/results/checked-summary.json),
and the [grokking direction report](grokking_raw_direction_2026-09-09.md).
Different recipes are not pooled replicates. Common classification, probability
loss and rare recognition answer different questions.

![All three seeds show the difference between preserving common accuracy and acquiring a rare correct class.](../output/2026-09-09-spectral-selectivity-boundary/results/learning.png)

This supports a useful intervention in a specified failure regime. It does
not establish better validation-selected performance, reliable rare learning,
an alignment benefit or a general optimizer recommendation. Those stronger
claims are not prerequisites for acknowledging the observed benefit.

### Augmentation adds a learning boundary

The user's later [augmentation test](../output/2026-09-10-spectral-augmentation/results.md)
adds a useful boundary rather than erasing these positives. Across three fresh
seeds, random masking improves raw rare accuracy/CE in Clean, Shared and Sham,
but worsens native rare CE in each seed/cell. Targeted cue erasure reduces
some registered cue contrasts while harming Shared common and rare competence.
Part of the contrast reduction comes from greater wrong-class bias without a
cue. Native nevertheless retains lower patched-image loss and higher patched
common accuracy than raw in every seed under every mode. This is a learning
tradeoff, not complete semantic rejection or a failure to do anything useful.

The conceptual distinction (artifact not distributed in this public snapshot)
is frequency versus reliability: erasing half the cues lowers occurrence
frequency but leaves surviving inserted cues perfectly associated with their
assigned wrong target. Reducing an isolated toy cue covariance is not enough
to predict improvement in the evolving optimizer. All72trajectories and their
saved-logit check are complete; no further mask tuning is selected.

### Ordinary augmentation separates local selectivity from practical benefit

The subsequent clean, all-class translation study improves raw accuracy
92.93→95.65% but worsens the stable rank32 endpoint 87.03→82.35%. With 80%
fixed actually wrong labels, translation improves raw 24.65→62.01% and worsens
native 53.01→48.64%. All three seeds agree in accuracy and CE in each study.
Yet native genuinely improves both metrics after its own warmup in every
seed/mode and has useful unaugmented noisy-label protection. The practical
finding is that this combined recipe loses, not that it does no useful learning.

A six-clean-parent diagnostic identifies a favorable equal-actual-step-size
native direction for translated inputs/readouts after unaugmented warmup,
alongside all-seed adverse original-readout contrasts. Preferential retention
of mean/between-image over within-view variation is separately measured, not
semantic selection or trajectory mediation. A fresh four-view observer test
then improves translated accuracy by 1.88/1.44/1.52 points over native1, with
favorable CE in all three seeds and actual progress from warmup. Primary
original accuracy changes +0.16/+0.70/−0.26 points, with mixed CE and almost
unchanged mean loss; raw4 still reaches 95.63% versus observer4's 81.59%.

These bounded gains belong beside the negative practical result. Historical
stable rank200 reached 78.77% versus matched raw AdamW's 38.93% at seed42,
with different model, normalization, data roles and exposure. The completed
[strong bridge](../output/2026-09-10-spectral-strong-augmentation/results.md)
now tests that larger configuration with translation and a separate validation/
reporting split; it is not an exact historical reproduction.

Unaugmented native reaches 79.753% reporting accuracy versus raw 32.000%,
learning +43.053 points beyond its own update100 warmup with lower CE in every
seed. Its validation-selected accuracy remains better, 82.973% versus 72.487%,
but selected CE is worse in every seed. Translation alone reaches 84.973%,
whereas native+translation reaches 66.820%: every seed favors augmented raw
on accuracy and CE at fixed final, own validation-selected and fixed late
readouts. Native+translation still learns +40.700 points after warmup; useful
learning does not imply a practical gain over the ordinary control. It also
fits the actually wrong assignments more than either single intervention,
not less, by both subset metrics in every seed. Thus the adverse augmentation
result is not confined to small rank32, but its mechanism remains unidentified.
The [sequence section](spectral_augmentation_sequence_2026-09-10.md) retains
all four absolute endpoints, every primary sign, sources, costs and limits.

## 2. The mathematical steelman is selective continuation

There is a precise favorable case in which directional restriction achieves
more than one global stopping time. In fixed-design linear regression, take
an isotropic Hessian λI and noisy training optimum w = θ* + ζ. The clean
target θ* and corruption component ζ are nonzero and orthogonal. Define
A = ‖θ*‖² and B = ‖ζ‖². Every global scalar endpoint αw has clean risk

<pre>
R_scalar(α) = (λ/2) [ A(1−α)² + Bα² ]
min_α R_scalar(α) = (λ/2) AB/(A+B) > 0.
</pre>

An ideal fixed orthogonal projector Π that retains θ* and removes ζ instead
allows useful learning to continue while preventing that corruption component:

<pre>
θ_(t+1) = θ_t − η Π ∇L_train(θ_t),    θ_0 = 0,  0 < ηλ < 1
R_projected(t) = (λA/2) (1−ηλ)^(2t) → 0.
</pre>

The comparison is strict under these assumptions. It is not a theorem that
the native optimizer discovers Π, nor a superiority result against a method
already told the useful subspace. The earlier derivation (artifact not distributed in this public snapshot)
includes warmup and direction-error bounds, a batch-heterogeneity construction
that exposes the useful direction, and the failure of full-batch isotropic
gradient covariance to discover it. The algebra is existing conditional theory,
not a new theoretical contribution claimed by this synthesis.

The scientific task is to determine when this promising separation is
realized in actual learning. A negative result on one neural recipe does not
invalidate the construction; a positive toy does not establish the neural bridge.

## 3. What the implemented statistic actually sees

The native observer uses a running mean and centered innovations of consecutive
batch gradients. It does not inspect whether an example is truthful. A
constant component is absent from ideal centered covariance; a finite EMA
initialized elsewhere can retain transient innovations. A changing useful
feature can create a large covariance direction, as can a changing shortcut.

At a fixed model, let D contain the gradient of every indexed training example
as a column, and let a batch's weight vector be w_t. Then

<pre>
g_t = D w_t
mean(g_t) = D mean(w_t)
Cov(g_t) = D Cov(w_t) D^T.
</pre>

This identity explains why equal expected objectives do not fix the selection
statistic: the sampler changes Cov(w_t). It concerns fixed-parameter gradient
covariance, not an exact description of a moving, centered, finite-memory,
rank-truncated observer. With two deterministic groups it reduces to

<pre>
g_t = μ_common + q_t (μ_rare − μ_common)
Cov(g_t) = Var(q_t) (μ_rare − μ_common)(μ_rare − μ_common)^T.
</pre>

The selected direction is a contrast between groups, not automatically the
rare group's useful direction. Correctness is absent from the expression.
Frequency, gradient magnitude, competing modes and temporal estimation all
matter. The [reviewed batch-composition analysis](../output/2026-09-09-spectral-next-mechanism/batch-composition-theory.md)
provides the general within/between-group decomposition and its native-estimator
qualifications. These are standard identities used to motivate a test.

## 4. The sampler intervention makes that explanation testable

The completed three-seed batching experiment keeps the exact indexed example
occurrences, repetitions and assigned labels identical inside each 50-update
block. It either spreads rare examples across batches or concentrates them
in fewer batches. Neither schedule is parameter clustering or extra rare
weighting. Model and optimizer histories subsequently diverge, as intended.

Under noisy training, grouping raises spectral common accuracy from 47.92%
to 56.64%, with gains of 9.20, 10.07 and 6.91 points. The native-versus-raw
schedule interaction is favorable in every seed. This is an additional useful
effect from how identical exposure is presented. Rare accuracy nevertheless
stays zero in every noisy native branch. Clean rare gains are uneven and
cost common performance. An apparently favorable noisy rare interaction
comes from raw deteriorating—not spectral rescuing the rare class.
[All paired outcomes and audit](../output/2026-09-10-spectral-batch-composition/results.md).

![All seed schedule effects show both useful noisy common preservation and the absence of a general rare-learning rescue.](../output/2026-09-10-spectral-batch-composition/results/schedule-effects.png)

This identifies a sampler–policy interaction. It does not by itself establish
that covariance, rather than base-optimizer batching effects or their
interaction, mediates the endpoint. That motivated one local intervention.

## 5. Direction access, delivered signal and actual learning are distinct

Two observer copies watch differently grouped histories at the same frozen
model, then act on identical block-mean gradient bytes. Each delivered action
takes one step from the same copied Adam state. Raw and explicit-zero actions
are active references. The three parents are reused early states, not fresh
replications of the long-run experiment.

For a true-label training-probe gradient q, delivered action h and actual
parameter displacement Δθ, the diagnostic stages ask different questions:

<pre>
Access:        ‖Πq‖² / ‖q‖²         Can this space represent q?
Delivery:      q^T h                 Does this input produce a helpful gradient?
Actual step:   −q^T Δθ               Is the resulting motion locally helpful?
Finite result: L_hold(θ) − L_hold(θ + Δθ)   Did heldout loss improve?
</pre>

The training probe and heldout examples are different populations. Agreement
between their signs is not a Taylor decomposition of one identical loss.

For a fixed orthogonal Π, q^TΠg = (Πq)^Tg. A large ‖Πq‖ alone does not
determine that cross term. The implemented action uses the actual stored V,
and inherited Adam further changes its geometry. A zero supplied gradient
can still cause motion; it is not the same as leaving the model unchanged.

The completed intervention finds a favorable grouped-minus-interleaved rare
loss contrast in all three Clean parents: two greater improvements and one
reduction in damage. Under Diffuse the contrast is adverse in two parents,
despite a large increase in rare-direction access. Saved-vector accounting
finds that those two adverse relative orderings already appear in delivered
alignment and agree with the actual Adam/loss signs. Rare input alignment is
positive throughout, so absence or reversal of that input is not the explanation
at these states. Other control comparisons still reverse order through Adam.
[Observer experiment](../output/2026-09-10-spectral-observer-pathway/results.md);
[signal accounting and all controls](../output/2026-09-10-spectral-observer-signal/interpretation.md).

This is a useful local history pathway and a specific failure of an
accessibility-only account. It does not identify the cause of rare failure
at step 2,000. High later retention, helpful individual bursts and poor final
recognition may coexist without establishing which intervening process is
responsible. The saved-vector calculation had source review and internal
numerical checks, not a new independent scientific audit.

## 6. What is new enough to argue, and what is not?

The paper should argue for this linked empirical characterization of one
temporal-subspace method—not that gradient coherence, projection or
gradient-history filtering are new ideas. [Coherent Gradients](https://arxiv.org/abs/2002.10657)
already motivates reducing overfitting through agreement between example
gradients. [Grokfast](https://arxiv.org/abs/2405.20233) already accelerates
grokking by amplifying slowly varying gradients. They are close precedents;
the latter needs a matched comparison if we choose a grokking-method claim.
The present claim is about conditional learning behavior and a controlled
observer pathway, not defeating either method.

The safety connection is the need to predict which learning an intervention
suppresses. Memorization is not intrinsically bad: [Feldman's long-tail
model](https://arxiv.org/abs/1906.05271) gives conditions where it is necessary
for good generalization. Our rare-class experiment is not a direct test of
that theorem. Nor is a patch cue an emergent-misalignment assay. An actual
behavioral safety benefit at maintained competence remains unestablished.

Novelty and venue readiness are still open judgments. A focused comparison
against the nearest subspace/gradient-coherence work is load-bearing for that
judgment; more local dot products are not a substitute. Existing scalar-control
successes, clean underfitting and legacy/stable contradictions remain in the
full draft and cannot be removed to make the story simpler.

## 7. Next work: finish the claim before adding another study

The current evidence supports the conditional claim without resolving every
mechanism. **Neither a familiarity nor a persistence experiment is an automatic
gate before writing it up.** A methods-complete working manuscript and bounded
nearest-prior check have now been assembled. Review and refinement concern the
specific contribution and its closest comparisons, not another compulsory
local diagnostic. The secondary evidence appendix keeps both positive learning
effects and the limits visible; a working layout is not submission readiness.

The typeset appendix integrates the earlier masking result and the subsequent
ordinary clean/wrong-label, fixed-state and multiview sequence. The working
PDF agrees with the Markdown evidence and methods; its separate theory appendix
connects the loss and covariance calculations without changing those results.
The [unified hypotheses](spectral_optimizer_unified_hypotheses_2026-09-10.md)
retain older scalar/cross-optimizer findings beside these newer boundaries.
The [strong-regime bridge is complete](../output/2026-09-10-spectral-strong-augmentation/results.md),
with a PASS saved-array audit and independent scalar/interpretation review.
The paper retains its substantial native unaugmented learning alongside the
adverse augmented primary; changed split/exposure, selected-metric disagreement
and non-replication audit scope remain explicit. It does not establish which
gradient components explain the deficit or select another experiment.

As another optional, unselected learning-history question, a useful design is
early-versus-late exposure of the same complete indexed **(image, assigned-label)**
occurrences, with timing crossed with native/raw policy. Hold total updates
and the full occurrence multiset fixed, and measure rare/common competence
at the phase boundary and fixed endpoint. This is not yet a registered or
launched study; seeds, timing, switch policy and resource accounting would
have to be fixed before execution.

Moving rare exposure before filtering begins can let Adam learn it first and
filtering preserve it. That would be a useful history-dependent regime, not
proof that filtering acquired the rare feature or that familiarity alone
mediated the effect. Moving images across clean/noisy phases while regenerating
labels would change corruption dose and invalidate equal-exposure attribution.
No demonstrated early competence would make a negative familiarity result
uninformative. Persistence of a one-step perturbation is a different question
and would not establish mediation of the original long trajectory.

The original contribution synthesis did not launch new science. The subsequent
user-prioritized augmentation sequence is completed and reported, without paid
reservation or public submission. This consolidation adds no acquisition. The
overall investigation continues; completed experiment handles must not restart.
