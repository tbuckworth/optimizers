# What the spectral optimizer is doing: a unified hypothesis report

Codex — Spectral Optimizer Investigation · 10 September 2026

## Executive view

**The optimizer does something useful.** In selected noisy-label tasks it protects previously acquired classification performance while ordinary continued training increasingly fits corruption. In a controlled modular-arithmetic continuation, restricting the gradient direction improves held-out learning beyond the tested raw-direction scaling rule. These are positive learning results, not merely failures to beat a baseline.

**The most coherent explanation is a history-dependent change in the allocation of learning.** A learned space, the gradient mean, the speed and magnitude of response, and the base optimizer's memory jointly determine which changes remain easy. That allocation sometimes favors the structure we want; it can also obstruct useful rare learning and retain a coherent wrong shortcut. Confidence is high in these bounded policy effects, moderate in this unifying account, and low in any uniquely identified long-run mediator.

Two earlier simplifications should be retired. The method is not established to be “just a moving average”: directional controls matter in grokking, and constructive examples demonstrate directional advantages. Nor is it established that covariance eigenvectors distinguish generalizable features from memorized facts: the rare-class and wrong-cue experiments directly constrain that interpretation.

The paper's strongest present contribution is **understanding conditional control of learning**, with useful protection, costs, and a controlled local history intervention. It is not a general training-speed result or a demonstrated alignment defense. The user's ordinary-augmentation question now has completed clean and fixed-wrong-label comparisons: augmentation helps raw AdamW but worsens the small stable rank32 endpoints, despite genuine native progress after warmup. A saved-state directional positive and a fresh multiview secondary gain do not overturn that practical finding. The [completed strong-regime bridge](../output/2026-09-10-spectral-strong-augmentation/results.md) preserves the larger rank 200 positive—36.70→79.75% reporting accuracy after warmup—but raw translation reaches 84.97% versus 66.82% for the combination. All three seeds favor raw translation on accuracy and CE at fixed, validation-selected and late readouts. Thus useful learning is real, while an additive augmentation benefit is absent in both tested regimes. The [sequence](spectral_augmentation_sequence_2026-09-10.md) and working paper (artifact not distributed in this public snapshot) now include this result; the earlier delivered unified-report PDF remains unchanged.

## 1. Scope and how to read the evidence

The working paper's Appendix E now collects the augmentation-objective and
feature-geometry calculations discussed below, including the signed-utility
and signal-retention boundaries. These remain conditional theory. The
[integration record](../output/2026-09-10-spectral-manuscript/theory-integration.md)
records the typesetting and citation checks, not a new scientific audit.

Initial report assembly performed no training, model replay, raw-tensor measurement or new literature search. Subsequently linked theoretical supplements include their own bounded primary-source checks and fabricated algebra checks, not new neural evidence. Numerical experimental claims below use completed reports linked to their raw artifacts. Three paired seeds do not become independent replications by crossing several treatments; multiple continuations from one parent remain dependent. Later analyses on reused parents or saved streams are explicitly exploratory. Auditing recorded calculations does not establish external validity.

“Native” refers to the particular recorded filtering recipe, **not one algorithm shared by every study**. Most recent classifier studies use stable global hard rank 32 before ambient-coordinate AdamW. The positive grokking action branches inherit legacy states; the older scalar studies use SGDm and modified mean/history actions. Those distinctions prevent false contradictions.

## 2. The mathematics: what is observed, selected and delivered

### 2.1 Temporal covariance is not signed agreement

Let gₜ be a flattened minibatch gradient over p selected parameters. In the regular idealized update, with decay β and rank truncation Tᵣ:

```text
μₜ = β μₜ₋₁ + (1−β) gₜ
zₜ = gₜ − μₜ = β(gₜ − μₜ₋₁)
Cₜ ≈ Tᵣ[β Cₜ₋₁ + (1−β) zₜ zₜᵀ]
Πₜ = Uₜ Uₜᵀ,  with UₜᵀUₜ = I
hₜ = Πₜ gₜ                    (ideal hard delivery after warmup)
```

The code stores a factor, not a dense p×p matrix. Initialization gives the first nonzero innovation exceptional weight; rejection, recursive truncation, repair and finite precision qualify the recurrence. It updates the observer with the current gradient **before** filtering it. The [source](../spectral_filter.py) implements these facts; its introductory “consistently pointing” and subspace-step wording should not be read as a theorem. Constant signed progress can have vanishing centered variation; large oscillation can have large covariance.

For historical nonorthogonal V = QΣWᵀ, the implemented action is:

```text
V Vᵀ g = Q Σ² Qᵀ g,            not generally Q Qᵀ g.
```

Thus repairing numerical geometry changes the learning action, not just its accuracy as a calculation. Stable code is the maintained implementation; legacy code is needed to reproduce legacy evidence. Storage is O(pr), but basis rotation and the small eigensolve introduce O(pr²) and O(r³) work. “Low rank” does not imply faster training. See the [mathematical reconstruction](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-audit.md).

### 2.2 The strongest version of the shared-feature intuition

At a fixed model, let a random example gradient have group label Z, group mean a_z, within-group covariance Σ_z and overall mean ā. For B independent occurrences:

```text
Cov(g_batch) = (1/B) [ sum_z p_z Σ_z
                      + sum_z p_z (a_z−ā)(a_z−ā)ᵀ ].
```

Here Σ_z is a within-group covariance matrix. Shared features can create aligned group means and therefore a low-dimensional between-group term. That is a concrete route by which the original intuition can work. It requires the useful variation to dominate competing variation; it does not require every semantic feature to correspond to one eigenvector.

For a simplified recurring contribution A·v, present with probability q and otherwise absent, the isolated covariance is q(1−q)A²vvᵀ, divided by B for independent batching. Rarity alone therefore does not fix spectral prominence: amplitude, frequency, within-group dispersion and batch-count correlations all matter. Grouping examples changes covariance even at unchanged total exposure. Model motion and observer memory add further temporal variation. The [batch-mixture analysis](../output/2026-09-09-spectral-next-mechanism/batch-composition-theory.md) spells out these assumptions.

This also explains the semantic limit without dismissing the idea. A useful recurring rule and a recurring wrong shortcut can both generate coordinated gradients. Geometry contains information about the data and task, but the covariance statistic has no external criterion for which learned association ought to be trusted.

The [feature-geometry specialization](spectral_feature_geometry_2026-09-10.md)
provides another affirmative route. At a frozen last layer, uniform predictions
and fresh uniform labels give `Cov(vec G) = E[hhᵀ] ⊗ (I−11ᵀ/K)/K`.
Noise can therefore reveal the representation's feature second moment,
despite zero expected class signal at full randomization. With partial label
replacement and favorable feature scales, an explicit top-direction projection
preserves all expected class signal while reducing nuisance variance. Reversing
those scales discards all signal. Class-contrast degeneracy, nonzero feature
means and parameterization matter. This is conditional mathematics with
K-FAC/Jacobian precedents, not identified semantic clusters, an Adam trajectory
theorem or equivalence between native global rank200 and feature PCA. Together
with the augmentation loss decomposition in§3.7, it sharpens a possible
complementary restriction mechanism without changing the live comparison.

### 2.3 Retaining a useful direction does not guarantee a useful delivered gradient

Let q be the gradient of a desired probe loss and h = Πg. A sufficiently small SGD step has first-order utility ηqᵀh. Write g = a q + r with qᵀr = 0:

```text
qᵀΠg = a ‖Πq‖² + (Πq)ᵀr.
```

Even if projection retains more of q, its signed interaction with the rest of the incoming gradient can change. Accessibility, ‖Πq‖²/‖q‖², omits that interaction. It is not a useful-gradient fraction. This identity accommodates the observer result: grouping increased rare-direction access under both label regimes, but improved relative rare delivery in all three Clean parents and only one of three Diffuse parents. This is an explanatory accounting framework, not a newly measured mediation decomposition. See the [saved-vector interpretation](../output/2026-09-10-spectral-observer-signal/interpretation.md).

For AdamW the actual displacement adds another stage:

```text
bₜ = β₁ bₜ₋₁ + (1−β₁) hₜ
vₜ = β₂ vₜ₋₁ + (1−β₂) (hₜ ⊙ hₜ)
Δθₜ = −η b̂ₜ / (√v̂ₜ + ε) − ηλθₜ.
```

Carried momentum, coordinatewise division and decay need not remain in the current learned space. In the fixed-model observer study, 84.84–95.39% of squared actual displacement lay outside it. Zero input was an **active Adam step**, not frozen parameters. The [complete-state report](../output/2026-09-10-spectral-observer-pathway/results.md) measures these distinctions. Nevertheless, two adverse Diffuse *relative* rare effects were already present before Adam; Adam cannot be their first point of appearance.

### 2.4 Why mean restoration and smoothing are related but not equivalent

An exponential moving average is μₜ = βμₜ₋₁ + (1−β)gₜ. A finite-window simple moving average instead assigns equal weight to the last W observations. The scalar experiments mainly concern exponential temporal response and momentum, not a universal result about every moving average.

A mean-preserving spatial action is:

```text
h = μ + Π(g−μ) = Πg + (I−Π)μ.
```

It supplies the current gradient in the retained space and a historical mean outside it. It is neither hard projection nor purely scalar smoothing. The mean can contain useful adaptation and unwanted realization-fitting information together. With moving Π, different momentum transports and normalization, even matching constant-input gain does not match transient response, realized step size or effective regularization. The [I15 mathematics](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/mathematical-interpretation.md) and I17 moving-gain caveat (artifact not distributed in this public snapshot) separate these cases.

## 3. What the learning experiments actually establish

### 3.1 Useful protection, sometimes coupled to foregone useful adaptation

The repeated noisy-label finding is real, but endpoint and attainable performance are different claims. In [I11](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-011/results.md), longer continuation eventually favored filtering on clean CE in every seed mean after an earlier adverse comparison. Much of the benefit was raw deterioration while filtered performance was preserved; a general advantage over hindsight-selected raw stopping was absent.

[I10](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-010/results.md) removes a tempting explanation: replacing persistent wrong labels with soft targets or fresh redraws did **not** restore native adaptation. Adding the filter to a different, late raw-trained parent could help, however. [I12](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-012/results.md) shows that fresh Adam moments do not remove the deficit; apparent advantages of zeroing second moments largely reflected damaging startup shocks, not a successful remedy.

Restoring the mean in [I13](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-013/results.md) recovered substantial genuine soft/redraw progress—aggregate accuracy gains of 27.28/23.56 points over native—but also near-raw fitting of persistent corrupt labels. That is evidence for mixed useful/unwanted information in the suppressed response, not evidence that all useful learning was already preserved.

The [fresh selectivity study](../output/2026-09-09-spectral-selectivity-boundary/results.md) makes the cost concrete. Under diffuse corruption, common-digit accuracy was 62.01% for native versus 33.97% for raw, but newly introduced rare-digit recognition was 0% versus 51.73%. Common CE was essentially tied in mean with mixed signs. Rare CE still improved from warmup, so “zero recognition” is not “no learning.” A shared wrong cue remained highly learnable, alongside favorable bounded cue contrasts and patched-loss protection. This study has three paired seeds, not one replication per cell.

### 3.2 Other optimizers and the corrected moving-average conclusion

[I14](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-014/results.md) found fixed-label endpoint protection with **SGD momentum as well as AdamW**, but not plain SGD; clean comparisons were adverse across the tested bases. Adaptive second moments are therefore **not necessary** for that protection. The base optimizer still matters through history, movement and the raw trajectory available to protect.

[I15](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-015/results.md) weakens a stronger “old momentum simply substitutes for the removed mean” account: projecting old momentum did not abolish protection. Explicit mean restoration under projected history allowed useful progress with continuing clean underfitting; unrestricted mean restoration admitted raw-like wrong-label fitting.

In [I16](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-016/results.md), seven of eight registered selected mean comparisons favored the scalar temporal family. The simplest k=0 arm also beat the spectral endpoint on both accuracy and CE in every seed for both target regimes, while making real progress.

But [I17](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-017/results.md) normalized spectral stationary gain and substantially improved its fixed-target endpoint accuracy, **53.633% → 62.527%**, reversing the earlier k=0 endpoint ordering in all four mean metrics. Six of eight joint-selected means still favored the scalar family, including all four fixed-target selections. These studies reuse a small three-seed panel and offer the scalar family 24 (k,horizon) choices versus six spectral horizons; time-varying dose and history are not fully matched.

The correct conclusion is that **temporal response and scale are serious explanatory competitors, and useful simple controls exist**. Neither “a moving average is generally better” nor “the same performance proves the same mechanism” follows.

### 3.3 Grokking supplies positive direction-specific evidence

The [fresh five-seed confirmation](grokking_stable_confirmation_2026-09-08.md) separates the older and repaired implementations. Mean sustained 90% attainment was 4,040 updates for AdamW, 2,810 for legacy, and 3,980 for stable. Legacy's improvement was consistent; stable's mean advantage was small and variable. Both filters took longer measured time in every paired run. Numerical repair did **not** establish an extra efficiency gain.

The [inherited-state action test](grokking_action_mechanism_2026-09-09.md) starts all branches from complete legacy step-1,500 states. Plain orthogonal projection lost performance; restoring the incoming native gradient norm to the orthogonal projection improved all five final comparisons on accuracy, loss and margin versus native. A subsequent [raw-direction control](grokking_raw_direction_2026-09-09.md), with the same functional norm rule recomputed on its own trajectory, lost every final primary comparison to the projected action, despite perfect training accuracy.

This is a useful **conditional directional policy effect**, beyond that particular scalar rule. It does not show that every magnitude-only optimizer must fail, that legacy anisotropy is necessary, or that this learned orientation uniquely selects rule features. Earlier timepoints were mixed and not all branches grokked. The [common-state one-step measurement](../output/2026-09-09-spectral-function-response/interpretation.md) also found local off-span suppression helpful for held-out loss while norm amplification partly reversed that benefit. A single local sign need not predict a thousand-step trajectory ranking.

### 3.4 Equal exposure and observer interventions locate a real history pathway

In the [batch-composition study](../output/2026-09-10-spectral-batch-composition/results.md), grouping conserved example occurrences and labels within each 50-update block. Native noisy common accuracy rose 47.92% → 56.64%, with a favorable native-minus-raw interaction in every seed. Rare recognition stayed zero. A favorable rare interaction arose because raw deteriorated, not because native rescued it.

The [observer-only discriminator](../output/2026-09-10-spectral-observer-pathway/results.md) then held model, Adam and action input fixed while changing the observed ordering. Clean grouping improved relative rare held-out loss in all three reused parents, with two absolute improvements and one reduction in damage. Diffuse grouping improved that relative contrast in only one parent, despite strongly increasing rare access. The [saved signal analysis](../output/2026-09-10-spectral-observer-signal/interpretation.md) locates the same three-versus-one sign pattern in delivered alignment before Adam.

Thus historical observations can causally change a useful next step. This goes beyond a covariance correlation on differing trajectories. It does **not** isolate population covariance from the full finite observer history, or establish mediation of the earlier final accuracy effect.

### 3.5 One masking study does not settle augmentation generally

The [72-continuation augmentation study](../output/2026-09-10-spectral-augmentation/results.md) answered the user's random-masking suggestion with three fresh paired seeds. Random masking improved raw rare accuracy and rare CE in every seed under Clean, Shared and Sham; native rare CE worsened in every corresponding comparison. Shared rare accuracy changed from 57.27% to 63.80% for raw and 9.20% to 3.40% for native.

Targeted cue masking reduced native's patch-excess statistic but worsened ordinary common and rare competence in every seed. Part of the lower excess came from a higher wrong-class prediction rate on **unpatched** images. Native nevertheless retained better patched common accuracy and CE than raw in every seed under every mask mode. Both sides belong in the account.

The prospective covariance note (artifact not distributed in this public snapshot) explains why this was a plausible but limited intervention. For a constant additive cue gradient, reducing occurrence probability from .1 to .05 reduces its covariance coefficient from .09 to .0475. Yet a surviving inserted cue still perfectly predicts the assigned wrong label. Frequency reduction is not reliability reduction. We did not measure covariance mediation.

There is a distinct, constructive mathematical reason to test general augmentation. At a fixed model, let g(i,T) be the gradient for example i under a label-preserving random transform T, let a_i = E_T[g(i,T)], and let W_i = Cov_T(g(i,T)). With B independently sampled examples and m conditionally independent views per example, averaged before the optimizer observes the batch:

```text
Cov(g_batch, m views) = (1/B) [ Cov_i(a_i) + E_i(W_i)/m ].
```

This is the law of total covariance, not a measured native mechanism. Shared transformation-stable structure could become easier to isolate if nuisance-specific variation is reduced. But single-view augmentation (m=1) can also add high-variance nuisance directions, and changing the augmentation distribution changes the mean objective too. The native moving, centered, truncated observer does not equal this fixed-model covariance. This motivated the subsequently completed separate multiview study below; it was not an extra arm silently added to the first comparison, and its additional gradient-evaluation cost is recorded.

### 3.6 Ordinary augmentation and multiview observation: useful effects, no small-recipe rescue

The [six-clean-state diagnostic](../output/2026-09-10-spectral-augmentation-state/results.md)
finds a favorable equal-actual-Adam-data-step-size native direction for translated
input/readout after unaugmented warmup, all three seeds. Translated inputs on
original readouts favor raw in all three under both warmups even after size
matching. Pre-update mean/between-image energy is preferentially retained over
within-view variation, but these are not semantic components or identified
trajectory mediators. Neither pure size collapse nor universally bad native
direction describes those local results.

In the [fresh multiview test](../output/2026-09-10-spectral-multiview-clean/results.md),
observing four-view mean gradients while delivering projected first-view
gradients improves translated accuracy over native1 by 1.88/1.44/1.52 points
and CE in all three seeds, with actual own-warmup progress. Primary original
accuracy changes +0.16/+0.70/−0.26 points and CE benefits are
+0.011308/−0.001730/−0.009662: mixed seeds and near-zero mean CE difference,
not equivalence. Raw four-view delivery reaches 95.633% versus observer4's
81.593%, and wins both readouts/metrics in every seed. Extra observation views
therefore yield a bounded secondary benefit, not a practical clean rescue or
tested noisy-label protection. The full [sequence section](spectral_augmentation_sequence_2026-09-10.md)
keeps all local signs, costs and historical-regime qualifications.

### 3.7 Augmentation supplies a consistency objective, not just observer noise

The [independently reviewed loss-geometry note](spectral_augmentation_loss_geometry_2026-09-10.md)
adds an exact connection to established augmentation/Bregman theory. Average
view cross-entropy equals the loss at mean logits plus a nonnegative penalty
`E KL(softmax(mean logits) || softmax(view logits))`. At the same fixed model,
the penalty does not depend on assigned labels. Splitting those labels into
their corruption-law expectation and fixed realization yields three terms:
softened-target fit, realization force and view consistency. The latter is
not a truth detector: consistently wrong predictions can have zero penalty.

This makes the restricted-learning hypothesis more precise. The filter might
obstruct useful consistency updates as well as particular label fitting.
Between/within-view gradient covariance and standalone retention do not identify
that pathway; a signed total-update calculation is needed. The note includes
a concrete high-retention counterexample and a proposed common-state diagnostic,
not a new measurement or arm. The completed strong-regime comparison below
does not identify that proposed pathway; the conceptual supplement is not an
explanation established by the unfavorable combined trajectory.

### 3.8 Strong-regime bridge: substantial learning, no additive augmentation benefit

The [fresh three-seed bridge](../output/2026-09-10-spectral-strong-augmentation/results.md)
uses the larger stable rank 200 configuration, standardized pixels and fixed 90%
uniform replacement (about 81% actually wrong), with separate 50k/5k/5k
training/validation/reporting roles and 72 passes. It matches historical 3.6 million
exposures but changes population/repetition, so it is not exact reproduction.
The [once-only saved-array audit](../output/2026-09-10-spectral-strong-augmentation/audit.json)
verifies all validation-only choices before computing reporting outcomes.

Final mean accuracy is 32.000% raw without augmentation, 79.753% native without,
84.973% raw translation and 66.820% native translation. Native improves from
its own warmup by 43.053 points without augmentation and 40.700 with it; both
accuracy and CE improve in every native seed. This rules out explaining these
positives solely as preserving a competent frozen warmup.

The primary combined-minus-raw-translation contrast is nevertheless adverse
in every seed on both metrics at final, validation-selected and late windows:
final accuracy differences −20.44/−19.18/−14.84 points, selected mean −17.320.
Final wrong-subset assigned accuracy is 3.762% combined, versus 2.230% raw
translation and 2.517% native without augmentation. The combined policy fits
wrong assignments more than either comparator by both wrong-subset metrics
in all seeds. Its lower all-assigned accuracy is not extra noise suppression.
Unaugmented native retains a selected-accuracy advantage over raw, but selected
CE is worse in every seed. Claims about early stopping must preserve this split.

This fairly carries the historical protection phenomenon into a prospective
internal-holdout design while extending the practical augmentation boundary.
It is not an identified consistency-gradient, covariance or Adam mechanism;
neither the strong positive nor adverse combination establishes safety transfer.
The reviewed plot report is delivered (artifact not distributed in this public snapshot),
all science handles are consumed, and no additional acquisition is selected.

## 4. Constructive theory and finite performance must remain separate

There are genuine steelmans for learning in selected directions. The [I8 actual-filter toy](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-008/results.md) beats the entire raw-SGD stopping curve in favorable fixed-design regression, with eight paired order seeds under each of two rotations. Frozen learned bases also work. Reversed variation harms and tied variation is mixed: selection helps when variation identifies the right distinction.

The I19 stochastic construction (artifact not distributed in this public snapshot) gives a second, distinct example. A changing signal in one direction and noisy stationary observations in another benefit from different temporal responses. A fixed directional route has asymptotic error .499048, below the .658872 lower bound for the stated class of deterministic shared unit-gain causal linear temporal filters, including signed ones. This is a conditional estimation theorem, not neural convergence or a native-filter guarantee.

The corresponding [finite I19 test](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-019/results.md) loses to strong scalar controls despite learning a favorable direction. [I18](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-018/results.md) has late but not whole-run success in a different affine-drift setting. [I20](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-020/results.md) improves the strong stochastic cell by separating direction-memory and response timescales on the same saved streams: MSE .620737 versus EMA .9 at .694234 and common Kalman at .663074. Weak/no-signal cells fail in every seed. I20 is outcome-informed reuse of 32 bundles, not fresh or neural confirmation.

Together these results preserve a meaningful possibility: spatially differentiated response can beat shared smoothing when the task warrants it. They also show why population eigenvectors, finite tracking quality and useful learning cannot be collapsed into one claim.

## 5. Clustering, J-Lens and application transfer

### A graph can add a grouping assumption, not a semantic oracle

A true clustering method adds an affinity construction and group assignment. For a fixed group-pattern matrix R, an amplitude-aware group action could be R(RᵀR)⁻¹Rᵀg when its columns are independent; equal-amplitude cluster means are a special case. This is a different inductive assumption from arbitrary leading covariance directions. The proportional-group note (artifact not distributed in this public snapshot) is theory, not a measured remedy.

The [neural clustering pilot](../output/2026-09-09-spectral-clustering-mnist/results.md) did work in the bounded preservation sense: noisy endpoint accuracy 48.93% versus raw 29.37%, with less wrong-label fitting. Stable hard spectral reached 50.80% and beat clustering on accuracy and CE in all seeds in both conditions. Clustering's noisy warmup accuracy was already 49.90%; clean learning was restricted and the prototype was slower. Low-memory graph feasibility therefore does not establish that its groups match semantic features or improve the original filter.

### J-Lens is a separate, frozen-model interpretability direction

The latest meaningful-completion pilot (artifact not distributed in this public snapshot) at worktree commit `14353f3` uses 48 authored rows, 32 fit and 16 held out, with only eight unique held-out content pairs. At two residual layers, activation rank-four PCA achieved 14/16 and 15/16 topic classification versus 15/16 for full activations at both layers. Multi-token completion-gradient PCA achieved 6/16 at both layers, with clustering ARI zero; full gradients achieved 9/16 and 8/16.

Meaningful targets did not rescue this residual-gradient clustering route. Activation compression remains interesting, but not better than full-vector classification here. Topic classification is not itself a J-Lens score. The [subsequent blinded readout comparison](../output/2026-09-10-jlens-simple-comparison/results.md) directly tests signed activation eigenvectors: J-Lens matches3/4 held-out extreme pairs versus2/4 plain, with the same tally for secondary random axes. PC1/2 match with both decoders, PC3 with J-Lens only, PC4 with neither. This preserves some useful readouts but establishes neither a reliable spectral advantage nor four distinct concepts. Two fresh same-model raters supply one judgment per item; the outcome-informed, previously inspected single-model panel and uncertified reconstructed random basis remain explicit limitations.

The shared mathematics is covariance eigendecomposition. The objects differ: frozen residual activations or loss gradients, versus temporal **parameter** gradients during learning. No J-Lens run updated model weights or tested optimizer selectivity. A forward Jacobian map on directions is also distinct from the transpose pullback of a gradient covector. These pilots should neither certify nor discredit the optimizer's learning effects.

### Practical and safety transfer remain unproved

The application evidence (artifact not distributed in this public snapshot), summarized with direct source links in application transfer (artifact not distributed in this public snapshot), contains adverse CIFAR adversarial-robustness results, weak/confounded Numerai transfer and inconclusive emergent-misalignment comparisons. Front-loaded adapter loss reduction is not measured wall-clock acceleration. Per-matrix temporal covariance, full-parameter covariance, LoRA rank and the singular spectrum of an accumulated adapter are distinct objects.

The [source-checked safety synopsis](spectral_safety_contribution_2026-09-10.md) states the appropriate ambition: understand controllable generalization, then test actual undesirable behavior with preserved benign competence and useful new learning. Merely lowering bad-answer rates by preventing adaptation would not establish the hoped-for safety intervention. None of this is a prerequisite for honestly reporting today's positive conditional results.

## 6. Ranked hypotheses, with their strongest challenges

The ranks are explanatory priorities, not numerical posterior probabilities. The hypotheses overlap.

| Rank | Current hypothesis | Support | Strongest boundary / discriminating implication |
|---|---|---|---|
| 1 | **History-conditioned restriction redistributes learning, often favoring preservation over adaptation.** | Multi-seed noisy protection, soft/redraw costs, mean-restoration tradeoff, rare-class deficits, source-state reversal and genuine but restricted learning in both small and larger augmentation regimes. High confidence in the policy pattern; moderate in this unifying account. | Cannot explain every positive as merely stopping: I8, grokking and native warmup progress demonstrate useful continuation. Raw augmentation wins over the combined recipe in both tested regimes; the causal mediator and familiarity versus frequency remain uncertain. |
| 2 | **Spatial orientation and temporal response jointly matter, with task-dependent marginal value.** | Grokking direction discriminator, SGDm transfer, I16/I17 scalar/gain results and constructive directional theory. Moderate. | No single scalar rule is ruled out universally; no universal spatial advantage. Fairly tuned temporal controls belong in any future practical-superiority claim, not as a gate before characterizing a phenomenon. |
| 3 | **Group recurrence can make useful features accessible, but usefulness depends on the whole gradient and parent state.** | Equal-exposure batching and the controlled Clean observer pathway; conditional group-mixture algebra. Moderate for local mechanism, low for endpoint mediation. | Rare access can increase while relative useful delivery worsens; shared wrong cues survive. A measure of retained energy alone cannot predict beneficial learning. |
| 4 | **Observer memory and admission dynamics create a familiarity/novelty bias.** | Truncation theory and synthetic delay, common-only warmup followed by difficult rare acquisition, history intervention. Provisional for neural learning. | Rarity and novelty are confounded in current classifier tests; late rare retention is already high. Prediction must distinguish acquisition history from permanently excluding rare directions. |
| 5 | **Base-optimizer history and coordinate scaling modify the intervention.** | Exact algebra and complete-state displacement/zero-input observations. Structural interaction is established. | Adam second moments are not necessary for protection; resets do not restore adaptation; some adverse contrasts begin before Adam. No identified long-run mediation fraction. |

Claims that should no longer lead the research narrative: “top variance means useful semantics”; “all low-variance learning is memorization”; “the native filter implements a softened population objective without adaptation loss”; “adaptive second moments are required”; “a moving average generally wins”; “stable repair improves grokking efficiency”; and “high retention proves the useful rare gradient was delivered.” Each exceeds or conflicts with the completed evidence.

## 7. A coherent next plan

### First: consolidate the completed focused mechanism diagnostic

The formerly proposed small ordinary-augmentation comparison, wrong-label
extension, saved-state diagnostic and clean multiview study are **complete**.
Do not restart them. The subsequent [strong-regime bridge](../output/2026-09-10-spectral-strong-augmentation/results.md)
is also complete, audited and reported: substantial native post-warmup learning
and unaugmented protection, but no additive translation benefit. Fixed, selected
and late comparisons all preserve the adverse combined-policy ranking, while
selected unaugmented accuracy and CE disagree. Its historical motivation and
changed evaluation roles remain recorded in the [regime comparison](../output/2026-09-10-spectral-multiview-clean/regime-comparison.md)
and [prospective protocol](../output/2026-09-10-spectral-strong-augmentation/protocol.md).

Manuscript/knowledge consolidation is complete, with [main PDF/source review](../output/2026-09-10-spectral-manuscript/strong-bridge-main-review.md).
The [design-only assessment](../output/2026-09-10-spectral-strong-augmentation/next-decision.md)
is now complete, with a [main implementation decision](../output/2026-09-10-spectral-component-utility/design-decision.md).
The distinct question is whether native restricts useful consistency or
softened-target learning, or restricts realized-label fitting while useful
learning survives. The [fixed protocol](../output/2026-09-10-spectral-component-utility/protocol.md)
uses twelve existing native warmup/final parents, two matched translated
action draws, actual inherited-state Adam displacements and exact 25-view
S/F/C changes, paired with separate original/transformed clean readouts.
Signed progress is new relative to earlier energy/direction measurements;
it still does not establish mediation of the full trajectory or a fresh
efficacy result. No intermediate restorable model exists.

[Strict restoration, panel/action and objective helpers are now prepared](../output/2026-09-10-spectral-component-utility/helper-acceptance.md),
independently reviewed with48 fabricated CPU tests passing. The [fixed runner,
independent saved-array audit and guard are now accepted](../output/2026-09-10-spectral-component-utility/implementation-acceptance.md),
with109 fabricated CPU tests passing, including cross-implementation checks.
[The diagnostic and independent audit are now complete](../output/2026-09-10-spectral-component-utility/results.md).
All twelve parents,48 actions and144 endpoints are retained; saved-array audit
PASS has50,388 checks. At translated final states, raw already worsens the
consistency objective in all three seed averages. Native-minus-raw clean-CE
effects are mixed and tiny; in the two clean-adverse seeds native causes
slightly less consistency deterioration. At warmup, native improves relative
consistency yet worsens clean CE in both settings/all three seeds. Actual
wrong-subset CE also favors the wrong targets relative to raw there, whereas
final-state fitting effects are mixed and F is not a pure memorization measure.
The [all-cell interpretation](../output/2026-09-10-spectral-component-utility/interpretation-analysis.md)
preserves full/tenth, finite/linear, tiny and per-draw qualifications.

This does not support the local lost-useful-consistency account. It does not
identify the accumulated mechanism or overturn the earlier useful learning.
Only native-history parents were tested with inherited Adam; outcome-informed
state reuse is not fresh confirmation. **Do not restart either completed unit.**
The result is now [integrated into the paper](../output/2026-09-10-spectral-manuscript/component-utility-main-review.md)
and [reconciled with the earlier history/pathway evidence](spectral_action_history_synthesis_2026-09-10.md).
That reviewed elementary identity separates immediate effects from terminal
hybrid-policy effects; neither observer-history mediation nor a new theorem
is claimed. The two-plot HTML result report is delivered.
Reporting demonstrated learning is not conditional on finding one universal
mechanism, conducting a tuning sweep or requesting another approval.

All completed acquisition/audit handles are consumed. No rank/LR sweep, noisy
observer4 transfer, paid reservation or broad speed benchmark is selected.

### Second: distinguish unfamiliar learning from low-frequency learning

This asks whether the native deficit is particularly severe for an omitted
class, or persists similarly for already learned low-frequency classes. Report
warmup competence, each policy's absolute correct-learning progress and all
endpoint CE/accuracy values before interpreting the planned interactions.
Baseline subtraction cancels within policy pairs but does not remove ceiling
or representation differences across warmups. The fixed competence annotation
never selects parents or extends warmup. Frequency changes exposure to the
other focal class in the opposite direction; the eight-background stream
stays fixed. Thus results would concern full-history dependence under a paired
exposure swap, not an isolated frequency effect or pure covariance mediation.

This remains an extension, not a prerequisite for the present paper. The
user's augmentation priority and the resulting diagnostic/report are complete.
No source/fixture preparation is now selected for this proposal; no new
scientific result, launch or paid reservation is implied by the design.

### Later: test useful learning on a matched competence frontier, if a practical claim is pursued

Use a fresh confirmation panel and equal validation-selection opportunities for native, a small scalar temporal family and ordinary stopping/regularization. Plot newly acquired useful competence against preserved competence and unwanted fitting, not one aggregate winner. This would ask whether native offers an allocation unavailable to the selected simpler policies. I16/I17 motivate the controls; they do not already settle this fresh, balanced comparison. No broad language-model speedrun or expensive tuning campaign is selected.

### Separate extensions: interpretability and behavioral safety

**J-Lens is now the sole current priority under the user's redirection.**
The first meaningful paired readout comparison above is complete and its
five-plot HTML report delivered. The existing Astra worktree has completed a
[qualitative display](../output/2026-09-10-jlens-individual-comparison/review.md)
of those same four signed PC pairs beside the individual
J-Lens readouts of their exact endpoint examples. Equal display length is not
equal underlying information: PCs summarize fit rows, while individual
activations directly encode held-out examples. Fit-individual controls are
not retained. This is an illustration, not a new superiority score or fresh
confirmation. Several individual lists echo prefix verbs while some PC lists
suggest broader themes, but several signed poles mismatch their endpoints.
The [subsequent fresh-text test is complete](../output/2026-09-10-jlens-fresh-content/results.md):
24new authored prefixes/12pairs, one common directional-ordering task,
fit-only references and all144 judgments committed before grading. Direct
direction tokens score31/48, individual fit-exemplar tokens33/48 and raw
fit exemplars34/48. Direct per-axis11/12,9/12,9/12,2/12 supplies a useful
three-direction demonstration but a misleading fourth and no overall
advantage over examples. All four axes remain in the result. This is lexical
transfer within familiar topics, not144 independent samples, four semantic
clusters, natural-corpus generalization or a causal explanation. No further
model acquisition is selected; the plot-led report is delivered (artifact not distributed in this public snapshot).

The [subsequent saved-fit geometry analysis](../output/2026-09-10-jlens-fit-geometry/results.md)
adds a finer distinction without new model work. Topic-mean shares on PC1–PC4
are61.67%,85.35%,10.56%,15.10%; both PC3 and PC4 mainly vary among contents
within topics, while paired framing contributes less than1% on every axis.
Independent arithmetic and exact128-score checks pass. Thus the useful third
is not mainly a topic-mean direction, but this coarse geometry does not
explain why the fourth failed. These are reused fit rows, and the earlier
fresh test compared different topics. The [completed within-topic comparison](../output/2026-09-10-jlens-within-topic/results.md)
now supplies a useful third-direction signal: direct-token11/12 versus9/12
example tokens and8/12 example text. Direct all-axis7/12,7/12,11/12,6/12 keeps
the weak fourth, and overall31/48,31/48,33/48 is no general advantage. This
uses12new pairings of24reused texts and new locked responses, not independent
content or a causal comparison removing topic information. The [mathematical note](jlens_covariance_interpretation_2026-09-10.md)
explains the population-slope steelman and why neither variance dominance nor
readable tokens guarantees individual interpretive accuracy.

For stronger J-Lens evidence, independently sampled content and reader replication would test the limited fresh-text positive; that extension is not queued. For safety, the key outcome is less absolute undesirable behavior at maintained benign competence, including rare groups, and demonstrable useful adaptation beyond a frozen parent. Neither requires relabeling an existing classification contrast as alignment. Both need separate prospective constructs before new acquisition.

Do not automatically repeat lagging, moment resets, tracking timescale sweeps, arbitrary mask tuning or the completed clustering/completion pilots. Muon ordering, rank-one high-learning-rate training, CoinRun and language-model speedruns remain ideas, not necessary next steps for this safety-focused account. Closest-method performance and publication-priority claims also require their own evidence; the [bounded prior check](spectral_subspace_prior_check_2026-09-10.md) already identifies substantial architectural overlap and a partial-access TAGD lead.

## Bottom line

The original intuition has a defensible core: coordinated variation can expose structure that supports selective learning. The investigation now also identifies why that need not be useful structure, why useful rare learning can suffer even with high late access, and why optimizer history and response speed cannot be omitted. The honest paper is not “PCA solves memorization” or “nothing beyond smoothing.” It is a positive, conditional account of **which learning a spectral policy preserves, admits and obstructs**, and what interventions make that account testable.
