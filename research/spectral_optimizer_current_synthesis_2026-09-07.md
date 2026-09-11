---
title: "Spectral optimizer: current scientific synthesis"
author: "Codex — Spectral Optimizer Investigation"
date: "7 September 2026"
---

# Spectral optimizer: current scientific synthesis

## Executive summary

Confidence is high that the optimizer is a **history-dependent, task-dependent
directional regularizer whose delivered gradient is subsequently transformed
by AdamW**; confidence is only moderate in this as the dominant causal account.
It learns a low-rank space of recent *centered gradient variation*, then filters
the raw current gradient. It is not, by construction, a detector of clean
labels, semantic signal, persistent signed agreement, Hessian curvature, or
the useful part of a gradient.

The central contradiction is immediate: hard rank-32 versus AdamW at each
method's validation-accuracy-selected checkpoint is **−2.38 points in I4 but
+4.827 points in I6**, across different three-seed bundles. The strongest
stable result is instead late anti-memorization. The historical three-seed
noisy-MNIST comparison has a 42.52-point final advantage, but only 1.69 points
at each method's test-selected peak. The evidence supports suppression of late
fitting, not universal improvement in attainable performance or replacement of
ordinary early stopping.

The continuations sharpen rather than dissolve that conclusion. Wider
covariance storage clearly reduces truncation-induced adaptation delay on a
synthetic switch and tracks a finite-history rank-32 reference more faithfully
on common neural gradient streams. Neither result establishes better learning.
In endogenous filtered training, widening sometimes helps accuracy and
sometimes hurts; its measured clean-minus-corruption retention advantage is
small and inconsistent. A scalar gradient-norm control does not reproduce the
endpoint preservation, but it also does not match the resulting AdamW step.
Current-observation self-inclusion is geometrically large, yet lagging the basis
does not reliably help: it harms clean performance in both tested seed bundles,
while the noisy effect reverses sign between the primary and fresh bundles.

Cross-repository outcomes impose strong boundary conditions. Clean CIFAR can
underfit; sparse-parity grokking slows while modular-addition grokking speeds;
a near-clean-accuracy-matched CIFAR robustness recipe is worse on every tested
radius in the arm means; the final Numerai spectral candidate transfers poorly
to later eras amid major recipe confounds; and emergent-misalignment results do
not isolate a general safety benefit. Noise can itself be low-dimensional,
structured, curvature-aligned, or temporally persistent.

The leading scientific hypothesis is therefore learned directional
regularization, with Adam history and coordinatewise normalization as a likely
material mediator. Confidence is high in the algebraic distinctions, moderate
in this broad account, and low in any uniquely identified causal mechanism.
The highest-value next evidence is not
another broad performance sweep: it is complete-state, same-batch one-step
measurement of raw versus filtered AdamW displacements and independent clean,
fixed-corruption and soft-target probe losses, followed by independent-batch
diagnostics that separate conditional batch variability from mean surprise.

No experiment is currently running. Iteration 007's one-shot CPU preparation
failed with main-process exit 2 and produced no scientific measurement; the
consumed failure is preserved. Replacement would require explicit new scope.
The optimizer, defaults, original 6 September Markdown/PDF, and broader
continuing investigation remain unchanged.

## Abstract

This report integrates the reconstructed implementation, classical experiments,
cross-repository applications, six focused continuations, and the 7 September
mathematical analyses of the spectral optimizer. The canonical method maintains
a recursively truncated covariance sketch of centered temporal gradient
innovations, uses its leading space to filter the raw current gradient, and
then relies on AdamW for the parameter update. This construction selects high
recent variation, not useful information by definition. Replicated noisy-label
experiments show substantial preservation of late clean accuracy, but clean
underfitting, task-dependent grokking, negative adversarial-robustness results,
weak temporal transfer in the final Numerai comparison, and unresolved language-
model behavior studies rule out a general denoising interpretation. Newer
controlled evidence further separates endpoints from selected checkpoints,
covariance fidelity from utility, and projected gradients from actual updates.
In particular, the sign of the validation-accuracy-selected noisy-MNIST effect
against AdamW reverses between two three-seed bundles, and lagged-basis effects
on noisy data reverse between a primary and fresh bundle. Conditional covariance
identities, fixed-Jacobian analysis, and label-noise geometry support a more
qualified account: the filter changes which local function directions remain
easy to learn, while Adam can rotate and rescale the intervention. The leading
hypothesis is task-dependent directional regularization with unresolved Adam
mediation. Concrete same-state tests are specified to distinguish it from
simple restraint, covariance-tracking, truncation, and self-inclusion accounts.

## 1. Scope, provenance, and evidence levels

This is a bounded synthesis of evidence already present in the repository and
the audited cross-repository investigation. It adds no training run, raw-data
reanalysis, external search, optimizer modification, or new default. The
original 6 September investigation (artifact not distributed in this public snapshot)
remains the detailed reconstruction of repositories, worktrees, reports,
emails, code and experiment history. The
final review resolution (artifact not distributed in this public snapshot)
records the corrections accepted after independent review. In particular, the
production Numerai spectral refits used constant learning rate rather than a
completed cosine schedule; the configured schedule horizon was inert.

Source priority is committed code and raw metrics; focused reports tied to
those artifacts; maintained synthesis; then planning notes and recollection.
Numerical claims link to the closest focused report or raw result. The original
inventory covered 391 current classical JSON paths and 36 archive-only paths,
but these are coverage counts, not independent replications.

Four evidence labels are useful:

- **Established** means an implementation fact, exact derivation, audited
  reconstruction, or empirical pattern replicated across relevant seeds or
  settings.
- **Supported** means direct evidence exists but is limited to a small number
  of seeds, one task, one model, or a selected recipe.
- **Provisional** means a mechanism or generalization is compatible with the
  record but has not survived a discriminating intervention.
- **Historical or invalid for the stronger claim** preserves provenance without
  treating old, confounded, malformed, or superseded evidence as confirmation.

The six focused continuations are not six independent confirmations of one
effect. Iterations 001 and 002 are controlled mechanism examples; 003, 004 and
006 are small MNIST policy studies with different paired seed bundles; 005 is
a passive replay on the iteration-004 AdamW streams. Repeated checkpoints and
probe observations are dependent within trajectories. Test accuracy from the
same official MNIST test set does not become new independent evidence each time
a checkpoint is evaluated.

## 2. The algorithm—and the spectral objects it is not

Let \(g_t\in\mathbb R^p\) be the flattened batch gradient and β the decay. The
canonical filter updates an exponential mean and forms a centered innovation.
For an idealized regular update with rank truncation, the covariance recurrence is:

\[
m_t=\beta m_{t-1}+(1-\beta)g_t,\qquad
c_t=g_t-m_t=\beta(g_t-m_{t-1}),
\]

\[
C_t=\mathcal T_r\!\left[\beta C_{t-1}
 +(1-\beta)c_tc_t^T\right],\qquad P_t=V_tV_t^T.
\]

Here \(P_t\) is idealized as an orthoprojector. The native finite-precision
action is \(V_t(V_t^Tg_t)\), which need not be an exact orthoprojector; startup,
residual rejection, pruning and repair also qualify the displayed recurrence.
After warmup, hard mode delivers \(h_t=P_tg_t\) in this ideal notation. The current raw
gradient is used to update the basis before that same gradient is projected:
the observer is self-inclusive. The covariance state is represented by
\(V_t\operatorname{diag}(S_t^2)V_t^T\), avoiding a dense \(p\times p\) matrix.
The stable code uses an orthonormal augmented basis, a small float64 symmetric
eigensolve, scale-aware rejection, reorthogonalization and periodic repair.
These improve numerical geometry; they do not turn recursive truncation into
exact PCA of the complete history. The source of record is the
[canonical filter](../spectral_filter.py), with reconstruction and executable
counterexamples in the
[mathematical audit](../output/2026-09-06-spectral-optimizer-investigation/analysis/mathematical-audit.md).

Startup and memory matter. The first nonzero innovation initializes with
\(c_jc_j^T\), not \((1-\beta)c_jc_j^T\); at β=.99 this initially gives that
innovation 100 times an ordinary observation's covariance weight. Repeated
rank truncation creates a distinct hysteresis: evidence for a new weak direction
can be discarded at every step before it accumulates enough mass to displace an
incumbent mode. Storage and projection are \(O(pr)\), but rotating the basis and
solving the reduced eigenproblem introduce \(O(pr^2)\) and \(O(r^3)\) work.

The name “spectral” covers at least four different objects in this research
history, and conclusions do not transfer automatically among them:

| Spectral object | Space and statistic | Intervention |
|---|---|---|
| Per-example gradient Gram matrix | Sample or microbatch similarity at one model state | Reweight examples or project a mean through a sample-space eigensystem |
| Temporal covariance of flattened gradients | Parameter-space variation across training observations | Filter the current gradient through a learned historical basis |
| Singular spectrum of one matrix gradient or momentum | Row/column geometry within a layer | Matrix preconditioning, polar update, or low-rank coordinate compression |
| Singular spectrum of a LoRA/effective weight change | Geometry of accumulated parameter change | Describe or regularize the trained adapter |

The canonical optimizer is the second. “Per-matrix” means separate temporal
covariance sketches for vectorized weight matrices; it is not an SVD filter of
the current matrix gradient. LoRA rank, covariance storage rank, delivered
projection rank and effective adapter rank are different quantities. The older
normalized sample-space consensus algorithm is different again: its eigenvectors
come from row-normalized example gradients while its update uses raw rows.

Soft mode is also qualitatively different from hard projection. With residual
enabled, tracked directions receive weights
\(w_i=(\lambda_i/\lambda_{\max})^\alpha\), while the untracked complement remains
at weight one, followed by raw-gradient norm restoration. Increasing α thus
suppresses weak *tracked* modes while preserving the complement; it does not
converge to a single leading direction. Without residual, α=0 gives a
norm-restored hard projection, not the unnormalized hard method. Results from
these variants must keep their intervention labels.

Most importantly, the projector constrains AdamW's input, not necessarily its
parameter displacement. Coordinatewise adaptive scaling does not generally
preserve a rotated subspace, momentum carries previous directions, and weight
decay adds another component. Therefore “projected gradient,” “step in the
subspace,” “matched gradient norm,” and “matched parameter-step norm” are not
synonyms.

## 3. What the cross-repository outcomes establish

### 3.1 Random-label anti-memorization is real but conditional

The strongest positive evidence is that selected hard temporal filters can
prevent large losses of clean accuracy late in training on randomly relabelled
MNIST and CIFAR. In the historical three-seed MNIST run at nominal 90%
replacement, Adam ended at 37.22% mean clean-test accuracy and legacy global
hard rank 200 at 79.74%, a 42.52-point endpoint difference. Their test-selected
peaks were 83.19% and 84.88%, only 1.69 points apart. A fixed random projector
ended at 56.03%, and the LoRA control at 31.94%. These values and seed curves
are recorded in the
classical evidence ledger (artifact not distributed in this public snapshot).

The stable one-seed 60-epoch comparison gives AdamW .3893, global rank-200
.7877 and per-matrix rank-64 .8174 clean-test accuracy, with 179.40 MiB versus
57.41 MiB basis storage for the two spectral layouts. The matrix rank was
chosen on a scout using that seed, so this is supporting evidence, not an
unbiased global-versus-matrix estimate; see the
[raw summary](../results/noisy_mnist_hard_curves/summary.json).

Nominal 90% uniform replacement does not mean 90% wrong labels. With ten
classes, replacement returns the original class one tenth of the time, so the
expected incorrect fraction is 81%. The newer runs observe wrong-label fractions
near 80.2–81.7%. Their “corruption residual” is the gradient difference under
clean and fixed corrupted labels at a state trained using the corrupted labels;
it is not automatically independent zero-mean noise.

The two-seed CIFAR study shows the same benefit/cost boundary: at 0%, 40% and
80% nominal replacement, spectral-minus-Adam final clean accuracy is −6.07,
+18.48 and +23.95 points. A separate rank/decay sweep changes the ranking across
noise levels; rank 1200/decay .999 nearly restores clean accuracy but loses much
of the high-noise gain. There is no universal rank or decay.
The classical ledger (artifact not distributed in this public snapshot)
identifies the underlying CIFAR runs and configurations.

### 3.2 Useful weak structure can be suppressed

Grokking results reject a universal “spectral accelerates generalization”
story. In five-seed modular addition, first 90% test crossing occurs at a mean
3,720 epochs for AdamW and 2,550 for temporal spectral, 31.45% earlier; switching
to spectral after memorization averages 2,520. Without weight decay, all three
tested single-seed recipes fit training but fail to generalize by 40,000 epochs.
Sparse parity reverses the pattern: AdamW crosses at 650/750/750 epochs, while
the original spectral recipe crosses at 2,050/2,150/2,000. Some low-rank runs
later regress or never cross. A restriction that blocks late noisy-label fitting
can also block rare or weak features needed for clean generalization. The
[grokking report](grokking_v2_findings.md) retains the task-specific runs.

The LoRA evidence reinforces this boundary. Soft rank four approximately
matches the clean LoRA baseline, hard filtering underfits, and selected soft
filtering does not improve the noisy-LoRA task. These are different parameter
spaces and capacities, not contradictory measurements of one common rank.

Coherent unwanted structure is another important limit. The
[targeted backdoor-ablation experiments](targeted_ablation_findings.md)
test probe-guided removal, not a canonical filter-versus-Adam backdoor comparison.
The linear intervention succeeds much more than the nonlinear one, but that
does not demonstrate canonical-filter backdoor amplification. One covariance
direction was selected by trigger-probe alignment, not simply leading variance;
a separate proposed trigger direction was a softmax gauge direction. Adam's
post-projection motion further complicates the nonlinear failure. The safe
conclusion is that coherence does not imply desirability, not an identified
universal backdoor mechanism.

### 3.3 Robustness does not follow from low-rank restriction

The audits reaggregate stored arrays; they do not replay attacks because the
adversarial pixels were not retained. Global versus per-matrix layout, horizon,
seeds and selection changed together, so “raising rank caused the reversal” is
not identified. The evidence does rule out treating earlier small-rank margin
findings as a generic robustness result. The
external audit (artifact not distributed in this public snapshot)
links the raw payloads and explains why AURAC intervals from different studies
cannot be pooled.

### 3.4 Financial and behavioral “noise” can be structured

Numerai is not random-label classification. Low signal-to-noise, regime shift,
persistent spurious factors and rare useful predictors all give temporal
covariance structure. A corrected small-MLP five-seed study on one fold has
CORR .000609 for spectral versus .016863 for AdamW despite similar late training
MSE. Larger high-rank runs show delayed overfitting and better selected validation
correlation, but the final later-era Diagnostics comparison reports CORR .014690
for the deployed spectral candidate versus .036058 for AdamW. The original
Diagnostics export is missing, so those scores remain report-level evidence.

Language-model emergent-misalignment evidence is likewise unresolved. A Qwen3
sport trajectory is a descriptive near-null against an interpolated authors'
Adam curve. A keep-versus-ablate contrast at one closely loss-matched checkpoint
shows −11.81 alignment points for ablation, but 19.014% of ablate-94 answers
contain nonempty text before a closing-only `</think>` tag versus .264% for
keep. The scoring helper fails to strip that form, so the corrected effect is
unmeasured. Rank-one adapter observations compare unfiltered AdamW and Lion;
the endpoint alignment difference is +.134 with a paired question-cluster
bootstrap interval [−1.196,+1.507]. This is a local near-null under a known
measurement defect, not equivalence or evidence about a filtering intervention.

The selected single-seed Qwen2.5 point has medical NLL 1.413 versus 1.416, MMLU
58.3% versus 67.3%, and EM 19.0% versus 10.3% for AdamW versus spectral, with
only 58 judged answers (11/58 versus 6/58 raw). The wider sweep fails to achieve
broad loss matching. Its legacy global microbatch filter also differs from
Qwen3's stable per-block post-accumulation filter. No general safety benefit is
established. Direct provenance and measurement limits are in the
application audit (artifact not distributed in this public snapshot)
and EM scoring audit (artifact not distributed in this public snapshot).

## 4. What continuations 001–006 add

The focused sequence is most coherent when read as a chain of increasingly
specific challenges rather than an accumulating performance leaderboard.

Sources are the focused reports for
[I1](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-001/results.md),
[I2](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-002/results.md),
[I3](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-003/results.md),
[I4](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-004/results.md),
[I5](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-005/results.md), and
[I6](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-006/results.md).

Two reversals must remain explicit. First, the noisy Current-32 comparison
against AdamW at each method's maximum-validation-accuracy checkpoint is −2.38
points in I4 but +4.827 points in I6. These are different three-seed bundles;
neither should overwrite the other, and pooling them post hoc would not repair
the instability. Second, I6's lagging intervention is adverse on noisy data in
the primary bundle but favorable in the separately predeclared fresh bundle.
The correct result is heterogeneity, not zero effect and not a selected positive.

I5's ordering measurement explains why the lag test was worth doing. Across
four common-stream states, the current gradient retains about 84% of its energy
after updating the basis but about 41% under the previous basis. Yet a large
geometric effect is not necessarily a beneficial contamination. I6 finds that
lagging lowers clean performance in both bundles and gives unstable noisy
effects. Norm restoration in I6 matches the norm of the contemporaneous current
projection on the restored arm's own state. It does **not** match Current-32's
cross-trajectory gradient, raw-gradient norm, or actual AdamW step: on noisy
data, restored and current decay-subtracted step norms are .035467 and .043200.

![Common-stream covariance fidelity, retention and self-inclusion diagnostics from iteration 005. All four prescribed states are shown; bands are three-seed ranges, not confidence intervals. The lower-right panel shows means only. The upper-left deficit is relative to the rank-32 optimum, not total covariance energy.](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-005/common-stream-diagnostics-v2.png)

## 5. Mathematical synthesis

The [mathematical mechanisms report](spectral_optimizer_mathematical_synthesis_2026-09-07.md)
and [label-noise follow-up](spectral_label_noise_identifiability_2026-09-07.md)
provide the detailed derivations, primary-source connections and prior reviews;
both accompany this synthesis in the current PDF edition.

### 5.1 What creates the covariance

Condition on current parameters, fixed training labels, past batches and the
observer state. Write \(g_t=\bar g_t+\xi_t\), where \(\bar g_t\) is the
conditional batch-gradient mean, \(\operatorname{Cov}(\xi_t)=\Sigma_t\), and
\(d_t=\bar g_t-m_{t-1}\). Then

\[
\mathbb E[c_tc_t^T\mid\mathcal F]
=\beta^2(d_td_t^T+\Sigma_t).
\]

Thus new covariance mass combines **mean surprise** \(d_td_t^T\) and fresh
conditional batch variability Σ. Mean surprise can reflect objective drift,
representation change, curvature-driven motion, regime change, or noise retained
by the lagging mean. It is not pure clean signal. Fresh variability can itself
be structured. The identity describes an ideal regular-weight proposal before
native startup, rejection, pruning, repair, rounding and truncation, and
expectation does not commute with eigenvector selection.

At one fixed state, let \(m=m_{t-1}\) be the fixed pre-observation observer
mean and \(d=\bar g-m\). Draw \(g,g'\) conditionally independently from the
same fixed-training-objective batch distribution. Then

\[
\mathbb E[(g-g')(g-g')^T/2\mid\mathcal F]=\Sigma,
\qquad
\mathbb E[(g-m)(g'-m)^T\mid\mathcal F]=dd^T.
\]

Projected scalar versions of these quantities can identify whether retained
activity is dominated by fresh variability or mean surprise without forming a
dense covariance. They still do not label either component useful.

### 5.2 PCA solves the wrong objective for general denoising

At a fixed state, let the clean objective gradient be μ and the corrupted
training gradient \(g=\mu+b+\xi\), with fixed-label bias \(b\) and zero-mean
fresh noise ξ of covariance Σ. For an orthogonal projector chosen before that
fresh draw, with the state and target fixed,

\[
R(P)=\mathbb E\|Pg-\mu\|^2
=\|\mu\|^2+\operatorname{tr}\!\left[P(bb^T+\Sigma-\mu\mu^T)\right].
\]

The rank-\(r\) oracle for this MSE retains leading eigenvectors of
\(\mu\mu^T-bb^T-\Sigma\), not leading covariance eigenvectors. With varying
useful signal second moment \(M_s\) and independent zero-mean additive noise,
the analogous oracle sees \(M_s-\Sigma\), whereas uncentered PCA sees
\(M_s+\Sigma\). For example, useful variance
\(\operatorname{diag}(4,0)\) and nuisance variance
\(\operatorname{diag}(0,9)\) make rank-one PCA select pure nuisance. Leading
variance is not a usefulness score.

A favorable special case exists: desired gradients lie in a low-dimensional
subspace, nuisance is independent and isotropic, desired coefficients vary
enough to appear after centering, and estimation is independent enough not to
overfit the same draw. Under those assumptions population PCA can recover the
desired space. The current optimizer only approximates them. A constant useful
mean may disappear from centered innovations, whereas a sign-alternating
nuisance direction can have large covariance.

Even better gradient MSE does not guarantee a better immediate loss step. For
plain SGD with an unbiased gradient and unchanged learning rate, projection
cannot improve the first-order clean-descent term beyond the full clean
gradient. A benefit can instead come from reducing curvature cost, suppressing
bias-driven fitting, changing finite-step stability, or altering future feature
learning. These mechanisms require distinct measurements.

### 5.3 A fixed-Jacobian view: restriction of learnable functions

For square loss with residual \(r=f-y\), fixed Jacobian \(J\), and
\(g=J^Tr/n\), projected SGD gives

\[
\Delta f=-\eta JPJ^Tr/n.
\]

The local kernel becomes \(K_P=JPJ^T\), with
\(0\preceq K_P\preceq JJ^T\). In a linear model this is exact; in a nonlinear
network it is first-order and local. The projector changes which residual
components are easy or impossible to fit through that step. This provides a
cohesive explanation for both noisy-label preservation and clean underfitting:
the same restriction helps when wrong-label fitting depends on excluded
directions and hurts when useful features do.

It is not a proof that PCA finds the right function modes. If \(J=USV^T\),
then \(C_g=VS(U^TC_rU)SV^T/n^2\); residual covariance and Jacobian geometry
jointly determine the parameter-space spectrum. Nor is \(K_P\) an AdamW
kernel. Adding a fixed diagonal preconditioner produces \(JDPJ^T\), generally
nonsymmetric; actual Adam has historical and input-dependent state.

### 5.4 Adam limits gradient-space interpretations

At a common pre-step Adam state, absorb bias correction into constants and
write the data displacement coordinatewise as

\[
u_i(h)=-\eta\frac{A_i+B h_i}{\sqrt{C_i+D h_i^2}+\epsilon}.
\]

Changing \(h\) changes numerator and denominator. On the first step with zero
moments and negligible ε, the displacement is approximately
\(-\eta\operatorname{sign}(h_i)\), so input magnitude can largely disappear.
Later, old momentum can move parameters even under zero current gradient;
diagonal scaling can rotate a vector out of a non-coordinate subspace; and
decay contributes independently. I2 supplies an exact existence example, while
I3 shows that corresponding leakage-associated direction reversals are rare in
the measured noisy-MNIST trajectory. Mathematical possibility and empirical
frequency are separate conclusions.

I4 and I6 show why norm language must be precise. Scaling the raw gradient to
the norm of its hypothetical projection does not match the filtered arm's Adam
moments or step. Restoring the lagged projected-gradient norm does not restore
the current arm's step. Conversely, I6 clean filtering has larger mean data-step
norm than AdamW while learning less. Neither “smaller steps” nor “direction
alone” is identified. The base optimizer is part of the intervention.

### 5.5 Fixed corruption, fresh labels, and the retention ceiling

For logits \(z\), probabilities \(p\), parameter Jacobian \(J\), clean target
\(y\), and a fresh uniform-replacement label \(\tilde y\) at rate ρ,

\[
g_y=J^T(p-e_y),\quad
q=(1-\rho)e_y+\rho\mathbf1/K,\quad
g_{\tilde y}=g_q+\varepsilon_{\tilde y},
\]

where \(g_q=J^T(p-q)\) and
\(\mathbb E[\varepsilon_{\tilde y}\mid\theta,x,y]=0\) only for the indicated
fresh independent redraw. The existing trained fixed-corruption residual does
not inherit this conditional identity. Old labels helped determine θ; drawing
new minibatch indices does not make those assignments fresh. The missing
soft-target gradient \(g_q\) cannot be reconstructed from retained clean/residual
aggregates and requires a new probe acquisition.

Fresh label randomness is structured after the Jacobian:

\[
\operatorname{Cov}(g_{\tilde y}\mid\theta,x,y)
=J^T[\operatorname{diag}(q)-qq^T]J.
\]

For \(K=10,\rho=.9\), the nonzero label-space eigenvalues are .09 on the eight
wrong-class contrast directions and .171 on the true-versus-others direction.
At \(p=q\) and with locally twice-differentiable logits, this covariance equals
the per-input expected cross-entropy Hessian: the residual-weighted logit-Hessian
term vanishes. This is not a claim about the observed temporal covariance.
Thus a covariance direction can look
curvature-like precisely because of random labels; curvature alignment alone
is not evidence of clean semantic signal.

At uniform predictions, the expected fresh corruption residual is exactly
\(-\rho g_y\). Any fixed homogeneous linear filter retains the same normalized
energy of these antiparallel mean vectors. This illustrates a sign-blindness
limit, not an assertion that I5 predictions were uniform or its realized
residuals equal their fresh-label means.

The strong clean/residual opposition in I5 also has a precise limit. For
nonzero vectors \(c,r\), cosine γ, and any ideal nontrivial orthoprojector \(P\),

\[
\left|
\frac{\|Pc\|^2}{\|c\|^2}-
\frac{\|Pr\|^2}{\|r\|^2}
\right|\le\sqrt{1-\gamma^2}.
\]

Across all twelve retained I5 states, mean cosine is −.881220 and the mean
ideal absolute-gap ceiling is .469728. Observed signed clean-minus-residual gaps
are only .058674 for width32 and .063387 for width128. At final states, the
corresponding values are cosine −.896945, ceiling .441290, and gaps .064382/
.072703. Opposition limits possible selectivity but leaves substantial ideal
headroom; it does not force the observed gaps to be small. The ceiling is an
oracle geometric maximum, not a native guarantee or promised performance gain.
The maximizing projector need not be learnable, stable, or good for training.
See the complete
[identifiability analysis](spectral_label_noise_identifiability_2026-09-07.md).

## 6. Ranked hypotheses and falsifiable tests

The hypotheses overlap; the ranking expresses current explanatory value, not
posterior probabilities. Confidence concerns the existing evidence, not a
claim that one mechanism acts alone.

### 1. Learned directional regularization — moderate confidence

The filter changes which gradient information enters AdamW, suppresses late
wrong-label fitting in multiple selected recipes, and does something the tested
scalar attenuation controls do not reproduce. It also underfits clean MNIST and
CIFAR, delays parity, and fails to transfer uniformly. This broad account fits
both benefits and costs without assuming semantic eigendirections.

**Potential contradiction:** accuracy-selected effects against AdamW reverse
between I4 and I6, and clean/residual retention selectivity is small and
heterogeneous. A learned orientation could be epiphenomenal while effective
capacity, history, or optimizer-state changes cause the outcome.

**Falsifiable test:** at identical complete states, compare filtered and raw
data displacements after artificially matching *actual displacement norm*, then
add identical decay. Evaluate finite loss changes on independent clean,
fixed-corruption, fresh-redraw and soft-target probes. If learned orientation
does not improve any predeclared clean-versus-corruption utility contrast, or
if random orientations with matched displacement perform equivalently within
prespecified precision, that local directional account weakens. A null one-step
effect would not exclude a mechanism operating through later states.

### 2. Adam history and adaptive scaling mediate the effect — interaction certain, causal contribution unresolved

Noncommutation and moment carryover are mathematical facts; neural trajectories
show large out-of-subspace step energy and failures of input-norm matching to
match steps. What remains unknown is how much this interaction causes the
long-run anti-memorization rather than merely accompanying it.

**Potential contradiction:** the filter changes learning even though Adam often
leaves its nominal subspace, which could mean the important effect occurs in
the moments' information content rather than geometric confinement. Rare
observed ascent reversal argues against using the I2 pathology as a generic
explanation.

**Falsifiable test:** clone parameters, buffers, observer, Adam moments,
counters, scheduler and RNG at the same state. Feed raw, current-basis,
previous-basis and zero gradients on the same batch. Record data-only and total
displacements, projected leakage, logits and independent probe losses. Repeat
with separately specified SGD or reprojected-Adam displacement controls. A
small, stable difference between gradient-space predictions and realized utility
would weaken the chosen immediate-step mediation prediction; a sign-changing
difference would support it. Neither alone bounds Adam's long-run causal role.

### 3. Covariance primarily tracks a mixture of mean drift, curvature-driven motion, and batch variability — moderate mathematical, low neural confidence

The conditional identity includes mean surprise and fresh variability,
and the centered filter emphasizes temporal change. A scalar quadratic shows
that equal injected noise can create different gradient variance as curvature
and learning-rate dynamics approach oscillation. Structured label covariance
can itself resemble curvature.

**Potential contradiction:** these sources may explain the spectrum while
remaining irrelevant to which information affects generalization.

**Falsifiable test:** at frozen states, use independent batch pairs to estimate
fresh-variability and mean-surprise energy along retained, discarded and random
comparison directions. Only if mean surprise dominates should a second stage
measure gradient temporal autocorrelation and Hessian-vector response. Failure
of these quantities to predict same-state utility would reject them as useful
mediators even if they reconstruct covariance energy.

### 4. Recursive truncation causes useful path-dependent learning delay — high estimator, low learning confidence

I1 establishes the delay on controlled streams, and I5 shows wider storage is
more faithful to the chosen finite-history reference. But exact or wider
covariance is not an oracle, width can be nonmonotone in analytical examples,
and filtered trajectories show adverse width seeds.

### 5. Self-inclusion is mainly a tracking mechanism, not useful denoising — low-to-moderate confidence

I5 measures a very large current-versus-prior retention change, but I6 finds no
dependable benefit from lagging. This favors the view that self-inclusion helps
the basis follow the current regime, though its utility varies.

**Potential contradiction:** I6 is small and whole-policy; on noisy data its
sign reverses across bundles. Lagging changes direction, magnitude, moments and
future states together.

**Falsifiable test:** in complete-state one-step clones, compare current and
prior bases with reciprocal displacement matching and independent probes. Test
whether the self-included component disproportionately benefits training-batch
loss while harming independent objectives. Do not treat current-gradient
retention as independent evidence.

### 6. Ordinary training restraint explains most endpoint gains — plausible alternative, currently weakened but not ruled out

The collapse from endpoint to accuracy-selected advantage makes early stopping
an important alternative. The I4 scalar control fails to reproduce endpoints,
so one own-trajectory gradient-norm policy is insufficient. That does not cover
validation-tuned stopping, learning-rate schedules, weight decay, update-norm
controls or reduced capacity.

**Falsifiable test:** predeclare a fair tuning budget and compare spectral
training with validation-accuracy and validation-CE early stopping, warmup-stop,
learning-rate/weight-decay baselines, reduced-width models, and actual-step-
matched controls. Keep an untouched final test and report endpoint and selected
estimands separately. If simpler restraint matches the Pareto frontier, the
spectral geometry has limited incremental value.

### Prioritized resources

The first two diagnostic families fit the existing small MLP scale and can be
CPU/GPU bounded after a short pilot; they require complete-state acquisition
because old model-only checkpoints cannot reconstruct exact Adam and observer
counterfactuals. A three-to-five-paired-seed whole-policy follow-up could run on
the local RTX 3090 or free MATS L40 partition only after those diagnostics and a
frozen analysis plan. CIFAR attack replay, fresh purged Numerai folds, and 7B
language-model training or rejudging are later and require separate runtime,
cost and metric approval. No paid compute is implied here.

## 7. Limitations and their disposition

Several apparently reassuring quantities must not be overread. The I5 ideal
rank-32 reference is a covariance approximation target, not semantic truth.
The label-retention ceiling is an oracle upper bound, not a native guarantee or
performance headroom. An ideal recurrence with regular observation weighting is
not the exact native algorithm at startup or under rejection, repair, truncation
and floating point. Numerical audits validate recorded computations; they do
not turn three seeds into population inference.

## 8. Related primary work

The closest literature clarifies concepts; none validates this optimizer by
analogy.

- [Gur-Ari, Roberts and Dyer](https://arxiv.org/abs/1812.04754) report gradient
  concentration in small Hessian-related subspaces. Their result motivates
  low-dimensional tracking but does not make centered temporal covariance a
  useful-gradient oracle.
- [Song, Ahn and Yun](https://arxiv.org/abs/2405.16002) show that projecting
  onto dominant Hessian directions can stall learning while removing them can
  preserve progress. High energy and productive descent are not equivalent.
- [Jacot, Gabriel and Hongler](https://arxiv.org/abs/1806.07572) provide the
  neural-tangent-kernel framework used for the fixed-Jacobian analogy. The
  present finite ReLU/AdamW trajectories are not a frozen-kernel theorem.
- [Arpit et al.](https://arxiv.org/abs/1706.05394) document early pattern
  learning and later memorization of random labels, making selective slowing a
  plausible outcome. They do not identify this filter's mechanism.
- [Kunstner, Balles and Hennig](https://arxiv.org/abs/1905.12558) explain why
  empirical gradient second moments should not casually be called Fisher or
  Hessian matrices. The warning is stronger for centered temporal covariance.
- [Ghashami et al.](https://arxiv.org/abs/1501.01711) give Frequent Directions
  sketch guarantees, helping separate covariance approximation from downstream
  utility. Those guarantees do not apply to this recursively truncated EMA.
- [GaLore](https://arxiv.org/abs/2403.03507),
  [Shampoo](https://proceedings.mlr.press/v80/gupta18a.html), and
  [SOAP](https://arxiv.org/abs/2409.11321) show why matrix layout and optimizer-
  state placement must be specified. Their spectral objects and interventions
  differ from temporal global filtering.
- [Wilson et al.](https://arxiv.org/abs/1705.08292) support treating adaptive
  optimization as part of the inductive bias.
- [Feldman](https://arxiv.org/abs/1906.05271) shows why memorization of rare
  examples can support generalization. Suppressing memorization is not
  intrinsically beneficial.
- [Szegedy et al.](https://arxiv.org/pdf/1512.00567) introduce the soft-target
  view used in label smoothing; [Müller, Kornblith and Hinton](https://papers.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)
  and [Lukasik et al.](https://arxiv.org/abs/2003.02819) study its calibration,
  representation and noisy-label effects. None shows that the spectral filter
  implements label smoothing.

“Spectral bias” in Fourier function learning, sample-gradient agreement,
low-rank optimizer-state compression and adapter singular spectra are useful
neighboring ideas but not interchangeable claims. This investigation makes no
publication-standard novelty or state-of-the-art claim.

## 9. Conclusion and current status

The evidence supports one robust practical statement: selected temporal hard
filters can strongly suppress late memorization of randomly corrupted training
labels. It does not support the broader statement that dominant temporal
gradient-covariance directions are generally clean, useful, robust, aligned or
economically persistent. Noise can be structured; useful means can vanish under
centering; recursive memory can delay new features; and Adam can transform a
projected gradient into a step outside the nominal space.

The current synthesis is therefore conditional. The optimizer most likely
acts as learned directional regularization, altering which local function
changes remain learnable, with Adam history and adaptive scaling mediating the
real intervention. This account accommodates endpoint anti-memorization,
clean underfitting, opposing grokking results, negative robustness, unstable
selected-checkpoint gains, and application-transfer failures. It remains a
provisional mechanism because no same-complete-state causal measurement yet
connects learned covariance directions to independent clean-versus-corruption
utility after actual displacement is controlled.

The next decisive question is not “does PCA keep the good gradient?” but:
**which task-relevant function changes become easier or harder after the
history-dependent filter and AdamW interact, and can those changes be predicted
from measurements made before the outcome?** Complete-state one-step comparisons
and independent-batch covariance-source probes are the shortest path to an
answer. Only a positive predictive bridge would justify a further whole-policy
or expensive application study.

As of 7 September 2026, no experiment is running. Iteration 007 preparation
made no native training run and produced no new scientific evidence: its fixed
one-shot service has MainPID 0, exited with status 2, and left an invalid
zero-byte measurement file. The failure is preserved and consumed; any repaired
replacement requires explicit authorization and a new scope. This report does
not announce a launch, permission, retry, default change, or completion of the
broader continuing investigation.
