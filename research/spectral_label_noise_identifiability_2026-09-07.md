# What the clean-versus-corruption retention gap can identify

Codex / Spectral Optimizer Investigation — 7 September 2026.

## Result in brief

The existing “corruption residual” is not pure random noise. That was already
established in the earlier decomposition (artifact not distributed in this public snapshot).
This follow-up adds two useful distinctions:

1. Even fresh random label noise has structured covariance after passing through
   the model's Jacobian; in a special correctly matched case it equals local
   expected-loss curvature. Covariance/curvature alignment is therefore not, by
   itself, evidence of useful signal.
2. A sharp geometric bound quantifies when clean/residual retention comparisons
   have little room to distinguish directions. Applying it to **all twelve
   retained iteration-005 states** finds substantial remaining room. Strong
   opposition alone does **not** explain away the observed small retention gaps.

This is theory plus post-hoc scalar analysis of existing evidence, not new
training or a causal explanation of the optimizer's performance. It refines the
[mathematical synthesis](spectral_optimizer_mathematical_synthesis_2026-09-07.md);
the leading hypothesis remains task-dependent directional regularization with
unresolved Adam mediation.

## 1. The actual label recipe and its conditioning

The [source helper](../output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-003/neural_harness.py)
draws one replacement decision and digit per training example and reuses those
fixed labels. Replacement includes the original class. It measures clean and
corrupted gradients on the same probe examples and calls their difference the
fixed training-corruption residual. Iterations 004–006 reuse this recipe.

Let \(z(\theta,x)\) be K logits, \(p=\operatorname{softmax}(z)\),
\(J=\partial z/\partial\theta\), and \(u=\mathbf1/K\). At identical input,
parameters and forward state, unweighted cross-entropy gives

\[
g_y=J^T(p-e_y),\qquad r=g_{\tilde y}-g_y=J^T(e_y-e_{\tilde y}).
\]

For a **fresh independent replacement draw** at rate \(\rho\), the target mean
is \(q=(1-\rho)e_y+\rho u\). Define \(g_q=J^T(p-q)\). Then

\[
g_{\tilde y}=g_q+\varepsilon_{\tilde y},
\quad \varepsilon_{\tilde y}=J^T(q-e_{\tilde y}),\quad
\mathbb E[\varepsilon_{\tilde y}\mid \theta,x,y]=0,
\]
\[
\mathbb E[r]=\rho J^T(e_y-u)
=\rho(g_u-g_y),\qquad g_u=J^T(p-u).
\]

These are recalled identities, not new discoveries in this follow-up. The
zero-mean statement requires the indicated fresh draw. For the old labels that
helped determine \(\theta\), the algebra still holds but conditional unbiasedness
does not follow. Independent minibatch indices do not turn the old fixed
assignments into fresh independent labels.

Expected cross-entropy over fresh replacement is the deterministic soft-target
loss. This connection appears explicitly in
[Szegedy et al., section 7](https://arxiv.org/pdf/1512.00567).
It equates a fixed-state expectation, not entire stochastic training trajectories:
in particular \(\mathbb E[\operatorname{Adam}(g)]\) need not equal
\(\operatorname{Adam}(\mathbb E[g])\), even at the same previous Adam state.

## 2. Random labels can create structured, curvature-like covariance

A categorical one-hot target with mean \(q\) has covariance
\(C(q)=\operatorname{diag}(q)-qq^T\). Therefore

\[
\operatorname{Cov}_{\tilde y}(g_{\tilde y}\mid\theta,x,y)=J^TC(q)J.
\]

For \(K\ge2\), write \(a=1-\rho+\rho/K\) for the true-class probability and
\(b=\rho/K\) for each other class. The label-space spectrum is exactly:

| Subspace | Eigenvalue | Dimension |
|---|---:|---:|
| Common shift \(\mathbf1\) | 0 | 1 |
| Wrong-class contrasts: zero true-class coordinate, other coordinates sum to zero | \(b\) | \(K-2\) |
| True class versus the others, \(e_y-u\) | \(Kab\) | 1 |

The proof is direct substitution into \(C(q)v\) on these three orthogonal
subspaces. At \(\rho=.9,K=10\), the two nonzero eigenvalues are **.09 and .171**.
At \(\rho=1\) they coincide at \(1/K\); at \(\rho=0\) the covariance vanishes.
These are exact label-space values, not measured neural covariance eigenvalues.
The Jacobian pullback generally changes both spectrum and ordering.

Even fully random labels, isotropic on the label-contrast space, produce
\(J^T(I-\mathbf1\mathbf1^T/K)J/K\), which is generally anisotropic in parameter
space. Randomness can thus have large, repeatable directions reflecting the
model's response geometry. Low-dimensional or curvature-aligned covariance
does not by itself show that retained gradients are semantically useful.

For twice-differentiable logits locally, the soft-target loss Hessian is

\[
\nabla_\theta^2\ell_q
=J^TC(p)J+\sum_k(p_k-q_k)\nabla_\theta^2 z_k.
\]

At a state satisfying \(p=q\), this equals the fresh-label gradient covariance:
the residual-weighted logit-Hessian term vanishes and \(C(p)=C(q)\). This is a
sufficient special condition, not a description of the observed trajectory.
Affine logits remove the second term but still leave \(C(p)\), not \(C(q)\).
Nor is this per-input fresh-label covariance the filter's centered, truncated,
temporal batch-gradient covariance.

For independently redrawn labels on each of B fixed input occurrences, the
batch-mean covariance is \(B^{-2}\sum_i J_i^TC(q_i)J_i\). Repeated use of fixed
assignments, especially repeated occurrences of the same item, requires a
different conditioning/cross-covariance calculation. Do not substitute this
formula for the actual fixed-dataset batch covariance.

## 3. A sign-blind diagnostic can miss an exact opposition

If \(p=u\), then \(g_u=0\) and

\[
\mathbb E[r]=-\rho g_y.
\]

For nonzero \(g_y\), \(\rho>0\), and any fixed homogeneous linear action L,
the normalized retentions of these two **mean vectors** are identical:

\[
\frac{\|L\mathbb E[r]\|^2}{\|\mathbb E[r]\|^2}
=\frac{\|Lg_y\|^2}{\|g_y\|^2}.
\]

Thus an exactly antiparallel systematic corruption effect cannot be selectively
removed by a linear subspace projector while preserving the clean vector.
Squared-energy retention is blind to the sign. This illustrates an
identifiability limit; it does not assert that the measured finite-batch
residual equals its fresh-label mean, or that trained predictions are uniform.
Retention of an expected vector also differs from expected retention of random
vectors.

A near-uniform prediction regime makes this a useful hypothesis for why clean
and residual vectors oppose each other. Other model/label configurations can
produce opposition too. It is not proof that filtering implements label
smoothing or suppresses only realization-specific noise.

## 4. The sharp bound—and what the retained data say

For nonzero vectors c and r, let \(x=c/\|c\|\), \(y=r/\|r\|\), and
\(\gamma=x^Ty\). For any ideal orthoprojector P,

\[
\left|\frac{\|Pc\|^2}{\|c\|^2}
-\frac{\|Pr\|^2}{\|r\|^2}\right|
\le \sqrt{1-\gamma^2}.
\]

Proof: \(D=xx^T-yy^T\) is traceless with the two possible nonzero eigenvalues
\(\pm\sqrt{1-\gamma^2}\). The gap is \(\operatorname{tr}(PD)\); its largest
absolute value is the positive eigenvalue, attained by projecting onto the
corresponding eigendirection. At fixed nontrivial rank \(1\le r_{\rm rank}<p\),
orthogonal null directions can fill the remaining rank. Identity or zero
projectors have zero gap. Alignment and opposition impose the same bound.

For a fixed general L, replace the right-hand side by
\[
[\lambda_{\max}(L^TL)-\lambda_{\min}(L^TL)]\sqrt{1-\gamma^2}.
\]
Native \(VV^T\) need not be an exact orthoprojector, and its floating-point
evaluation is not literally a fixed real-linear map. The ideal bound is
therefore used below as a geometric benchmark, not a native numerical
certificate.

### Complete existing-state panel

The derived scalar record (artifact not distributed in this public snapshot)
uses all twelve original JSON snapshots: seeds 3,4,5 at steps 200,500,1000,2000,
both widths. Every input hash matches the committed iteration-005 summary, and
each original record equals its copied raw record there. Only scalar Gram
entries and norms were read; no tensors, models or datasets were loaded.
The exact read-only calculation (artifact not distributed in this public snapshot)
is retained.

| Existing-state panel | Mean raw cosine | Mean ideal absolute-gap ceiling | Actual signed gap, width 32 | Actual signed gap, width 128 |
|---|---:|---:|---:|---:|
| Four states per seed, then three-seed mean | −.881220 | .469728 | .058674 | .063387 |
| Final state, three-seed mean | −.896945 | .441290 | .064382 | .072703 |

These numbers are retention fractions, not accuracy points. Bounds were
computed per vector pair before averaging, not from the aggregate cosine.
Individual ceilings range **.408218–.560310**. Mean per-state absolute-gap/
ideal-ceiling ratios are **.129209/.138059** for widths 32/128, or
**.146726/.165389** on final states. These are descriptive headroom ratios, not
fractions of a demonstrated achievable learning benefit.

The conclusion is limited but decisive for this proposed explanation:
**opposition does not mathematically force the gaps to be as small as observed.**
Substantial ideal-projector headroom remains. The maximizing projector uses
oracle knowledge of these two probe vectors and need not be reachable by the
native observer or useful for learning. Lack of saturation is not an optimizer
defect or an argument to optimize this ratio.

Saved maximum basis-orthogonality and native/represented-action diagnostics
are approximately \(2.06\times10^{-6}\) and \(1.75\times10^{-7}\), respectively.
They are observed diagnostics, not a new global floating-point proof. No
historical audit or training run was reexecuted.

## 5. Consequence for the hypotheses and tests

The broad directional-regularization account survives. This follow-up weakens
a specific shortcut: neither “the residual is pure nuisance” nor “a small gap
is inevitable because the vectors oppose each other” is adequate here.

The next same-state mechanism measurements should separate:

- Clean \(g_y\), soft-target \(g_q\), and fixed-realization
  \(\varepsilon=g_{\rm fixed}-g_q\), with signed cross terms.
- Fresh batch variability around the **fixed training** objective, which the
  paired-batch diagnostic identifies, versus fresh replacement-label variability,
  which is a different randomization.
- Gradient retention, actual Adam displacement and independent loss changes.

The existing c/r Gram entries cannot generally reconstruct the missing
soft-target gradient. Computing it would require a separately scoped probe
acquisition; this note does not perform or authorize it. Clean labels used for
that research diagnostic are oracle information, not a deployable noisy-label
training assumption. A fixed-realization component at a learned state must not
be called conditionally zero-mean merely because its definition resembles a
fresh-label residual.

A later fixed-label versus fresh-redraw versus deterministic-soft-target policy
comparison could distinguish persistent assignment fitting from label-gradient
stochasticity. It changes the training data process and interacts with Adam;
it would be a separately designed experiment, not an equivalent rerun or
automatic causal decomposition.

Related work reinforces the distinction. [Müller, Kornblith and Hinton](https://papers.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)
report calibration and representation changes from label smoothing.
[Lukasik et al.](https://arxiv.org/abs/2003.02819) connect smoothing to
label-noise loss correction and find competitive results in their settings.
Neither establishes smoothing as the spectral filter's mechanism. Higher
accuracy with worse cross-entropy is compatible with multiple confidence and
classification changes; it does not identify underconfidence by itself.

## Scope and review