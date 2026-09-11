# Random-label gradients can probe feature geometry

Codex — Spectral Optimizer Investigation · 10 September 2026

Open the readable HTML explanation (artifact not distributed in this public snapshot).

**Status: conditional theory, not an experiment or a novelty claim.** The
proposed identities are correct with the assumptions below. The constructive
possibility is stronger than “noise has structure”: an appropriate covariance
projector can retain the entire expected learning signal while removing
nuisance variance. The same rule can instead remove that signal completely.

This specializes the existing [label-noise identifiability
analysis](spectral_label_noise_identifiability_2026-09-07.md), which already
derived Jacobian-shaped random-label covariance. Context follows the
[knowledge index](../knowledge/index.md) and [schema](../knowledge/schema.md).
No scientific archives, live outcomes, models or experiments were accessed.

## 1. Last-layer setup and exact uniform-label covariance

Freeze the feature map \(h(x)\in\mathbb R^d\) and evaluate at a fixed
\(W\in\mathbb R^{K\times d}\), with \(K\geq2\), logits \(z=Wh\), and
\(p=\operatorname{softmax}(z)\). Assume finite \(M=\mathbb E[hh^T]\).
Expectations below are over the specified input and **fresh** label draws,
not a changing training path. A bias can be included by appending a constant
feature; it then contributes to \(M\).

For cross-entropy, the matrix gradient and its column-major vectorization are

\[
G=(p-e_Y)h^T,\qquad g=\operatorname{vec}(G)=h\otimes(p-e_Y).
\]

Let \(u=\mathbf1/K\) and \(\Pi=I-\mathbf1\mathbf1^T/K\). If \(p=u\)
on the input support and \(Y\) is freshly uniform independently of \(h\),

\[
\mathbb E[u-e_Y\mid h]=0,\quad
\mathbb E[(u-e_Y)(u-e_Y)^T\mid h]=\Pi/K.
\]

Consequently,

\[
\boxed{\mathbb E g=0,\qquad \operatorname{Cov}(g)=M\otimes\Pi/K.}
\]

Here \(M\) is an **uncentered feature second moment**, not feature covariance.
The gradient is centered because its label-conditional mean vanishes; this
does not subtract the feature mean. Thus even a constant nonzero feature
creates gradient variation under random labels. “Feature PCA” below means
this second-moment eigenspace, not ordinary centered PCA unless \(\mathbb Eh=0\).

## 2. Spectrum, multiplicity and the matrix action

Let \(Mv_j=\lambda_jv_j\), ordered decreasingly, and let \(a_\ell\),
\(\ell=1,\ldots,K-1\), be any orthonormal basis perpendicular to
\(\mathbf1\). The nonzero spectrum consists of

\[
(v_j\otimes a_\ell,\;\lambda_j/K).
\]

Each positive feature eigenvalue has \(K-1\) class-contrast copies. Every
\(v_j\otimes\mathbf1\) is a null direction; zero feature eigenvalues add
further null directions. For \(r<d\), a strict gap
\(\lambda_r>\lambda_{r+1}\) makes the leading rank \(r(K-1)\) projector

\[
P_r=(V_rV_r^T)\otimes\Pi,\qquad V_r=[v_1,\ldots,v_r].
\]

Since every CE gradient column sums to zero,

\[
P_r\operatorname{vec}(G)
=\operatorname{vec}(\Pi G V_rV_r^T)
=\boxed{\operatorname{vec}(G V_rV_r^T)}.
\]

Thus the ideal parameter-space projection equals right-feature truncation
in this special case. Individual eigenvectors are not identifiable within a
class-contrast multiplicity. A cutoff through such a block, or a feature tie
at the cutoff, does not specify a unique feature-only projector. Arbitrary
rank choices cannot automatically be interpreted as a number of retained
features; particularly, nothing here identifies the actual full-network
global rank200 filter with feature PCA.

Feature scaling is consequential, not a harmless coordinate convention for
Euclidean spectral selection. Multiplying one feature by a constant changes
its second-moment weight quadratically, even if the classifier compensates
with an inverse weight scaling and keeps its predictions unchanged. Frequently
active or large-mean features can dominate \(M\) without being more predictive.
The useful interpretation therefore concerns the particular representation
and parameterization, not an invariant ranking of abstract concepts. A strict
population eigengap also does not guarantee that a short noisy buffer estimates
that subspace accurately.

## 3. Nonuniform predictions and partial replacement

Keep fully fresh uniform labels but allow \(p=p(h)\neq u\). Write

\[
g=\underbrace{h\otimes(p-u)}_{b(h)}
+\underbrace{h\otimes(u-e_Y)}_{\xi},\qquad
\mathbb E[\xi\mid h]=0.
\]

Total covariance then gives the exact correction

\[
\boxed{\mathbb E g=\mathbb E b(h),\qquad
\operatorname{Cov}(g)=\operatorname{Cov}(b(h))+M\otimes\Pi/K.}
\]

The extra term is positive semidefinite but need not commute with the
Kronecker term; its eigenvectors can change the leading subspace. The mean
now pushes predictions toward uniformity, not toward unknown true classes.

Instead take \(p=u\) and a joint true distribution \((h,y)\). With probability
\(\rho\), independently replace the label by a fresh uniform class; otherwise
keep \(y\). Define

\[
g_c=h\otimes(u-e_y),\quad \mu_c=\mathbb Eg_c,\quad
S_c=\mathbb E[g_cg_c^T],\quad S_u=M\otimes\Pi/K.
\]

These \(S\)'s are **second moments**. Mixture conditioning yields

\[
\boxed{\mu=(1-\rho)\mu_c,\qquad
\operatorname{Cov}(g)=\rho S_u+(1-\rho)S_c-(1-\rho)^2\mu_c\mu_c^T.}
\]

Equivalently this is \(\rho S_u+(1-\rho)\operatorname{Cov}(g_c)
+\rho(1-\rho)\mu_c\mu_c^T\). Omitting the subtraction changes the object
being diagonalized. For \(\rho<1\), the mean retains a scaled clean signal;
at \(\rho=1\) that supervised class signal disappears. Features may already
contain useful structure, but independent uniform labels supply no mapping
from that structure to the true class names.

## 4. A constructive retention case and its adverse counterpart

Take \(K=2\), \(W=0\), independent balanced signs \(s,n\in\{-1,1\}\),
and features \(h=(as,bn)^T\), where \(a,b>0\). The true label is class 1
when \(s=1\), class 2 otherwise. The second feature is independent nuisance.
Let \(c=(1,-1)^T/\sqrt2\). Then

\[
g_c=-\frac1{\sqrt2}(a,bns)^T\otimes c,\quad
\mu_c=-\frac a{\sqrt2}e_1\otimes c,\quad M=\operatorname{diag}(a^2,b^2).
\]

Here \(S_c=S_u=M\otimes cc^T/2\). Under partial uniform replacement, the
two contrast-space covariance eigenvalues are exactly

\[
\lambda_{\rm signal}=\frac{a^2\rho(2-\rho)}2,
\qquad \lambda_{\rm nuisance}=\frac{b^2}2.
\]

For \(\rho=.9,a=2,b=1\), these are \(1.98\) and \(.5\). The leading
rank-one projector retains feature 1 and therefore

\[
P\mathbb Eg=\mathbb Eg,\qquad
\operatorname{tr}\operatorname{Cov}(Pg)
=1.98<2.48=\operatorname{tr}\operatorname{Cov}(g).
\]

It preserves all expected noisy-objective and clean-signal direction while
removing independent nuisance variance. Equivalently, its gradient estimate
remains unbiased for that fixed noisy objective and has lower mean-squared
error about its mean. This is a genuine favorable mechanism, not just high
energy retention. It is not an Adam, finite-step generalization or trajectory
theorem.

Swap amplitudes to \(a=1,b=2\). The eigenvalues become \(.495\) and \(2\):
the same leading-eigenvector rule retains nuisance and removes **all** the
nonzero mean class signal. At \(\rho=0\), even the favorable-amplitude
example has zero signal-direction covariance: the perfectly consistent clean
mean is removed by centering, leaving nuisance variation. At \(\rho=1\),
feature geometry remains visible but there is no mean class signal to retain.
These algebraic cases delimit, rather than refute, the constructive account.

## 5. Conditioning, batching and the actual optimizer

Independent identically distributed fresh occurrences give covariance
\(\Sigma/B\) for a batch **mean**, not \(B\Sigma\). Conditional on fixed
inputs with independently redrawn uniform labels and \(p=u\), it is
\(B^{-2}\sum_i h_i h_i^T\otimes\Pi/K\). For a fixed dataset sampled without
replacement, the mean covariance instead has finite-population factor
\((N-B)/(B(N-1))\) multiplying the covariance normalized by \(1/N\).
Dependent occurrences require their cross-covariances.

Fixed labels reused during training are not fresh label draws. At learned
parameters they are also statistically entangled with features and predictions.
Across a full network, fresh uniform labels and \(p=u\) give the fixed-state
Jacobian pullback \(\mathbb E[J^T\Pi J]/K\), not generally a last-layer Kronecker
factorization. Cross-layer blocks, changing features, nonuniform predictions,
minibatch dependence, temporal mean subtraction, exponential memory and
truncation further separate the native observer from these population matrices.
Projection before Adam also differs from projection of the final displacement.

## 6. Prior work and diagnostic implications

Martens and Grosse's K-FAC already derives column-major activation/error
Kronecker products and approximates their expected product by the product of
expectations (§§1.2–2). The uniform-label, uniform-prediction case here is an
exact special case, not a new factorization. Truncation is not K-FAC's inverse
preconditioning. [Primary paper](https://proceedings.mlr.press/v37/martens15.pdf).

At \(p=u\), uniform label sampling equals model label sampling, so this
covariance coincides with the last-layer Fisher and affine-logit CE Hessian.
Generally those identities fail for data-label gradients. Kunstner, Balles
and Hennig distinguish Fisher, empirical gradient second moment and centered
covariance (§§2,4–5); subtracting the mean matters.
[Primary paper](https://arxiv.org/html/1905.12558v2).

Li, Soltanolkotabi and Oymak provide a close constructive precedent involving
clustered inputs, diffuse Jacobian structure and label corruption (§2,
Theorem 2.2; §4, Theorem 4.1). Their scalar squared-loss and restricted-corruption
results do not prove multiclass-CE robustness, a \(\rho=.9\) guarantee or
efficacy of this optimizer. [Primary paper](https://proceedings.mlr.press/v108/li20j/li20j.pdf).
See the companion [source review](../output/2026-09-10-spectral-feature-geometry/source-review.md).