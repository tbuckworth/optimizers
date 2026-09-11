# Mathematical and implementation audit of the spectral optimizer

Evidence cut: 2026-09-06. Canonical repository commit inspected: `95c48ed208a144c5cf9b857dd6b6cd5caa96208e`. This note distinguishes algebraic facts, implementation facts, measured numerical examples, and empirical hypotheses. Paths below are repository-relative; line references refer to that commit. The associated classical evidence audit (artifact not distributed in this public snapshot) reconstructs the task results. analytical-checks.json (artifact not distributed in this public snapshot) records bounded CPU checks against the actual current implementation, without model training. Reproduce all eight checks with `python3 output/2026-09-06-spectral-optimizer-investigation/analysis/analytical_checks.py` from the repository root; the [script](analytical_checks.py) prints results and asserts the relevant invariants without changing recorded evidence.

## 1. The defensible central interpretation

The canonical hard filter is an **adaptive restriction of incoming gradients to a low-rank subspace of recent gradient variation**. It learns this subspace from the batch-mean gradient before filtering it. Consequently it changes the direction, norm, optimizer state, and subsequent trajectory of training. It can preserve useful structure while reducing late fitting of independently corrupted labels, but it has no mathematical test for semantic goodness, label correctness, or causal features.

“Coherence amplifier” is at most a mnemonic. The estimator is centered, its eigenvectors ignore sign, its state is repeatedly truncated, and the current observation participates in constructing its own projector. Those facts do not imply that a constant useful mean survives or that a transient, large-norm nuisance disappears. They also prevent interpreting the filter as a general low-pass smoother of the gradient sequence. The mean subtraction is itself a temporal high-pass operation; the subsequent operation projects in parameter space.

The right mechanistic question is therefore: **when does the covariance subspace estimated along this particular training trajectory retain the useful gradient and exclude gradients that drive undesirable fitting, after the base optimizer transforms both?** Existing accuracy curves provide strong evidence about selected outcomes, but do not by themselves measure this alignment.

## 2. Exact state update: what is being estimated?

Let `g_t ∈ R^p` be the flattened, raw batch-mean gradient over the trainable parameters selected by the base optimizer. Let `β` denote the covariance/mean decay, distinct from Adam's moment decays. `spectral_filter.py:322–326` first updates the mean and then subtracts it:

\[
m_1=g_1,\qquad m_t=\beta m_{t-1}+(1-\beta)g_t,
\]
\[
c_t=g_t-m_t=\beta(g_t-m_{t-1})=\beta\delta_t.
\]

Thus the first centered observation is exactly zero. The raw gradient, rather than `c_t`, is eventually projected. Ignoring finite precision, repeated truncation, and the initialization exception, the represented covariance evolves as

\[
\widetilde C_t=\beta C_{t-1}+(1-\beta)c_tc_t^T,\qquad
C_t=\mathcal T_r(\widetilde C_t),
\]

where `T_r` keeps the largest allowed positive eigenmodes. Its rank can also be limited by eigenvalue tolerances or the configured energy threshold. This is a low-rank **recursive approximation**, generally not the top-rank truncation of the complete historical covariance computed afterward.

### Relationship to a genuine exponentially weighted central covariance

Define normalized exponential weights with initialization at `g_1`, and let `M_t` be their exact central covariance. The weighted Welford identity is

\[
M_t=\beta M_{t-1}+\beta(1-\beta)\delta_t\delta_t^T,\quad M_1=0.
\]

With zero covariance initialization and without truncation, the code's recurrence would give `C_t = β M_t`: post-update centering changes the covariance by a common scalar, so it would not change eigenvectors. Calling the method centered temporal covariance is consequently justified; calling it the empirical Fisher or an uncentered gradient second moment is not.

There is an initialization exception in both stable and legacy paths: the first nonzero innovation, at step `j`, sets `V=c_j/||c_j||` and `S=||c_j||`, without the usual `sqrt(1−β)` observation factor (`spectral_filter.py:328–333`, `249–254`). Therefore, before truncation,

\[
C_t=\beta M_t+\beta^{t-j+1}c_jc_j^T,\quad t\ge j.
\]

The first innovation's covariance contribution is overweighted by `1/(1−β)` relative to a normal observation: 100 at `β=.99`, 1,000 at `.999`. Its excess decays, but repeated truncation can make early selection consequential after the excess itself becomes small. This is an implementation fact, not evidence that historical performance is caused by this effect. The CPU audit gives represented eigenvalue `.9801` versus regular EMA contribution `.009801` for the first change from `(0,0)` to `(1,0)` at `.99`, and checks the exact central-covariance-plus-initialization identity to Frobenius error `2.32e−16`.

### Memory and statistical scaling

For a scalar EMA of independent observations, the e-folding time is `−1/log β`, half-life is `log(.5)/log β`, and normalized-weight effective sample size is `(1+β)/(1−β)`. At `.99` these are approximately 99.5, 69.0, and 199 observations. These are not hard rank ceilings. A full exact EMA can have rank up to `min(p,t−1)` with arbitrarily small nonzero modes. A practical numerical rank also depends on eigenvalue thresholds, anisotropy, repeated truncation, observation scale, and whether training repeatedly visits the same batch.

For i.i.d. stationary `g_t = μ + ε_t`, with `Cov(ε_t)=Σ` and after the initialization transient,

\[
\operatorname{Cov}(c_t)=\frac{2\beta^2}{1+\beta}\Sigma.
\]

The stationary mean `μ` disappears. For independent mini-batch examples at a fixed model state, `Σ_batch≈Σ_example/B`; the mean term does not scale down in an uncentered second moment, but here it is centered away. Changing batch size changes the stochastic covariance amplitude, steps per epoch, amount of examples represented by an EMA window, and training dynamics at once. Such changes cannot be interpreted as changing only noise strength.

If all gradients are rescaled by a fixed positive factor `a`, ideal covariance scales as `a²` and hard eigenspaces are unchanged. The initialization also scales homogeneously; initialization alone does not break that invariance. Absolute cutoffs, residual thresholds, numerical precision and time-varying rescaling can break exact invariance. Adaptive rules operate on the retained spectrum, not on the unknown full spectrum.

## 3. Centering, drift and persistence: resolving the apparent contradiction

Ignoring initial conditions, the transfer function from the gradient sequence to the innovations is

\[
H(z)=\frac{\beta(1-z^{-1})}{1-\beta z^{-1}}.
\]

It has zero response at zero frequency. For a slowly varying gradient, `c_t` is approximately `β/(1−β)` times the change per step, provided that change is slow relative to the EMA window. A fixed direction whose coefficient changes during optimization can therefore dominate centered covariance even though its strictly constant component does not. In full-batch training, where mini-batch sampling noise is absent, covariance comes from optimization dynamics, model curvature, and any other actual stochasticity in the forward pass.

For a locally quadratic deterministic objective, `g_{t+1}−g_t≈Hessian_t Δθ_t`. The temporal covariance can consequently emphasize directions of changing gradients associated with curvature and optimizer motion. That observation does not establish equality with a Hessian, empirical Fisher, natural gradient metric, or semantic feature covariance. The Hessian can be indefinite; this covariance is positive semidefinite.

A useful decomposition is `g_t=μ_t+ε_t`. Innovations then mix drift of `μ_t`, mini-batch fluctuations, estimation lag, and their correlations. A coherent feature may be represented through variation in its strength across batches or over time. An unwanted backdoor may have exactly those properties. Repetition of a **direction with varying coefficient** can produce covariance; repetition of an exactly constant vector cannot. Sign is irrelevant to `c_t c_t^T`: alternating `+u` and `−u` can dominate even though their signed mean is zero.

### A decisive counterexample

Take `g_t=(1,ξ_t)` with varying zero-mean `ξ_t`. Initialize the running mean from the first gradient exactly as the code does. The first coordinate of every `c_t` is zero. A rank-one covariance basis is therefore `e_2`, and hard projection returns `(0,ξ_t)`: it removes the perfectly persistent useful mean `(1,0)` and retains the fluctuating nuisance. Running the current class on `ξ=[1,−1,2,−2]` gives final filtered gradient `(0,−2)` from `(1,−2)`.

Conversely, if every raw gradient is exactly the same vector from the beginning, the basis never initializes and the implementation returns the raw gradient. “Centering removes the mean” concerns which directions enter the covariance, not unconditional removal of the mean from every delivered gradient. These two cases explain why both a blanket persistence claim and a blanket claim that constant gradients are always deleted would be wrong.

## 4. The streaming eigensystem and numerical behavior

The state represents `C≈V diag(S²) V^T`, with `V∈R^(p×k)`. The stable updater (`spectral_filter.py:340–392`) computes `a=V^T c`, a residual `r=c−Va`, a second Gram–Schmidt correction, and, if significant, `q=r/||r||`. In the augmented coordinates `[V,q]` it diagonalizes

\[
K=\begin{bmatrix}
\beta\operatorname{diag}(S^2)+(1-\beta)aa^T & (1-\beta)a\|r\|\\
(1-\beta)\|r\|a^T & (1-\beta)\|r\|^2
\end{bmatrix}.
\]

Without a significant residual it solves only the upper-left block. Heavy operations stay on the model device; the small symmetric eigensystem is solved on CPU in float64. Residual acceptance requires `||r|| > max(1e−12, 10 eps_dtype ||c||)`; default eigenvalue retention requires `λ_i > 1e−8 λ_max`, with optional absolute floor (`182–204`, `348–352`). Periodic QR repair computes the small covariance `R diag(S²) R^T` after `V=QR`; thus it preserves the represented covariance before configured truncation, rather than merely replacing `V` by `Q` and silently changing eigenvalues (`206–224`).

The legacy path forms the Gram matrix of `A=[sqrt(β)Vdiag(S), sqrt(1−β)c]` and reconstructs left vectors via `A U Λ^(−1/2)` (`239–311`). That is algebraically valid with orthonormal `V`, but division by small singular values and the assumption that the old basis is orthonormal amplify finite-precision errors. Its eigensystem is usually float32 and its positivity threshold is a fixed `1e−12`. Historical experiments before the August 12 stability change used this implementation; their results should not be described as replicated current-stable behavior merely because current harnesses default to stable mode.

The stored numerical benchmark shows orthogonality error falling from `3.8461e−5` to `7.7135e−7`. This validates geometry, not a universal accuracy gain. Orthogonality drift also means a supposed hard projector need not be exactly idempotent or norm non-increasing. Rank-deficient, mixed-precision use needs diagnostics rather than relying solely on the mathematical projector idealization.

The dominant basis storage is `O(pk)`. Projection and residual updates are `O(pk)`; rotating an existing `p×k` basis by a dense `k×k` rotation costs `O(pk²)`, and the small eigensolve costs `O(k³)`. Scheduled QR similarly costs `O(pk²)`. “A tiny eigensolve” therefore does not imply the entire algorithm costs only `O(pk)`, nor is a historical approximate 2× optimizer overhead a general scaling guarantee.

### Repeated truncation can suppress accumulation of a real emerging direction

Suppose the retained rank-one covariance begins at `e_1 e_1^T`. With `β=.99`, feed innovations `c_t=e_2` repeatedly. These innovations can be realized by choosing `g_t=m_(t−1)+e_2/β`. At each update, the existing mode has variance `.99^t`, while the newly proposed `e_2` mode has variance `.01`; until the former falls below `.01`, the latter is discarded. After 100 updates, the exact full covariance is `diag(.3660,.6340)`, whose dominant vector is `e_2`; the actual rank-one sketch still represents only `.3660 e_1e_1^T`. The CPU audit reproduces this.

## 5. Warmup, current-observation inclusion, and rank adaptation

`filter_grad()` increments the count, updates the covariance using the current **unfiltered** gradient, then projects that same gradient once `step_count>warmup` (`spectral_filter.py:491–500`). The estimator receives observations during warmup, but the model is allowed unrestricted base-optimizer steps. Warmup does not guarantee the rank cap has filled: with 100 warmup steps and rank 200, at most 100 innovation directions exist when filtering first starts, before numerical or adaptive reductions.

Including the current observation is statistically consequential. With `P_(t−1)` independent of a fresh sample under a suitable stationary approximation, one can analyze retained noise using an independent projector. `P_t`, fitted using the very gradient being filtered, violates that assumption. A sufficiently large new innovation can enter a leading mode and largely preserve itself. In the CPU example, old covariance is `e_1e_1^T`, mean is zero, `β=.5`, and the new gradient is `10e_2`. Lagged projection returns zero, while current-update projection returns the entire `10e_2`. A one-step lag or independent covariance batch is therefore an essential falsification control, not merely a speed optimization.

\[
k_{\mathrm{eff}}=\mathrm{round}\{\exp[-\sum_i p_i\log p_i]\},\quad
p_i=\lambda_i/\sum_j\lambda_j,
\]

and the gap rule chooses the largest difference of adjacent log eigenvalues. These are spectral concentration statistics, not estimates of the number of directions necessary to learn a function. A low-variance direction may be indispensable. A low effective rank likewise does not prove that the learned circuit has that many parameters, features, or Fourier modes.

## 6. Hard, soft, residual and normalized policies are materially different

For an orthonormal basis selected for projection, let `P=VV^T`. Hard filtering delivers `h=Pg`. This never increases the Euclidean gradient norm; its retained squared-norm fraction is `g^T P g / ||g||²`. Partial strength `η_f<1` gives `h=(1−η_f)g+η_f Pg`, with eigenvalue 1 in the retained space and `1−η_f` outside. Here `η_f` is filter strength, not learning rate.

For soft weighting (`spectral_filter.py:438–455`),

\[
w_i=\min\{10^3,[\max(\lambda_i/\lambda_{\max},10^{-12})]^\alpha\}.
\]

With `soft_residual=False`, the preliminary vector is `z=V diag(w) V^T g`. With the **default `soft_residual=True`**, it is

\[
z=(I-P)g+V\operatorname{diag}(w)V^Tg.
\]

Both are then rescaled to `h=||g|| z / max(||z||,1e−12)`. Exactly zero remains zero; extremely tiny norms also break exact norm matching because of the clamp. For ordinary nondegenerate vectors, the raw-gradient norm is matched. Subsequent partial-strength mixing can change that norm again.

The important consequences are:

- For positive `α`, tracked low-eigenvalue directions are attenuated. With residual enabled, the **untracked complement remains at weight 1**, the same as the leading eigenvector before norm matching. This is not monotonically increasing weighting over the complete spectrum.
- `α→∞` retains the leading tracked eigenspace **and the untracked complement** when residual is enabled. It does not generally collapse to the top direction, contrary to comments near `spectral_filter.py:76–77`. In the numerical example `V=[e_1,e_2]`, `S=(2,1)`, `g=(1,1,1)`, `α=100`, output is `(1.2247,0,1.2247)`.
- `α=0` with residual is an identity control. Without residual it is a **norm-matched** hard projector, not the unnormalized hard mode; the code comment calling it simply equivalent to hard filtering omits that scale distinction.
- Negative `α` upweights weaker retained modes subject to clipping. It is not a natural-gradient method merely by resemblance: the covariance is centered temporal covariance in raw coordinates, the complement policy is different, and Adam further transforms the output.

The normalized policy (`459–475`) first forms the existing low-rank factor `L=Vdiag(S)`. Variance normalization uses `D_ii=(LL^T)_ii`; degree uses `D_ii=|sum_j(LL^T)_ij|`. After flooring `D`, it forms `A=D^(−1/2)L` and delivers

\[
h=A(A^TA+10^{-6}I)^{-1}A^Tg.
\]

This is a regularized projection onto the column space of the **diagonally transformed existing low-rank approximation**. In the zero-ridge idealization, that space is the nonzero eigenspace of `D^(−1/2) C_lowrank D^(−1/2)`. It is not generally the leading eigenspace obtained by first normalizing the full unknown covariance and then truncating. The implementation neither tracks that full normalized covariance nor recovers discarded modes. The branch ignores `proj_k` and does not execute the soft-weighting branch. Signed-covariance row sums are not graph degrees in the usual nonnegative-affinity setting; taking absolute values is an implementation choice, not a canonical spectral clustering identity.

## 7. Adam and AdamW: gradient subspaces are not update subspaces

If `h_t=P_t g_t`, Adam forms coordinatewise moment states

\[
a_t=\beta_1a_{t-1}+(1-\beta_1)h_t,\qquad
b_t=\beta_2b_{t-1}+(1-\beta_2)h_t\odot h_t,
\]
\[
\Delta\theta_t=-\gamma_t\,\widehat a_t/(\sqrt{\widehat b_t}+\epsilon).
\]

A diagonal transformation generally does not preserve an arbitrary rotated subspace: `DP≠PD`. Even if `P` is fixed and momentum remains in its range, the final preconditioned update can escape. If `P_t` rotates, momentum already combines several historical subspaces. AdamW adds `−γ_t λ_wd θ_t`, which need not lie in the current subspace either. The global filter acts before `base_optimizer.step()` (`spectral_filter.py:518–520`), and the random control uses the same placement (`experiments/random_subspace_optimizer.py:58–64`).

The two-dimensional example is explicit. Project gradients onto `u=(1,2)/sqrt(5)`. Adam's first update from zero state is approximately `−γ(1,1)`, and 31.62% of its norm lies outside `span(u)`. This is observed in the CPU check. For ablation of a perpendicular direction, the same phenomenon means that projecting an unwanted direction out of the raw gradient does not remove it from the parameter update.

Thus the random-control comment that parameters stay in `θ_0+span(P)` is false for its actual Adam use. That statement would hold for a fixed projected-gradient basis under plain SGD without outside-subspace terms; it also holds for deliberately parameterizing `θ=θ_0+Vz` and optimizing `z`. Coordinate-space Adam in `z` is yet another optimizer, not equivalent to ambient Adam on `VV^Tg`.

Similarly, matching the raw filtered gradient norm does not match the actual optimizer-step norm or optimizer state. Adam approximately cancels a **constant positive** rescaling of its entire gradient history when epsilon is negligible, but per-step scaling, per-block scaling, changed directions, epsilon, and evolving moments defeat a blanket scale-invariance argument. Claiming that `alpha` is completely decoupled from effective learning rate is too strong.

Required observables are `||(I−P_t)Δθ_t||/||Δθ_t||`, raw and filtered gradient norms, moment norms, preconditioned-step norms, and the weight-decay contribution. A geometry control should project actual updates, with a clearly specified policy for moments and decay. That control changes the algorithm and should be evaluated as a distinct condition.

## 8. Per-matrix and LoRA geometry

The matrix wrapper partitions optimizer-owned trainable parameters, builds a separate temporal covariance per matrix, and optionally combines matching affine biases (`matrix_spectral_filter.py:73–156`). It does not take an SVD of each gradient matrix. A matrix is flattened into a parameter block and its **temporal covariance** is tracked. With blocks `j`, hard filtering is a block-diagonal projector `diag(P_1,…,P_J)` on filtered blocks, with identity on intentionally unfiltered vectors.

Storage is `Σ_j p_j k_j`; basis rotations cost approximately `Σ_j p_j k_j²`, and small eigensolves `Σ_j k_j³`. Smaller ranks can save substantially, but using the same rank cap everywhere is not automatically cheaper than a global basis, and total retained dimension can be `Σ_j k_j`. Cross-block temporal covariance is discarded. Rank allocation determines which layers may vary independently; matching global rank to per-block rank numerically is not a matched-capacity experiment. Matching storage, total dimension, clean accuracy, gradient energy retention, and actual-step distortion answers different questions.

For LoRA, `W=W_0+sBA`, so

\[
\Delta W=s(B\Delta A+\Delta B A+\Delta B\Delta A).
\]

The first-order effective-weight tangent is `s(BΔA+ΔB A)`; a model-output tangent additionally applies the model's Jacobian with respect to `W`. Independent projectors on flattened `A` and `B` neither impose a common global effective-weight subspace nor directly bound functional change by their nominal covariance rank. There is a gauge freedom `(A,B)→(RA,BR^(−1))` for invertible `R`; this preserves `BA` but generally changes raw parameter-gradient covariance, diagonal optimizer scaling, and the spectral filtering intervention. The canonical method is not invariant under this reparameterization.

At usual initialization `B=0`, `∂L/∂A=0` whereas `∂L/∂B` can be nonzero. Separate blocks consequently see different early histories, especially with the first-innovation initialization exception. Soft norm preservation occurs **within each block**, not just globally, so it changes relative layer behavior differently from a single global rescaling. The code supports optimizer subsets and excludes frozen weights; this makes storage tractable but does not establish a generic LoRA quality advantage.

## 9. Historical algorithms: similar names, different estimands

The earliest `SpectralConsensusFilter` (`experiments/spectral_optimizer.py:72–112`) forms per-example gradients as rows of `G∈R^(B×p)`, normalizes each row, and diagonalizes their cosine-similarity Gram matrix `S=G_norm G_norm^T`. For a selected sample-space projector `Q=U_kU_k^T` and uniform sample weights `q=1/B`, it returns

\[
h=G^TQq.
\]

Because eigenvectors come from **row-normalized** `G` but the weighted update uses **raw** `G`, this is not generally the same as orthogonally projecting the batch-mean gradient onto the leading parameter-space singular vectors of raw `G`. With equal row norms, the usual SVD identity recovers that equivalence. Unequal sample gradient norms matter. The sample weights can be negative and need not sum to one. A mode with high eigenvalue but zero overlap with the uniform weight vector contributes no update. The code's threshold is `mp_factor × mean eigenvalue`, not an exact Marchenko–Pastur upper edge derived from aspect ratio and noise assumptions. Calling it an MP test should remain heuristic.

The historical hard branch forces at least one eigenvector to survive (`experiments/spectral_optimizer.py:107–110`), even if none exceeds the heuristic threshold. In particular, for batch size one the filter is the identity on the example gradient: the executable eighth CPU check returns `(1,−2,3)` unchanged. There is no theorem that it cannot learn random labels. The soft sigmoid branch also retains nonzero finite-temperature weights. High-dimensional approximately orthogonal examples can produce low aggregate consensus under a particular threshold, but that is a contingent geometric regime, not an inability to fit arbitrary labels by construction.

An intermediate implementation, `experiments/legacy/weight_cov_optimizer.py:70–102`, computes **within-batch mean-centered per-example gradients**, stacks their scaled rows with the decayed historical factor, and recomputes an SVD. It estimates an EMA of within-batch gradient covariance, distinct from both normalized sample-space Gram filtering and temporal covariance of batch means. The initial factor uses unscaled centered rows, whereas subsequent batches use `sqrt((1−β)/B)`; this gives another initialization scale exception. Results under `results/weight_covariance/` belong to this intermediate variant and should not be relabeled as current temporal-filter trials.

The later `PerSampleCovarianceFilter` (`experiments/persample_cov_optimizer.py:44–107`) instead tracks

\[
C'_t=\beta C'_{t-1}+(1-\beta)B^{-1}\sum_i g_{ti}g_{ti}^T
\]

through a rank-B streaming Gram solve. This is an **uncentered per-example second moment**, so it contains `μ μ^T` as well as covariance. It is not the canonical centered batch-mean temporal method. It also uses legacy-style singular-vector reconstruction, so estimator and numerical differences coexist in historical parity comparisons. The raw parity failures of this implementation are evidence against those tested recipes, not proof that all per-example second-moment filtering is inferior.

## 10. When can filtering help, and what would falsify the explanation?

For a projector independent of fresh noise, write `g=s+ε`, with `Eε=0`, `Cov ε=Σ`. Then `E[Pg]=Ps`, retained signal power is `||Ps||²`, and retained noise power is `tr(PΣ)`. A useful filter needs high alignment with the desired gradient and low retention of damaging noise or nuisance. Covariance magnitude alone does not impose either property. In particular, in the stationary additive-noise counterexample, the covariance is entirely noise covariance.

For plain SGD and a smooth objective with gradient `s`, the first-order expected decrease from `Pg` is proportional to `||Ps||²≤||s||²`. Benefits can arise by reducing second-order curvature/noise cost, avoiding bad finite-sample fitting, changing later representation learning, or interacting with regularization. Under Adam these quantities must be evaluated after preconditioning. The filter does not need to improve one-step clean descent to improve eventual held-out accuracy, and a reduction in raw gradient norm is not by itself evidence of selective denoising.

The following tests discriminate competing explanations and should be preregistered before looking at their main outcomes:

| Hypothesis | Measurements and intervention | Outcome that weakens it |
|---|---|---|
| Useful gradients lie in retained covariance directions | At matched checkpoints, estimate clean-label gradients and corrupted-label/trigger gradients on independent probe batches; compare projected energy and cosines with the mean and covariance basis | Useful-gradient retention is no higher than nuisance retention while gains persist |
| Temporal variation adds information beyond a mean direction | Compare centered covariance, uncentered temporal second moment, mean-only direction, and mean-augmented covariance basis with matched total rank and update norms | Mean-only or scalar norm controls recover all gains |
| Covariance filtering suppresses fresh noise rather than preserving its own current observation | Use one-step-lagged basis and independent covariance batches; measure current-minus-lagged retained energy | Only same-observation filtering helps, or spikes systematically install themselves |
| Learned rotation matters beyond learned orientation and warmup | Compare fixed random, random rotating, covariance learned during warmup then frozen, live covariance, and matched warmup for all | Frozen learned basis matches live basis; random rotating or matched learning-rate controls suffice |
| Rank selection, not recurrent evidence loss, explains weak-feature failure | Hold projection rank fixed while varying estimation rank; include exact covariance on a small model | Wider estimation basis rescues weak features without changing delivered rank |
| Accuracy gains are not merely reduced effective optimization | Match raw-gradient norms, separately match actual-step norms and decay/data-step ratios, and compare at matched clean accuracy and training loss | Advantages disappear under these controls |
| Backdoor failure reflects distributed representation rather than optimizer leakage | Project actual Adam updates or use matched SGD, track forbidden-direction leakage, compare gradient/activation probes and retraining from step zero | Eliminating update leakage makes small-direction prevention work |
| LoRA results reflect functional geometry rather than arbitrary factor scaling | Apply function-preserving invertible changes of LoRA coordinates; compare trajectories and effective-weight updates | Results change markedly under equivalent factorizations |

Norm controls require care: a shadow optimizer updated with counterfactual gradients defines one-step counterfactuals, while a separately trained baseline defines an actual competing trajectory. Those are different estimands. Weight-decay-to-data-update balance should be logged in grokking, because shrinking or redirecting the gradient while retaining AdamW decay may change regularization pressure without discovering a special generalizing direction.