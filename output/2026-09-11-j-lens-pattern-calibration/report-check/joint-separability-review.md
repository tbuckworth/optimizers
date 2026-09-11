# Joint directions and separable groups: independent algebra review

Verdict: the proposed account is correct, with the singular-case and distinguishability qualifications below. This is algebra only; no arrays, models, readers, grades or empirical measurements were loaded or run.

## Marginal versus joint patterns

Let centered x have finite covariance C, let UᵀU=I, set z=Uᵀx and S=UᵀCU. If S is positive definite, minimizing E‖x−Az‖² gives the normal equations AS=CU and the unique solution A=CU S⁻¹. Regressing x separately on each scalar z_j instead gives d_j=Cu_j/S_jj, hence D=CU diag(S)⁻¹. Therefore

```text
UᵀA = I,                  UᵀD = S diag(S)⁻¹,
A = U + (I−UUᵀ)CU S⁻¹.
```

Each marginal pattern contains the associations with other retained scores; each joint coefficient describes the linear association conditional on the other scores. These are regression descriptions, not causal interventions or unconditional semantic meanings. If the retained subspace is C-invariant, CU=US, so A=U even when S has nonzero off-diagonal entries. Conversely, positive definiteness makes A=U equivalent to that invariance. Correlated retained scores alone therefore do not force a new joint direction. Indeed A=D exactly when S is diagonal: multiplying a proposed equality by Uᵀ proves necessity, and the formulas prove sufficiency.

## Singular score covariance

If S is singular, some score combination has zero variance. For v in ker S, E(vᵀz)²=0, so vᵀz=0 almost surely and CUv=0. Joint coefficients are not uniquely identified. The minimum-Frobenius-norm solution is A₀=CU S⁺; every minimizer has form A₀+B(I−SS⁺), and all give the same predictions almost surely. Now UᵀA₀=SS⁺, not I, and

```text
A₀ = U SS⁺ + (I−UUᵀ)CU S⁺.
```

Marginal patterns remain defined for individual positive-variance scores even if the joint scores are linearly dependent; a zero diagonal entry makes that marginal pattern undefined. A pseudoinverse or ridge convention must be declared rather than silently treated as the invertible case. Near-singularity can make ordinary inverse coefficients sensitive to estimation error; ridge changes the estimand and loses UᵀA=I.

## Orthogonality is not token or cluster separation

For fixed linear L, the readout-column Gram matrix is (LU)ᵀ(LU)=UᵀLᵀLU. Its off-diagonal entries need not vanish. It determines inner products and whether L is injective on the retained subspace: positive definiteness means no retained nonzero contrast maps to zero. This is only linear distinguishability. Even distinct, linearly independent readout columns may induce identical vocabulary rankings. Top-token truncation, normalization and finite precision discard further information. Neither this Gram matrix nor input orthogonality establishes separated groups, readable names or semantic classes.

## Strong clustering steelman and covariance non-identifiability

Take 0<ε<1, independent equiprobable signs G,H, independent η∼N(0,ε²I₂), and a=√(4−ε²), b=√(1−ε²). The distribution x=(aG,bH)+η has mean zero and covariance diag(4,1), exactly matching N(0,diag(4,1)). Consequently the covariance, eigenvalues, eigenaxes and any fixed-L eigenvector readouts are identical. Covariance-based marginal/joint patterns also coincide for a common U. Yet the Gaussian is unimodal while the mixture concentrates around four corner means for small ε.

Joint coordinates retain both signs and permit arbitrarily low component-classification error as ε→0. A single PC cannot identify all four components: each first-coordinate distribution is shared by the two H values, and each second-coordinate distribution by the two G values. At every ε>0, Gaussian tails overlap, so perfect separation is false. The constructed latent components are not established semantic classes. This demonstrates a genuine reason to inspect several coordinates jointly, and why second moments alone cannot decide whether those groups exist.

## No automatic remedy

Joint calibration changes marginal reconstruction coefficients into conditional ones; it does not optimize score prediction from words or cluster discovery. If the retained subspace is invariant it changes nothing, and otherwise ill-conditioning, omitted nonlinear structure, vocabulary compression and reader inference still matter. Several eigenvectors can retain useful grouping information without individually naming groups. That is a principled possibility, not a new empirical result or a selected experiment.
