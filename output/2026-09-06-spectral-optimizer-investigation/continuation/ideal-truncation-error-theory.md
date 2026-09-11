# Ideal recursive truncation: error, capture regret and a width counterexample

6 September 2026. Separate theoretical follow-up, not an iteration005 result or
protocol amendment. The identities below are exact-arithmetic statements; the
counterexample uses exact rational covariance entries. There is no MNIST,
optimizer-policy, generalization or GPU evidence here.

## 1. The PSD error recursion is exact under explicit assumptions

Fix any common innovation sequence c_t, decay 0<beta<1, and nonnegative
observation weights eta_t. Let C be the uncapped reference and A the ideal
rank-k recursive estimate:

    C_t = beta C_(t-1) + eta_t c_t c_t^T
    B_t = beta A_(t-1) + eta_t c_t c_t^T
    A_t = T_k(B_t),

where T_k keeps the largest k eigenvalues/eigenvectors of a PSD matrix without
altering retained eigenvalues. Define discarded covariance R_t=B_t-A_t and
reference error E_t=C_t-A_t. Subtraction gives

    E_t = beta E_(t-1) + R_t.

Every R_t is PSD. Hence E_s PSD implies E_t PSD, and A_t is a PSD underestimate
of C_t. With matching startup E_s=0,

    E_t = sum_(j=s+1)^t beta^(t-j) R_j,
    trace(E_t) = sum_(j=s+1)^t beta^(t-j) trace(R_j).

For canonical initialization, s is the first accepted nonzero innovation,
eta_s=1 and eta_t=1-beta thereafter. With k>=1, A_s=C_s=c_s c_s^T and R_s=0.
Earlier zero innovations contribute nothing. Regular-weight initialization also
satisfies the theorem if the reference and estimate both use that convention;
mixing conventions does not.

The statement needs neither stationarity nor independent samples: it is
conditional on the same realized innovations and weights. It cannot compare
estimators that induce different training trajectories by silently substituting
one stream for the other. Variable caps or additional exact positive-eigenmode
deletion are also allowed provided each deletion leaves a PSD remainder.

## 2. PSD underestimation bounds matched-rank captured-energy regret

For 1<=r<=p, let P_C and P_A be rank-r orthogonal projectors maximizing captured
energy for C and A respectively. Write F_r(M)=sum of its r largest eigenvalues.
The captured-reference-energy regret is

    L_r = F_r(C) - trace(P_A C) >= 0.

Because P_A maximizes trace(P A),

    L_r = trace((P_C-P_A) A) + trace((P_C-P_A) E)
        <= trace(P_C E) - trace(P_A E)
        <= F_r(E) - trace(P_A E)
        <= F_r(E).                         [last inequality uses E PSD]

Thus the proposed bound **holds**. In particular,

    L_r <= F_r(E) <= min(r ||E||_op, trace(E)),
    L_r <= sum_j beta^(t-j) F_r(R_j).

The last line follows from the subadditivity of the largest-r-eigenvalue sum
on PSD matrices. For F_r(C)>0, divide the bounds by F_r(C) to bound the loss in
optimal-energy fraction. These are capture bounds, not clean/noise selectivity
bounds. Small eigenvalue gaps can still permit substantially different leading
spaces with similar capture.

The proof does not require a unique rank-r eigenspace: any maximizing
orthoprojector works at a tie. If A has fewer than r positive modes, a rank-r
maximizer mathematically completes it with zero-eigenvalue directions. A native
filter that instead uses only its available columns is a different-rank object;
the statement must not be applied to it as though the missing directions existed.

For signed E, the sharper intermediate bound F_r(E)-trace(P_A E) remains valid,
but dropping the second term is unjustified. A uniform signed-error bound is
F_r(E) minus the sum of its r smallest eigenvalues, at most 2r||E||_op. Native
approximately orthogonal operators are not automatically eligible projectors.

## 3. Wider storage does not guarantee better matched-rank capture

Here is a counterexample using only diagonal matrices, three axes A/B/C, and
six nonzero innovations. There are no relevant positive-eigenvalue boundary ties
and no rotation or numerical instability. Both estimators receive the same stream.
The evaluated rank is r=1; storage widths are k=1 and k=2.

Set beta=9/10. The **unit covariance additions** follow axes A,B,A,B,C,C.
Under canonical startup, the first innovation has norm 1 and subsequent
innovations have norm sqrt(10), so that eta_t||c_t||^2=1 each time. These are
not equal raw innovation norms. In actual centered-mean indexing, prepend the
initial zero innovation and take these six nonzero innovations at steps 2..7.
Any desired innovations can be realized in exact arithmetic by
g_t=m_(t-1)+c_t/beta, followed by the canonical mean update.

Each tuple below is a diagonal in fixed A,B,C order. The terminating decimals
are exact; every fraction and all three decay cases are retained in the checks JSON.

| Nonzero innovation | Uncapped C | Storage 1 | Storage 2 |
|---|---|---|---|
| A | (1, 0, 0) | (1, 0, 0) | (1, 0, 0) |
| B | (.9, 1, 0) | (0, 1, 0) | (.9, 1, 0) |
| A | (1.81, .9, 0) | (1, 0, 0) | (1.81, .9, 0) |
| B | (1.629, 1.81, 0) | (0, 1, 0) | (1.629, 1.81, 0) |
| C | (1.4661, 1.629, 1) | (0, 0, 1) | (1.4661, 1.629, 0) |
| C | (1.31949, 1.4661, 1.9) | (0, 0, 1.9) | (1.31949, 1.4661, 0) |

At the same state, the wider **full covariance** estimate has smaller squared
Frobenius error: 361/100 versus 38905030701/10000000000. This is an explicit
case of better unequal-capacity covariance reconstruction but worse rank-one
capture. It does not prove monotonic covariance-error improvement in general.

The **capture reversal** holds whenever beta^5+beta^3>1. This condition does
not guarantee the preceding better-Frobenius-error pattern. The final reference is

    diag(beta^5+beta^3, beta^4+beta^2, 1+beta).

For every 0<beta<1, the third entry exceeds the second, which exceeds the first.
The stated additional condition makes both C observations be discarded by the
width-two estimate. At beta=.99, the wider capture fraction is exactly
194069601/199000000 = 0.9752241256281408, still below 1. Its later innovation
norms are 10 rather than the startup norm 1.

Each estimator individually satisfies 0<=A<=C, but the two estimates need not
be Loewner-ordered: their final difference has two positive diagonal entries and
one negative entry. The PSD theorem provides no cross-width ordering. With
consistent eigenbasis/tie choices, a single truncation of one common matrix has
nested top spaces; independently selected optimal tied spaces need not be nested.
Recursive estimators
have different intermediate matrices. That distinction defeats the naive
monotonic-width argument.

## 4. Why the ideal PSD theorem is not exact for production

These code statements refer to `spectral_filter.py`, SHA256
`9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`.

- Mean arithmetic and startup occur at lines 322–333. A shared reference on
  saved rounded innovations avoids changing the data stream, but normalized
  float32 V and the rounded norm S do not exactly reconstruct the real-arithmetic
  c_s c_s^T. Thus the initial error need not be exactly zero or PSD.
- The residual floor at lines 348–352 rejects small **innovation components**,
  not merely PSD eigenmodes. Even in otherwise exact arithmetic, replacing
  c=u+v by its represented component u gives observation error
  eta(uv^T+vu^T+vv^T), generally indefinite.
- For example, take an old exact basis e_1 and c=(1,delta), where nonzero delta
  falls below that floor. Before any new rank truncation, the reference minus
  the floor-rejected update is eta[[0,delta],[delta,delta^2]]. Its determinant
  is -eta^2 delta^2<0, so it has a negative eigenvalue. Tiny rejection can break
  the *exact* PSD-underestimation property even if its norm is small.
- In contrast, exact positive-eigenvalue pruning at lines 182–204 does preserve
  PSD underestimation; it contributes additional discarded PSD mass. It should
  not be blamed indiscriminately for the preceding signed error.
- Exact QR repair at lines 206–224 preserves V diag(S^2) V^T before its spectral
  pruning. If it discards PSD J before the next update, the error recursion can
  include beta J. Production QR/rotation and small-matrix construction use
  mixed precision (lines 209–222, 356–391), and nonorthogonal stored V makes the
  coefficient algebra only approximate. Those defects need not be PSD.
- A reset followed by fresh overweighted initialization does not follow the
  single-initialization reference convention. Such an event must be separately
  modeled or gated, as in the iteration005 design.

For an arbitrary implemented A_t one can always *define*
R_t=beta A_(t-1)+eta_t c_t c_t^T-A_t and recover the algebraic identity. What is
lost is the guarantee R_t>=0 and its interpretation as discarded spectral mass.
Moreover, QR of the first r nonorthogonal stored columns need not be the exact
top-r eigenspace of the complete represented covariance, and native B B^T need
not be an orthoprojector. The ideal regret theorem should not be relabeled a
certified bound on those production diagnostics without checking its assumptions.

## 5. Reproduction and evidential status

The rational counterexample is a proof by construction, not a seed-selected
stochastic finding. [check_ideal_truncation_error.py](check_ideal_truncation_error.py)
prints the exact tables and also checks the error recursion, PSD property and
sharper regret inequality on 27 tiny NumPy float64 streams: 324 updates and
540 rank-specific inequalities. Those checks all pass at tolerance 1e-11;
maximum relative recursion residual is 1.29e-16 and maximum normalized sharper-
bound violation is 4.36e-16. Floating residuals are diagnostics, not the proof.
The script does not import or execute the production optimizer, train a model,
read MNIST or write experiment outputs.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 output/2026-09-06-spectral-optimizer-investigation/continuation/check_ideal_truncation_error.py
```

Exact output and source hash are in
ideal-truncation-error-checks.json (artifact not distributed in this public snapshot).
No iteration005 files, knowledge pages or prior frozen artifacts are changed.
