# Review of the covariance-tracking steelman

Independent theory review, 8 September 2026. I read the complete note, its rational checker, and the canonical observer's post-ingest mean and stable/legacy covariance updates. I did not read live I17 outcomes, load scientific tensors, or run a model, dataset, acquisition or audit.

## Verdict

**PASS with no mathematical or interpretive defect found.** The note establishes a precisely conditioned population tracking example, not a neural result or a guarantee about the native finite-rank estimator.

Reviewed hashes:

- `covariance-tracking-steelman.md`: `45a8fb545b80f93bc87d1d4a690cc2aca1ba270c49513c917e7a7dc8164b63b8`
- `check_covariance_tracking.py`: `72068edd81210dc0adb030789bad4f0fde321d4bc6f8613dd1a988024e1d4073`

## Algebra checked

For post-ingest \(\mu_t=\beta\mu_{t-1}+(1-\beta)g_t\) and \(z_t=g_t-\mu_t\), linear drift gives \(E[z_t]=\beta a/(1-\beta)\). Writing the EMA noise as \(u_t=\beta u_{t-1}+(1-\beta)\varepsilon_t\) retains the current-sample correlation \(\operatorname{Cov}(\varepsilon_t,u_t)=(1-\beta)\Sigma\). Consequently

\[
\operatorname{Cov}(z_t)
=\Sigma+{1-\beta\over1+\beta}\Sigma-2(1-\beta)\Sigma
={2\beta^2\over1+\beta}\Sigma.
\]

Adding the nonzero residual mean outer product gives the displayed second moment. This is exactly the post-mean centering order in `spectral_filter.py`; treating the current gradient and updated mean as independent would be wrong.

At \(\beta=.99\), \(a=(.03,0)\), and \(\Sigma=\operatorname{diag}(1,4)\), the exact diagonal entries are 9.80592512562814 and 3.94010050251256, so the first coordinate is the unique population rank-one direction. The threshold calculation is also exact:

\[
a_1^2>{6(1-\beta)^2\over1+\beta}=0.0003015075376884422.
\]

The routed risk 0.14563208145993123 follows from \(81(.03)^2+1/19+4/199\). For a shared-decay EMA with \(L=q/(1-q)\), the risk is \(.03^2L^2+5/(2L+1)\). The split at \(L=14\) gives the universal lower bound \(5/29=0.1724137931034483\), strictly above routed risk: when \(L\le14\), the noise term alone supplies the bound; when \(L\ge14\), the drift term is at least 0.1764. This covers every \(q\in[0,1)\).

The standard-library rational check passed and reproduced every displayed coefficient, threshold and inequality.

## Scope preserved

The note correctly distinguishes the limiting expectation of an untruncated exponentially weighted residual second moment from the actual estimator, whose initialization, finite memory, rank truncation, numerical representation and moving neural trajectory need not recover that direction. The drift contribution is explicitly a residual mean-square term, not centered noise covariance.

The dominance claim is limited to uniform single-decay EMAs in this exogenous two-coordinate tracking problem. It does not cover all isotropic temporal filters, the I17 scalar-mixture family, trend estimators, moving actions, covariance learnability in finite samples, or neural clean-risk convergence. The note neither interprets the live I17 run nor admits another arm.
