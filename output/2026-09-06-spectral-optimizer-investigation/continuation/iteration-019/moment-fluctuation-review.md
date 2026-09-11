# Independent review of I19 moment-fluctuation theory

**Status: PASS**
**Date:** 8 September 2026

## Reviewed artifacts

- `moment-fluctuations.md`: SHA-256
  `27ffdd5464ba8eff68f7fbee75461137128a13cf6702d1d133df05423da01e3a`
- `moment-fluctuation-check.py`: SHA-256
  `d72a527dc0ced75391c6e9aacd11eeab01cea2a07cf653b73b388a3967e7df1f`

I read both artifacts completely, independently re-derived the displayed
autocovariance and weighted-moment identities, and ran the exact deterministic
checker. It returned `status: pass`. I did not read I19 scientific arrays or
outcomes, invoke the native observer, or modify any frozen acquisition source.

## Findings

The residual representation and both autocovariance formulas are correct for
the infinite-past stationary Gaussian residual process in the diagonal
generating coordinates. Applying Isserlis's identity and summing both temporal
orientations gives the stated diagonal and off-diagonal EW-moment variances.
The exact fractions reproduce the reported population gap `1.970050251`,
`SD(M11)=4.030810647`, `SD(M22)=0.395327802`, and
`SD(M12)=0.291327409`.

The separate covariance-decay result is also correct:

\[
\operatorname{CV}^2=2\frac{1-\lambda}{1+\lambda}
\frac{1+\lambda\beta^2}{1-\lambda\beta^2}.
\]

At `lambda=beta` its limit as `beta` approaches one is `2/3`; at fixed
`beta<1`, its limit as `lambda` approaches one is zero. The note correctly
qualifies finite initialization, slower convergence, other joint limits,
measurement noise, eigengap and off-diagonal effects, and the distinction
between the full diagnostic and native truncated rank-one state.

During review, one sentence incorrectly suggested that elapsed run length
alone improves the precision of an across-seed current-state mean at fixed
seed count. The final pinned note corrects this: elapsed time removes startup
effects, while independent seeds improve across-seed precision and a temporal
average would require a separate serial-dependence calculation.

No remaining mathematical or interpretive defect was found. In particular,
the note does not infer a wrong-direction probability or claim measured native
learnability from a population gap and marginal moment variances.
