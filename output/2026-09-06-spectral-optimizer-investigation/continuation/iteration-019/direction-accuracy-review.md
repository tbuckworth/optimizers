# Independent review of the fixed-direction accuracy boundary

**Status: PASS**
**Date:** 8 September 2026

## Bindings and scope

- `direction-accuracy-boundary.md`: SHA-256
  `06ab3ee8e3b3b31fc7b4c69629dffdc3b78488e6b941b1c713f2e82479a4f4dc`
- `check_direction_accuracy.py`: SHA-256
  `8622228d1c647b6393cab43977dc18676d10d6e733623011bb4d1b65cddc664e`

I read the note completely, independently re-derived the trace risk and
alignment threshold, and ran the deterministic checker. It returned
`status: pass` with 22 checks. The checker uses no RNG, native observer or
experimental array. I also checked the two quoted alignment means against the
accepted I19 summary; both are available over 32 independent seeds.

## Mathematical findings

For a fixed rank-one orthoprojector `P=uu^T`, the process contributions have
squared energies `a=(u^T e1)^2` and `1-a` in the two orthogonal response
subspaces. The measurement contributions are
`tr(PR)=aR1+(1-a)R2` and `tr((I-P)R)=(1-a)R1+aR2`. Because the delivered error
vectors lie in orthogonal subspaces, their cross term vanishes pointwise.
These facts give exactly the displayed trace risk and the affine identity

`J(a)=a J_useful+(1-a) J_nuisance`.

In both registered fast-decay cases,
`J_useful < J_common < J_nuisance`; therefore
`J(a)<J_common` is equivalent to
`a>(J_nuisance-J_common)/(J_nuisance-J_useful)`. Direct evaluation reproduces:

| Fast decay | Useful risk | Nuisance risk | Alignment threshold | Angle |
|---:|---:|---:|---:|---:|
| .9 | .499047871 | 5.140677070 | .965567161 | 10.6938° |
| .729843788 | .290256714 | 5.554845801 | .929982070 | 15.3437° |

The endpoint checks are also coherent: `a=1` and `a=0` recover the useful and
nuisance routes, while setting both response decays equal makes risk
independent of projector orientation.

## Interpretation

The empirical late native alignment `0.755563436` cannot be inserted into this
formula: the native action moves, is observation-dependent and is coupled to
its response history. The accepted full-moment late alignment is lower,
`0.615853754`, so the evidence does not support blaming rank truncation alone
or claiming that longer covariance memory is already known to fix the result.
The note states both boundaries explicitly and treats memory separation and a
saved-stream counterfactual as proposed discriminators rather than established
remedies.

No mathematical defect or causal overclaim remains in the pinned note.
