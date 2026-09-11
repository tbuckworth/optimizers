# I16 prospective design review

7 September 2026. This is a read-only review completed before any I16
acquisition root or attempt existed. I inspected the fixed scientific question,
the scalar recurrence, its implementation, the acquisition runner, and the
synthetic tests. I did not run training, inspect I16 outcomes, or replay old
models or forwards.

Status: PASS

## Frozen review inputs

- `protocol.md`: `dfa8c3202516349931f8242b9f5d896199785abeb222aed5a9ee6eb14f77a346`
- `predictions.md`: `c65e8535f8b89a9f40135302a7e027dfbf2175ccf0da4dc20039cbfbd089d3ee`
- `scalar_core.py`: `5145feb59beb87a4df16d68164f4b9fc839150c9d95452726067a39b7ff0efee`
- `test_scalar_core.py`: `5be34cbe77f9a351ee0a49796d402fe73058f9112e8cd142152884445e16713a`
- `run_scalar_controls.py`: `23d3a9c42ed28f2a047246735e97ed76cd1416cd2863f9187ebec83685a68f68`
- `test_scalar_runner.py`: `7cf4207b16a36f9e2d40dcd631ba2f28cda871f7d5e39019201fc263bb593414`
- supporting `temporal-regularization-note.md`:
  `31709f6cd0a7f755931a7a675a3d33ef3e041ccdab49a42305de34ab7362ffbe`

The core owner reported 7/7 CPU tests passing. The independent runner reviewer
reported the combined scalar-core/runner suite at 17/17 passing after the
binding and numerical-failure fixes. I did not duplicate those executions.

## Scientific and operational consistency

The implementation matches the registered intervention. It computes one raw
gradient, advances the canonical observer once, uses the literal post-ingest
mean, scales the pre-existing SGDm buffer by $k$, delivers
$kg+(1-k)\mu$, applies the unchanged manual decay, and then takes the ordinary
PyTorch momentum step. The endpoints receive special handling: $k=0$ is the
literal EMA-only recurrence and test-only $k=1$ is exactly the inherited I14
raw SGDm update. The latter has full-state one-step parity coverage and is
reused, not rerun, scientifically. Diagnostics retain the intended recurrence,
actual-versus-ideal action errors, components, signed dots, and both registered
energy windows.

The runner now directly binds the six I15 mean/projected reference branch JSONs,
their 30 checkpoint records/files, exact branch membership and parent digests,
and equality of every saved six-horizon scalar curve to the pinned audited
summary. It also binds the six I14 raw references and loads only the six shared
h100 parents. All six parent state/evaluation/RNG seams are checked before the
first scientific update, then checked again on branch restore. First-step raw
gradient, observer, and old-buffer digests are paired across the three new
$k$ arms. Typed numerical failures retain partial evidence; structural,
resource, serialization, and provenance errors abort rather than enter a
survivor analysis.

The primary comparison is appropriately limited: validation jointly selects
$(k,h)$ within the four scalar choices and separately selects the spectral
horizon, after which auxiliary CE and accuracy are read out. This tests a strong
isotropic temporal-response family, but it is neither an auxiliary oracle nor
a formal equivalence test. The fixed-$k$ curves, common-h100 progress, both
targets, both selectors, and all seed values remain mandatory, so clean
underfitting or selection noise cannot be hidden by the envelope.

## Interpretation limits retained

The transfer functions and effective-regularization formulas in the supporting
note are exact only for their stated fixed-stream or stationary idealizations.
The real observer action moves and the learned trajectories are endogenous;
therefore a scalar win would show that spatial selectivity was unnecessary for
this panel, not that the mechanisms are globally equivalent. Conversely, a
spectral win would support anisotropic gain as a useful account without by
itself identifying semantic clean-gradient selection.

This is an adaptive continuation on the same seeds, data split, and auxiliary
panel used in I14 and I15. Validation-to-auxiliary selection remains a valid
conditional comparison here, but it is not fresh-dataset confirmation or a
prospectively fixed research program. Finally, scalar branches retain the
canonical eigenspace observer for matched state and geometry even though it is
not needed for their delivery. Recorded elapsed times therefore measure this
study's feasibility, not the efficiency of a spectrum-free scalar algorithm.
