# Next discriminator: can the native observer realize the positive tracking case?

8 September 2026. Prospective design after the first passing I17 integrity
audit. **Not an acquisition protocol, launched run or additional I17 arm.**
Finalize membership, source review and a bounded resource plan before use.

## Candidate experiment

Use a two-coordinate external gradient stream `g_t=s_0+a*t+epsilon_t`, with
independent Gaussian noise covariance `diag(1,4)`. Prespecify drift in coordinate1
at zero, .01 and .03. With beta=.99, these span no useful change, useful change
below the population direction-selection threshold, and the constructive case
above it. Do not choose drift strengths after native outcomes are seen.

Run the unchanged stable native rank-one covariance observer once per stream.
Feed its actual action to both the I17 normalized-buffer recurrence and the
prospective constant-preserving delivery-state recurrence, using the identical
raw stream and observer trajectory. Unlike endogenous neural branches, this
permits a genuine common-stream comparison of response rules. Prespecify
initial-state mapping and burn-in; retain startup results separately, not only
the favorable late window. Exact-projector theory must not silently replace
the actual native action.

Candidate controls are uniform EMAs at decay.9/.99 and, for nonzero drift, the
theoretically optimal shared decay for the known generating distribution; the normalized I17 scalar
mixtures; a fixed useful-direction oracle route; and a fixed nuisance-direction
route. The theoretical optimum uses generating parameters, not observed native
performance, and is explicitly an oracle comparator. With zero drift the
stationary-risk infimum occurs as decay approaches1 and is not attained within
the decay<1 family; do not invent a finite optimal EMA for that cell.
A small prespecified
additional decay grid could test sensitivity, but do not label it the entire
class of scalar temporal filters.

Track the full untruncated 2×2 post-mean residual second moment diagnostically
alongside the native rank-one estimate. This distinguishes lack of population
signal, finite-memory noise and rank-truncation/path effects; it is not a
replacement optimizer. A fixed coordinate rotation of the entire construction
would be a useful prespecified equivariance check if the resource plan permits.

## Discriminating predictions

- **Positive learned-routing account:** above threshold, native action aligns
  with the changing coordinate and the constant-preserving route beats the
  strongest prespecified uniform controls in tracking MSE after burn-in.
- **Finite-estimator limitation:** the population/full-moment diagnostic favors
  the useful direction but the native rank-one observer does not reliably align;
  the useful-direction oracle still wins. Do not attribute this to lack of
  directional headroom.
- **Response-transport limitation:** native alignment is useful but the I17
  moving-buffer normalization loses to constant-preserving delivery on the
  exact same action/gradient stream.
- **Necessary adverse boundary:** zero/weak drift allows nuisance variance to
  dominate the spectrum. Preserve resulting tracking harm, not just the strong-
  drift cell. If neither estimated nor oracle routing wins, revisit the precise
  finite-horizon assumptions before drawing a broad negative conclusion.

Report paired seed-level tracking errors, mean biases, variances, action
alignment, orthogonality/idempotence defects and response-rule residuals. Choose
seed count, horizons, scalar grid, ties/failures and complete reporting before
execution; no best-seed or best-window selection. Use CPU-only tiny tensors with
an explicit short wall-time/memory cap and zero cloud spend. No old neural
checkpoint or completed experimental handle is loaded or restarted.

## What this cannot establish

Even a clean success is an exogenous tracking result, not neural optimization,
clean-risk generalization, label-noise identification or a reason to deploy the
filter. It directly tests a concrete positive mechanism and its learnability.
Fresh-task neural replication and equal whole-program tuning budgets remain
separate necessary steps, as does a genuinely different signal model such as
curved/nonlinear drift or persistent noise. None is silently folded into this
small proposed discriminator.
