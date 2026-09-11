# Independent mathematical review of the working paper

9 September 2026. **Qualified PASS:** the central algebra and conditional
interpretation are correct. Small specification and evidence-scope corrections
below should be applied before delivering this draft. This is not a
submission-readiness or novelty verdict.

Reviewed [draft](../../research/spectral_paper_draft_2026-09-09.md), 295 lines,
SHA-256 `297997bb52aff76195061e167b5999de317025382ed18dceba8f39d8e1cc9828`.
Line references refer to that version; related-work edits are being handled
separately. No training, inference, empirical reanalysis, or benchmark was run.
The review used hand derivation and the experiment-specific sources below;
it is not another independent raw-result audit.

## Mathematical checks

**Equal-norm raw-SGD inequality — PASS.** With the compact SVD
`V = Q Sigma Wᵀ`, let `z = Qᵀg`, `lambda_i = sigma_i²`, and
`w_i = z_i² / ||z||²`. Orthonormality gives
`||q||² = ||z||²`, `||r_native||² = sum_i lambda_i² z_i²`, and therefore
`c = sqrt(E_w[lambda²])`. Also `gᵀq = ||q||²` and
`gᵀr_native = ||q||² E_w[lambda]`. RMS–mean establishes the displayed
inequality. Equality holds when the gains are constant on gradient-active
singular directions. Cauchy–Schwarz also proves that the normalized projected
gradient maximizes the first-order dot product among all equal-norm vectors
in that span. The draft correctly excludes finite-step, curvature and Adam
trajectory orderings; the existing
[prospective note](../2026-09-09-spectral-grokking-action/adam-action-interpretation.md)
contains a valid finite-step quadratic counterexample.

**Retained-norm angle and distance identities — PASS with explicit domain.**
Let `Pi = Q Qᵀ`, `q = Pi g`, `a > 0`, and `q != 0`. Then
`gᵀq = ||q||²`, both normalized actions have norm `a`, and
`uᵀv = a² ||q||/||g|| = a² rho`. This yields
`cos(u,v) = rho` and `||u-v||² = 2a²(1-rho)` exactly.
Here `0 < rho <= 1`; retained squared energy is `rho²`, not `rho`.
The identities do not hold exactly after arbitrary casting, clipping or
nonorthogonal projection. The paper correctly treats them as ideal local
geometry and does not infer equality of unrun Adam trajectories.

**Estimator, clustering and Adam distinctions — PASS.** A deterministic
shared mean cancels from a centered covariance; the second moment retains its
outer product. Temporal innovations additionally mix sampling variation with
model motion. Neither statement gives semantic labels to eigenvectors.
An affinity construction followed by clustering is a different operation;
the anchor pilot review (artifact not distributed in this public snapshot)
supports factorized graph feasibility and conditional synthetic recovery,
not neural or safety efficacy. Likewise, carried moments, coordinatewise
Adam scaling and decay can move outside the current input span. Whole-history
positive scaling from zero moments is the appropriate near-invariance case;
scaling one current action with old moments is not.

## Concrete corrections

1. **Lines 75–88: initialization and statistic.** Keep the displayed recurrence
   as an intended innovation-second-moment update, but add: “Initialization is
   special: the mean is seeded with the first gradient, and the first nonzero
   innovation initializes `S=||z||` rather than `sqrt(1-beta)||z||`, so its
   covariance contribution lacks the later `(1-beta)` weight. Thus this
   is not literally a standard exponentially weighted covariance, even before
   later truncation and numerical drift.” The first-observation exception is
   explicit in [legacy code](../../spectral_filter.py) at lines 242–253 and the stable
   path. Calling this only a truncation/numerical-error qualification omits
   that implementation fact. No result changes.
2. **Lines 97–105 and 213–214: define the retained SVD.** Replace “a thin SVD”
   with “a compact SVD over the retained positive singular values, with
   orthonormal columns of Q”. State `q != 0` when defining `c`. A thin SVD
   that includes zero-singular-value left vectors does not necessarily make
   `Q Qᵀ` the projector onto `col(V)`. All audited common first-step bases
   are full-column rank, so this is a general-method specification fix, not
   an empirical objection. Numerical truncation/casting remain separately
   qualified in the experiment-specific report.
3. **Lines 225–227: domain and terminology.** Write “retained norm fraction”
   rather than “retained energy”; explicitly set `Pi = Q Qᵀ` for the same
   retained span and require `a > 0` and `q != 0`. The current phrase
   “nonzero denominators” mostly covers this, but the explicit domain avoids
   undefined cosine when both actions are zero. If energy is discussed,
   name its fraction `rho²`.
4. **After lines 179–193: archived-reference caveat.** Add one sentence:
   “Native endpoints are archived rather than contemporaneously replayed;
   known microscopic CUDA replay differences remain a numerical
   reproducibility caveat for the later native comparison.” This caveat is
   in the [full action report](../../research/grokking_action_mechanism_2026-09-09.md)
   and should survive condensation into the paper. The norm-matched versus
   orthogonal comparison is within the new acquisition.
5. **Lines 156–157 and 179–204: presentation only.** Restore missing spaces
   such as “at 2500”, “all five”, and “exactly 1000”. Define the table's SE
   explicitly as the sample standard error across five paired seeds, not
   a confidence interval. Clarify “never-trained probe splits” as
   “model-training-held-out pairs with separate probe-fit and probe-evaluation
   halves”; the linear probe itself is necessarily fitted.

## What the five-seed result identifies

The statement at lines 206–211 is appropriately scoped: the unequal native
within-span gains are not necessary for these later endpoints under the
norm-matched replacement policy, conditional on the inherited legacy state.
The replacement does not remove learned geometry: both its span and its norm
target depend on the branch's own estimator, while subsequent gradients,
spans and Adam histories diverge. Thus it is not fixed-projector mediation,
proof of scalar sufficiency, an explanation of pre-fork formation, or a safety
result. Retaining mixed step-2000 behavior and incomplete step-2500 accuracy
is essential and done correctly. The paper's primary numerical claims agree
with the focused action, representation and confirmation reports inspected.

The retained-norm identity provides a useful prospective test calibration,
not a known-outcome loophole: it fixes a one-step geometric comparison, while
the inherited-state Adam continuation outcome remains genuinely empirical.
No new acquisition is needed to resolve any correction in this review.

## Revision acceptance

I checked the corrected mathematical and methods passages in the 321-line
revision with SHA-256
`ea26d61e4fd5f046b2a8d1bfafc95fb0e369e5f1b1742643911a074d49d079b5`.
The compact positive-singular-value SVD, explicit nonzero action domain,
`Pi=Q Qᵀ`, norm-versus-energy distinction, archived CUDA sensitivity,
paired-SE definition and probe-data clarification are all incorporated.
The special initialization is now stated too. One final wording refinement
was sent to the main agent: distinguish the missing `sqrt(1-beta)` amplitude
factor in `S` from the missing `(1-beta)` covariance contribution. My initial
suggestion was insufficiently explicit about that distinction; the corrected
recommendation above spells out both quantities.

**Accepted after that wording refinement for this bounded mathematical and
interpretive scope.** The newer related-work paragraphs were independently
handled by the main agent and another reviewer; forthcoming trajectory
statistics require their separate saved-data audit. Neither is claimed as
validated by this mathematical review. The original reviewed hash and
correction history remain above.
