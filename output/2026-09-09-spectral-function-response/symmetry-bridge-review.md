# Independent symmetry-bridge review

Codex · 9 September 2026.

Reviewed `symmetry-bridge.md`, SHA-256
`7f3fd6a8520868c3dabcb5076970064f4fc3f33f5112e3f4aa031aa6c556ba59`.
This was a theory-only review. I opened no experimental tensor, array or
numerical result and ran no inference, update or acquisition.

## Verdict

**Pass.** The group-average, Jensen, counterexample and parameter/function
bridge claims are correct under their stated assumptions. The note does not
turn the diagnostic's by-construction symmetry metric into optimizer evidence.

## Algebra

- The maps `T_h(a,b)=(a+h,b−h) mod p` form the additive group action whose
  orbits are exactly the `p` ordered input pairs sharing each modular sum. The
  protocol's `R` is the uniform finite-group average and an orthogonal projector
  under the uniform full-grid inner product.
- Within an orbit, every example has label `s`. Convexity of
  `ℓ_s(z)=logsumexp(z)−z_s` therefore gives the stated per-orbit Jensen
  inequality and, after averaging orbits, `L(Rf)≤L(f)`.
- CE has only the all-class-shift null direction. Once every input's logits are
  class-centered, its Hessian is positive definite on the centered subspace.
  Jensen equality therefore requires identical centered logit vectors within
  each sum orbit, exactly as stated.
- If `f=Rf`, linearity and idempotence give `R(f+δ)=f+Rδ`; applying the preceding
  inequality yields `L(f+Rδ)≤L(f+δ)`. The full-grid CE gradient is then invariant
  over each orbit and orthogonal to `(I−R)δ`, so residual first-order utility is
  zero. This neither says the retained update improves on doing nothing nor
  supplies an analogous guarantee on an arbitrary held-out subset.
- For noninvariant centered `f=Rf+h`, choosing `δ=−h` makes the unfiltered result
  `Rf`, while filtering the update gives `f` because `Rδ=0`. Since centered
  `h≠0`, strict Jensen applies: this is a valid function-space example where
  suppressing residual update blocks correction. It correctly refutes the idea
  that residual-update energy is intrinsically harmful.

## Parameter/function bridge

The note correctly separates the parameter projector `P` from the task
projector `R`. In an ideal first-order SGD model, `J_θPg≈RJ_θg` is the bridge
needed for raw-to-truncated filtering. With
`u=αg`, `t=Pu`, `ρ=‖Pg‖/‖g‖` and `v=t/ρ`, it implies
`J_θv≈ρ⁻¹RJ_θu` for the norm-matched projected action. The extra factor is why
truncation and retained-amplitude concentration require separate contrasts.

No low-rank or covariance fact establishes this bridge. The actual carried-Adam
map depends nonlinearly on the delivered action and inherited first and second
moments, while decay adds movement not constrained to `range(P)`. Consequently,
the finite-response diagnostic can measure local behavior and task-aligned
function structure without treating `P` as if it were `R` or claiming local
mediation of the later trajectory.

## Interpretation

The temporal proposal is appropriately conditional: if parameter filtering
approximately realizes function-space restriction, it could preserve an
existing invariant computation yet hinder correction of an earlier
noninvariant function. The one-step diagnostic can assess the local response
only. It cannot establish that temporal account, semantic feature selection,
memorization cleanup or a safety mechanism.

The related-work paragraph attributes only the established group-orbit and
variance-reduction framing to Chen, Dobriban and Lee; the modular CE derivation
is explicitly presented as this project's elementary specialization. The final
safety distinction is justified: a known label-preserving task symmetry does
not imply that learned covariance structure in a language model is truthful,
benign or safe.
