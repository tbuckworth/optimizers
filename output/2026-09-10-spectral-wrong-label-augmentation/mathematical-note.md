# Why augmentation could help—or preserve the wrong thing

Codex — Spectral Optimizer Investigation · theoretical reasoning, not results

Let (xᵢ, ỹᵢ) be a training example with its fixed, possibly wrong label, and
T a random translation. The augmented objective is

L(θ) = (1/N) Σᵢ E_T ℓ(f_θ(Txᵢ), ỹᵢ).

Augmentation supplies different views of **the same assigned label**, not new
independent votes about its truth. It can make memorizing a single untransformed
image harder while also encouraging an entire orbit of views to share that
image's wrong label. Hence a favorable or adverse result is plausible without
an implementation bug.

For ideal independent symmetric corruption with actual wrong fractionρ and
Kclasses, the expected target vector conditional on the true class y is

q = [1 − ρ − ρ/(K−1)]e_y + [ρ/(K−1)]1.

Atρ=0.8,K=10, this is (1/9)e_y+(4/45)1: q_y=0.20 and q_other≈0.0889.
The truth-favoring difference is1/9. Thus the population target still has the
right argmax; it is much softer than the clean target. The present balanced,
exact-count finite sample is not a fresh independent target draw at each step.
Learning particular assignments can defeat the population interpretation.

At a **fixed θ**, write gᵢ,T=∇_θℓ(f_θ(Txᵢ),ỹᵢ). The total-covariance identity is

Covᵢ,T(gᵢ,T) = Covᵢ(E_T[gᵢ,T|i]) + Eᵢ[Cov_T(gᵢ,T|i)].

The first term measures differences between each example's mean direction;
the second measures variation across views of the same example. Neither term
is automatically “generalization” or “memorization”: both depend on the fixed
assigned labels, representation and current model. Averaging m independent
views **of one example** before forming its update divides its within-example
variance by m while keeping its mean. This pilot instead uses one view per
occurrence; it does not perform that averaging intervention.

The actual optimizer tracks centered **temporal minibatch gradients along a
changing model trajectory**, then projects the raw gradient into its retained
span before AdamW. It is therefore not simply diagonalizing the fixed-state
covariance above. Mean displacement, view noise, path change, rank, normalization
and Adam's coordinate scaling can all affect the delivered update. This identity
organizes a possible saved-state diagnostic; it does not identify the mechanism
from final accuracy or covariance retention alone.

The first useful empirical question is consequently the user's direct one:
does augmentation improve clean held-out learning under wrong labels? Next
compare its progress from its own warmup and its wrong-target fitting. Only
then use a targeted diagnostic to distinguish plausible mechanisms, rather
than treating reduced memorization or a favorable relative rank as sufficient.
