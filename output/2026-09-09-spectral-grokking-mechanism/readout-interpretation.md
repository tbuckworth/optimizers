# Why sum decodability and output symmetry are related

9 September 2026. Conditional algebra, not an empirical circuit identification.
Written before reading the intermediate-state results. It explains why the
registered measurements should not be counted as independent mechanism proofs.

Let `s=(a+b) mod p`, let the final hidden vector be `h(a,b)`, and write its
linear unembedding as `l(a,b)=h(a,b)W+b`. Suppose, as an idealized limit, that
row-centered logits have the exactly equivariant form

`z_c(a,b)=q(c−s mod p)`.

For frequency `k≠0`, define `ω=2π/p` and
`Q_k=Σ_r q(r) exp(iωkr)`. Changing variables `r=c−s` gives the identity

`Σ_c z_c(a,b) exp(iωkc) = exp(iωks) Q_k`.

The left side is a fixed complex linear functional of logits, and hence an
affine function of `h`. If `Q_k≠0`, division by this constant makes the real
and imaginary parts exactly the cosine and sine targets of the registered
probe. Row centering does not alter this Fourier coefficient, because
`Σ_c exp(iωkc)=0` for nonzero k. Thus an ideal equivariant score function with
nonzero frequency amplitude already implies affine decodability of that sum
mode. Finite ridge, noise and finite-data fitting can lower its measured R².

This implication does not require correct classification. A template whose
largest score occurs at offset37 rather than0 is still equivariant and still
encodes the sum, but predicts the wrong answer. Conversely, arbitrary
input-dependent positive temperatures can preserve correct classes while
violating the fixed-template assumption. These are precisely the complementary
warnings illustrated by the symmetry fixtures.

The reverse implication is also insufficient: an independently fitted probe
may read sum information from hidden coordinates that the trained unembedding
does not use appropriately. High probe R² can therefore coexist with poor
model accuracy. That separation is scientifically useful when it occurs
**before** the behavioral transition, but does not establish the hidden
information's causal role in later learning.

## What a temporal result would mean

- If readable sum information rises earlier under a filter while behavior is
  still poor, the filter changes an internal progress marker before the final
  task metric. This strengthens the formation hypothesis without identifying
  a circuit or proving that covariance specifically selected it.
- If all arms already have a similar high readout before their accuracies
  diverge, later readout use, residual interference, margins or weight-decay
  dynamics become more attractive explanations than earlier formation alone.
- If only the per-state-selected panel differs, inspect the fixed panel and
  all frequency identities. Different rule representations may use different
  modes; a panel selected on seed100 is not a universal dictionary for all
  seeds or an optimizer-neutral circuit locator.

The correct paper claim should identify the actual measured temporal ordering
and preserve its limitations. A common-state perturbation, not a second
correlated readout, is the next route to a causal claim.
