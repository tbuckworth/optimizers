# Independent review of the readout interpretation

9 September 2026

## Verdict

The conditional algebra is correct and appropriately scoped. I found no sign,
centering, reconstruction, or logical-equivalence error.

With the registered shift convention
`(S_delta z)_c = z_(c-delta mod p)`, increasing either input by `delta` sends
`q(c-s)` to `q(c-s-delta)`, so the template has the claimed equivariance. For
the positive-exponent Fourier convention and `r=c-s`, the change of variables is

`sum_c q(c-s) exp(i omega k c) = exp(i omega k s) sum_r q(r) exp(i omega k r)`.

The phase sign is therefore positive, matching the registered cosine/sine target
`[cos(omega k s), sin(omega k s)]`. A negative-exponent convention would reverse
the sine sign, but that is not the convention used in the note.

For nonzero `k mod p`, the classwise constant removed by row centering contributes
zero because the corresponding roots of unity sum to zero. The Fourier
coefficient of centered logits consequently equals that of the original logits.
Since unembedding is affine in the real hidden vector, its complex Fourier
coefficient is a complex affine functional of that vector. When, and only when
for this implication, `Q_k` is nonzero, division by the fixed complex scalar
`Q_k` gives `exp(i omega k s)`; its real and imaginary components are two real
affine readouts reconstructing the cosine and sine targets. The note correctly
does not claim decodability for a mode with `Q_k=0`.

The interpretation boundaries are also sound:

- Equivariant shifted templates need not classify correctly; a template maximum
  at nonzero offset is the stated counterexample.
- Correct classification need not imply a fixed equivariant template; an
  input-dependent positive rescaling preserves argmax while changing the
  template amplitude.
- Hidden-state affine decodability does not imply that the trained unembedding
  uses that information, much less that a particular Fourier circuit causes the
  behavior.
- The probe and output symmetry are mathematically related readouts, not
  independent mechanism proofs. Temporal precedence remains observational and
  is not mediation.

One notation detail is harmless but should remain understood: PyTorch's
`nn.Linear` stores the matrix in the transpose orientation relative to the
paper-style `hW+b`; this does not change the affine argument. Likewise, the
registered frequencies 1 through 56 are the nonredundant real representatives
for `p=113`; the root-sum identity applies to every nonzero residue modulo 113.

No change is required before using this note to qualify the saved-array results.
