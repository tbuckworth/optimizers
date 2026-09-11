# Independent raw-direction interpretation review

Codex · 9 September 2026.

Reviewed:

- `research/grokking_raw_direction_2026-09-09.md`, SHA-256
  `4f4af8287a8cb1deff151136cde25651e6d8334f617e0a14305ac426cea610d0`;
- saved analysis summary, SHA-256
  `23359c850a17c51d178b431594bafbdd40e57c2cc803267c69a066cd81352165`;
- the accepted earlier action report and the prospectively reviewed direction
  interpretation.

This was a saved-summary and claim review only. I did not recompute numerical
results, open tensors or NPZ files, run inference or training, restart any job,
or launch a discriminator.

## Verdict

**Pass.** The report states the strongest supported constructive conclusion and
keeps the important causal and semantic limits. Its numerical claims agree with
the stored summary values inspected directly. I found no remaining blocking
interpretation or mathematical defect.

## Strongest supported conclusion

For these five inherited step-1,500 states and this fixed modular-addition task,
continued retained-span projection has a useful conditional post-fork policy
effect relative to the norm-law-matched raw-direction alternative. At step
2,500, raw minus projected is adverse for all five seeds on each primary:
held-out CE `+3.8245 ± 0.6632`, correct-class margin
`−4.5054 ± 1.0246`, and selected-five final-hidden held-out probe R²
`−0.4357 ± 0.0695`. Raw is also adverse to archived native on all three
primaries in all five seeds. At step 2,000, CE and margin have the reported
seed-101 exception while probe R² favors projection in all five seeds.

The contrast is about useful fixed-endpoint behavior, not basic interpolation:
all raw branches have 100% training accuracy at step 2,500 while held-out
accuracy ranges only 0.22–2.13%. The earlier action experiment supplies the
complementary conditional result that norm restoration helps while projection
is retained. Together these results support the tested combination of retained
direction and operator-derived scale; they do not establish a generic
factorial interaction or necessity beyond the tested policies.

The report correctly treats the archived-reference design and CUDA sensitivity
as limitations on a strict simultaneous-control reading. It also correctly
says that the scale *function*, rather than the realized scalar sequence or
Adam displacement norm, is shared after trajectories diverge.

## Mechanism boundaries and live hypotheses

The report does not overread the lower raw probe scores. They show lower linear
readability under the frozen probe recipe, not absence of modular-sum
information, erasure, a uniquely identified circuit, or proof of memorization.
The positive raw selected-frequency R² values and high training accuracy remain
compatible with several mechanisms.

The main live hypotheses are appropriately retained:

1. The projected policy may suppress off-span changes that interfere with
   shared arithmetic behavior or favor training-specific fitting.
2. Equal-norm projection also concentrates more drive in the retained
   direction. Exactly, at a common state, `P d_raw = ρ d_proj`; the present
   comparison cannot separate harmful off-span motion from insufficient
   retained-span drive, or their interaction.
3. Curvature, adaptive preconditioning and carried Adam/decay dynamics can
   change how incoming actions map to finite parameter and function changes.
4. The intervention changes the subsequent gradients, basis, scale and Adam
   history. Those feedback pathways are part of the policy effect and are not
   mediation controls.

Accordingly, the results do not establish generic learned-geometry necessity,
the necessity of this learned rather than a generic or frozen span, effects on
pre-fork formation, exactly matched branch-local scalars, memorization cleanup,
safety, broad transfer, or time-to-grokking.

## Review of the proposed next discriminator

The proposed function-space test is mathematically coherent if its future
protocol preserves the report's qualifications. On the complete uniform
`p²` input grid, for every output coordinate,

<pre>
R δf(a,b) = p⁻¹ Σᵤ δf(u,(a+b−u) mod p)
</pre>

is the orthogonal projection onto functions constant within each modular-sum
class. Centering logits across output classes at each input is necessary to
remove softmax-null all-class shifts before comparing energies. `R δf` is only
sum-consistent, not necessarily label-beneficial; `(I−R)δf` is within-sum
variation, not automatically an example-specific exception or memorization.

The intermediate action should be defined exactly as
`d_trunc = P d_raw = a Pg/‖g‖ = ρ d_proj`. Raw versus `d_trunc` holds the
retained incoming component fixed while removing the off-span component;
`d_trunc` versus projected keeps the retained direction fixed while increasing
its magnitude. All three must use the identical inherited model, filter and
Adam state. Identical decoupled weight decay is shared, but the carried moments
interact nonlinearly with each delivered action and remain part of the measured
local policy effect.

For any scalar outcome evaluated on the same three counterfactual states, the
two successive pairwise differences telescope exactly to the raw-versus-
projected difference. That algebra does not license additive causal attribution
of “off-span suppression” and “retained amplification” through nonlinear Adam
and network responses, nor additive squared energies across separately realized
counterfactual effects.

The bounded test should report total centered function-change magnitude,
`R` and residual energies, signed first-order held-out-loss utility, and finite
held-out CE/margin for each complete counterfactual action. If component-only
finite losses are also shown, they must be labeled post-hoc logit-space
counterfactuals rather than outputs of separately realized Adam steps. Because
the same held-out split and task define these quantities, they remain mechanism
diagnostics on the existing benchmark, not fresh validation or proof of
trajectory mediation.
