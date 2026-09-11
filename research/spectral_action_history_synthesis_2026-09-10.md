# Local utility, accumulated history, and what the paper can explain

Codex — Spectral Optimizer Investigation · 10 September 2026

**Evidence synthesis plus an elementary conditional identity, not a new
experiment or a novel theorem.** The completed component diagnostic removes
one local explanation from the leading narrative. It does not remove the
useful learning phenomenon or establish that history is its remaining cause.

## 1. Three different questions have now been measured

| Question | Intervention and evidence | What follows, and what does not |
|---|---|---|
| Does filtering change useful learning over training? | [Strong bridge](../output/2026-09-10-spectral-strong-augmentation/results.md): three paired seeds, full trajectories, validation-only selection. | Native makes substantial useful post-warmup progress under corruption; the combination with translation loses to translated raw. These are learning outcomes, not a decomposition of their cause. |
| Can observer history change useful delivery at a fixed model? | [History intervention](../output/2026-09-10-spectral-observer-pathway/results.md) and [saved-vector accounting](../output/2026-09-10-spectral-observer-signal/interpretation.md): fixed model/Adam/input, grouped versus interleaved observer histories. | Clean rare utility improves relatively in all three reused parents; noisy effects are mixed. This establishes a conditional local pathway, not that it mediates the strong bridge. |
| Which objective changes accompany the present filtering choice? | [Component utility](../output/2026-09-10-spectral-component-utility/results.md): same full parent, raw/native action, exact S/F/C and clean readouts. | At translated final parents, raw consistency itself worsens; the local lost-useful-consistency account is unsupported. Warmup relative consistency and clean utility move oppositely. This does not estimate accumulated history effects. |

The observer-history study uses different seed bundles and operating points;
the component diagnostic deliberately reuses strong-bridge native states.
All three interventions and estimands differ. They cannot be assembled into
one measured causal chain by joining their headline numbers. In particular,
the history study's clean positive and the component study's noisy warmup
cost are not contradictory effects of the same treatment on the same state.

## 2. Why one-step clean effects cannot simply be added up

Fix a horizon T and all exogenous data/augmentation draws. Let the full state
x include parameters, Adam moments/counter and observer memory/counter, together
with anything else needed to make the update deterministic. Write

```text
x_(t+1) = F_t^a(x_t),       a ∈ {N, R}
```

for native and raw transitions. H(x) evaluates the chosen terminal clean loss
from the model parameters; it need not depend directly on the stored moments.
The local diagnostic reports an immediate clean benefit of native,

```text
d_t(x) = H(F_t^R(x)) − H(F_t^N(x)).
```

Now define V_t^R(x) as the terminal H obtained by applying raw at every
remaining step t,...,T−1 from x, with the same fixed future draws. Then

```text
V_T^R(x) = H(x)
V_t^R(x) = V_(t+1)^R(F_t^R(x)).
```

Along the full native trajectory x_t^N, define the *future-evaluated* cost
of the present native choice,

```text
c_t = V_(t+1)^R(F_t^N(x_t^N)) − V_(t+1)^R(F_t^R(x_t^N)).
```

Substitution gives c_t = V_(t+1)^R(x_(t+1)^N) − V_t^R(x_t^N).
The terms telescope exactly, so for a common initial state,

```text
H(x_T^N) − H(x_T^R) = sum[t=0,...,T−1] c_t.
```

This is a deterministic telescoping identity, not an estimated mediation
formula or a novelty claim. Each c_t compares one native versus raw choice
from the same native-history parent **and then the same raw continuation**.
It includes how the changed state affects later gradients and updates. It is
not −d_t(x_t^N), except at the final transition or under additional restrictive
conditions making the future evaluation equal the immediate H difference.

Consequently, small immediate differences at a few states do not bound the
terminal policy gap without an appropriate bound on future response. Nor does
this identity prove that the missing gap comes from observer memory rather
than representation, Adam, changing inputs, or their interaction. Those are
components of the full transition, not additive causal contributions.

No V_t or c_t was computed in the new study. There are only warmup/final native
snapshots, not all intermediate restorable states. At unaugmented parents the
diagnostic also uses a hypothetical translated action, not that trajectory's
actual next input. With the reported training horizon T=56,304, its final-parent
probe is the hypothetical transition56,304→56,305, not the identity's last
transition56,303→56,304. Its warmup probe is100→101. Thus even the identity's
last-transition equality does not turn the final-parent probe into a measured
last summand of the reported endpoint gap.
The identity clarifies an estimand; it does **not** authorize
reconstructing missing states or launching T continuation experiments.

## 3. The component decomposition is a different accounting identity

On the fixed training/view panel the study measures L = S + F + C, hence for
each endpoint E_L = E_S + E_F + E_C and for paired actions D_L = D_S + D_F + D_C.
It does not follow that clean held-out D_H has those three terms. H uses true
labels and a different reporting panel; S itself mixes true-target and uniform-
target losses, F is a full-panel assignment residual, and C is view disagreement.

The [exact objective derivation](spectral_augmentation_loss_geometry_2026-09-10.md)
already permits consistently wrong predictions with C=0. The new measurement
adds actual policy evidence: better relative C accompanies worse original and
translated clean CE at warmup in both training settings/all three seeds.
At translated final states native's F advantage does not uniformly predict
actual wrong-subset fitting. Neither identity yields a percentage of the
clean training gap attributable to “memorization” or “consistency.”

## 4. Consequence for the next decision

The paper should retain the positive learning case and its controlled local
pathway, then state this new explanation boundary. It should not replace an
unsupported consistency story with an unmeasured history story. The existing
[unified hypotheses](spectral_optimizer_unified_hypotheses_2026-09-10.md) remain
provisional where they go beyond the identified intervention effects.

No additional experiment is selected here. Finish integrating the audited
component result and reconcile the contribution, methods and source map before
choosing another acquisition. A later history/continuation question would need
to specify which state is changed, which future policy is shared, and whether
the outcome is useful adaptation, preservation, or unwanted fitting. It need
not estimate every telescoping term, and it is not a prerequisite for reporting
the useful phenomena already demonstrated. Broad speedruns and capability
claims remain outside the current safety/understanding-focused scope.
