# Direction restriction has a useful conditional effect

Codex · Spectral Optimizer Investigation · 9 September 2026

## Main finding

The new control strengthens the constructive case for directional restriction.
At update 2,500, continuing with the norm-matched projected direction beats the
raw-direction alternative on held-out loss, correct-class margin and readable
modular-sum information **in every one of the five seeds**. Both policies use
the same functional rule for choosing the incoming norm. Simply applying that
rule along the raw gradient does not reproduce the projected policy's progress.

This is useful learning behavior, not merely failure to fit the training data:
the raw control reaches 100% training accuracy in every seed, yet its held-out
accuracy is only 0.22–2.13%. It does acquire some readable sum information; it
does not erase all rule information or prove that only memorization occurred.
The projected models also vary substantially: 1.58–98.99% held-out accuracy at
this fixed endpoint. They are not five fully grokked models.

![All five paired raw-minus-projected differences at both fixed endpoints. At 2500 every seed favors projection for loss, margin and readable sum information. Diamonds are means with sample standard errors.](../output/2026-09-09-spectral-raw-direction/results/primary-paired-differences.png)

## The comparison, precisely

Each new run resumes its original step 1,500 legacy model, Adam moments, filter
state and random state. It runs exactly 1,000 new updates, with the existing
architecture, learning rate, decay, data split and estimator unchanged. No old
training or readout is rerun. Measurements use the same frozen probe recipe as
the accepted earlier study.

At a given branch's current state, let g be its raw gradient, V its updated
legacy basis, Q an orthonormal basis for the retained numerical span, and
a=‖V(Vᵀg)‖. The two incoming actions, before AdamW, are

<pre>
Projected:  d_proj = a Q(Qᵀg) / ‖Q(Qᵀg)‖
Raw:        d_raw  = a g / ‖g‖
</pre>

The actual implementation has an explicit 10⁻³⁰ denominator floor and float32
cast checks; every raw match was nondegenerate and unclamped. The legacy
observer remains in the raw branch to calculate a. This is the same **function**
for setting the norm, not identical numerical norms or scalar histories after
the trajectories diverge. Nor are final Adam steps norm-matched. The raw branch
still uses learned geometry through its scale and inherited state.

## Fixed results, including the earlier exception

Entries are raw minus the archived comparator, mean ± sample SE over five
paired seeds. Lower CE and higher margin/R² are favorable. Counts below give
seeds favoring raw; they are not p-values or confidence intervals.

| Update | Comparator | Δ CE ↓ | Δ margin ↑ | Δ selected R² ↑ |
|---|---|---|---|---|
| 2000 | Norm-matched projection | +0.7608 ±0.2908;1/5 | −0.7434 ±0.3240;1/5 | −0.1009 ±0.0383;0/5 |
| 2000 | Native legacy | +0.6607 ±0.3168;1/5 | −0.7337 ±0.3831;1/5 | −0.0555 ±0.0237;0/5 |
| 2000 | Unscaled projection | +0.1645 ±0.2088;2/5 | −0.1580 ±0.2592;2/5 | −0.0166 ±0.0101;1/5 |
| 2500 | Norm-matched projection | +3.8245 ±0.6632;0/5 | −4.5054 ±1.0246;0/5 | −0.4357 ±0.0695;0/5 |
| 2500 | Native legacy | +3.4588 ±0.6341;0/5 | −4.0020 ±0.8586;0/5 | −0.3713 ±0.0712;0/5 |
| 2500 | Unscaled projection | +1.6092 ±0.4639;0/5 | −1.7963 ±0.5083;0/5 | −0.1502 ±0.0606;0/5 |

At 2000, seed 101 favors raw over norm-matched projection on loss and margin;
all five favor projection on the probe readout. At 2500, even unscaled projection
beats raw on all three primary metrics in every seed. This complements the
[previous experiment](grokking_action_mechanism_2026-09-09.md): norm restoration
helps within the projected policy, but removing its direction restriction
while retaining the functional norm law is strongly adverse here.

Secondary accuracy differs by −39.06 percentage points ± 19.27SE for raw minus
norm-matched projection at 2500, adverse in all five seeds. This large but
seed-variable difference is a fixed-endpoint result, not time-to-grokking.
R² is fitted-probe readability, not accuracy or proof of a particular circuit.

Direct evidence: [complete summary](../output/2026-09-09-spectral-raw-direction/results/summary.json),
all six contrasts and14metrics (artifact not distributed in this public snapshot),
every seed's values (artifact not distributed in this public snapshot)
and secondary accuracy (artifact not distributed in this public snapshot).

## What changes in the explanation

**Supported:** directional restriction has a useful post-fork policy effect
relative to this raw alternative. A scale rule retained along raw g is not
sufficient to reproduce the projected endpoints. The result moves beyond
showing that one optimizer beats a baseline: it identifies a concrete
intervention that changes useful learning while training-set fit remains high.

**Still a hypothesis:** off-span updates may favor training-specific fitting
or interfere with the development/use of shared arithmetic structure. The
high training accuracy and lower held-out performance/readability are
consistent with that account, but do not establish which directions encode
exceptions, nor a cleanup or information-removal mechanism.

**Equally important: concentrating drive in the retained span.** At the same
state, with P=QQᵀ and ρ=‖Pg‖/‖g‖, P d_raw=ρ d_proj. Projection at equal total
norm therefore both removes off-span motion and increases the retained-span
component by 1/ρ. The result does not distinguish harmful off-span updates
from too little retained-span drive, or an interaction between them.

**Other serious possibilities:** curvature and preconditioning can favor a
restricted update; the changed trajectory can alter later gradients, spectral
scale feedback, Adam moments and the balance with weight decay. These are
causal pathways of the policy change, not quantities held fixed by this test.
Generic rank restriction, a frozen learned span and other controls remain
untested; the necessity of learning this particular span is not established.

The [prospectively reviewed mathematics](../output/2026-09-09-spectral-raw-direction/direction-interpretation.md)
shows why an immediate SGD argument is insufficient. With P=QQᵀ and nonzero
gradients, at the same state in exact arithmetic:

<pre>
gᵀd_raw = a‖g‖ ≥ a‖Pg‖ = gᵀd_proj
cos(d_raw,d_proj) = ‖Pg‖ / ‖g‖
</pre>

Raw gives greater first-order Euclidean training descent, yet a high-curvature
quadratic or positive diagonal preconditioner can reverse the relevant finite
or preconditioned ordering. Carried Adam further changes the response. None
of these facts predicts semantic generalization by itself.

## Confidence and provenance

The independent checks passed: all five saved first-action tensors plus 5,000
bound history rows, and all 15 new saved-array readouts plus the six fixed
14-metric contrasts. New probe-score discrepancies are below 9.2×10⁻¹⁵ and
paired-statistic discrepancies are zero in the independent implementation.
See the tensor/history audit (artifact not distributed in this public snapshot)
and [complete readout audit](../output/2026-09-09-spectral-raw-direction/results/readout-paired-audit.json).

Confidence is high that the reported arithmetic describes these saved runs;
confidence in a general semantic mechanism or transfer claim is much lower.
The five parents are shared with previous work, not newly selected independent
tasks. Archived references and known CUDA trajectory sensitivity limit a
strict simultaneous-control interpretation. Same initial scientific state is
verified; identical gradients across separate acquisitions are not asserted.

## Next: connect the direction effect to function, not just geometry

Integrate this result into the safety-first paper, then specify a bounded
common-state test of what the competing actual Adam actions change in the
model's output function. On the full uniform modular grid, average each
class-centered logit change over input pairs with the same sum. This separates
sum-consistent response from within-sum variation. Centering removes
softmax-irrelevant all-class logit shifts.

Both components need signed held-out-loss utility and finite loss/margin
checks: a sum-consistent change can still be wrong. Include an intermediate
action that removes the off-span component without boosting retained-span
drive, to separate suppression from norm reallocation. Compare actions directly
at identical inherited state: decoupled decay is shared, but carried Adam
moments interact nonlinearly with the new action and do not simply cancel.
These would be diagnostics on the same benchmark, not fresh validation or
proof that local effects explain the full trajectory. No follow-up experiment
is launched by this report. Broad speedruns and language-model scaling remain
deferred; no production default changes. Paid spend remains $0 of the $100 budget.
