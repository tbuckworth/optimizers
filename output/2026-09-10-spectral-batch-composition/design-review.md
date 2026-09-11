# Independent batch-composition design review

Codex — Spectral Optimizer Investigation, 10 September 2026.
Prospective review; no new experiment or old scientific audit was executed.
Read the [exact protocol](protocol.md), the prior
next discriminator (artifact not distributed in this public snapshot)
and [interpretation](../2026-09-09-spectral-selectivity-boundary/interpretation.md).

## Assessment

This is a coherent constructive test, not baseline chasing. The previous study
established a useful preservation effect and an important acquisition cost.
The new experiment asks whether the same correct rare examples can become
more learnable when their contributions co-occur, without adding examples,
loss weight or a filter modification. Success is not fixed by construction:
grouping could help, hurt, or change raw AdamW as much as native filtering.

The fixed 36-trajectory roster is appropriately small: three fresh seeds,
Clean/Diffuse, two schedules and three policies. Fresh warmups are explicitly
distinguished from the prior note's reuse suggestion; no completed experiment
is restarted. The comparison is between two new scheduling policies, **not**
a reproduction of the earlier iid baseline.

At fixed parameters, exact block-multiset equality preserves the average
loss gradient. The two-group identity in the protocol correctly gives
`Var(q) * (mu_rare - mu_common)(mu_rare - mu_common)^T`. The selected direction
is a difference of means, not necessarily the rare mean itself, and not a
semantic feature by definition. This supplies a plausible covariance pathway
without assuming the moving native observer will realize it.

The strongest favorable result would be actual rare recognition and CE gains
under grouped native delivery, alongside useful common classification and
diffuse-corruption protection. A gain that consumes the preservation benefit
is a tradeoff, not a cost-free rescue. Report all seeds and both metrics;
the registered continuous comparisons avoid choosing a successful checkpoint
or retroactively declaring a small loss negligible.

## Concrete failure modes and prospective safeguards

| Risk | Warning sign | Safeguard and remaining limit |
| --- | --- | --- |
| “Same exposure” accidentally means only equal class counts, or repeats are lost. High impact; avoidable. | Different per-example multiplicities within a block. | The protocol requires the exact 3,200 indexed draw occurrences, including repeats. Independently rebuild streams 6–8 and compare sorted multisets, labels and all batch arrays. |
| Generic burst/curriculum effects are attributed solely to covariance. High interpretation risk. | Raw and norm-control schedule effects resemble native; moment histories differ. | Preserve the raw/control interactions. Shared within-group order and independently randomized batch locations reduce gratuitous differences, but do not equate moving gradients, Adam moments or signed updates. |
| A rare-learning gain hides common-class damage or extra wrong-label fit. High interpretation risk. | Better rare accuracy with worse common CE/accuracy or greater corruption fitting. | Report the outcome vector and changes from common warmup. No weighted victory score or unregistered “no sacrifice” margin. |
| Count-selected high/low events are treated as matched-state causal contrasts. Medium interpretation risk. | Anchor timing differs across schedules, or all counts tie. | The first-max/first-min-excluding-max rule yields distinct, outcome-independent anchors; ties remain recorded. Compare descriptive response, not a counterfactual at identical model states. |
| Retention alone is called successful learning. High risk given prior evidence. | High retained rare mean energy but poor recognition or adverse local loss. | Keep mean gradients, native versus numerical-span retention, actual Adam displacement, signed utility and finite losses separate. No per-example coherence inference from a group-mean vector. |

The prior class-blocked wrong-probe limitation is materially improved: new
wrong probes sample the actual changed-label population with a separate fixed
RNG, while the majority probe is explicitly stratified. These remain small
training subsets, not held-out causal estimates. Rarity and digit-8 difficulty
remain confounded; this experiment tests presentation at fixed task/exposure,
not a universal statement about rare knowledge.

## One clarification to freeze in implementation

The original protocol specified majority-probe membership but needed to make its loss
targets explicit in Diffuse: **majority and rare probes use true labels**;
only the wrong-assigned probe uses modified targets. Wrong-corrected uses
true targets on the identical inputs. This preserves the intended useful-
learning interpretation of the rare-minus-majority mean gradient. Main and
producer were notified before acquisition. Main then added the explicit target
contract to the protocol. **Resolved before source freeze.** It is a labeling
clarification, not a new arm, threshold or approval gate.

## Audit feasibility and disposition

There are **72 native diagnostic events**, not the prior study's 36: 12 native
trajectories times six anchors. Streaming one event at a time keeps the saved
state, mean-vector and prediction arithmetic compatible with the proposed
one CPU / 4 GiB / five-minute audit envelope. No resource expansion is currently
needed; actual runtime is not promised before execution. The independent
checker must bind the inherited arithmetic helper source and reconstruct the
new sampler/anchors rather than invoking producer generation functions.

The pre-mortem guidance influenced the concrete safeguards above; it did not
add workflow approval gates or an outcome-dependent stopping rule.
