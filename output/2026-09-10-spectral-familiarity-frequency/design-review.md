# Independent review: familiarity under a paired frequency swap

Codex — Spectral Optimizer Investigation · 10 September 2026

**PASS for inert implementation preparation. No blocking scientific-design or
arithmetic error found. This is not acquisition admission or implementation
acceptance, and is a scientific-design verdict only, not the current work
priority.** The proposed study adds meaningful information beyond the
completed absent-rare-class and local-action studies. Two small documentation
precisions should be resolved before source/protocol freeze; neither requires
another arm, longer warmup, new experiment or user approval.

Read `design-decision.md` and `protocol.md` completely, followed by
`source-feasibility.md` and `design-assessment-independent.md`. Review is based
on documentary design/source assessments only. No scientific arrays, model,
checkpoint, raw acquisition, test, experiment or audit was opened/executed.
The pre-mortem guidance was used to find concrete failure modes, not to invent
additional gates or demand a new failure-first research workflow.

## 1. Physical versus logical design

The core crossing is correct. For one seed, label condition and policy, the
four history/schedule trajectories supply the following two focal readouts:

| Warmup focal class | Subsequent class3 exposure | Digit3 readout | Digit8 readout |
|---|---|---|---|
| 3 | Low | Familiar / low | Omitted / high |
| 3 | High | Familiar / high | Omitted / low |
| 8 | Low | Omitted / low | Familiar / high |
| 8 | High | Omitted / high | Familiar / low |

Thus3 seeds ×2 warmups ×2 schedules ×2 label conditions ×2 policies is
**48 physical continuations**, with **96 dependent logical focal-class rows**.
Both focal digits are crossed within every seed; digit and seed are not
confounded. Six warmups are trained once and shared, not charged again for
each continuation. Three initial states, six parents and48 final states are
57 physical snapshots. Six×100 +48×1900 = **91,800 training updates**.

This is three independently initialized seed bundles, not six independent
class replicates or96 runs. The two digit readouts from each trajectory are
joint observations. Averaging the two digit contrasts within seed only after
retaining the digit-specific cells is an appropriate descriptive reduction.

## 2. Pools, exposure and labels

Warmup slots:712 focal +8×711 background =6400. Each continuation block:
64+640+8×712 =6400, for19 blocks. Per focal class, the totals are1216 versus
12,160; each background digit gets13,528 in either schedule. Background total
108,224 plus focal total13,376 =121,600 =1900×64.

The64 permanent slots for each focal class plus576 exchangeable slots give
the specified704 focal slots. A common full-slot permutation allows the low
occurrences to be an exact position-matched subset of high occurrences while
leaving background IDs and chronology unchanged. There is no accidental
same-multiset claim across the two schedules and no need for a grouping arm.
The remaining focal occurrences must map to the exchangeable slots in a fixed
order in the implementation, not be independently reshuffled per policy.

The550 warmup-only /550 continuation-only /500 reporting pools require1600
IDs per true digit and are explicitly disjoint within seed. The future numeric
plan must check this availability and all memberships; it was not inspected
here. Equal eligible support removes the old50-versus550 support asymmetry.
It does not equalize realized unique exposure or arrival/spacing, which are
recoverable from the stored occurrence plans and correctly retained as
measured exposure features rather than additional matched factors.

The Diffuse law is coherent: fixed per-ID background selection with probability
0.9, then uniform target among eight background classes including truth.
Expected wrong fraction is0.7875 within background. Since background occupies
89% of continuation slots, the expected whole-stream fraction is0.700875,
not78.75% or90%. The balanced continuation *pool* instead has80% background,
so its expected whole-pool wrong fraction is63%; realized pool and stream
rates need not equal their expectations. Recording both prevents a misleading
single corruption percentage. Neither focal class receives an assigned wrong
label or serves as a wrong target.

Implementation should apply this fixed label law to continuation-pool IDs;
warmup and reporting remain clean. That follows the protocol's phase rules,
not a request to invent corruption in additional pools. Repeated IDs must
retain assignments across histories, frequency schedules and policies.

## 3. State, clocks and estimands

Two warmups per seed share initialized parameter bytes, compute count and
background stream but legitimately produce different weights, moments and
observer contents. Raw-delivery warmup with passive observation gives both
clocks100. Native's first differing delivery is after observation101 and its
final Adam/observer count is2000. Raw inherits the same Adam state and reaches
Adam2000, but must not be described as observer2000 if that observer was
discarded after fork verification. Full-state/RNG ownership and the first
paired raw-gradient equality are needed, not just matching weight digests.

The policy contrasts have consistent beneficial signs: CE_raw−CE_native and
accuracy_native−accuracy_raw equal native progress minus raw progress because
each policy pair shares its baseline. The protocol explicitly avoids treating
this cancellation as adjustment for the different familiarity parents.
J_low/J_high and K identify interactions of policy with the **specified
prior-class replacement and later exposure swap**, not observer-only memory
or isolated target frequency. Increasing one focal dose lowers the other's;
making one familiar omits the other. Neither coupling disappears with RNG
pairing, baseline subtraction or class-conditional evaluation.

The competence annotation is defensible as a modest, frozen operational
description: exposed-class reporting accuracy≥50% and CE below its initialized
value. It is not guaranteed by100 warmup updates. Retaining all branches if it
fails preserves an honest acquisition result without silently converting it
into preservation. Initial/warmup accuracy and CE must remain visible; the
annotation alone should never certify mastery, matched competence or removal
of ceiling effects. No adaptive warmup, competence-selected checkpoint or
headroom normalization is warranted.

## 4. Exact prospective storage and resource arithmetic

There are3 initial +6 parent +48×19 continuation readouts =921 physical
logit sets. At10,500 examples ×10 classes ×4 bytes:

| Component | Bytes |
|---|---:|
| 921 logit sets | 386,820,000 |
| 57 snapshots ×7,597,528-byte conservative tracked-state allowance | 433,059,096 |
| Numeric plans, scalar metrics, headers and receipt allowance (256 MiB) | 268,435,456 |
| **Total** | **1,088,314,552** |
| 2-GiB cap | 2,147,483,648 |
| Remaining margin | 1,059,169,096 |

This is about1.014GiB, as stated. Initialization/warmup readouts are shared;
storing them separately for every logical focal row would be erroneous
duplication. All scheduled class/background/wrong-subset reductions are
recoverable from the proposed logits and label/role plans without per-example
gradient storage. The snapshot allowance conservatively treats even raw and
initial states as populated rank32 states.

The inherited local envelope and10–20-minute planning allowance are plausible
against the documentary references, not measured performance or reservation.
Before acquisition, the actual implementation must enumerate every artifact,
enforce bytes while writing, preserve a failure-footer allowance, bound input
loading and reader memory, and verify live owners and no-restart service caps.
The five-minute independent-audit hard limit is provisional, not independently
validated by this document. Nothing in this review admits a smoke run or retry
to calibrate those resource assumptions.

## 5. Concrete precision fixes before source freeze

1. **Primary-status mismatch between documents — medium impact, modest risk.**
   The earlier independent assessment calls J_low/J_high the primary CE
   contrast vector and accuracy secondary. The selected protocol instead
   names endpoint focal CE and accuracy jointly primary, with interactions
   descriptive. This is a documentation priority mismatch, not an algebraic
   flaw. Smallest fix: explicitly state that the selected protocol's §6
   supersedes that earlier primary-ranking suggestion, or align the earlier
   assessment text. Do not leave both available for post-outcome promotion.

2. **Audit evidence versus producer assertions — medium impact, modest risk.**
   “Check snapshot identities/schema/non-alias receipts and actual clocks”
   needs one implementation-facing distinction. A byte hash plus producer
   metadata does not independently establish stored clocks or parameter
   ownership. Specify whether bounded restricted snapshot decoding checks
   schema/counters, or whether these remain labelled producer claims whose
   receipts are checked. Runtime non-alias checks are necessarily producer
   execution checks unless separately reproduced; no neural replay is implied.
   This is important for accurate final audit scope, not a demand to broaden it.

Small convention to freeze during implementation: retain accuracy as a
proportion in sufficient-statistic arithmetic, state whether displayed
contrasts are proportions or100×percentage points, and use0.5 for the50%
annotation. This prevents a100-fold presentation mismatch without changing
any scientific estimand.

## 6. Information value and interpretation limits

**Strongest constructive outcome:** demonstrated familiar low-frequency
competence is retained under native while unfamiliar correct behavior still
improves and background wrong-target fitting is reduced, with competent
background performance. This would add a useful conditional preservation/
adaptation regime and directly separate an omission-only interpretation from
the older rare-class result. More limited familiar preservation is still
informative even if unrestricted unfamiliar acquisition does not coexist.

**Strongest adverse outcome:** native loses acquired low-frequency competence
or restricts familiar and omitted classes similarly, and making the omitted
class frequent fails to restore its useful learning relative to raw. That
would weaken a simple novelty-specific account without erasing background
protection or earlier useful strong-regime results.

**Main way the completed study could remain ambiguous:**100-step warmups may
not establish the declared competence, high-frequency cells may saturate, or
digit-specific effects may oppose. Those are interpretable retained outcomes,
not design failures to repair after observing them. Neither positive
interactions driven by raw damage nor small estimates with inadequate
precision establish useful selection or equivalence. The full joint readouts
and no-exclusion rule already guard these possibilities; no additional arm or
prerequisite experiment is needed for inert preparation.

The design earns its scope as a new learning-level answer, not as proof of
the covariance observer's mediator or as a condition before publishing the
already completed paper. No familiarity/frequency implementation or acquisition
is the next action: the user's J-Lens redirection takes precedence. These
prospective considerations apply only if this archived design is later resumed.

## Reviewed document identities

| Document | SHA256 at review |
|---|---|
| `design-decision.md` | `1b986346ec0eeaadc8542a571ac3b0e7dac2d2c674bad04a7d3c52dbeb4c2346` |
| `protocol.md` | `4e266325a2e8133a833a47a030c29adf2372b6ca0663c8d15761e213dae9ffda` |
| `source-feasibility.md` | `e35853df25fee6bcc36c81c684f99c6a9d513d7cdd2b7045482722cec2fd9ced` |
| `design-assessment-independent.md` | `bd4548485e26444f4f624baafd8470e02be94d3479a930831be1eca83598a5de` |