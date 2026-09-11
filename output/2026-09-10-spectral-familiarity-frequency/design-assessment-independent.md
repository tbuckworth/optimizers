# Familiarity × continuation frequency: independent design assessment

Codex — Spectral Optimizer Investigation · 10 September 2026

## Recommendation and distinct information

This follows the [design-only next decision](../2026-09-10-spectral-component-utility/next-decision.md) and the [unified hypotheses, §§6–7](../../research/spectral_optimizer_unified_hypotheses_2026-09-10.md). The research-workflow skill supplies evidence discipline, not a new workflow or approval gate. Only documents were inspected; no model, dataset array, saved numerical readout, experiment, or scientific audit was executed.

The [final prospective protocol](protocol.md), particularly [§6](protocol.md#6-readouts-contrasts-and-competence-annotation), is authoritative for preparation and the readout hierarchy. Recommendations and alternative designs discussed here are subordinate to that protocol, not competing specifications.

| Completed evidence | What remains unseparated |
|---|---|
| [Selectivity protocol](../2026-09-09-spectral-selectivity-boundary/protocol.md) and [results](../2026-09-09-spectral-selectivity-boundary/results.md): one designated digit, absent during warmup, lower unique training support and lower continuation frequency. | Class identity, prior exposure, support and continuation frequency were coupled. Zero argmax accuracy coexisted with substantial CE improvement; preservation cannot be inferred for a class that started unrecognized. |
| [Batching protocol](../2026-09-10-spectral-batch-composition/protocol.md) and [results](../2026-09-10-spectral-batch-composition/results.md): same indexed occurrences, changed grouping/order, same common-only warmup. | This tested temporal arrangement at equal exposure, not prior familiarity or dose. Favorable noisy rare interactions could arise entirely from raw deterioration. |
| [Observer protocol](../2026-09-10-spectral-observer-pathway/protocol.md) and [results](../2026-09-10-spectral-observer-pathway/results.md), with [signal accounting](../2026-09-10-spectral-observer-signal/interpretation.md). | These identify a conditional fixed-model history pathway, not the effect of having learned the designated class before continuation. Accessibility did not fix useful-delivery signs. |
| [Component diagnostic](../2026-09-10-spectral-component-utility/results.md) and [action/history synthesis](../../research/spectral_action_history_synthesis_2026-09-10.md). | Present-step component effects do not identify accumulated learning or a familiarity interaction. Repeating those probes would not fill this gap. |

## Smallest useful counterbalance

Predeclare two focal digits, 3 and 8, and use **both digits in every fresh seed bundle**. Merely assigning a different digit to each of three seeds confounds digit and seed. This two-digit design supports those designated classes, not a population claim over all ten digits.

For each seed, create two equal-clock, clean raw-delivery warmups with a passively recording canonical observer. Couple initialization and every background occurrence/slot exactly. Swap only which focal class fills the ninth-class slots:

| Warmup | Eight background digits | Focal 3 | Focal 8 |
|---|---|---|---|
| A | Identical indexed occurrences and slots | Represented | Absent |
| B | Identical indexed occurrences and slots | Absent | Represented |

This fixes the replacement policy: removing one focal class adds the other focal class, not extra updates or an unspecified increase in background exposure. It is a *swapped prior-exposure intervention*, not an intervention on observer memory alone. Use equal source-pool sizes for every class; remove the old 50-versus-550 unique-support asymmetry.

The selected candidate uses disjoint warmup-only and continuation-only example pools within each class—550 IDs per role, plus 500 independently disjoint held-out IDs—to target **class familiarity**. This prevents exact-image recurrence from being bundled into the treatment and adds little storage relative to the existing small-model workflow. Reusing the same 550/class pool in both phases would instead test prior class-and-example exposure; that alternative is not selected. No old source state supplies both counterbalanced warmups; these would be genuinely new parents, not a reason to replay completed parents.

### Two continuation schedules are enough

The main agent's proposed paired focal-frequency swap is preferable to my initial three-schedule option (balanced, low-3, low-8). It uses fewer branches and, importantly, leaves all eight background exposures fixed:

| Schedule, per 100-update block of 6,400 slots | Focal 3 | Focal 8 | Each background digit |
|---|---:|---:|---:|
| Low-3 / high-8 | 64 (1%) | 640 (10%) | 712 (11.125%) |
| High-3 / low-8 | 640 (10%) | 64 (1%) | 712 (11.125%) |

Keep identical background occurrence IDs, chronological positions/order, and common random slot permutations across the two schedules. Couple each focal class's low stream to a declared subset/prefix of its high stream within each block; high exposure adds occurrences from the same fixed pool, not extra eligible examples. Do not separately group low-frequency examples into bursts: that would reopen the completed batching factor. Persist exact quotas, indexed occurrences, duplicates, unique IDs encountered, and first-arrival times.

Every full warmup state branches onto both schedules and raw/native. With three seeds this is **24 physical clean continuations**, yielding 48 logical focal-class rows because both digits are evaluated in every trajectory. The logical rows are dependent, not 48 replications. No separate balanced reference is essential for the stated contrast. Its absence limits attribution: increasing one focal class's dose necessarily reduces its competitor's dose. This is a paired *exposure-allocation* effect, not an isolated target-frequency effect.

Matched Clean and Diffuse background-label conditions would make **48 physical continuations / 96 logical focal-class rows**. That extra factor is justified if the objective includes preservation under unwanted fitting, rather than just clean acquisition. Keep both focal classes correctly labeled. In Diffuse, fixed 0.9 uniform replacement among the eight background labels has expected actually-wrong fraction 0.7875 **within background examples**; it is not 90% actually wrong and not the whole-dataset error rate. Use the identical fixed assignment per source example across schedules, warmups and policies, with true/assigned/wrong-subset readouts. Do not add cues, augmentation, ranks, or norm controls to this minimal factorial.

## Estimands and pairing

Let J be held-out class-conditional CE, f ∈ {familiar, absent}, q ∈ {low, high}, and p ∈ {native, raw}. For each seed, designated digit and background-label condition, define:

```text
P(f,q,p) = J(warmup_f) − J(endpoint_f,q,p)       [absolute CE progress]
D(f,q)   = P(f,q,native) − P(f,q,raw)           [native benefit]
I(q)     = D(familiar,q) − D(absent,q)          [familiarity interaction]
K        = I(low) − I(high)                    [three-factor interaction]
```

Positive D favors native CE; positive I means its relative effect is more favorable after represented warmup. Under authoritative protocol §6, **focal reporting endpoint CE and accuracy are joint primary behavioral readouts**, per seed/class/cell, with warmup values and absolute changes mandatory. I(low), I(high), and K are **planned descriptive paired contrasts**, not separate primary outcomes; I(low)/I(high) here correspond to the protocol's J_low/J_high. Report them for both metrics, with all underlying P/D cells, retaining accuracy/CE disagreements. An isolated positive interaction cannot establish useful learning. For accuracy use endpoint-minus-warmup progress, rather than the CE sign convention. Average the two designated-digit contrasts within each seed first, then report the three seed summaries alongside all digit-specific values. Do not pool the two digits, label conditions, or branches into additional independent seeds.

Within each (seed, warmup), preserve the exact parent weights, optimizer moments/counters, observer state/counter and RNG before every branch. Raw and native share identical continuation inputs/labels within each schedule and label condition. All branches have identical update count, batch size, learning-rate/decay schedule, evaluation steps and opportunity for selection. The native observer advances once per native update; raw bypasses it. The warmup observer has the same fixed age in the two familiarization states, but its contents legitimately differ.

A minimal continuation of the accepted small recipe is raw-delivery warmup 100, continuation 1,900, batch 64, fixed endpoint 2,000, unchanged stable rank32 and AdamW. This supplies 19 exact quota blocks and equal clocks; it is not a proposal to extend warmup until results look satisfactory. Freeze the finite evaluation schedule before acquisition. Save baseline, warmup and endpoint logits and intervening fixed curves for each focal digit, every background digit, background macro, and balanced total. In Diffuse additionally retain actual wrong-target fit. No validation-selected winner or adaptive duration is needed for this question.

## Competence and interpretive guards

**Warmup-normalized progress does not remove ceiling confounding.** Since raw/native share their parent, D = J(raw endpoint) − J(native endpoint): the warmup subtraction cancels. Comparing D across familiarization states still compares different representations, confidence/headroom, moments and histories. Do not divide gains by baseline error or accuracy headroom, match checkpoints using outcomes, or treat regression adjustment for baseline competence as an identified direct effect.

Main's selected competence annotation is familiar-class held-out accuracy at least 50% **and** CE below that class's CE under the same seed's initialized model. This checks useful acquired competence rather than assuming that exposure alone established it. It is not an empirically calibrated threshold or a criterion for a good optimizer. Keep every predeclared parent/branch if it fails; mark the competent-preservation interpretation unavailable for that case, without replacing seeds/classes, extending warmup or excluding it from the main table. Report the annotation for every seed/focal class, the continuous represented-versus-absent baseline gap, available accuracy/CE headroom, and full curves. No additional gap cutoff, headroom normalization, or outcome-dependent parent selection is recommended. Freeze this definition before outcomes rather than search several guards afterward.

The following outcomes are meaningfully distinguishable:

Class-conditional evaluation removes evaluation-prior weighting, **not training distribution shift**. Familiar-low moves from roughly one-ninth warmup prevalence to 1%; absent-low moves from zero to 1%. High exposure changes the size/direction of that shift as well. The absent class's output weights have already received negative-class softmax gradients, and shared features may already exist; “absent” is not an untouched parameter direction or proof of semantic novelty. Fixed clocks also mean unequal focal example exposure by design. Record dose, distinct examples and spacing; do not relabel matched-update comparisons as exposure-controlled frequency effects. Matched-dose horizons would change clocks and define another study, not a hidden correction.

## Boundedness and decision

The learning-level factorial cannot distinguish model familiarity, optimizer-state effects and observer-memory mediation. That is an honest boundary, not a defect requiring the previous observer experiment to be repeated. No gradient replay or another local retention table is necessary for this first behavioral answer.

With the suggested 100/1,900 clocks, 48 continuations plus six shared warmups mean 91,800 physical training updates; the clean-only option means 46,200. Using 21 saved states, 5,500 continuation-training and 5,000 held-out logits at float32/10 classes gives about 423 MB of logical logit payload for 48 arms before shared-baseline deduplication. Full endpoint/warmup states and manifests require a separate exact inventory; a provisional 2 GiB archive cap is plausible, not verified admission. The [earlier batching run](../2026-09-10-spectral-batch-composition/results.md) establishes that similar small-model work has been bounded locally, not a runtime guarantee for this new design. Main owns any final host/GPU/time caps and source verification. No old scientific source needs mutation.