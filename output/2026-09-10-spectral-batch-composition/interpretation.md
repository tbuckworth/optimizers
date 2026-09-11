# Grouping makes useful spectral behavior more accessible—but does not rescue noisy rare recognition

Codex — Spectral Optimizer Investigation, 10 September 2026.
Evidence: **three paired fresh seeds; one fixed 36-trajectory experiment;
independently checked saved results.** This interpretation reads the accepted
audit summary and all72 receipt-bound diagnostic JSON reports. It does not load
models, tensors or logits, rerun training/inference/gradients/auditing, or change
the registered experiment. Diagnostic synthesis below is descriptive,
post-outcome interpretation of all registered anchors, not a new significance
test or outcome-selected interval.

## Strongest constructive result first

Changing batch composition while preserving each50-update block's exact example
multiset materially improves the native filter's **common-digit protection under
Diffuse corruption**. Grouped-minus-Interleaved common accuracy gains are
**+9.20,+10.07,+6.91 percentage points** across seeds121/122/123, mean **+8.73**
(sample SE0.94). The corresponding native-minus-raw schedule interaction is
**+10.09,+8.56,+6.38 points**, mean **+8.34** (SE1.08). This is not an effect that
appears only after averaging opposite signs.

Grouped native common accuracy reaches56.64%, versus47.92% Interleaved native
and35.57% Grouped AdamW. Wrong-target training fit falls from7.93% to6.40%, with
the reduction present in every seed. The best steelman is therefore that batch
composition provides real headroom for a useful, geometry-dependent learning
restriction without adding clean labels, rare examples or loss weight.

There is a second constructive observation: on Clean data, native rare accuracy
increases from2.27% to10.27%. All three paired accuracy changes are positive,
but **+0.8,+0.6,+22.6 points** makes the heterogeneity essential. Grouped rare
accuracy is0.8%,0.6%,29.4%, not three similar rescues. Clean rare CE improves in
two seeds and worsens in one; mean change−0.486, SE0.330. Native common accuracy
falls0.133 points and common CE rises0.00693, both adverse in every seed. Those
costs are small but cannot be called “no sacrifice.” Native rare recognition
also remains well below the roughly52–53% Grouped raw-policy means.

## The requested rescue is not established

Under Diffuse corruption, **native rare accuracy remains zero for every seed
under both schedules**, on heldout data and the50 rare training examples.
Rare CE does improve from the inherited warmup in every seed, so zero accuracy
is not zero learning. Grouping lowers native rare CE by−0.502,+0.008,−0.088
(mean−0.194): two favorable signs, one unfavorable. It is not a successful
noisy rare-recognition rescue.

Common probability quality remains a qualification too. Native Diffuse common
CE moves from1.9095 to1.8941, improving in two seeds but worsening in one. It is
still worse than Grouped AdamW's1.8531 and the own-norm raw control's1.8364.
Grouped balanced heldout CE is2.1672 native versus1.9684 AdamW. Common accuracy
protection and poorer probability predictions coexist.

The endpoint interaction has an important trap: native-minus-raw's Diffuse rare
accuracy interaction is a seemingly favorable **+34.4 points**, but the entire
value comes from **raw deterioration**, not native improvement. Grouping drops
AdamW rare accuracy51.8%→17.4% and own-norm raw52.67%→21.4%; all three seed
accuracy changes are adverse for both controls. Native is0%→0%. The raw rare-CE
changes are adverse in every seed, whereas the norm-control rare-CE changes are
adverse in two and favorable in one. A positive policy interaction is not itself
recovery of useful competence.

All schedules still begin at the same majority-only warmup. Native Diffuse
common accuracy falls40.40 points under Interleaved and31.67 under Grouped.
The gain is **less deterioration**, not absolute improvement on that inherited
classification task. Clean native rare accuracy genuinely improves from the
zero-accuracy warmup, but Diffuse recognition does not.

## What the native diagnostics add

The anchor rule chooses high/low rare-count batches in blocks0,7,37, independently
of outcomes. Across these selected blocks, Grouped high batches contain23–38
rare occurrences, Interleaved high batches contain1, and lows contain0. These
are deliberately different inputs and generally different chronological steps
and parameter/Adam/observer states. The high/low label is not a randomized,
matched-state intervention or a representative sample of all1,900 updates.

Nevertheless the positive mechanistic observation is substantial: **all18
Grouped high-count events improve the fixed rare training-probe CE after the
actual Adam update**—all9 Clean and all9 Diffuse. These are finite before-minus-
after losses, not just positive gradient alignment. At the first Grouped high
events, native rare mean-action energy retention is98.20–99.19% Clean and
96.79–98.10% Diffuse. Corresponding finite rare CE improvements are:

| Cell | Seed121 | Seed122 | Seed123 |
| --- | ---: | ---: | ---: |
| Clean, first Grouped high event | +0.51559 | +0.38263 | +0.60674 |
| Diffuse, first Grouped high event | +0.29782 | +0.39157 | +0.21198 |

Thus concentrated rare batches can coincide with high retained incoming signal
and genuinely helpful actual steps. A claim that the filter simply cannot
learn from rare data, or that only a covariance statistic changes, is too strong.
The signed linear utilities agree in sign for these positive high-event results;
the finite loss measurements provide the direct local evidence.

The contradiction is equally informative: **all9 Diffuse Grouped high events
worsen the common true-label probe CE**, while long-run common endpoint accuracy
improves. For Clean,6/9 Grouped high events worsen common-probe CE,3/9 improve.
Locally helping rare examples can temporarily oppose familiar classification;
the selected positive/negative steps do not determine their accumulated balance.

For completeness, counts of positive finite CE improvements across every fixed
anchor are below. Each cell contains9 distinct events, three per seed. They are
dependent descriptive counts, not nine independent replications.

| Cell / schedule | High: rare helps | Low: rare helps | High: common helps | Low: common helps |
| --- | ---: | ---: | ---: | ---: |
| Clean / Interleaved | 6/9 | 4/9 | 4/9 | 3/9 |
| Clean / Grouped | 9/9 | 4/9 | 3/9 | 5/9 |
| Diffuse / Interleaved | 7/9 | 6/9 | 5/9 | 3/9 |
| Diffuse / Grouped | 9/9 | 5/9 | 0/9 | 5/9 |

### Late signal accessibility is not sufficient

The final-block native rare mean-energy retention is already96.45–97.75% in
Clean Interleaved and99.22–99.78% in Clean Grouped. Under Diffuse it is
61.63–69.49% Interleaved but **96.29–99.18% Grouped**, pooling both prescribed
high and low events. The Grouped result therefore substantially weakens a
simple *permanent exclusion of the rare mean gradient* account. Late rare-minus-
majority difference retention is also95.84–98.69% in Diffuse Grouped.

Yet both Diffuse native endpoints retain zero rare recognition. Passing the
incoming mean into the delivered gradient space does not ensure sufficient
accumulated logit change, useful representations or eventual argmax recognition.
These diagnostics do not identify which intervening updates undo/compete with
rare improvements. Sparse local measurements cannot establish a forgetting
trajectory between bursts.

Nor is actual Adam movement confined to the retained gradient span. At all six
late Diffuse Grouped events, **60.89–68.78% of adaptive displacement energy** lies
outside the current numerical span. The corresponding range in Interleaved is
62.95–68.41%. These are squared-energy fractions, not norm ratios; decay has
already been separated using the saved rounded FP32 AdamW multiplication.
High incoming retention and substantial outside-span Adam movement coexist.
The report therefore does not call the optimizer update a projector step.

The randomized wrong-probe geometry also resists a blanket rejection story.
For each seed, averaging its six fixed Diffuse anchors, wrong-assigned mean
retention is75.88%,68.16%,68.46% Interleaved and74.97%,66.43%,65.08% Grouped;
wrong-minus-corrected retention remains roughly75–78% across those seed means.
The lower wrong-target fitting rate is not explained here by near-zero passage
of the measured wrong mean/residual. These random training subsets are better
distributed than the previous class-blocked convenience probes, but remain
small training probes, not a population-level causal attribution.

## Most plausible accounts, in order of evidential support

1. **Composition affects useful filtering, but learning remains a dynamic
   competition.** The consistent common-accuracy schedule interaction, useful
   rare burst updates and heterogeneous long-run rare outcomes support this
   broad account. This is behavior plus a local mechanism clue, not mediation.
2. **Concentrated rare events make the relevant incoming direction accessible.**
   The high native mean/difference retention is compatible with the proposed
   covariance contribution. However the observer sees the current training
   gradient before measurement; that gradient itself contains the rare burst.
   High retention is partly an immediate post-observation property. There is
   no matched-state observer-only comparison here.
3. **Between-event competition, novelty/history or Adam feedback limits
   accumulation.** Rare mean retention and positive burst utilities can coexist
   with zero final recognition; raw controls also suffer under noisy grouping.
   This makes a schedule-wide issue plausible, not a filter-exclusive inability.
   Which of these mechanisms dominates is unknown; no full intervening state
   or signed-utility trajectory was measured.

The fixed-parameter multiset mean identity is mathematical, but gradients
along different neural trajectories are not equal. Grouping changes temporal
spacing, optimizer history and gradients as the weights move, as well as
covariance statistics. The rare digit's identity, low support and absence during
warmup remain entangled. No coherent-cue safety result transfers automatically
from this two-cell study, and none was newly measured.

Use **one matched-parent observer-only diagnostic**, rather than another broad
training roster. At the already saved step100 parents for all three fresh seeds
and both label cells, freeze model weights and Adam state. Compute the same
first block's two schedule sequences of gradients at that fixed parameter
vector. Feed each sequence into a separate copy of the identical inherited
observer, without optimizer steps. Then compare the resulting native actions
on one common, prospectively defined block-mean gradient, and the finite
rare/common losses after one copied-state Adam update driven by each action.

That would newly isolate whether schedule-induced observer geometry alone can
change delivered direction and useful local movement at a common model/Adam
state; it does not replay either completed training trajectory. Identical block
loss weight, shared test gradient, explicit current-action norm differences,
same Adam state and all seeds/cells must be retained. A positive result would
identify a concrete local observer channel, not prove endpoint rescue. A null
would redirect attention to co-evolution/history rather than establish that
covariance never matters. Specify fixed numerical tolerances and a small local
resource envelope before any execution. **No such work has been launched.**

## Source boundary

The load-bearing summary is the `independent_summary` in the
single audit result (artifact not distributed in this public snapshot),
SHA256 `3640269898d6c4115adbac925586fb54a6f1d47b0000b7e511a2fa1f1fcdeb6b`.
It reports PASS,514,404 checks,756 evaluations,36 trajectories and72 native
events in34.3879 seconds. Every diagnostic JSON used above was size/hash-checked
against that audit's input receipts before reading. The direct diagnostic files
are `acquisition-001/diagnostic-sSEED-CELL-SCHEDULE-native32-uSTEP.json` under the
same parent; their exact update identities and receipt hashes are in that audit.
The [protocol](protocol.md) and [implementation note](implementation.md) define
the prospective sampling and arithmetic conventions. Main owns the numerical
results report, plots and durable knowledge integration.
