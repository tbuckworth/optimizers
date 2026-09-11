# Decision: separate prior familiarity from subsequent frequency

Codex — Spectral Optimizer Investigation · 10 September 2026

The assessment had recommended preparation of one bounded learning-level
study, without acquisition admission. That recommendation is superseded by
the redirection above. The completed design assessment finds a real gap:
the earlier rare digit was both infrequent and absent from warmup, with fewer
available training examples. Those results do not tell us whether filtering
can preserve a genuinely learned skill when it becomes rare, or whether it
also obstructs unfamiliar learning when that learning becomes frequent.

## Smallest useful comparison

Within each of three fresh seed bundles, swap which of two focal digit classes
(3 and 8) appears in a fixed clean warmup. Keep the other eight classes'
indexed warmup examples and slot positions identical. Then swap which focal
class receives 1% versus 10% of subsequent examples, leaving the eight-class
background stream identical. Both focal classes always have correct labels.

Each continuation is paired between ordinary AdamW and native stable rank32
filtering from the exact same full parent. Run both a clean background and a
fixed-corruption background. This gives **six new parents and 48 physical
continuations**; the two focal readouts yield 96 logical factorial rows, not
96 independent runs or more than three independent seed bundles. The
[proposed protocol](protocol.md) fixes the roster, estimands and limitations.

Warmup and continuation use disjoint images with equal available support per
class. Thus familiarity means prior exposure to the digit class, not repeated
training images. It still changes the complete learned model, Adam moments
and observer history; it is not a pure observer-memory intervention. The
omitted class has already received negative-class softmax gradients.

The two continuation schedules exchange exposure between the focal classes.
That holds background exposure exactly fixed, but couples a focal class's
frequency to the other focal class's opposite change. The claim must concern
this **paired exposure swap**, not an isolated frequency effect. A third
balanced schedule would add 24 physical runs across the two label settings
without removing that fundamental fixed-budget replacement question. It is
not selected.

## What would actually be learned

- **Constructive boundary:** native retains demonstrably acquired low-frequency
  competence while still improving unfamiliar correct learning, with reduced
  background wrong-label fitting. This would identify a useful preservation/
  adaptation regime in this small model, not semantic selection or safety.
- **Qualified trade-off:** protection of familiar performance accompanies
  restricted unfamiliar learning, especially at low frequency. This would
  strengthen an acquisition-history account at the learning level, without
  identifying the observer as its causal mediator.
- **Broader restriction:** familiar and unfamiliar correct learning both lose
  similarly, or making the unfamiliar class frequent does not help. That would
  weaken a simple unfamiliarity-specific explanation in this regime.
- **Ambiguous but retained:** weak initial competence, strong starting-state
  differences, mixed classes/seeds or disagreement between accuracy and CE
  can prevent a preservation conclusion. Report those outcomes; do not extend
  warmup, replace seeds, retune rank or select a favorable checkpoint.

Every interpretation must inspect absolute warmup and final performance,
progress since warmup, and raw/native contrasts. An interaction alone is not
a useful result: it can arise because raw gets worse. Subtracting each
parent's baseline does not equalize starting competence across familiarity.

## Feasibility and limits

The [source assessment](source-feasibility.md) confirms reusable model,
optimizer, observer and complete-state primitives, but existing plans and
group metrics hard-code digit 8. New role-aware plans and genuinely new parents
are necessary. No scientific arrays, model or checkpoints were opened during
this assessment; source and scalar/receipt metadata were sufficient.

The proposed 100-step warmup and 1,900-step continuations require 91,800
physical training updates. A conservative output inventory is about 1.09 GB
before its 2 GiB cap; analogous completed local acquisitions suggest a
10–20-minute planning allowance, not a measured runtime prediction. One local
RTX3090 is sufficient in expectation. Paid spend/reservation remains **$0 of
$100**; no GPU service or reservation has been created for this proposal.

If explicitly resumed in future, prepare inert source and fabricated CPU fixtures, independently check
the selected design and implementation, then freeze exact source/artifact
and resource admission before any scientific run. This is autonomous work
under the existing user authority, not another approval request. No sweep,
language-model experiment, optimizer-default change or manuscript expansion
is selected here.

Evidence inspected: prior [selectivity protocol/results](../2026-09-09-spectral-selectivity-boundary/results.md),
[batching results](../2026-09-10-spectral-batch-composition/results.md),
[observer pathway](../2026-09-10-spectral-observer-pathway/results.md), their
immutable source and completion metadata, and the
[preceding bounded decision](../2026-09-10-spectral-component-utility/next-decision.md).
This document adds a prospective design, not a new empirical result.
