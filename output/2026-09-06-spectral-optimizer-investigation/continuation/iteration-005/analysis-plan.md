# Frozen-analysis specification for the common-gradient replay

Prepared before any iteration005 MNIST replay or worst-size resource pilot.
This formalizes [analysis-intent.md](analysis-intent.md) and the final
[protocol](protocol.md); it does not authorize execution. Source/schema tests
must be complete and bound before the separately approved pilot.

## Primary

Exactly one primary metric: `span_energy_fraction` at step 2000, computed on
rank-32 QR spans with the same algebraic finite-history reference. For seeds
3,4,5, report width128-minus-width32, all three signed differences and their
mean, median, minimum, maximum and descriptive sample SD. Group statistics
require all three valid pairs; otherwise null with explicit unavailable seeds
and reasons. Do not substitute a different time, a different rank, a full-width
covariance error or an available-case mean as the primary result.

The reference boundary-gap gate does not by itself invalidate this energy
fraction. A unique-reference-projector distance or its probe retentions may
be null when the energy fraction and observer probe metrics remain defined.
Keep numerical failure separate from a scientifically undefined diagnostic.

## Mandatory secondary reporting

Retain all twelve seed/snapshot records (seeds 3,4,5; steps 200,500,1000,2000),
both widths, every stored scalar metric, null reason, numerator/denominator and
reference validity diagnostic. The comparison layers remain distinct:

1. Full stored covariance error: widths 32 and 128 have unequal capacity.
2. Rank-32 QR-span energy and gated projector distance: matched rank, analysis
   span, not an alteration of the stored float32 delivery operator.
3. Native observer action/probe retention, clean-minus-fixed-corruption gap,
   noisy-gradient energy/cancellation and current-gradient self-inclusion.

At each fixed step, show all valid paired differences and counts. A descriptive
available-case mean for a secondary metric must be separately labeled and
must state its exact seed mask. Also report complete-case three-seed summaries
with unavailable seeds. For each seed, a four-snapshot mean of a metric or its
paired difference requires all four scheduled values; no partial-prefix average
is silently substituted. Across-seed summaries of those means require all three
complete seeds. Report mean/median/range/descriptive sample SD and preserve
all signed individual values.

Arithmetic means of per-probe retentions are the ordinary retention summaries.
Retain summed original/projected energies across each seed's four snapshots
and their separately named ratio as a secondary energy-weighted quantity.
Zero total input energy yields null; do not fill undefined per-probe ratios
with zero. Comparing paired arithmetic means requires matching validity masks.
Any reference-probe summary remains distinctly named, rather than being pooled
with the native observers or used to hide a boundary-tie null.

## Validation and interpretation

The summarizer must reject an incomplete confirmation, duplicate/missing seed
or snapshot, unexpected width/metric schema, non-finite non-null metric or a
failed required reproduction/numerical gate. It reads completed metadata only,
never pilot outcomes, and refuses output overwrite. Input/source hashes and
exact completion status accompany its output. It must not modify bulk files,
the raw records or existing experiment summaries.

These are three reused randomized gradient streams, not three new training
policy outcomes, twelve independent prefixes, or independent dataset/test-set
replication. No new test evaluation occurs. There are no p-values, confidence
intervals based on repeated states, significance, equivalence or uniquely
identified denoising claims. Better covariance capture and worse or unchanged
clean/corruption retention selectivity are compatible findings. Report both.
