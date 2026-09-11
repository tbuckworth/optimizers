# The later contrast is smaller and differently oriented

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

The saved states support **both smaller contrast magnitude and changed
orientation**, not a pure fading account. Across all eight cases, the later
activation difference has only 14–20% of its verb-position norm and is nearly
perpendicular to the original difference. This is coordinate geometry in one
layer, not measured information loss, a causal rotation mechanism or behavior.

## What was newly calculated

No model ran. We reused the frozen 16 shortened texts at two saved positions
plus the original 16 full texts at their final position. For every unchanged
O−P pair, δ is its full 1024-dimensional activation difference. The analysis
retains all four fixed directions and all three endpoint transitions.

| Contrast / template | Verb norm | Sentence-end norm | Later norm | Later / verb | Cosine: verb versus later δ |
|---|---:|---:|---:|---:|---:|
| Examined − readied / active | 2.1901 | 0.7922 | 0.4101 | 0.1872 | −0.0047 |
| Examined − readied / passive | 2.2195 | 0.7677 | 0.4222 | 0.1902 | +0.0734 |
| Logged − fulfilled / active | 1.7918 | 0.8877 | 0.3472 | 0.1938 | −0.0346 |
| Logged − fulfilled / passive | 2.1207 | 0.6560 | 0.3059 | 0.1443 | −0.0338 |
| Inventoried − dispensed / active | 2.0455 | 0.7211 | 0.3318 | 0.1622 | −0.0097 |
| Inventoried − dispensed / passive | 1.9691 | 0.6241 | 0.3982 | 0.2022 | +0.0237 |
| Catalogued − packaged / active | 1.8637 | 0.6012 | 0.3534 | 0.1896 | +0.0327 |
| Catalogued − packaged / passive | 1.7710 | 0.4628 | 0.2765 | 0.1561 | +0.0576 |

Norm decreases in every cell at both transitions. Median sentence-end/verb
ratio is 0.334 (range 0.261–0.495); later/sentence-end is 0.534 (0.391–0.638);
later/verb is 0.188 (0.144–0.202). These are medians of paired ratios, not
ratios of median norms. Full-vector cross-position cosines have medians 0.407,
0.152 and 0.009 respectively, with later/verb range −0.035 to +0.073.

Cosine zero means perpendicular in these coordinates, not that two meanings
are unrelated. Nonzero δ at a later endpoint is not proof that downstream
computation retains or uses the corresponding semantic distinction.

## Alignment with the unchanged PC4

| Contrast / template | Verb | Sentence end | Later ending |
|---|---:|---:|---:|
| Examined − readied / active | +0.0855 | +0.1074 | −0.0513 |
| Examined − readied / passive | +0.0821 | +0.1850 | −0.0086 |
| Logged − fulfilled / active | +0.1660 | +0.1010 | −0.0415 |
| Logged − fulfilled / passive | +0.1103 | +0.0589 | +0.0267 |
| Inventoried − dispensed / active | +0.2327 | +0.0655 | −0.0531 |
| Inventoried − dispensed / passive | +0.2252 | +0.1085 | −0.0257 |
| Catalogued − packaged / active | +0.1378 | +0.0283 | −0.0724 |
| Catalogued − packaged / passive | +0.0301 | +0.0195 | −0.0814 |

These are signed cosines c between δ and PC4, not the raw score gaps or
accuracy. From verb to first-sentence end, alignment improves in both
examined/readied cases while norm shrinks. It decreases in the other six.
All eight alignments decrease at the next transition, with seven sign reversals.
The earlier 8/8, 8/8 and 1/8 sign counts remain unchanged. Sign reversals already
implied an alignment change before this calculation; the new information is
the full-vector magnitude and cross-position geometry, not that implication.

All axes remain in the [complete result JSON](analysis/results.json). Their
median signed alignments (descriptive, not a winner selection) are:

| Direction | Verb | Sentence end | Later ending |
|---|---:|---:|---:|
| PC1 | +0.1252 | +0.0966 | −0.0138 |
| PC2 | −0.0140 | +0.0024 | +0.0082 |
| PC3 | +0.0872 | +0.1239 | −0.0073 |
| PC4 | +0.1241 | +0.0833 | −0.0464 |

## Mathematics: an accounting identity, not causal attribution

For the unchanged direction u, write n=||u||₂, v=u/n, ρ=||δ||₂,
q=vᵀδ and c=q/ρ. The original score gap is nρc in exact arithmetic.
The fixed source mean cancels from O−P. For nonzero endpoint norms:

    q_b − q_a = M + A
    M = (ρ_b − ρ_a)(c_a + c_b)/2
    A = (c_b − c_a)(ρ_a + ρ_b)/2

Stored terms use unit-direction projection units; multiply both by n for
original-gap units. They reconstruct all transitions within tolerance.
The split is a symmetric convention, not unique causal contributions.
Only total projection changes necessarily telescope; the terms separately do
not. No percentages or ratios of signed terms are reported. In particular,
shrinking norm can produce a positive M when average alignment is negative.
That is less negative projection, not increased semantic information.

## Useful interpretation and next decision

The prior local positive is still useful: the fixed J-Lens interpretation
predicted all eight signs at the verb and first sentence's end. A later
shared ending is not an interchangeable measurement location. The evidence
suggests reporting token roles explicitly and matching them when comparing
readout usefulness. It does not justify an endpoint-specific sign flip or a
general rule that every interpretation should be measured at a verb.

The next useful task is a compact synthesis and a new-verb,
matched-role replication design—not another model launch by default. Compare
the fixed interpretation's predictions with example-based references under
the same endpoint information, retaining all directions and failures. Any
new data/criterion must be specified before acquisition; the current four
reused authored contents cannot certify lexical or semantic generalization.
Here new means not yet tested in this investigation, not absent from pretraining.

Context, token identities and positions vary together; causal verb prefixes
have different partial propositions under the two templates. The extra
sentence adds time information, and BF16 forwards have different lengths.
Norms and angles are representation-dependent. The old external-text 19/24
and the completed small comparator results answer different questions and
are not pooled here. No optimizer or safety-performance claim follows.

## Provenance and checks

Protocol/source frozen `c042abf`; [independent source review](source-review.md)
and both seven-case fabricated tests passed before real array access.
One new CPU-only calculation completed 01:46:57 UTC, process exit 0. It checks
24 contrast vectors, 96 projections and 24 transitions, with scalar fsum
corroboration and exact previous-gap signs. Maximum previous-gap discrepancy
is below 5.56e−17. These validate arithmetic and bindings, not semantics.

The [result receipt](analysis/results.json) contains hashes for both producer
receipts, all eight producer outputs, original datasets/pairs, fixed direction
archive, protocol, source and prior three-role analysis. Result SHA:
`068c2dd8ec409452a622c6ebd90cf7d76de5930421edd1fad8d0a6bb8a7422f2`.
The exclusive attempt (artifact not distributed in this public snapshot) and result are consumed; do not
repeat this calculation or any prior model stage. Paid spent/reserved $0/$100.
