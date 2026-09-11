# Saved contrast geometry: fixed calculation

Codex — Spectral Optimizer Investigation · J-Lens · 11 September 2026

## Question and evidence status

Does the previously observed endpoint change accompany smaller activation
contrast magnitude, changed alignment with the fixed directions, or both?
This is descriptive reuse of known outcomes, not a new prospective semantic
test. The seven PC4 sign reversals already imply alignment sign changes when
the contrast norm is nonzero; rediscovering this is not a new mechanism.

Reuse exactly the eight O−P contrasts from `2026-09-11-jlens-endpoint-location`,
in their original order: four contents, active/passive forms. New-data archive
here means the already completed shortened-prefix archive, NOT a new forward.
Use its saved verb and sentence_end states and the previous content/template
archive's full-prefix final states. Preserve all four unchanged u32 directions.
No new texts, PCA, fits, model calls, decoding, judges, training or paid compute.

## Quantities and conventions

At each of three roles r, form δ_r = h_O,r − h_P,r in float64 after converting
the saved float32 states. For each fixed nonzero direction u, let

    n = ||u||₂,  v = u/n,
    ρ_r = ||δ_r||₂,
    q_r = vᵀδ_r,
    c_r = q_r/ρ_r  when ρ_r > 0.

Then the old unnormalized score gap equals n ρ_r c_r in real arithmetic.
Check recomputed uᵀδ against the immutable saved gap with absolute tolerance
2e−12 and identical signs; do not replace old outcome counts. This checks
arithmetic, not semantics. A fixed source mean cancels in exact pair differences.

For each transition verb→sentence_end, sentence_end→old_full_end, and
verb→old_full_end, retain the norm ratio ρ_b/ρ_a, cosine between δ_a and δ_b,
signed c_a/c_b values (not their ratio), and the exact symmetric decomposition

    q_b − q_a = M + A,
    M = (ρ_b−ρ_a)(c_a+c_b)/2,
    A = (c_b−c_a)(ρ_a+ρ_b)/2.

M and A are an algebraic allocation under this symmetric convention, not
unique causal contributions or percentages. They can reinforce or cancel.
Only total projection changes necessarily telescope across transitions;
the two allocated terms separately need not telescope.
The unnormalized-gap allocation is n times these terms. No fitting, best-axis
selection, pooled significance test or winner threshold. Report every cell;
minimum/median/maximum across the eight cells are descriptive only.

If a contrast norm is zero, record q=0, cosine/alignment as null and any
undefined ratio/decomposition as null; do not impute a direction. Nonfinite
values or zero u norms stop the calculation. No denominator epsilon or clipped
cosine will hide an invalid input. Allow only floating-point tolerance in
the unit-cosine bound. Save original computed values.

## Source binding and checks

Pin both producer receipts, the earlier three-role analysis and u archive.
Check all four producer output file hashes; match exact dataset rows, IDs,
O/P pairing, roles, model revision, weights, adapter and layer. Require each
shortened token sequence to be the old sequence's prefix and the exact removed
suffix to be ` This happened yesterday.`. No assumed row-order coincidence.

Check Euclidean norms and dot products independently with Python math.fsum
inside the same new saved-data calculation. Fabricated fixtures first cover
shrink-only, rotation-only, both changes, sign reversal and zero contrast.
The independent reviewer inspects source/math before the single actual run.
One CPU thread, CUDA hidden, 60-second wall limit. Create an exclusive attempt
before reading scientific arrays; never automatically retry consumed output.

Implementation follows official [NumPy 1.26 load documentation](https://numpy.org/doc/1.26/reference/generated/numpy.load.html)
(pickle disabled and archives closed) and [norm documentation](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.norm.html)
(explicit vector axis). Explicit float64 arithmetic and shape checks avoid
accidentally computing a matrix norm or subtracting in saved float32 precision.

## Limits and useful next decision

These coordinates and norms are representation-dependent, not semantic
information or downstream behavioral utility. Full block11 contrast remaining
does not establish that any later layer uses it. Endpoint/context/tokenization,
partial propositions and different forward lengths remain entangled. A change
of δ orientation is not a measured rotation operator transporting the state.
Four authored contents expressed twice in one frozen model/layer are not eight
independent content replications. Old natural-text results remain separate.

Use the results to sharpen the interpretation of the local positive and later
failure. Do not select a new model experiment merely to keep the loop running.
