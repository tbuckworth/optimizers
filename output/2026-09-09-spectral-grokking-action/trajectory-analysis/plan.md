# Saved continuation diagnostics: fixed descriptive plan

9 September 2026. **Post-outcome exploratory saved-JSON analysis**, written after
the action endpoint report was accepted. It is prospective only with respect
to this new summarization, not preregistered confirmation. Before writing it,
the author read the endpoint report, the source code, metadata/schema of the
seed100 history files, and the already audited first-step JSON. No later-history
numerical summaries have yet been computed.

## Fixed scope

Read all ten completed `seed100`–`seed104` histories, both `orthogonal` and
`norm_matched`, from the fixed batch
`/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD`.
Use every one of the 999 steps 1502–2500 per history. Report the whole interval
and the original endpoint-aligned windows 1502–2000 and 2001–2500. Step1501 is
separate: copy selected values from the existing accepted tensor-audit JSON,
never recompute tensors or combine that common-input step with later histories.

For every seed/policy/window report count, minimum, maximum, mean and median:
basis column count k, numerical rank, rank/k, largest and smallest singular
values, delivered/raw gradient norm ratio, and raw/delivered norms. For the
norm-matched histories also report c, hypothetical branch-local native/raw and
orthogonal/raw norm ratios, native-versus-orthogonal action cosine, and
post-cast absolute/relative norm mismatch. Preserve undefined ratios/cosines;
do not aggregate a selectively defined subset. Report full-column-rank,
degenerate, clamped, nonexact and tolerance-exceeded counts. Equal-weight
across-seed means of the within-seed means are descriptive; do not treat 999
temporally dependent updates as independent experimental replications.

No outcome-selected windows, correlations, p-values, new loss/readout analysis,
new training, inference, checkpoint replay, GPU, cloud, algorithm or
hyperparameter. No plot is required; a complete per-seed window table is the
smallest useful comparison and main may visualize the generated JSON.

## Admission and provenance

Bind the known accepted batch-completion hash and tensor-audit hash. Validate
all five seed completion receipts, exact seven-checkpoint roster and parent
identity; validate each history's schema/seed/policy/endpoint/source-map and
checkpoint receipt against that accepted completion. Hash the ten JSON history
files before and after processing; bind all source files and this plan, and
verify the ten original scientific source hashes still match the accepted map.
Validate all 999 contiguous step identities, dimensions, singular-value/rank
bookkeeping, finite numeric fields, coefficient dimensions, expected
policy-specific metric availability and norm-matching arithmetic/flags.

**The trajectory JSON files were not individually hashed in the original batch
completion receipt.** Their metadata/checkpoint/source bindings and current
hashes establish internally consistent present artifacts, not retrospective
cryptographic proof that every history scalar was immutable since acquisition.
The accepted checkpoint receipts are compared as metadata; this analysis does
not reopen or hash the large checkpoint payloads, already accepted previously.
No claim of an independent tensor corroboration of later histories follows.

## Boundaries of interpretation

Each c is `||V(V^T g)|| / ||QQ^T g||` computed at the norm-matched branch's own
then-current state, with the frozen denominator-floor convention. It does not
match the archived native trajectory's norm or the other branch's norm. It is
not an actual Adam displacement or effective learning-rate multiplier. Later
histories do not save raw gradient/basis orientations or actual Adam movements:
they cannot identify cross-branch span angles, subspace turnover, later Adam
dose, moment mediation, curvature alignment or feature semantics. Full column
rank concerns the stored thin V, not all P parameter directions. Within-branch
native/orthogonal cosines are unavailable for orthogonal histories by design.

## Execution envelope

One stdlib-only Python process, one CPU thread, 2GiB memory, no swap, five-minute
external runtime limit and 280-second cooperative deadline. New output cap
100MiB with 1GiB free-space reserve. Exact exclusive output:
`/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD/trajectory-analysis-001`.
Preserve any failed/partial output, with no automatic retry. Review the source
and plan before one launch; main records the service's actual execution receipt.
Compact accepted outputs may then be copied into this repository directory.
