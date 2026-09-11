# Frozen-shape analysis contract for implementation review

Codex parent, Spectral Optimizer Investigation, 6 September 2026.
Status: prospective specification; implementation approval is recorded separately.
Read with [scientific design](common-state-design.md), the anchor schema and
numerical contract. No new scientific outcomes were consulted in this document.

## Units, membership and interpretation

The observational unit is a named **current-policy-state** anchor `(bundle,t)`.
Primary bundles are `[71001,71002,71003]`, sensitivity `[71901]`; each has
times `[101,500,1000,2000]`. These sets are immutable once execution is approved.
Development bundle 71990 at `[101,200]` never enters scientific tables. Each
record carries `evidence_role` as `development`, `primary`, or `sensitivity`;
an independent numerical audit is an artifact role, not another scientific set.

Six branch keys, in output order: `raw,current,lagged,restored,reciprocal,zero`.
Four loss keys: `batch_noisy,train_probe_noisy,train_probe_clean,auxiliary_clean`.
The auxiliary-clean mean over all 5,000 reserved images is primary. Each of its
ten disjoint 500-image chunks is a diagnostic with its own before/after/delta;
do not treat chunks or times as independent replications. Training probes keep
duplicate sampled indices and their order. No test or accuracy field is allowed.

Finite native tensors and exact same-platform replay are engineering gates.
Positive, negative, mixed, weak-leverage and valid undefined scientific cells
are all reportable outcomes. No passing engineering test certifies benefit.

## Candidate construction and state intervention

Use one native raw minibatch gradient and one cloned observer transition per
anchor. The old basis is copied before that transition. The stored current
and lagged candidate vectors are shared immutable inputs to all branches;
do not reobserve in each branch. Raw/current/lagged preserve their native bits.
Restored and reciprocal use float64 unit direction times target norm, then one
float32 cast. Zero is a full assigned tensor, not None and not a skipped step.

Apply the scientific design's exact-zero truth table before division. Finite
positive-target/zero-direction is the sole expected branch-domain null. There
is no broad exception-to-null conversion. Overflow, nonfinite input, failure
of the 1e-6 relative norm/unit-direction gate or positive-target underflow stops
execution with evidence retained. A zero target must produce exact zeros.

For every valid branch, reload the identical model, optimizer and RNG state,
assign the candidate, execute one AdamW step and retain actual native endpoint
parameters. Compare current against the **live source-step** witness, including
moments/counters and observer state. A replay-generated witness is not valid.
The raw input uses filtered-history moments. The zero input includes existing
moment motion; neither is a separate source trajectory or additive mediator.

## Tensor-derived quantities

Flatten parameters in recorded ordered-name order. Work on CPU float64 copies
for numerical summaries, with `delta=after.double()-before.double()` and
`delta_data=delta+0.001*0.01*before.double()`. Never form delta in float32 first.
Report actual native endpoints as well as both displacement vectors. The
nominal decay subtraction does not remove historical moments or float roundoff.

Use float64 accumulation for vector norm, squared norm and dot. No cosine
clipping; zero-norm cosines are null with a reason. For each of the 15 unordered
branch pairs, report delivered-vector, full-delta and decay-subtracted-delta
distance and cosine. The latter two pairwise vector differences should agree
up to the numerical contract because their common nominal decay term cancels.

Leverage fields per anchor are `current_norm`, `lagged_norm`, signed norm
difference, relative absolute norm separation, ratio nc/nl, unit-direction
distance/cosine, and the branch-pair delivered distances. Relative norm
separation is `abs(nc-nl)/max(nc,nl)` when the denominator is positive; otherwise
null. Norm leverage requires separation >2e-6. Direction leverage requires
both norms positive and unit-direction distance >2e-6. Keep all numeric
responses when leverage is weak; only factor interpretation is restricted.

## Defined loss functional

For each anchor/probe, evaluate and cache the before loss and its parameter
gradient once. Evaluate each branch's after loss using its saved endpoint.
Float64 functional: promote the exact already normalized float32 input values
and native float32 parameters; eval-mode Linear–ReLU–Linear; unsmoothed,
unweighted integer-label CE. Use parameter order `0.weight,0.bias,2.weight,2.bias`.
Auxiliary reduction is ten ordered sum-CE and sum-gradient chunks, divided by
5,000 after accumulation. The other probes are a single sum divided by count.
Do not normalize integer pixels directly in double, deduplicate indices or
reorder the pool. Probe evaluation must not mutate source gradients or RNG.

Let B be the shared before CE and A_b the after CE of branch b. Persist B, A_b,
`Y_b=A_b-B`, the before gradient q, `D_b=q^T delta_b`,
`Ddata_b=q^T delta_data_b`, and `R_b=Y_b-D_b`. R is a finite-response residual,
not a Hessian estimate or a pure smooth-curvature term across ReLU kinks.
Record native GPU-float32 B/A/Y separately as concordance diagnostics; their
disagreement with the CPU64 functional is retained, not converted to equivalence.

## Fifteen mandatory contrasts

Apply these coefficients to scalar outcomes Y, D, Ddata and R, and to full/data
displacement vectors. The primary family is auxiliary-clean Y. All coefficients
sum to zero; for Y, compute from **after losses** directly so the shared B
cancels. Independently verify agreement with the corresponding differences of
stored Y values under the numerical contract. Keep cancellation-aware bounds.

| Contrast key | Nonzero branch coefficients |
|---|---|
| ordering | lagged +1, current -1 |
| direction_at_current_norm | restored +1, current -1 |
| direction_at_lagged_norm | lagged +1, reciprocal -1 |
| norm_at_current_direction | current +1, reciprocal -1 |
| norm_at_lagged_direction | restored +1, lagged -1 |
| interaction | restored +1, lagged -1, current -1, reciprocal +1 |
| current_minus_raw | current +1, raw -1 |
| lagged_minus_raw | lagged +1, raw -1 |
| restored_minus_raw | restored +1, raw -1 |
| reciprocal_minus_raw | reciprocal +1, raw -1 |
| raw_minus_zero | raw +1, zero -1 |
| current_minus_zero | current +1, zero -1 |
| lagged_minus_zero | lagged +1, zero -1 |
| restored_minus_zero | restored +1, zero -1 |
| reciprocal_minus_zero | reciprocal +1, zero -1 |

Interaction is both `norm_at_lagged_direction - norm_at_current_direction`
and `direction_at_current_norm - direction_at_lagged_norm`; check both identities.
Negative Y contrast means the first policy lowers that exact probe loss more.
Do not interpret sign as favorable for every secondary quantity or probe.

## Masks and descriptive summaries

Missing branch membership is malformed input, not a domain null. A present
undefined branch has `defined=false`, its exact reason and no endpoint/results.
A contrast is defined iff all its nonzero-coefficient branches are defined.
For each contrast/probe/time, retain the ordered three-bundle values and mask.
Compute mean/min/max only if all three values are defined. Otherwise all three
summary fields are null with `incomplete_predeclared_primary_mask`; never average
an available subset. Missing execution anchors make the run incomplete, not
another eligible mask. Sensitivity uses the identical schema but no pooled mean.

Weak leverage is separate from numerical validity and domain masks. Do not
remove weak-leverage cells from numeric summaries. Factor-level claims require
the appropriate leverage fields; interaction needs both. Report exact named-state
signs and auditor resolution status, without CI, p-value, equivalence or implied
population frequency. The primary table precedes any interpretation.

## Decisions and engineering fixtures

Follow the scientific design's precedence: validity, domain, leverage, complete
contrasts, interpretation. A resolved interaction establishes only finite-grid
nonadditivity at the relevant anchor. Any sign/probe disagreement blocks a
universal local-direction account. The all-12-anchor concordance heuristic can
only prioritize a new source-state proposal. No result automatically authorizes
another run, seed, longer window, production policy or reciprocal training arm.

Required dataset-free fixtures before neural GO: four zero cases, tiny positive
norms and underflow rejection; missing/nonfinite/type/shape inputs; owned clones
and mutation isolation; known factorial tables with nonzero interaction and
shared-baseline cancellation; every required-cell mask, missing membership,
incomplete primary set, sensitivity separation and weak-leverage retention;
exact CPU source replay, moment-order rejection and RNG continuation; analytical
CE/gradient fixtures including ReLU kinks, chunking and duplicate examples.
These fixtures validate code paths, not sampled neural effect sizes.
