# Saved-vector signal accounting

Codex — Spectral Optimizer Investigation · 10 September 2026.

Status: source preparation; no scientific tensor calculation or job launched.
This implements the previously selected
[decision](../2026-09-10-spectral-observer-pathway/next-decision.md),
design (artifact not distributed in this public snapshot) and
[review](../2026-09-10-spectral-observer-pathway/next-design-review.md), not a
new training experiment or user-approval gate. Source, fabricated fixtures,
input inventory and numerical thresholds must be frozen before arithmetic.

## Question and scope

The completed fixed-model study found that grouped batch history increased
rare-probe accessibility under Clean and Diffuse labels, but improved the
relative rare-loss step consistently only under Clean. We now ask where the
local chain differs: incoming signal, filtered delivery, or actual Adam motion.
Preserve the favorable Clean finding and all contradictory/common-class costs.
This is outcome-informed accounting of three reused parents, not replication.

Seeds 202609121/122/123; cells Clean/Diffuse; probes rare/common; actions native
interleaved/native grouped/raw/zero. There are six cases and 48 logical
action/probe entries backed by 21 physical saved steps. Zero is physically
shared across label cells for each seed. No case, probe or control is dropped.

Only the saved O151 delivered action is used. Let g be the common block input,
q the rare or common true-label training probe, and h_s the native delivery.
Compute vectors in FP64, including conversion BEFORE every subtraction:

    B_q = qᵀg
    F_q,s = qᵀh_s
    K_q,s = qᵀ(h_s − g)
    D_filter,q = qᵀ(h_grouped − h_interleaved)

Report signed products, both vector norms and cosine separately. Cosine is
undefined at zero norm, never silently zero. Explicit control identities:
F_raw=B, F_zero=0, K_raw=0, K_zero=−B. No division by B to manufacture a
“useful signal retention fraction.” Also compute the six paired label-input
products qᵀ(g_Diffuse−g_Clean), with matched-probe/model identities checked.

Join, without replaying or recomputing actual steps, the accepted audit scalars:

    J_q,s = −qᵀDelta_s                  actual saved Adam linear utility
    U_q,s = CE_before − CE_after        actual full held-out group loss change
    E_q,s = J_q,s − J_q,zero            matched input intervention over zero

Include absolute values, native group-minus-interleaved contrasts and both
native actions versus raw/zero. Raw/zero's own absolute utilities remain
visible. All six action-pair contrasts for each case/probe give 72 rows; the
12 grouping contrasts are the primary explanatory subset, not extra samples.
Publish three-seed rows before means, sample SD/SE and signed/ambiguous counts.
Do not pool labels or probes as independent parents. Preserve already checked
retention and decay/adaptive scalars as joined quantities, not new measurements.

## Prospective numerical rules

For every newly calculated FP64 product, compare the ordinary reduction with
an independent compensated math.fsum reduction of its FP64 element products.
Operational tolerance: 128×eps64×sum(abs(products))+1e−12. This is a frozen
consistency check, not a general worst-case summation theorem.

Compare direct difference products with differences of paired products using
a cancellation-aware tolerance based on all constituent absolute-product sums;
the exact expression, near-zero sign rule and fabricated cancellation/zero/
negative cases are frozen in the reviewed source before scientific arrays.
Do not alter a tolerance from observed outcomes. Nonfinite or failed numerical
checks terminate with failure, preserving partial evidence and no automatic
retry. An unresolved tiny sign is not evidence for a directional hypothesis.

## Sources, loading and receipts

Immutable acquisition root:
`/tmp/spectral-experiment-artifacts/spectral-observer-pathway-20260910.jTwt14/acquisition-001`.
Its completion SHA256:
`b1c4837438051dc6ed51665879b6c37d7bd35673a86f8c4f363878c3f022464f`.
Accepted audit:
`/tmp/spectral-experiment-artifacts/spectral-observer-pathway-20260910.jTwt14/audit-001/result.json`,
SHA256 `d9b05160c9a58952d4f0899699f79de2ee2d919c5319f2bb96d687ea448d0d35`;
PASS 33,032 checks / zero errors. Reading those scalars does not rerun that audit.

Only 21 tensor archives: three oracles, six common-action and 12 observer files,
189,384,513 bytes in existing receipts. Bind every path to regular-file identity,
recorded size and SHA256. Read one archive at a time with CPU-only restricted
loading; extract only probe/input/O151 action vectors. Incidental members do
not authorize model, image, logit, optimizer or observer-state analysis.
No model construction, inference, gradient evaluation, SVD, stream replay,
observer update, Adam step or training is allowed.

Record exact source/test/protocol pins and selected input receipts. Recheck
pins before success. Expected output is small JSON tables/summaries plus
source/input/resource/terminal receipts; do not copy tensor archives or create
new large arrays on disk. Default import/help and fixture runs are inert.

## One-shot execution envelope

After main source review and a clean frozen commit, main owns one execution:
unit `spectral-observer-signal-001.service`, new exclusive parent under the
verified `/tmp/spectral-experiment-artifacts` mount, new child `analysis-001`. Type=exec,
Restart=no, KillMode=control-group, CPUQuota=100%, MemoryMax=4 GiB,
MemorySwapMax=0, RuntimeMax=180s. Cooperative deadline 150s starts at entry;
CUDA hidden, all math-library threads 1, output at most 100 MiB plus 1 GiB free
reserve. Source must enforce the cooperative/output limits; the unit enforces
hard resource limits. No automatic retry, extension or paid compute.

No output parent or service is allocated during preparation. Record actual
command, source freeze, unit/PID/invocation and completion/failure before
interpreting results. Existing acquisitions/audits/emails remain consumed.

## Interpretation boundary

Positive qᵀh concerns a hypothetical sufficiently small negative-gradient
step, not actual Adam. Positive D_filter with nonpositive D_Adam locates an
ordering mismatch, not a specific momentum or second-moment mechanism.
Probe J and held-out U concern different populations, so disagreement cannot
separate probe mismatch from finite-step nonlinearity. Accessibility is not
retained memory across training. None of these scalars alone explains the
step-2000 endpoint, semantic generalisation or safety selectivity.
