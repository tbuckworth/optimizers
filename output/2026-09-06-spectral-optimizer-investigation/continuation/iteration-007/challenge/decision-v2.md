# Replacement design review — corrections accepted, detailed freeze next

Codex parent, Spectral Optimizer Investigation, 6 September 2026.
Status: second design challenge complete; **no implementation or execution GO**.

The three independent second-round reviews are
[assumptions](assumption-analysis-v2.md), [mentor](mentor-review-v2.md), and
pre-mortem (artifact not distributed in this public snapshot). Each read the new base design without peer reviews.
The mentor now judges the design construct-valid with MINOR_REVISIONS. The
assumptions and pre-mortem reviews require explicit analysis and capture rules.
They do not certify code, data, numerical results or runtime feasibility.

The reviewed candidate is preserved in commit `2f83d71`; the
[current design](../common-state-design.md) includes the parent corrections below.
An automatic artifact commit `3297191` archived the second assumption review;
that commit itself is not approval. Earlier studies remain unchanged.

## Accepted corrections

1. Name the estimand **current-policy-state conditional 2x2 delivered-gradient
   response**. Two sampled directions/norms are not general mechanisms, and
   input-norm matching is not actual-step matching.
2. Use all 5,000 reserved auxiliary examples for primary clean loss, with all
   ten fixed disjoint chunk results retained. This removes the 256-example
   subsampling limitation, not finite-pool or cross-bundle state/pool dependence.
3. Add a zero-input next-step reference to show existing-moment motion plus
   decay. It adds no source trajectory; every parameter receives a zero tensor
   and AdamW still steps. It is not an additive decomposition of history.
4. Specify exact-zero cases, undefined cells and masks; tiny positives remain
   positive. Numerical underflow/nonfinite/gate failures cannot be converted
   to expected domain nulls. No available-case aggregate substitutes missing cells.
5. Report factor leverage with every anchor. Contrasts below the delivery-gate
   resolution cannot support factor claims. All anchors and actual numeric
   differences remain visible; treatment doses differ between anchors.
6. Define float64 evaluation on promoted normalized float32 inputs/parameters,
   ordered unsmoothed mean CE and fixed reduction chunks. Preserve native-
   float32 concordance separately. Use promoted endpoint differences, not a
   rounded float32 delta. Save actual native before/after parameter tensors.
7. Require the replay witness from the real next source step, not replay itself.
   Capture is pre-forward after zero_grad; all counters are t-1. Restore RNG
   after construction and test RNG continuation in addition to hash equality.
8. Give interpretation a precedence: validity, defined cells, leverage, complete
   contrast table. Remove “little separation means no large effect.” The exact
   12-anchor concordance condition only prioritizes a possible new source-state
   question; it is explicitly not a stability or population-sign criterion.
9. Count zero-reference endpoints and float64 deltas in the provisional byte
   ledger. Include full-pool losses/gradients and native concordance in any
   later timing-only pilot, and separately budget independent CPU audit.

On numerical versus sampling uncertainty, the parent adopts the assumptions
review's distinction: audit discrepancy is numerical; variation among auxiliary
chunks is finite-pool sensitivity. Do not merge them into an unexplained scalar
“resolution” or mistake either for a powered equivalence margin.

## Concrete work remaining before implementation

- Freeze the strict anchor/result schema, including ordered optimizer mapping,
  buffer/mode/null-gradient metadata, mixed device/dtype restoration, primitive
  NumPy RNG encoding, source witness and six-branch endpoint membership.
- Freeze separate independent-audit tolerances for native Adam arithmetic,
  float64 before/after loss, delta loss, gradient dots and factorial contrasts.
  Derive contrast bounds from their component errors; do not reuse the older
  checkpoint CE tolerance for much smaller one-step effects. Exact same-platform
  source replay and serialized tensor restoration have no tolerance escape.
- Specify the aggregate byte/time guards and the dataset-free correctness
  fixtures. The 560-MiB ledger remains an estimate, not measured serialization.
- Review the resulting analysis/schema contract before implementation; then
  implement dataset-inert modules and tests, review code, and record any pilot
  and primary/sensitivity/audit GO separately. No repeated whole-study restart.

Validation at handoff: six synthetic reminder-state/deduplication tests pass;
the actual dry-run sees the active unfinished goal without queuing a message;
the timer remains enabled/active with the original 19:18:19 BST next trigger;
knowledge lint passes all 14 pages. Research changes in this checkpoint are
design/review documents only, not verified implementation or empirical results.
