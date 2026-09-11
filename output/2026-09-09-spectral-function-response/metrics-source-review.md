# Independent finite-response metrics source review

Codex · 9 September 2026.

Reviewed:

- `experiments/grokking_function_response.py`, SHA-256
  `61df74453141a52ae232266ff781f7b1394ea626068f6c70151a6e9921427bdc`;
- `tests/test_grokking_function_response.py`, SHA-256
  `a079325e8d7cfd0b655061f400df04435d938067d7e73d6e1bbc6fdbb9a3ffb4`;
- prospective protocol, SHA-256
  `7c291cf6418ae08508be41a63c190f5a038c5cfe816b1848b799f9866262c21f`;
- launch guard, SHA-256
  `98d887413c8eb64362bc6cbe6b14ff443257e401306c7a7d027bfb3542e44376`;
- guard fixture, SHA-256
  `0edffc93c7d6efcdd5b8c5c5b447907d538d0473d3b69144e1951a97749fa847`.

This review opened no scientific tensors or arrays and ran no inference,
optimizer step, acquisition or saved-result analysis. The seven pure synthetic
metric fixtures and three pure guard fixtures pass.

## Verdict

**Pass for the pure metric layer and prospective protocol.** The formulas,
normalizations, signs and construct checks implement the reviewed design. I
found no blocking scientific defect. Collector provenance, exact seed binding,
action construction and state restoration remain separate integration-review
obligations.

## Metric correctness

- Every response is converted to float64, checked finite and centered over
  output classes per input. The behavior metrics are invariant to this common
  class shift.
- `sum_projection` requires a `p²` by `p` grid and exactly `p` observations in
  every supplied sum class. Its group mean is the uniform-grid orthogonal
  projector. The outer collector must still bind the supplied row roster and
  sum labels to the accepted ordered modular grid.
- Energy is consistently the full-grid mean squared class-vector norm.
  Total, sum-consistent and within-sum energies, their cross inner product,
  Pythagorean residual, idempotence, residual group mean and class mean are all
  returned and now actively enforced.
- The fixed negative CE gradient is `one_hot(y) − softmax(f_before)`, so its
  inner product with a finite logit difference is positive when the baseline
  CE derivative predicts improvement. The same baseline is reused for every
  arm and contrast. This is an output-space directional derivative along a
  finite chord, not a parameter JVP and not an exact finite loss change.
- Actual arm improvements have a uniform favorable sign: before CE minus after
  CE, and after minus before for margin and accuracy. Pairwise finite values are
  explicitly named right minus left; the stored semantics correctly warn that
  negative CE but positive margin/accuracy is favorable.
- The CE convexity remainder has the correct sign:
  linear utility minus actual CE improvement is nonnegative in exact
  arithmetic. It is checked only for before-to-action responses, where the
  common baseline tangent supplies the convexity bound.
- All reviewed primary and zero-reference pairwise logit responses receive the
  same energy and utility decomposition. Raw-to-truncated plus
  truncated-to-projected finite scalar differences telescope to
  raw-to-projected, and fixed-baseline utility telescopes componentwise. Both
  identities are saved and enforced. The implementation correctly does not
  impose a false additive identity on squared pairwise energies.
- Train and held-out indices must each be nonempty, unique and together form an
  exact partition of the full grid. Behavior and utility use their own mean
  normalization.

## Acceptance and interpretation

The scale-aware energy, cross-inner-product, projector, centering and utility
tolerances are suitable for double-precision construct validation. The fixed
`10⁻¹²` scalar telescoping and `−10⁻¹⁰` CE convexity bounds are defensible under
the protocol's bounded logit domain. These are arithmetic acceptance checks,
not empirical CUDA reproducibility bounds or thresholds for declaring a local
effect nonzero. The protocol correctly requires caution for tiny contrasts
between archived and new inference invocations.

The recursive paired summary requires exactly five structurally identical
results and retains every positional value, mean, sample SE and sign count. It
does not itself carry seed identifiers. The outer collector must enforce the
exact seed order `[100, 101, 102, 103, 104]` and retain that order and the seed
labels in the final artifact; the protocol already requires all five with no
selection.

The zero-input response is correctly secondary. It represents one AdamW step
with decayed carried moments, advanced optimizer time and decoupled weight
decay, rather than no movement. Action-minus-zero comparisons are conditional
references and do not isolate an additive momentum contribution.

The protocol keeps the strongest permissible local interpretation: finite
raw-to-truncated task improvement supports helpful local off-span suppression,
while finite truncated-to-projected improvement supports helpful local
retained-amplitude concentration, each conditional on the common inherited
state. Function energy alone is not usefulness, within-sum variation is not
identified memorization, and no one-step result establishes mediation of the
1,000-step endpoint.

## Guard note

The launch guard checks the exclusive large-volume output, interpreter,
single-thread environment, terminal predecessor units, actual cgroup limits,
service policy and committed source pins before execution. Under the collector
contract, `source_pins()` computes current on-disk hashes, so the original
comparison to committed `HEAD` already rejected working-tree drift. The added
explicit current-file hash comparison makes that invariant self-contained and
guards against a future static-map regression; it is defense-in-depth, not a
fix for an active bypass in the reviewed contract.
