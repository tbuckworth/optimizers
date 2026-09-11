# Review: available direction versus delivered signal

Codex — Spectral Optimizer Investigation, 10 September 2026.

**Verdict: worthwhile bounded post-hoc analysis; no conceptual blocker.**
Reviewed next-discriminator.md (artifact not distributed in this public snapshot), SHA256
`a3c6222868de4e9833fce9e48878db08bb1b02128c858fbafd5f5ca747e2dc72`.
This review reads design text and accepted JSON receipt/schema metadata only.
No tensor archives, scientific vector arithmetic, model work or jobs occurred.

## What it adds

The proposed cross terms are genuinely missing, not a relabeling of current
results. Probe retention says how an operator acts on the probe **itself**;
it does not determine the interaction between that probe and the actual
incoming gradient. Norms, retention and the already measured Adam utility
cannot recover `qᵀg` or `qᵀh`. Conversely, computing those products cannot
replace the accepted finite loss readout. Joining them supplies a useful
local distinction between signal availability, native delivery, and the
inherited Adam response.

The Clean reference and both true-label probes prevent this becoming
adverse-only explanation hunting. A favorable result would connect an
observed useful effect across additional local stages. An unfavorable result
would locate a mismatch in that chain, without disproving useful learning
elsewhere. Neither result identifies which mechanism caused the long-run
endpoint ranking.

## Roster and estimands

The fixed roster is coherent: three reused parents × two label cells, two
O151 native histories, raw and shared zero. Retain two probes per case and
keep the 24 logical versus 21 physical readouts distinct. There are:

- 12 input/probe products `B`;
- 24 native delivered/probe products `F` and 24 native changes `K`;
- 12 paired grouping contrasts `D_filter`;
- six paired label-intervention products;
- 48 logical action/probe entries when joining each of `J` and group-specific
  held-out `U`, with shared physical references explicitly identified.

These counts refer to scalar entries, not independent samples. Report three
seed values before summary SD/SE/sign counts; do not pool cells or probes.
The proposed O151-only scope is sufficient: O150 is already characterized and
would add another descriptive branch without answering the new question.

Before implementation, explicitly encode the control identities
`F_raw=B`, `F_zero=0`, `K_raw=0`, `K_zero=−B`. This requires no new readout
or inferred model action. Apply the same absolute/paired reporting to those
controls rather than displaying only whichever native alignment looks best.

## Signs, units and inference

The signs are correct: `qᵀh>0` predicts a helpful first-order change under a
hypothetical sufficiently small step `−eta*h`; `J=−qᵀDelta` is the signed
linear utility of the **actual** saved Adam displacement. Their dimensions
and step mappings differ, so the design correctly rejects a conversion ratio.
Positive `D_filter` with adverse `D_Adam` identifies a local ordering mismatch,
not a momentum/second-moment attribution. Positive `B` and nonpositive
`D_filter` means grouping did not improve that delivery projection; it does
not imply that either delivered vector has negative absolute utility.

“Weaker Diffuse input” should be reported separately as a smaller signed dot
product and as any change in cosine: reduced amplitude and poorer orientation
are not interchangeable. Preserve the vector norms. The paired label contrast
is well defined under the matched model/occurrences/probe premise, but is not
an exact contribution of rare examples. The full held-out `U` and training-probe
`J` refer to different populations; their disagreement still cannot separate
finite-step nonlinearity from probe mismatch.

## Availability and numerical implementation cautions

The accepted audit's receipt roster and acquisition completion metadata list
all proposed archives: three `oracles`, six `common-action`, and 12 `observer`
files, totaling **189,384,513 bytes**, below 200 MiB. The largest is 15,101,337
bytes. Paths, sizes and hashes are present, and the audit binds the exact
completion named in the proposal. This is metadata admission, not a new
verification of tensor contents or member keys. Future code must validate the
accepted schema and extract only the required gradients/actions; deserializing
an archive does not authorize using its incidental images or model states.

Two numerical details should be fixed prospectively in source/tests:

1. Convert saved vectors to FP64 **before** subtracting for `h−g`, `h_G−h_I`
   and the label contrast. Do not introduce an additional FP32 residual cast.
   Cross-check direct difference-vector products against paired product
   differences, accounting for cancellation and the parent products' errors.
2. `128*eps64*sum(abs(products))+1e−12` is a predeclared numerical check, not
   a general worst-case theorem for arbitrary summation of 50,890 terms.
   An independent compensated reference, such as `math.fsum` over FP64
   products, would make a discrepancy interpretable. Preserve numerical
   failures and unresolved near-zero signs; do not relax bounds after outcomes.

The one-archive-at-a-time, CPU-only resource proposal is proportionate to these
recorded sizes. Keep the no-retry and source/input receipt requirements. This
review supports preparing the bounded accounting; it does not authorize its
execution or another neural experiment. Research-workflow evidence guidance
informed the claim boundaries; the user's fixed-scope/no-rerun instructions
remain controlling.
