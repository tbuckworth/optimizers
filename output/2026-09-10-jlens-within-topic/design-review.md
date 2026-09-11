# Within-topic J-Lens comparison — design review

Codex — Spectral Optimizer Investigation · 10 September 2026

**PASS: no blocking design flaw for the stated exploratory comparison.**
This is a useful, proportionate follow-up, not a prerequisite for acknowledging
the earlier positive result. No additional arm, acquisition or experiment is
required by this review. Implementation execution and result verification are
outside this design verdict.

## What the comparison can establish

For a fixed direction, write a prefix score as its topic mean plus a residual:
`z(x) = mean(topic(x)) + residual(x)`. Within-topic subtraction cancels that
mean exactly. A reader who successfully orders these pairs therefore supplies
information beyond the four coarse labels alone. This does not distinguish
semantic content from subtopic, lexical or syntactic cues. The earlier FIT
variance decomposition motivates the question but does not predict accuracy
on these newly formed pairs.

## Roster and safeguards

The frozen pairs use all 24 original prefixes exactly once: adjacent indices
0–1, 2–3 and 4–5 in each of four topics. There are 12 disjoint pairs, three per
topic, and 48 axis/pair targets across all four axes. Each arm receives all 48
targets: 144 choices altogether. The allocation assigns each axis/arm to one
reader and gives each reader 48 choices. Reader identity remains partly
confounded with arm within an axis; the protocol correctly treats these as
single judgments, not replicated measurements of reader performance.

New isolated readers, outcome-free packet construction, common swaps, no
rerolling, and a committed complete response lock before key access are
proportionate safeguards. Keeping exact ties at half credit and all other
gaps, both constant-position baselines, all axes and all paired differences
avoids an outcome-selected subset. The key and texts already exist and the
researchers have seen earlier item outcomes: this is a new reader task on
reused measurements, not a globally blind or fresh-data confirmation.

## Interpretation cautions and small wording correction

- The references remain **global FIT extrema**, not examples selected to
  explain variation inside each topic. Success is useful transfer of those
  existing descriptions. Failure does not rule out readable within-topic
  structure, a better description, or J-Lens in general. No per-topic
  replacement arm is requested here.
- Comparing the new totals with the old cross-topic totals also changes the
  particular pairs, readers, allocation and presentation. It is not an
  isolated causal effect of removing topic-label differences or a
  matched-difficulty comparison.
- For reporting, replace “12 reused content pairs” with **“12 new pairings of
  24 reused texts.”** The reuse applies to the texts and scores, not the old
  pair assignments. This is a nonblocking precision correction.

The strongest positive case is that unchanged direct-direction descriptions
help order content inside the supplied topics, with the full per-axis record
retained, whether or not they outperform example descriptions. The adverse
case is that plausible direct token themes provide little usable ordering
information, especially if C succeeds on the same targets. If all arms are
weak, the design cannot separate description transfer, reader limitations,
small score gaps and poorly readable axes. None of these outcomes alone
establishes a semantic mechanism, four clusters, or an optimizer/safety benefit.

## Review scope and pins

Read the complete protocol and pairs, the fixed dataset, and the prior
fresh-content and fit-geometry narrative results. Used pre-mortem guidance to
identify concrete failure modes without adding workflow gates. No actual
scores, scientific arrays, model, new reader response, or new analysis was
opened or executed for this review. This is a separate design assessment by
an agent that previously authored the earlier grading implementation, not an
independent implementation-author audit or a blind result assessment.

Reviewed freeze: `f2c9aa34dc7fd6df068b42fbf6111482d619b824`.

- Protocol SHA256: `5de33f7d781aee09678efda50b7d8793d66245f0c1af8ca4cd72acef6a2259eb`.
- Pairs SHA256: `552b4090c2e2382a352a14d2b6be3bd28cad98a33f8f0c36dd49dde0a89fad7a`.
- Dataset SHA256: `c655a5531af3215a29e332ddc6b23ac4854b0bffdf91b8ad5e8a045748ff7cd9`.
