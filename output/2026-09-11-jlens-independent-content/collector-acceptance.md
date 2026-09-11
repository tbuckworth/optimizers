# Independent text collector — implementation acceptance

Codex — Spectral Optimizer Investigation · 11 September 2026 BST

Main read the complete collector, fabricated tests and design review. Nine
fabricated tests passed independently before any real collection. The source
matches the frozen four-root ordering, complete-catalogue requirement,
hash ranking, technical-only eligibility, exact 16-word prefixes and fixed
24-row/12-pair roster. The 1200-character setting is a request, not an exclusion.

Accepted source commit: `54357e43d002391906240d656e56a79c8bbbe373`.

- Collector SHA256: `337e9a208f491b22ec0deb3a591b1216669eaf42db4712e43f8e1bc97e98d347`.
- Fabricated tests SHA256: `24345cfa835350c49afb7574a1741cb185c2c18b5e8cc7a24823cf9280d6cb76`.
- Protocol SHA256: `78d6171dfcbd2e24ff39c4acf88138de67138003d970c43a3ef3580030c93051`.

Admission is one exclusive `catalogue` stage, followed by one exclusive
`excerpts` stage if the catalogue completes. Serial unauthenticated GETs,
20-second request timeout, 2 MiB response cap, 16 MiB combined output budget,
maxlag=5, descriptive User-Agent, no automatic retry. Failed attempts and
received bytes remain; do not delete a stage or resample after failure.
An oversized response is retained only to its explicit cap and marked
incomplete; normal responses are byte-exact. No public redistribution is
admitted before the fixed-roster attribution check.

This acceptance does not admit tokenizer execution, model inference,
judging, grading, cloud spending, or any change to the original directions
or references. Those boundaries remain separate from collection.
