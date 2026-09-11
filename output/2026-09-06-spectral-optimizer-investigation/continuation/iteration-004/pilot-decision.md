# Parent development-pilot decision

2026-09-06 11:47 UTC, before any iteration004 MNIST execution.

Approve only the protocol's runtime/invariant pilot: development seed 9877,
four fixed arms, 220 steps each with optional instrumentation off/on. The
parent has read the full design, harness and inherited helpers, implemented
the independent analysis and seven synthetic tests, and rerun all eleven
harness tests. The independent reviewer reports no unresolved material
blocker after independently passing all eighteen tests and an additional
synthetic checkpoint-selection integration fixture. The design auditor also
passed the final protocol/analysis alignment.

Use the local RTX3090 and cached training data; recent occupancy was 648 MiB
of 24,576 MiB with no identified research compute competing for the GPU.
Do not evaluate development accuracy, validation or test outcomes. Preserve
any failed attempt and stop for review; do not overwrite or tune the recipe.
The full twelve-run confirmation is **not yet approved**. It additionally
requires a passing pilot, unchanged source/data hashes, committed reviewed
source and a separate parent launch decision.
