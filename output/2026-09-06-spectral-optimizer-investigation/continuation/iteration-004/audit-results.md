# Independent iteration-004 completed-result audit

Verdict: **PASS**.

The auditor imports neither the iteration-004 harness nor its summarizer. No training or GPU work occurred.

Checkpoint replays: 96. Prespecified tolerance: CE absolute error at most 5e-5; accuracy at most one example per evaluated set.
Observed maximum CE error: 1.3427734391058266e-07; accuracy discrepancy: 0 examples.

Audit scope includes source/data/plan/checkpoint provenance, schedules, both strict selectors, common warmup evidence, all seed metrics and paired contrasts; completed checks and any failure are recorded in the JSON.
Stored scalar arithmetic is not a replay of unsaved gradients or optimizer moment trajectories. Saved checkpoint parameters and their evaluation metrics are independently replayed.

[Full audit record](audit-results.json), SHA256 `edf0f8d501805c598d1ed29078ac87f2db8264d6fd65085b8af7a7af49d93bfc`.
