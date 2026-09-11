# Independent saved-checkpoint verification

The parent verifier imports neither the producer nor its summarizer. It
reconstructs all seven RNG-plan streams, validates fixed corruption metadata,
parses raw IDX files, checks tensor structure and state/parameter fingerprints,
re-derives earliest strict selectors, and independently forwards the saved MLP
weights. It computes CE from log probabilities, accumulating in float64, rather
than calling the producer's evaluator. Both validation and test data are checked
for each of four checkpoints; final clean/noisy training scores are also checked.
This means 360 evaluations for the primary study and 60 for the separate rerun.
Tolerances were committed before outcome access in audit-reexecution-decision.md.

The best-practices-validator check supports explicit tensor-only CPU loading,
batch-sum/count aggregation, and prospective numerical tolerances. No missing
library requirement was identified for this small local tensor-only model.
CPU/GPU agreement is measured, not assumed. Official documentation checked:
[torch.load](https://docs.pytorch.org/docs/2.11/generated/torch.load.html),
[cross entropy](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cross_entropy.html),
[reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

This is independent evaluation and artifact verification, not a replay of the
unsaved optimizer/gradient trajectory and not independent fresh training. The
separate bundle60006 execution supplies the latter, within its stated scope.
Every metric discrepancy is retained. Audit errors preserve a diagnostic output
and require diagnosis; they do not authorize changing experimental evidence.
