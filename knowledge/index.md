# Optimizers knowledge base

This is the maintained synthesis layer for the repository. It compiles the
current view from code, raw result JSON, and experiment reports so later work
does not have to reconstruct the project from scratch.

Start with [Project synthesis](overview.md). For maintenance conventions, see
[Schema and workflows](schema.md). For chronology, see the append-only
[Knowledge log](log.md).

## Concepts

- [Temporal gradient-covariance filtering](concepts/temporal-gradient-covariance.md)
  — what the global `p×p` filter estimates, projects, and costs.
- [Stable streaming covariance update](concepts/stable-streaming-update.md)
  — why the stable update is the default and what the legacy escape hatch means.
- [Per-weight-matrix filtering](concepts/per-matrix-filtering.md)
  — scalable blocks, LoRA behavior, bias policy, memory, and limitations.

## Findings

- [Noise memorization](findings/noise-memorization.md) — the strongest result:
  filtering prevents late training-label memorization on MNIST and CIFAR-10.
- [Grokking is task-dependent](findings/grokking.md) — faster modular addition,
  slower sparse parity, and the role of weight decay.
- [Boundary conditions and negative results](findings/boundary-conditions.md) —
  where the mechanism underfits, amplifies unwanted coherent signals, or fails.

## Decisions and next work

- [Current recommendations](decisions/current-recommendations.md) — defaults,
  configuration choices, and when not to use the method.
- [Open questions](questions/open-questions.md) — unresolved empirical and
  scaling questions, ordered by expected information value.
- [Evidence map](sources/evidence-map.md) — canonical reports, raw results,
  implementations, and historical documents.

## Evidence status

- **Established:** replicated across seeds or tasks with direct raw results.
- **Supported:** clear but limited evidence, often one task or one seed.
- **Provisional:** design rationale or an empirical pattern awaiting stronger
  replication.
- **Historical:** superseded work retained for provenance, not current truth.

