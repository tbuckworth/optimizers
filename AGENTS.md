# Repository instructions

## Knowledge base

For research questions, experiment interpretation, or changes to the optimizer,
start with [`knowledge/index.md`](knowledge/index.md) and follow
[`knowledge/schema.md`](knowledge/schema.md).

- Treat committed code, raw result JSON, and experiment-specific research
  reports as evidence. The `knowledge/` directory is a maintained synthesis,
  not a replacement for those sources.
- When new evidence changes a durable conclusion, update the affected knowledge
  pages, `knowledge/index.md`, and the append-only `knowledge/log.md` in the same
  change.
- State whether evidence is single-seed, multi-seed, theoretical, or
  provisional. Preserve negative and contradictory results.
- Link numerical claims to the most direct local evidence available.
- Run `python3 scripts/lint_knowledge.py` after knowledge-base edits.

