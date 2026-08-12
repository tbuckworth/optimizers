# Knowledge-base schema and workflows

This project instantiates the persistent-wiki pattern described in Karpathy's
[LLM Wiki idea](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
The layout is adapted to an experiment repository: evidence already lives next
to the code that produced it, so it is referenced in place rather than copied.

## Three layers

1. **Evidence:** root implementations, `experiments/`, immutable committed JSON
   under `results/`, external papers, and experiment-specific reports under
   `research/`. These are the sources of record.
2. **Knowledge:** maintained Markdown under `knowledge/`. These pages combine
   evidence across experiments, expose contradictions, and record the current
   synthesis. Agents own this layer; humans primarily read and direct it.
3. **Schema:** this file plus the short entry points in `AGENTS.md` and
   `CLAUDE.md`. These define how future agents maintain the wiki.

The knowledge layer never silently overrides evidence. If a source is wrong,
correct it explicitly and preserve provenance in git history and the log.

## Page schema

Every content page below `knowledge/` except `index.md`, `schema.md`, and
`log.md` begins with YAML frontmatter:

```yaml
---
title: Human-readable title
type: overview | concept | finding | decision | question | source-map
status: current | provisional | historical
updated: YYYY-MM-DD
sources:
  - path/to/direct/evidence
---
```

Use ordinary relative Markdown links so pages work in GitHub, Obsidian, and
local viewers. A content page should link to related wiki pages and to its most
direct evidence. Numerical claims must be traceable to raw JSON or a report
that identifies the raw run. Explicitly distinguish:

- single-seed from multi-seed evidence;
- test-set selection from unbiased confirmation;
- measured findings from mechanisms inferred from those findings;
- current defaults from historical reproduction settings.

Do not erase a negative or contradictory result merely because a newer result
is positive. Explain the boundary between them.

## Source priority

When sources disagree, inspect them rather than resolving the conflict by
wording alone. Use this default priority:

1. raw committed metrics plus the code/configuration that produced them;
2. focused evaluation reports tied to those metrics;
3. current synthesis documents;
4. early findings and planning notes;
5. recollection in chat or commit messages.

Git history is part of the provenance trail. External claims require a direct
link and should be labeled as external rather than repository evidence.

## Operations

### Ingest

1. Identify the direct evidence and whether it supersedes earlier evidence.
2. Read `index.md`, then every page named by the relevant index section.
3. Update all affected concept, finding, decision, and question pages.
4. Add a new page only when the concept will be reused; avoid one page per run.
5. Update `index.md` if pages were added or their one-line descriptions changed.
6. Append a parseable entry to `log.md`.
7. Run `python3 scripts/lint_knowledge.py` and the relevant project tests.

### Query

Read `index.md` first, then drill into the relevant pages and their direct
sources. Cite evidence in the answer. File a query-derived synthesis back into
the wiki only when it is durable and adds a reusable comparison, decision, or
connection; do not archive routine answers.

### Lint

Run `python3 scripts/lint_knowledge.py`. It checks frontmatter, local links,
index coverage, and log heading format. A conceptual lint should additionally
look for stale claims, conflicting numbers, orphan concepts, unlinked raw
results, and open questions that existing evidence already resolves.

## Log format

`log.md` is append-only. Newest entries go at the top after the introduction:

```markdown
## [YYYY-MM-DD] ingest | Short source or result name
```

Allowed operation labels are `ingest`, `query`, `lint`, and `decision`. Each
entry names the evidence inspected and the pages changed.

