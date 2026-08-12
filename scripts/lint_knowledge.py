#!/usr/bin/env python3
"""Validate the local Markdown knowledge base without third-party packages."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
SPECIAL = {"index.md", "schema.md", "log.md"}
REQUIRED = {"title", "type", "status", "updated", "sources"}
ALLOWED_TYPES = {"overview", "concept", "finding", "decision", "question", "source-map"}
ALLOWED_STATUS = {"current", "provisional", "historical"}
LINK_RE = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}$")
LOG_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}\] (ingest|query|lint|decision) \| .+")


def frontmatter(path: Path, text: str, errors: list[str]) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        errors.append(f"{path.relative_to(ROOT)}: missing YAML frontmatter")
        return {}
    try:
        end = lines.index("---", 1)
    except ValueError:
        errors.append(f"{path.relative_to(ROOT)}: unterminated YAML frontmatter")
        return {}
    values: dict[str, str] = {}
    for line in lines[1:end]:
        if line and not line.startswith((" ", "-")) and ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    missing = REQUIRED - values.keys()
    if missing:
        errors.append(
            f"{path.relative_to(ROOT)}: missing frontmatter fields {sorted(missing)}"
        )
    if values.get("type") not in ALLOWED_TYPES:
        errors.append(f"{path.relative_to(ROOT)}: invalid type {values.get('type')!r}")
    if values.get("status") not in ALLOWED_STATUS:
        errors.append(f"{path.relative_to(ROOT)}: invalid status {values.get('status')!r}")
    if not DATE_RE.fullmatch(values.get("updated", "")):
        errors.append(f"{path.relative_to(ROOT)}: invalid updated date")
    source_items: list[str] = []
    in_sources = False
    for line in lines[1:end]:
        if line == "sources:":
            in_sources = True
            continue
        if in_sources and line.startswith("  - "):
            source_items.append(line[4:].strip())
            continue
        if in_sources and line and not line.startswith(" "):
            in_sources = False
    if not source_items:
        errors.append(f"{path.relative_to(ROOT)}: sources must contain at least one item")
    for source in source_items:
        if "://" not in source and not (ROOT / source).exists():
            errors.append(
                f"{path.relative_to(ROOT)}: missing local source {source!r}"
            )
    return values


def validate_links(path: Path, text: str, errors: list[str]) -> set[Path]:
    targets: set[Path] = set()
    for raw in LINK_RE.findall(text):
        target = raw.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (path.parent / target).resolve()
        targets.add(resolved)
        if not resolved.exists():
            errors.append(
                f"{path.relative_to(ROOT)}: broken link {raw!r}"
            )
    return targets


def main() -> int:
    errors: list[str] = []
    pages = sorted(KNOWLEDGE.rglob("*.md"))
    if not pages:
        print("knowledge base is empty", file=sys.stderr)
        return 1

    index_targets: set[Path] = set()
    for path in pages:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(KNOWLEDGE)
        if relative.as_posix() not in SPECIAL:
            frontmatter(path, text, errors)
        targets = validate_links(path, text, errors)
        if relative.as_posix() == "index.md":
            index_targets = targets

    content_pages = {
        path.resolve()
        for path in pages
        if path.relative_to(KNOWLEDGE).as_posix() not in SPECIAL
    }
    for orphan in sorted(content_pages - index_targets):
        errors.append(
            f"{orphan.relative_to(ROOT)}: content page is not linked from knowledge/index.md"
        )

    log = (KNOWLEDGE / "log.md").read_text(encoding="utf-8").splitlines()
    for line in log:
        if line.startswith("## ") and not LOG_RE.fullmatch(line):
            errors.append(f"knowledge/log.md: malformed entry heading {line!r}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"Knowledge lint failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print(f"Knowledge lint passed: {len(pages)} pages, {len(content_pages)} indexed content pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
