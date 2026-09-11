#!/usr/bin/env python3
"""Check publication integrity without running models or accessing private data."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(rb'/(?:home|Users|media)/titus/|<{7}|>{7}')


def main():
    manifest = json.loads((ROOT / 'PUBLICATION-MANIFEST.json').read_text())
    rows = manifest['exported'] + manifest.get('publication_added_or_updated', [])
    names = set()
    for row in rows:
        name = row['path']
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or name in names:
            raise ValueError('Invalid or duplicate manifest path: ' + name)
        names.add(name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['public_sha256']:
            raise ValueError('Publication hash mismatch: ' + name)
        if len(data) != row['bytes']:
            raise ValueError('Publication size mismatch: ' + name)
        if path.suffix not in ('.png', '.pdf') and PRIVATE.search(data):
            raise ValueError('Private path or merge-conflict marker: ' + name)
    print(f'Publication integrity passed: {len(rows)} files.')


if __name__ == '__main__':
    main()
