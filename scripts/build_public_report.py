#!/usr/bin/env python3
"""Build the offline public reader from the public Markdown and scientific plots.

Requires the Python `markdown` package. Does not launch any experiment, fetch
data, or read private archives. PDF rendering is a separate browser print step.
"""
from __future__ import annotations

import base64
import html
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/spectral_final_core_report_2026-09-11.md'
TARGET = ROOT / 'output/2026-09-11-spectral-final-core/reader.html'
PUBLIC = 'https://github.com/tbuckworth/optimizers/blob/main/'


class PublicLinks(HTMLParser):
    """Embed local images and make source links public, including PDF links."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        output = []
        for key, value in attrs:
            if value and key in ('src', 'href') and not urlsplit(value).scheme and not value.startswith('#'):
                parsed = urlsplit(value)
                path = (SOURCE.parent / parsed.path).resolve()
                relative = path.relative_to(ROOT)
                if key == 'src':
                    if path.suffix != '.png':
                        raise ValueError('Only public scientific PNGs may be embedded')
                    value = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
                else:
                    if not path.exists():
                        raise FileNotFoundError(relative)
                    value = PUBLIC + quote(relative.as_posix())
                    if parsed.fragment:
                        value += '#' + quote(parsed.fragment)
            output.append(key if value is None else f'{key}="{html.escape(value, quote=True)}"')
        self.parts.append('<' + tag + (' ' + ' '.join(output) if output else '') + '>')

    def handle_endtag(self, tag):
        self.parts.append(f'</{tag}>')

    def handle_data(self, data):
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append('&' + name + ';')

    def handle_charref(self, name):
        self.parts.append('&#' + name + ';')


def main():
    rendered = markdown.markdown(SOURCE.read_text(), extensions=['tables', 'fenced_code'])
    links = PublicLinks()
    links.feed(rendered)
    document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spectral Optimizer Investigation — final core report</title>
<style>
:root { color-scheme: light; }
body { margin: 0; color: #172638; background: #eef2f5; font: 17px/1.6 Georgia, serif; }
main { max-width: 960px; margin: 28px auto; padding: 48px; background: white; }
h1,h2,h3 { font-family: Arial,sans-serif; line-height: 1.2; color: #153954; }
h1 { font-size: 34px; } h2 { margin-top: 1.7em; } h3 { margin-top: 1.4em; }
a { color: #155883; } img { display: block; max-width: 100%; height: auto; margin: 22px auto; }
table { border-collapse: collapse; width: 100%; font: 14px/1.4 Arial,sans-serif; }
th,td { padding: 10px; border-bottom: 1px solid #d6dfe6; text-align: left; vertical-align: top; }
th { background: #e8eff5; } pre { white-space: pre-wrap; overflow-wrap: anywhere;
padding: 16px; background: #edf3f8; font: 14px/1.6 monospace; }
@media (max-width: 650px) { main { padding: 22px; margin: 0; } h1 { font-size: 27px; } }
@media print { @page { size: A4; margin: 16mm; } body { background: white; font-size: 10.5pt; }
main { padding: 0; margin: 0; max-width: none; } h1 { font-size: 23pt; }
h2,h3 { break-after: avoid; } img,pre,tr { break-inside: avoid; }
table { font-size: 9pt; } a { text-decoration: none; } }
</style></head><body><main>''' + ''.join(links.parts) + '</main></body></html>\n'
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(document)
    print(TARGET.relative_to(ROOT))


if __name__ == '__main__':
    main()
