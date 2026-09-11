#!/usr/bin/env python3
"""Package the mechanism report and its plots into one offline HTML document.

Reuses the existing paper-plan reader's Markdown/BeautifulSoup approach; it
performs no inference, numerical analysis, external upload, or email delivery.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import markdown
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", default="9 September 2026", help="Displayed report date; does not alter the source.")
    parser.add_argument("--title", default="Grokking mechanism measurements", help="Displayed browser title.")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing reader.")
    soup = BeautifulSoup(markdown.markdown(source.read_text(), extensions=["tables", "toc", "fenced_code", "sane_lists"]), "html.parser")
    references, images = {}, {}
    for img in soup.find_all("img", src=True):
        path = (source.parent / img["src"]).resolve()
        if not path.is_relative_to(REPO) or path.suffix != ".png" or not img.get("alt"):
            raise ValueError("Embed only local PNGs with an accessible description.")
        payload = path.read_bytes()
        images[str(path.relative_to(REPO))] = hashlib.sha256(payload).hexdigest()
        img["src"] = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    for anchor in soup.find_all("a", href=True):
        target = anchor["href"]
        if target.startswith(("https://", "http://", "#")):
            continue
        path = (source.parent / target).resolve()
        if not path.exists():
            raise ValueError("Unresolved evidence link: " + target)
        label = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        references.setdefault(label, f"evidence-{len(references)+1}")
        anchor["href"] = "#" + references[label]
    inventory = "".join(f'<li id="{key}"><code>{html.escape(path)}</code></li>' for path,key in references.items())
    navigation = "".join(f'<a href="#{heading["id"]}">{html.escape(heading.get_text())}</a>' for heading in soup.find_all("h2", id=True))
    css = """body{margin:0;background:#edf2f4;color:#263747;font:17px/1.65 Georgia,serif}
main{max-width:1050px;margin:30px auto;padding:40px 50px;background:white}
h1,h2,h3,nav,.kicker{font-family:Arial,sans-serif}h1{font-size:34px;line-height:1.2}
h2{margin-top:35px;padding-top:18px;border-top:1px solid #dae4e6;font-size:25px}
h3{font-size:20px}.kicker{color:#087d78;font-size:13px;font-weight:700}
nav{background:#f0f6f5;padding:15px;font-size:14px}nav a{display:block}
a{color:#087d78}img{display:block;width:100%;height:auto;margin:25px 0}
code,pre{font:13px/1.5 monospace;background:#f3f6f7;overflow-wrap:anywhere}
pre{white-space:pre-wrap;padding:15px}table{border-collapse:collapse;width:100%;font:14px/1.5 Arial,sans-serif}
td,th{padding:9px;border:1px solid #d9e2e5;text-align:left}th{background:#f0f5f5}
li{margin:7px 0}.small{font:13px/1.5 Arial,sans-serif;color:#536272}
@media(max-width:650px){main{margin:0;padding:24px 18px}body{font-size:16px}h1{font-size:28px}td,th{padding:5px;font-size:11px}}
@media print{main{margin:0;padding:0}body{background:white;font-size:11pt}nav{display:none}img,table{break-inside:avoid}}
"""
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codex · Spectral Optimizer Investigation · {html.escape(args.title)}</title>
<style>{css}</style></head><body><main>
<div class="kicker">Codex · Spectral Optimizer Investigation · {html.escape(args.date)}</div>
<p class="small">Offline report with embedded plots. No scripts, remote fonts, or internet connection are required.
Local evidence links lead to the inventory below; they are not public downloads.</p>
<nav aria-label="Contents">{navigation}</nav>{soup}
<h2 id="local-evidence">Local evidence inventory</h2><ol>{inventory}</ol>
<p class="small">Source: {html.escape(str(source.relative_to(REPO)))}<br>SHA-256: {source_hash}</p>
</main></body></html>'''
    check = BeautifulSoup(document,"html.parser")
    ids = {node["id"] for node in check.find_all(id=True)}
    if check.find("script") or any(a["href"][1:] not in ids for a in check.find_all("a",href=True) if a["href"].startswith("#")):
        raise ValueError("Broken navigation or an unexpected script dependency.")
    if not images:
        raise ValueError("Expected embedded plots.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(document)
    receipt = {"source_sha256": source_hash,"builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "display_date":args.date,"display_title":args.title,
               "html_sha256":hashlib.sha256(document.encode()).hexdigest(),"images":images,"local_references":references}
    with args.output.with_suffix(".receipt.json").open("x") as handle:
        json.dump(receipt,handle,indent=2)
        handle.write("\n")
    print(json.dumps({"html":str(args.output),"images":len(images),"bytes":len(document.encode())}))


if __name__ == "__main__":
    main()
