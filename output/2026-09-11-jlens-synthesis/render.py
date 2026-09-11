"""Assemble readable synthesis and prospective design without recomputing plots."""
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import unquote
import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYNTHESIS = ROOT/'research/jlens_current_synthesis_2026-09-11.md'
TRANSFER = HERE.parent/'2026-09-11-jlens-new-verb-transfer'
ROSTER = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-new-verb-transfer/proposed-roster.md')
PLOTS = [HERE.parent/'2026-09-11-jlens-endpoint-location/report-v3/positions.png',
         HERE.parent/'2026-09-11-jlens-contrast-geometry/report/magnitude.png']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def section(path):
    rendered = markdown.markdown(path.read_text(), extensions=['tables', 'fenced_code'])
    def absolute_link(match):
        href = html.unescape(match.group(1))
        if href.startswith(('https://', 'http://', '#', 'mailto:')):
            return match.group(0)
        return 'href="'+html.escape(str((path.parent/unquote(href)).resolve()), quote=True)+'"'
    return re.sub(r'href="([^"]+)"', absolute_link, rendered)


def main():
    files = [SYNTHESIS, TRANSFER/'protocol.md', TRANSFER/'dataset.json', TRANSFER/'pairs.json', ROSTER, *PLOTS, Path(__file__)]
    pins = {str(p): sha(p) for p in files}
    target = HERE/'report'; target.mkdir(exist_ok=False)
    (target/'attempt.json').write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(), 'inputs': pins})+'\n')
    body = '<p class="identity">Codex — Spectral Optimizer Investigation · J-Lens</p>'
    body += '<p><strong>Reader guide:</strong> Sections 1–6 synthesize completed evidence and maths. The appendix fixes a future test; its results do not exist yet. Existing plots are reused unchanged.</p>'
    body += section(SYNTHESIS)
    body += '<hr><h2>Two measured boundaries, unchanged</h2>'
    captions = ['The fixed local action contrast succeeds at the verb and first sentence end; the later ending is a retained failure.',
                'The full contrast becomes smaller and differently oriented; neither measurement establishes semantic forgetting.']
    for path, caption in zip(PLOTS, captions, strict=True):
        body += '<figure><img alt="'+html.escape(caption, quote=True)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
    body += '<hr><h2>Prospective appendix — no new model results</h2>'+section(TRANSFER/'protocol.md')
    body += '<hr>'+section(ROSTER)
    body += '<p>Machine-readable frozen inputs: <a href="'+str(TRANSFER/'dataset.json')+'">32 input rows</a>; <a href="'+str(TRANSFER/'pairs.json')+'">16 O−P pairings</a>.</p>'
    css = 'body{margin:0;background:#edf2f5;color:#233645;font:17px/1.65 system-ui,Arial,sans-serif}main{max-width:1080px;margin:24px auto;background:#fff;padding:36px}h1{font-size:30px;line-height:1.25}h2{margin-top:2em;font-size:23px}h3{font-size:19px}a{color:#146c94}table{border-collapse:collapse;width:100%;font-size:15px;display:block;overflow-x:auto}th,td{padding:9px;border:1px solid #d2dde3;text-align:left;vertical-align:top}th{background:#eef4f7}pre{padding:18px;background:#f2f6f8;overflow:auto;font-size:15px}code{font-family:ui-monospace,monospace}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption,.identity{font-size:14px;color:#536c7c}hr{border:0;border-top:1px solid #d4dfe5;margin:32px 0}@media(max-width:700px){main{padding:18px;margin:0}body{font-size:16px}}'
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Codex — Spectral directions through J-Lens</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
    with (target/'reader.html').open('x') as f:
        f.write(page)
    assert all(sha(Path(p)) == digest for p, digest in pins.items())
    with (target/'receipt.json').open('x') as f:
        json.dump({'status': 'complete', 'inputs': pins, 'outputs': {'reader.html': sha(target/'reader.html')},
                   'plots_reused_unchanged': True, 'model_calls': 0, 'email_sends': 0}, f, indent=2)
    print('Standalone synthesis rendered once; two existing plots embedded unchanged; no model, new plot or email')


if __name__ == '__main__':
    main()
