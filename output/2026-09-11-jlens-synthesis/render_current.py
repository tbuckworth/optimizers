"""Render the completed J-Lens synthesis without changing historical reports."""
import base64
from datetime import datetime, timezone
import html
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'research/jlens_current_synthesis_2026-09-11.md'
TARGET = HERE / 'current-report'
PLOTS = [
    (HERE.parent / '2026-09-11-jlens-endpoint-location/report-v3/positions.png',
     'Earlier four-content panel: the signed PC4 forecast works at the action and '
     'first sentence end, but not after the shared later ending. These are forecast '
     'signs, not reader accuracies; the eight cases repeat four contents.'),
    (HERE.parent / '2026-09-11-jlens-pattern-calibration/report/comparison.png',
     'Separate natural-text calibration comparison: 62/128 versus 61/128 overall. '
     'PC4 improves in both existing reader groups, but the numerical primary '
     'criterion fails. The disclosed transport deviation also remains. This is '
     'a different panel and measurement from the preceding plot.'),
]


def main():
    spec = importlib.util.spec_from_file_location('existing_synthesis_renderer', HERE / 'render.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    inputs = [SOURCE, HERE / 'render.py', Path(__file__), *[p for p, _ in PLOTS]]
    pins = {str(p): renderer.sha(p) for p in inputs}
    body = '<p class="identity">Codex — Spectral Optimizer Investigation</p>'
    body += ('<p><strong>Current through 11 September 2026, 09:46 UTC.</strong> '
             'Completed evidence, not a new experiment. All equations are plain '
             'Unicode text and remain visible without a math-rendering service. '
             'The two plots below are reused unchanged. Local evidence links '
             'require the repository; the narrative, equations and images work offline.</p>')
    body += renderer.section(SOURCE)
    body += '<hr><h2>Two different tests, not a combined success rate</h2>'
    for path, caption in PLOTS:
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        safe = html.escape(caption, quote=True)
        body += f'<figure><img alt="{safe}" src="data:image/png;base64,{data}"><figcaption>{safe}</figcaption></figure>'
    css = ('body{margin:0;background:#edf2f5;color:#233645;font:17px/1.65 system-ui,Arial,sans-serif}'
           'main{max-width:1080px;margin:24px auto;background:white;padding:36px}'
           'h1{font-size:30px;line-height:1.25}h2{margin-top:2em;font-size:23px}'
           'a{color:#146c94}table{border-collapse:collapse;width:100%;font-size:15px;display:block;overflow-x:auto}'
           'th,td{padding:9px;border:1px solid #d2dde3;text-align:left;vertical-align:top}th{background:#eef4f7}'
           'pre{padding:18px;background:#f2f6f8;overflow:auto;font-size:15px}code{font-family:ui-monospace,monospace}'
           'figure{margin:24px 0}img{max-width:100%;height:auto}figcaption,.identity{font-size:14px;color:#536c7c}'
           'hr{border:0;border-top:1px solid #d4dfe5;margin:32px 0}'
           '@media(max-width:700px){main{padding:18px;margin:0}body{font-size:16px}}')
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Codex — J-Lens completed research synthesis</title>'
            f'<style>{css}</style></head><body><main>{body}</main></body></html>')
    assert '<script' not in page and page.count('src="data:image/png;base64,') == 2
    assert 'argmin_d' in page and '62/128' in page and '16/16' in page
    assert all(renderer.sha(Path(p)) == value for p, value in pins.items())
    TARGET.mkdir(exist_ok=False)
    with (TARGET / 'reader.html').open('x') as handle:
        handle.write(page)
    with (TARGET / 'receipt.json').open('x') as handle:
        json.dump({'status': 'complete', 'utc': datetime.now(timezone.utc).isoformat(),
                   'inputs': pins, 'reader_sha256': renderer.sha(TARGET / 'reader.html'),
                   'plots_reused_unchanged': True, 'model_calls': 0, 'reader_calls': 0,
                   'email_sends': 0, 'historical_report_unchanged': True}, handle, indent=2)
        handle.write('\n')
    print('Current synthesis rendered; two saved plots; historical report unchanged; no model or email.')


if __name__ == '__main__':
    main()
