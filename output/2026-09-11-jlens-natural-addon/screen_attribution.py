"""One bounded notice screen of the already-fixed 32-article roster; no sampling."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parent
ATTRIBUTION_SHA = 'f48ef56bb706a4191351398900e0c4a8e8413f8e50c84db55c650590b04cae4a'
PATTERN = re.compile(r'\b(?:translat\w*|attribut\w*|copyright\w*|public.domain|copied|copying|copyvio\w*|merged|merging|imported|permission\w*|licen[sc]\w*)\b', re.I)


class Visible(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []
        self.links = []
        self.anchor = None

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if self.hidden:
            return
        if tag in ('div', 'p', 'li', 'tr', 'br', 'h1', 'h2', 'h3'):
            self.parts.append('\n')
        if tag == 'a':
            self.anchor = [dict(attrs).get('href', ''), []]

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        if self.hidden:
            return
        if tag == 'a' and self.anchor is not None:
            self.links.append({'href': self.anchor[0], 'text': ''.join(self.anchor[1])})
            self.anchor = None
        if tag in ('div', 'p', 'li', 'tr', 'h1', 'h2', 'h3'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)
            if self.anchor is not None:
                self.anchor[1].append(data)


def screen(raw, url):
    parser = Visible()
    parser.feed(raw.decode('utf-8', errors='strict'))
    visible = ' '.join(''.join(parser.parts).split())
    spans = []
    for match in PATTERN.finditer(visible):
        start, end = max(0, match.start()-220), min(len(visible), match.end()+320)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    contexts = [visible[a:b] for a, b in spans]
    links = [{**a, 'href': urljoin(url, a['href'])} for a in parser.links
             if PATTERN.search(a['text']+' '+a['href'])
             or (len(a['text'].strip()) >= 4
                 and any(a['text'].strip() in context for context in contexts))]
    return {'visible_characters': len(visible),
            'notice_contexts': contexts,
            'notice_links': links,
            'cc_by_sa_4_link_present': any('creativecommons.org/licenses/by-sa/4.0' in a['href'] for a in parser.links)}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def write(path, obj):
    data = (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    if len(data) > 2*1024*1024:
        raise ValueError('compact record cap')
    with path.open('xb') as handle:
        handle.write(data)


def run():
    attribution = (ROOT/'excerpts/attribution.json').read_bytes()
    assert hashlib.sha256(attribution).hexdigest() == ATTRIBUTION_SHA
    articles = json.loads(attribution)['articles']
    assert len(articles) == 32 and len({a['id'] for a in articles}) == 32
    roster = []
    for a in articles:
        for kind, url in [('article', a['url']), ('history', a['history_url']),
                          ('talk', 'https://en.wikipedia.org/wiki/Talk:'+quote(a['title'].replace(' ', '_'), safe=''))]:
            assert urlsplit(url).scheme == 'https' and urlsplit(url).netloc == 'en.wikipedia.org'
            roster.append({'id': a['id'], 'title': a['title'], 'kind': kind, 'url': url})
    assert len({r['url'] for r in roster}) == 96
    out = ROOT/'attribution-screen'
    out.mkdir()  # Existing attempts are never resumed/retried automatically.
    started = time.monotonic()
    write(out/'attempt.json', {'attribution_sha256': ATTRIBUTION_SHA,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'started_utc': datetime.now(timezone.utc).isoformat(), 'roster': roster,
          'max_requests': 96, 'timeout_seconds': 20, 'response_bytes': 2097152,
          'automatic_retry': False, 'automatic_redirects': False, 'html_saved': False})
    opener = build_opener(NoRedirect())
    for i, row in enumerate(roster, 1):
        if time.monotonic()-started > 2400:
            raise TimeoutError('stage wall cap')
        record = {**row, 'request_number': i, 'started_utc': datetime.now(timezone.utc).isoformat()}
        try:
            req = Request(row['url'], headers={'User-Agent': 'CodexSpectralInvestigation/1.0 (bounded attribution screen; contact: redacted@example.invalid)', 'Accept-Encoding': 'identity'})
            try:
                response = opener.open(req, timeout=20)
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                response = exc
            with response:
                raw = response.read(2097153)
                record.update(status=response.status, bytes_read=len(raw),
                              response_sha256=hashlib.sha256(raw).hexdigest())
                assert len(raw) <= 2097152, 'response cap'
                assert response.headers.get('Content-Encoding', 'identity') == 'identity', 'unexpected compression'
                record.update(screen(raw, row['url']))
            write(out/f'{i:03d}.json', record)
            print(f'{i}/96 {row["id"]} {row["kind"]} HTTP {record["status"]}', flush=True)
        except Exception as error:
            write(out/f'{i:03d}-failure.json', {**record, 'error_type': type(error).__name__, 'automatic_retry': False})
            raise
    write(out/'receipt.json', {'status': 'complete', 'requests': len(roster),
          'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
          'outputs': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(out.glob('[0-9][0-9][0-9].json'))]})


if __name__ == '__main__':
    run()
