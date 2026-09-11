"""Import-inert, serial, no-retry Wikipedia collection; catalogue and excerpts are separate stages."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

OUT = Path(__file__).resolve().parent
PROTOCOL_SHA = 'f7314607d2846707a2ee2e8b64c8f2f1fb3098a2366bb1f0d5b0bc0afdcbea3b'
API = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'SpectralJLensPilot/0.1 (research; https://github.com/tbuckworth)'
ROOTS = [('astronomy', 'Category:Astronomy'), ('cooking', 'Category:Cooking'),
         ('football', 'Category:Association football'), ('programming', 'Category:Computer programming')]
MAX_RESPONSE, MAX_OUTPUT, RESERVE = 2*1024**2, 16*1024**2, 16384


def require(ok, message):
    if not ok: raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()


def decode(raw):
    import math
    def pairs(items):
        result = {}
        for key, val in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = val
        return result
    def invalid(value): raise ValueError('nonfinite JSON constant')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(x):
        if type(x) is float: require(math.isfinite(x), 'nonfinite JSON number')
        elif type(x) is dict:
            for v in x.values(): finite(v)
        elif type(x) is list:
            for v in x: finite(v)
    finite(value)
    return value


def bounded_read(path, cap=MAX_OUTPUT):
    require(not path.is_symlink() and path.is_file(), 'input must be regular, not a symlink')
    with path.open('rb') as f: raw = f.read(cap+1)
    require(len(raw) <= cap, 'input byte cap')
    return raw


def fetch(params, cap):
    """One unauthenticated GET. HTTP error bodies are retained; no retry loop."""
    request = Request(API+'?'+urlencode(params), headers={'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'})
    try:
        response = urlopen(request, timeout=20)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read(cap+1), response.headers.get('Content-Encoding', 'identity')


class Stage:
    def __init__(self, root, name, prior_bytes=0):
        self.path = root/name
        self.path.mkdir(exist_ok=False)
        self.used, self.outputs = prior_bytes, []
        self.source_sha = sha(Path(__file__).read_bytes())
        self.write('attempt.json', {'stage': name, 'started_utc': utc(), 'source_sha256': self.source_sha,
                                   'python': sys.version, 'automatic_retry': False})

    def raw(self, name, raw, emergency=False):
        require(re.fullmatch(r'[a-zA-Z0-9_.-]+', name) is not None, 'unsafe output name')
        require(self.used+len(raw) <= MAX_OUTPUT-(0 if emergency else RESERVE), 'total output byte cap')
        with (self.path/name).open('xb') as f: f.write(raw)
        self.used += len(raw)
        record = {'path': name, 'sha256': sha(raw), 'size_bytes': len(raw)}
        self.outputs.append(record)
        return record

    def write(self, name, value, emergency=False):
        return self.raw(name, encode(value), emergency)

    def request(self, name, params, transport):
        params = {'action': 'query', 'format': 'json', 'formatversion': 2, 'maxlag': 5, **params}
        started = utc()
        cap = min(MAX_RESPONSE, MAX_OUTPUT-RESERVE-self.used-4096)
        require(cap > 0, 'no remaining request budget')
        try:
            status, raw, encoding = transport(params, cap)
        except Exception as error:
            self.write(name+'.request.json', {'endpoint': API, 'params': params, 'started_utc': started,
                'completed_utc': utc(), 'transport_error_type': type(error).__name__, 'user_agent': USER_AGENT})
            raise
        require(type(raw) is bytes, 'transport body must be bytes')
        body = self.raw(name+'.response.json', raw[:cap])
        self.write(name+'.request.json', {'endpoint': API, 'params': params, 'started_utc': started,
            'completed_utc': utc(), 'status': status, 'content_encoding': encoding, 'user_agent': USER_AGENT,
            'response': body, 'body_truncated_at_cap': len(raw) > cap})
        require(len(raw) <= cap, 'response byte cap; retained prefix is explicitly incomplete')
        require(status == 200, 'HTTP error')
        require(encoding in ('identity', ''), 'unexpected compressed response')
        value = decode(raw)
        require(type(value) is dict and not value.get('error') and not value.get('warnings'), 'API error/warning')
        require('continue' not in value, 'unexpected continuation')
        return value, body

    def fail(self, error):
        self.write('failure.json', {'status': 'FAILED', 'error_type': type(error).__name__,
            'reason': str(error), 'failed_utc': utc(), 'automatic_retry': False}, emergency=True)

    def complete(self, schema, extra):
        require(sha(Path(__file__).read_bytes()) == self.source_sha, 'source changed during stage')
        return self.write('receipt.json', {'schema': schema, 'status': 'complete', 'completed_utc': utc(),
            'source_sha256': self.source_sha, 'protocol_sha256': PROTOCOL_SHA, 'outputs': list(self.outputs), **extra})


def protocol(root):
    require(sha(bounded_read(root/'protocol.md')) == PROTOCOL_SHA, 'protocol hash mismatch')


def ordered_members(value, topic):
    require(type(value.get('query')) is dict, 'missing query')
    rows = value['query'].get('categorymembers')
    require(type(rows) is list and 6 <= len(rows) <= 500, 'incomplete/short catalogue')
    for row in rows:
        require(type(row) is dict and type(row.get('pageid')) is int and row['pageid'] > 0
                and type(row.get('ns')) is int and row['ns'] == 0
                and type(row.get('title')) is str and bool(row['title']), 'invalid catalogue member')
    require(len({r['pageid'] for r in rows}) == len(rows), 'duplicate catalogue page ID')
    return sorted(rows, key=lambda r: (sha(f"20260914|{topic}|{r['pageid']}".encode()), r['pageid']))


def catalogue(root=OUT, transport=fetch):
    stage = Stage(root, 'catalogue')
    try:
        protocol(root)
        groups = []
        for topic, category in ROOTS:
            value, response = stage.request(topic, {'list': 'categorymembers', 'cmtitle': category,
                'cmnamespace': 0, 'cmtype': 'page', 'cmlimit': 500}, transport)
            groups.append({'topic': topic, 'category': category, 'response': response,
                           'candidates': ordered_members(value, topic)})
        stage.write('catalogue.json', {'schema': 'jlens_independent_catalogue_data_v1', 'groups': groups})
        protocol(root)
        return stage.complete('jlens_independent_catalogue_v1', {'topics': 4})
    except Exception as error:
        stage.fail(error)
        raise


def load_catalogue(root):
    folder = root/'catalogue'
    receipt_raw = bounded_read(folder/'receipt.json')
    receipt = decode(receipt_raw)
    require(receipt['schema'] == 'jlens_independent_catalogue_v1' and receipt['status'] == 'complete'
            and receipt['protocol_sha256'] == PROTOCOL_SHA and receipt['source_sha256'] == sha(Path(__file__).read_bytes()),
            'catalogue source/protocol binding')
    used = len(receipt_raw)
    for record in receipt['outputs']:
        require(re.fullmatch(r'[a-zA-Z0-9_.-]+', record['path']) is not None, 'unsafe catalogue member')
        raw = bounded_read(folder/record['path'])
        require(sha(raw) == record['sha256'] and len(raw) == record['size_bytes'], 'catalogue artifact changed')
        used += len(raw)
    data = decode(bounded_read(folder/'catalogue.json'))
    require(data['schema'] == 'jlens_independent_catalogue_data_v1'
            and [(g['topic'], g['category']) for g in data['groups']] == ROOTS, 'catalogue root order')
    return data['groups'], used, sha(receipt_raw)


def eligible(value, requested, selected_ids, selected_prefixes):
    pages = value.get('query', {}).get('pages')
    require(type(pages) is list and len(pages) == 1 and type(pages[0]) is dict, 'malformed candidate response')
    page = pages[0]
    for marker in ['missing', 'invalid', 'redirect']:
        if marker in page: return None, marker
    if type(page.get('ns')) is not int or page['ns'] != 0: return None, 'non-main namespace'
    if 'disambiguation' in page.get('pageprops', {}): return None, 'disambiguation'
    if type(page.get('pageid')) is not int or page['pageid'] <= 0: return None, 'invalid page ID'
    require(page['pageid'] == requested, 'returned different page ID')
    if type(page.get('title')) is not str or not page['title'].strip(): return None, 'invalid title'
    url = page.get('fullurl')
    if type(url) is not str or urlsplit(url).scheme != 'https' or urlsplit(url).netloc != 'en.wikipedia.org': return None, 'invalid article URL'
    revisions = page.get('revisions')
    if not (type(revisions) is list and len(revisions) == 1 and type(revisions[0]) is dict
            and type(revisions[0].get('revid')) is int and revisions[0]['revid'] > 0
            and type(revisions[0].get('timestamp')) is str and revisions[0]['timestamp']): return None, 'invalid revision metadata'
    extract = page.get('extract')
    if type(extract) is not str or len(extract.split()) < 16: return None, 'short/missing extract'
    prefix = ' '.join(extract.split()[:16])
    if requested in selected_ids: return None, 'previously selected page ID'
    if prefix in selected_prefixes: return None, 'previously selected prefix'
    return {'prefix': prefix, 'pageid': requested, 'title': page['title'], 'url': url,
            'history_url': 'https://en.wikipedia.org/w/index.php?'+urlencode({'title': page['title'], 'action': 'history'}),
            'reported_revision_id': revisions[0]['revid'], 'reported_revision_timestamp': revisions[0]['timestamp'],
            'attribution': page['title']+' — Wikipedia contributors', 'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
            'transform': 'First 16 Python Unicode str.split words, joined with single ASCII spaces; no other editing.',
            'additional_attribution_notices': 'Not certified by API modules; fixed-roster publication check required.'}, None


def excerpts(root=OUT, transport=fetch):
    # Claim before any source read. The catalogue's byte count is added before requests.
    stage = Stage(root, 'excerpts')
    try:
        protocol(root)
        groups, prior_bytes, catalogue_hash = load_catalogue(root)
        stage.used += prior_bytes
        require(stage.used < MAX_OUTPUT-RESERVE, 'combined output byte cap')
        selected_ids, selected_prefixes, dataset, attribution, decisions = set(), set(), [], [], []
        for group in groups:
            accepted = 0
            for rank, candidate in enumerate(group['candidates'][:20], 1):
                topic, pageid = group['topic'], candidate['pageid']
                name = f'{topic}-{rank:02}'
                value, response = stage.request(name, {'pageids': pageid, 'prop': 'extracts|revisions|info|pageprops',
                    'explaintext': 1, 'exintro': 1, 'exchars': 1200, 'rvprop': 'ids|timestamp', 'rvlimit': 1,
                    'inprop': 'url'}, transport)
                article, reason = eligible(value, pageid, selected_ids, selected_prefixes)
                decision = {'topic': topic, 'candidate_rank': rank, 'pageid': pageid, 'response': response,
                            'selected': article is not None, 'skip_reason': reason}
                if article is not None:
                    ident = f'{topic}-wiki-{accepted}'
                    selected_ids.add(pageid); selected_prefixes.add(article['prefix'])
                    dataset.append({'id': ident, 'topic': topic, 'prefix': article['prefix']})
                    attribution.append({'id': ident, 'response': response, **article})
                    decision['id'] = ident
                    accepted += 1
                decisions.append(decision)
                stage.write(name+'.decision.json', decision)
                if accepted == 6: break
            require(accepted == 6, f'Fewer than six eligible pages within first 20 candidates: {group["topic"]}')
        require(len(dataset) == len(selected_ids) == len(selected_prefixes) == 24, 'final source roster')
        pairs = [{'id': f'P{i+1:02}', 'left': dataset[2*i]['id'], 'right': dataset[2*i+1]['id']} for i in range(12)]
        stage.write('dataset.json', dataset)
        stage.write('pairs.json', pairs)
        stage.write('attribution.json', {'schema': 'jlens_independent_attribution_v1', 'articles': attribution,
            'snapshot_scope': 'Exact response bytes, not an atomic historical revision reconstruction.',
            'reuse_scope': 'Excerpt-derived content retains applicable Wikipedia source terms; code/research are separate.'})
        protocol(root)
        require(load_catalogue(root)[2] == catalogue_hash, 'catalogue changed during excerpts')
        return stage.complete('jlens_independent_excerpts_v1', {'catalogue_receipt_sha256': catalogue_hash,
            'selected_count': 24, 'candidate_requests': len(decisions), 'additional_notice_check_before_publication': True})
    except Exception as error:
        stage.fail(error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['catalogue', 'excerpts'])
    args = parser.parse_args()
    {'catalogue': catalogue, 'excerpts': excerpts}[args.stage]()
