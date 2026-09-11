"""Inert, once-only fixed-role article collection. No selection or model entrypoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import time
import types
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

OUT = Path(__file__).resolve().parent
MAIN = Path('/private-artifacts/repositories/optimizers-launch-investigation')
STUDY = MAIN/'output/2026-09-11-jlens-pattern-calibration'
HELPER = MAIN/'output/2026-09-11-jlens-independent-content/collect_text.py'
HELPER_SHA = 'd277f27eccdbf85bdbf1907ece40fea40148d59dded23e806d5f4721e1756f7a'
PROTOCOL_SHA = 'a3b47578fad5c1075e7b2855d01cffe954d5d0fb6b5c4d7fd2e51262a42f0d4e'
MANIFEST_SHA = '8287106e8e3238ccf18d0c60bc0a4b59de894c7a8fd038c6a870a2cd1e94ed63'
SELECTION_RECEIPT_SHA = 'cbd406b04ce5c776c95091331cc8064cf16d38f7d50c5ca137b07e667c15ea78'
PLANNER_SHA = '82869d2acebec4524372247ecc3450f1a52c023a243ee2a5c7b8516affa2205a'
INVENTORY_SHA = 'e57b194d6ed0b6b0999097d64c12dc37198f7b73d9b27bb5c2dd31e51a54f7aa'
PREFIX_SHA = 'fd66b415caca0b563425b77e2ce0ee62c3c625849e6c9829c2b64153e062f6f7'
API = 'https://en.wikipedia.org/w/api.php'
UA = 'SpectralJLensPilot/0.1 (research; https://github.com/tbuckworth)'
TOPICS = ('astronomy', 'cooking', 'football', 'programming')
TARGETS = {'calibration': 32, 'evaluation': 16}
MAX_PAIRS, MAX_REQUESTS = 80, 160
MAX_RESPONSE, MAX_OUTPUT, RESERVE = 2*1024**2, 32*1024**2, 256*1024
MAX_SECONDS = 3600


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()


def snapshot(path, expected=None):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'input absent or symlink')
    with path.open('rb') as handle:
        raw = handle.read(MAX_OUTPUT+1)
    require(len(raw) <= MAX_OUTPUT, 'input byte cap')
    require(expected is None or sha(raw) == expected, 'input hash mismatch: '+str(path))
    return raw, {'path': str(path), 'sha256': sha(raw), 'size_bytes': len(raw)}


def helper():
    raw, _ = snapshot(HELPER, HELPER_SHA)
    module = types.ModuleType('pinned_pure_article_helpers')
    module.__file__ = str(HELPER)
    exec(compile(raw, str(HELPER), 'exec'), module.__dict__)
    return module


class Stage:
    def __init__(self, root):
        self.path = root/'excerpts'
        self.path.mkdir(exist_ok=False)  # Includes dangling-path refusal, before inputs.
        self.started = time.monotonic()
        self.used, self.outputs, self.requests = 0, [], 0
        self.dispositions = []
        self.source_sha = snapshot(Path(__file__))[1]['sha256']
        self.write('attempt.json', {'schema': 'jlens_pattern_calibration_collection_attempt_v1',
            'started_utc': utc(), 'source_sha256': self.source_sha, 'python': sys.version,
            'maximum_requests': MAX_REQUESTS, 'maximum_output_bytes': MAX_OUTPUT,
            'failure_reserve_bytes': RESERVE, 'automatic_retry': False})

    def raw(self, name, raw, emergency=False):
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', name) is not None, 'unsafe artifact name')
        require(self.used+len(raw) <= MAX_OUTPUT-(0 if emergency else RESERVE), 'output byte cap')
        with (self.path/name).open('xb') as handle:
            handle.write(raw)
        self.used += len(raw)
        record = {'path': name, 'sha256': sha(raw), 'size_bytes': len(raw)}
        self.outputs.append(record)
        return record

    def write(self, name, value, emergency=False):
        return self.raw(name, encode(value), emergency)

    def audit(self):
        require({p.name for p in self.path.iterdir()} == {r['path'] for r in self.outputs}, 'untracked artifact')
        for record in self.outputs:
            raw, _ = snapshot(self.path/record['path'], record['sha256'])
            require(len(raw) == record['size_bytes'], 'artifact size changed')
        require(sum(r['size_bytes'] for r in self.outputs) == self.used, 'artifact accounting')

    def fail(self, error):
        # The reserve is separate from all normal artifacts, including the success receipt.
        # Include any partial file left by a failed write, not only completed writes.
        actual = []
        for path in sorted(self.path.iterdir()):
            _, pin = snapshot(path)
            actual.append({**pin, 'path': path.name})
        self.used = sum(p['size_bytes'] for p in actual)
        self.write('failure.json', {'schema': 'jlens_pattern_calibration_collection_failure_v1',
            'status': 'FAILED', 'failed_utc': utc(), 'error_type': type(error).__name__,
            'reason': str(error)[:1000], 'source_sha256': self.source_sha,
            'article_requests_started': self.requests, 'elapsed_seconds': time.monotonic()-self.started,
            'artifacts_before_failure': actual, 'bytes_before_failure': self.used,
            'candidate_dispositions': self.dispositions, 'automatic_retry': False}, emergency=True)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise HTTPError(request.full_url, code, message, headers, fp)


def fetch(params, cap):
    # A socket timeout alone is not an end-to-end deadline. This lane is serial/main-thread.
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    require(previous_timer == (0.0, 0.0), 'existing process alarm; refusing to disturb it')
    def expired(signum, frame):
        raise TimeoutError('20-second article wall deadline')
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, 20)
        req = Request(API+'?'+urlencode(params), headers={'User-Agent': UA, 'Accept-Encoding': 'identity'})
        try:
            response = build_opener(NoRedirect()).open(req, timeout=20)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read(cap+1), response.headers.get('Content-Encoding', 'identity')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def request(h, stage, name, pageid, transport):
    require(stage.requests < MAX_REQUESTS, 'article request cap')
    require(time.monotonic()-stage.started < MAX_SECONDS, 'collection time cap')
    cap = min(MAX_RESPONSE, MAX_OUTPUT-RESERVE-stage.used-4096)
    require(cap > 0, 'no remaining request bytes')
    params = {'action': 'query', 'format': 'json', 'formatversion': 2, 'maxlag': 5,
        'pageids': pageid, 'prop': 'extracts|revisions|info|pageprops', 'explaintext': 1,
        'exintro': 1, 'exchars': 1200, 'rvprop': 'ids|timestamp', 'rvlimit': 1, 'inprop': 'url'}
    meta = {'endpoint': API, 'params': params, 'started_utc': utc(), 'user_agent': UA,
            'new_network_request': True}
    stage.requests += 1
    try:
        status, raw, encoding = transport(params, cap)
    except Exception as error:
        stage.write(name+'.request.json', {**meta, 'completed_utc': utc(),
            'transport_error_type': type(error).__name__})
        raise
    require(type(raw) is bytes, 'transport body must be bytes')
    body = stage.raw(name+'.response.json', raw[:cap])
    stage.write(name+'.request.json', {**meta, 'completed_utc': utc(), 'status': status,
        'content_encoding': encoding, 'response': body, 'body_truncated_at_cap': len(raw) > cap})
    require(len(raw) <= cap, 'response cap; saved prefix explicitly incomplete')
    require(status == 200 and encoding in ('', 'identity'), 'HTTP status or encoding')
    require(time.monotonic()-stage.started < MAX_SECONDS, 'collection time cap')
    value = h.decode(raw)
    require(type(value) is dict and 'error' not in value and 'warnings' not in value, 'API error/warning')
    if 'continue' in value:
        token = value['continue']
        require(type(token) is dict and set(token) == {'rvcontinue', 'continue'} and
                all(type(v) is str and v for v in token.values()), 'non-revision continuation')
    require(type(value.get('query')) is dict and 'redirects' not in value['query'], 'query schema/API redirect')
    return value, body


def validate_manifest(manifest, inventory):
    require(manifest['schema'] == 'jlens_pattern_calibration_candidate_manifest_v1' and
            manifest['seed'] == '20260921' and manifest['protocol_sha256'] == PROTOCOL_SHA and
            manifest['inventory_sha256'] == INVENTORY_SHA and manifest['source_sha256'] == PLANNER_SHA,
            'manifest provenance')
    require(manifest['targets'] == {'calibration_pairs': 32, 'evaluation_pairs': 16} and
            manifest['maximum_requested_pairs'] == MAX_PAIRS and manifest['maximum_article_requests'] == MAX_REQUESTS and
            manifest['eligibility_or_measurement_performed'] is False, 'manifest scope')
    available = {g['topic']: set(g['remaining_pageids']) for g in inventory['groups']}
    require(list(available) == list(TOPICS), 'inventory topic order')
    rows, seen = manifest['candidates'], set()
    require(len(rows) == 86 and manifest['candidate_role_counts'] == {'calibration': 58, 'evaluation': 28}, 'candidate counts')
    for i, row in enumerate(rows):
        require(row['candidate_id'] == f'K{i+1:03}' and
                row['role'] == ('evaluation' if i % 3 == 2 else 'calibration'), 'candidate role/order')
        for side in ('left_pageid', 'right_pageid'):
            ident = row[side]
            require(type(ident) is int and ident > 0 and ident in available[row['topic']] and
                    ident not in seen and ident not in inventory['excluded_pageids'], 'candidate ID/ownership/exclusion')
            seen.add(ident)
    tails = manifest['odd_tails']
    require(len(tails) == 1 and tails[0]['topic'] == 'programming' and
            tails[0]['pageid'] not in seen and tails[0]['pageid'] in available['programming'], 'odd unused tail')
    require(seen | {tails[0]['pageid']} == set.union(*available.values()), 'remaining pool coverage')


def load_inputs(h):
    paths = [(Path(__file__), None), (HELPER, HELPER_SHA), (STUDY/'protocol.md', PROTOCOL_SHA),
        (STUDY/'selection/manifest.json', MANIFEST_SHA), (STUDY/'selection/receipt.json', SELECTION_RECEIPT_SHA),
        (OUT/'inventory.json', INVENTORY_SHA), (OUT/'old-prefixes.json', PREFIX_SHA)]
    loaded, pins = [], []
    for path, digest in paths:
        raw, pin = snapshot(path, digest)
        loaded.append(raw); pins.append(pin)
    manifest, receipt, inventory, old = [h.decode(raw) for raw in loaded[3:]]
    validate_manifest(manifest, inventory)
    require(receipt['status'] == 'METADATA_SELECTION_COMPLETE' and receipt['network_calls'] == 0 and
            receipt['manifest_sha256'] == MANIFEST_SHA and receipt['inventory_sha256'] == INVENTORY_SHA and
            receipt['protocol_sha256'] == PROTOCOL_SHA and receipt['source_sha256'] == PLANNER_SHA, 'selection receipt')
    require(old['schema'] == 'jlens_pattern_calibration_old_prefix_inventory_v1', 'prefix inventory schema')
    records = []
    for i, source in enumerate(old['sources']):
        raw, pin = snapshot(source['path'], source['sha256']); pins.append(pin)
        require(pin['size_bytes'] == source['size_bytes'], 'old dataset size')
        value = h.decode(raw)
        rows = value if source['container'] == 'list' else value['rows']
        require(len(rows) == source['row_count'], 'old dataset count')
        records.extend({'source_index': i, 'row_index': j, 'id': r['id'],
                        'prefix': r[source['text_field']]} for j, r in enumerate(rows))
    prefixes = sorted({r['prefix'] for r in records})
    require(records == old['records'] and prefixes == old['unique_prefixes'] and
            len(records) == old['row_count'] == 296 and len(prefixes) == old['unique_prefix_count'] == 296,
            'exact old prefix source joins')
    return manifest, set(prefixes), pins


def collect(h, stage, manifest, old_prefixes, transport, targets=None, max_pairs=MAX_PAIRS):
    targets = dict(TARGETS if targets is None else targets)
    datasets, pairs = {r: [] for r in TARGETS}, {r: [] for r in TARGETS}
    used, attribution, visited = set(old_prefixes), [], 0
    stage.dispositions = [{**p, 'status': 'not_visited'} for p in manifest['candidates']]
    for decision in stage.dispositions:
        role, name = decision['role'], decision['candidate_id']
        if len(pairs[role]) == targets[role]:
            decision['status'] = 'capacity_skip_no_requests'
            continue
        require(visited < max_pairs, 'requested pair cap before both targets')
        visited += 1
        decision['status'], decision['articles'] = 'requesting', []
        articles = []
        for side in ('left', 'right'):
            ident = decision[side+'_pageid']
            value, body = request(h, stage, name+'-'+side, ident, transport)
            pages = value['query'].get('pages')
            require(type(pages) is list and len(pages) == 1 and type(pages[0]) is dict, 'article response shape')
            returned = pages[0].get('pageid')
            if type(returned) is int and returned > 0:
                require(returned == ident, 'different returned page ID, including technical skips')
            article, reason = h.eligible(value, ident, set(), set())
            entry = {'side': side, 'pageid': ident, 'response': body, 'eligible': article is not None,
                     'technical_reason': reason}
            decision['articles'].append(entry)
            stage.write(name+'-'+side+'.eligibility.json', entry)
            articles.append(article)
        reason = 'technical member exclusion' if any(a is None for a in articles) else None
        if reason is None:
            text = [a['prefix'] for a in articles]
            if text[0] == text[1]: reason = 'duplicate prefix within pair'
            elif any(p in used for p in text): reason = 'duplicate old or accepted prefix'
        decision['status'], decision['reason'] = ('rejected', reason) if reason else ('accepted', None)
        if reason is None:
            pair_id = ('C' if role == 'calibration' else 'T')+f'{len(pairs[role])+1:02}'
            pair = {'id': pair_id, 'left': pair_id+'-L', 'right': pair_id+'-R',
                    'topic': decision['topic'], 'candidate_id': name, 'role': role}
            pairs[role].append(pair); decision['accepted_pair_id'] = pair_id
            for side, article, evidence in zip(('L', 'R'), articles, decision['articles'], strict=True):
                ident = pair_id+'-'+side
                datasets[role].append({'id': ident, 'topic': decision['topic'], 'prefix': article['prefix']})
                attribution.append({'id': ident, 'pair_id': pair_id, 'candidate_id': name, 'role': role,
                                    'response': evidence['response'], **article})
                used.add(article['prefix'])  # Only a wholly accepted pair reserves strings.
        stage.write(name+'.decision.json', decision)
    require(all(len(pairs[r]) == targets[r] for r in targets), 'manifest exhausted before both targets')
    require(stage.requests == 2*visited <= MAX_REQUESTS, 'request/pair accounting')
    for role in TARGETS:
        require(len(datasets[role]) == 2*targets[role], 'accepted role count')
        stage.write(role+'-dataset.json', datasets[role]); stage.write(role+'-pairs.json', pairs[role])
    stage.write('candidate-dispositions.json', stage.dispositions)
    stage.write('attribution.json', {'schema': 'jlens_pattern_calibration_attribution_v1', 'articles': attribution,
        'snapshot_scope': 'Exact response bytes; not atomic historical revision reconstruction.',
        'publication_attribution_review_required': True, 'illustrations_copied': False})
    return {'accepted_pairs': {r: len(pairs[r]) for r in TARGETS}, 'selected_count': sum(map(len, datasets.values())),
            'requested_pairs': visited, 'article_requests': stage.requests,
            'candidate_dispositions': len(stage.dispositions), 'old_prefix_exclusions': len(old_prefixes)}


def run(root=OUT, transport=fetch):
    stage = Stage(root)
    try:
        h = helper()
        manifest, old_prefixes, pins = load_inputs(h)
        stage.write('bindings.json', pins)
        result = collect(h, stage, manifest, old_prefixes, transport)
        for pin in pins:
            raw, _ = snapshot(pin['path'], pin['sha256'])
            require(len(raw) == pin['size_bytes'], 'input size drift')
        require(snapshot(Path(__file__))[1]['sha256'] == stage.source_sha, 'producer source drift')
        stage.audit()
        stage.write('receipt.json', {'schema': 'jlens_pattern_calibration_excerpts_v1', 'status': 'complete',
            'source_sha256': stage.source_sha, 'protocol_sha256': PROTOCOL_SHA, 'manifest_sha256': MANIFEST_SHA,
            'completed_utc': utc(), 'elapsed_seconds': time.monotonic()-stage.started,
            'inputs': pins, 'outputs': list(stage.outputs), 'bytes_before_receipt': stage.used,
            'automatic_retry': False, 'tokenizer_or_model_called': False, **result})
        stage.audit()
        return result
    except Exception as error:
        stage.fail(error)
        raise


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), sort_keys=True))
