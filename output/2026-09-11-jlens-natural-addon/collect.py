"""New add-on collection; selection and serial excerpts are exclusive stages."""
import argparse
import hashlib
from pathlib import Path
import types
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / '2026-09-11-jlens-independent-content'
HELPER_SHA = 'd277f27eccdbf85bdbf1907ece40fea40148d59dded23e806d5f4721e1756f7a'
PROTOCOL_SHA = 'a510feaf2dfdd5cf6541d974d2c305188e24d04a4e9e185ac73675970b8a6246'
CATALOGUE_SHA = 'c98cebf2222fc93c0cee48e4442af277cfa8743f091594a964e50a14ca86b251'
OLD_DATA_SHA = '7391a1fa6c02872cdeabb2e8dffc1e63181b6379d80e2042f33b38a61308ff02'
TOPICS = ('astronomy', 'cooking', 'football', 'programming')
API = 'https://en.wikipedia.org/w/api.php'
UA = 'SpectralJLensPilot/0.1 (research; https://github.com/tbuckworth)'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def helper():
    path = OLD / 'collect_text.py'
    raw = path.read_bytes()
    require(sha(raw) == HELPER_SHA, 'historical helper changed')
    module = types.ModuleType('pinned_old_collection_utilities')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise HTTPError(request.full_url, code, message, headers, fp)


def fetch(params, cap):
    request = Request(API + '?' + urlencode(params), headers={
        'User-Agent': UA, 'Accept-Encoding': 'identity'})
    try:
        response = build_opener(NoRedirect()).open(request, timeout=20)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read(cap + 1), response.headers.get('Content-Encoding', 'identity')


def request(h, stage, name, params, transport):
    """Preserve exact bytes; accept only un-followed revision-history continuation."""
    params = {'action': 'query', 'format': 'json', 'formatversion': 2, 'maxlag': 5, **params}
    started = h.utc()
    cap = min(h.MAX_RESPONSE, h.MAX_OUTPUT - h.RESERVE - stage.used - 4096)
    require(cap > 0, 'no remaining request budget')
    try:
        status, raw, encoding = transport(params, cap)
    except Exception as error:
        stage.write(name + '.request.json', {'endpoint': API, 'params': params,
            'started_utc': started, 'completed_utc': h.utc(),
            'transport_error_type': type(error).__name__, 'user_agent': UA})
        raise
    require(type(raw) is bytes, 'transport body must be bytes')
    body = stage.raw(name + '.response.json', raw[:cap])
    stage.write(name + '.request.json', {'endpoint': API, 'params': params,
        'started_utc': started, 'completed_utc': h.utc(), 'status': status,
        'content_encoding': encoding, 'user_agent': UA, 'response': body,
        'body_truncated_at_cap': len(raw) > cap, 'new_network_request': True})
    require(len(raw) <= cap, 'response byte cap; retained prefix is explicitly incomplete')
    require(status == 200, 'HTTP error')
    require(encoding in ('identity', ''), 'unexpected compressed response')
    value = h.decode(raw)
    require(type(value) is dict and not value.get('error') and not value.get('warnings'), 'API error/warning')
    if 'continue' in value:
        token = value['continue']
        require(type(token) is dict and set(token) == {'rvcontinue', 'continue'} and
                all(type(v) is str and v for v in token.values()), 'non-revision continuation')
    return value, body


def snapshot(h, path, expected=None):
    path = Path(path).absolute()
    raw = h.bounded_read(path)
    digest = sha(raw)
    require(expected is None or digest == expected, 'input hash mismatch: ' + str(path))
    return raw, {'path': str(path), 'sha256': digest, 'size_bytes': len(raw)}


def recheck(h, records):
    for item in records:
        raw, record = snapshot(h, item['path'], item['sha256'])
        require(len(raw) == item['size_bytes'], 'input size changed')


def candidate_pairs(inventory):
    """Pure frozen pairing; never examines article text or scores."""
    require(inventory['schema'] == 'jlens_natural_addon_inventory_v1', 'inventory schema')
    groups = inventory['groups']
    require([g['topic'] for g in groups] == list(TOPICS), 'topic order')
    excluded = inventory['excluded_pageids']
    require(all(type(i) is int and i > 0 for i in excluded) and
            excluded == sorted(set(excluded)), 'exclusion IDs')
    excluded, owners, duplicates, pairs, tails = set(excluded), {}, [], [], []
    for group in groups:
        topic, available = group['topic'], []
        for row in group['candidates']:
            require(type(row.get('pageid')) is int and row['pageid'] > 0 and
                    type(row.get('ns')) is int and row['ns'] == 0 and
                    type(row.get('title')) is str and row['title'], 'invalid candidate')
            ident = row['pageid']
            if ident in owners:
                duplicates.append({'pageid': ident, 'owner': owners[ident], 'other_topic': topic})
                continue
            owners[ident] = topic
            if ident not in excluded:
                available.append(dict(row))
        available.sort(key=lambda r: (sha(f"20260919|candidate|{topic}|{r['pageid']}".encode()), r['pageid']))
        for i in range(0, len(available) - 1, 2):
            pairs.append({'topic': topic, 'left': available[i], 'right': available[i + 1]})
        if len(available) % 2:
            tails.append({'topic': topic, 'candidate': available[-1]})
    require(excluded <= owners.keys(), 'excluded ID outside catalogue')
    pairs.sort(key=lambda p: (sha(f"20260919|pair|{p['topic']}|{p['left']['pageid']}|{p['right']['pageid']}".encode()),
                              TOPICS.index(p['topic']), p['left']['pageid'], p['right']['pageid']))
    for i, pair in enumerate(pairs, 1):
        pair['id'] = f'K{i:03}'
    return {'schema': 'jlens_natural_addon_selection_v1', 'seed': '20260919',
            'pairs': pairs, 'odd_tails': tails, 'duplicate_ownership': duplicates,
            'excluded_pageids': sorted(excluded), 'available_count': 2 * len(pairs) + len(tails)}


def load_inputs(h, inventory_path, inventory_sha, root):
    inventory_raw, invpin = snapshot(h, inventory_path, inventory_sha)
    inv = h.decode(inventory_raw)
    records = [invpin]
    for path, digest in [(root / 'protocol.md', PROTOCOL_SHA), (Path(__file__), None),
                         (OLD / 'collect_text.py', HELPER_SHA),
                         (OLD / 'catalogue/catalogue.json', CATALOGUE_SHA),
                         (OLD / 'excerpts-resumed/dataset.json', OLD_DATA_SHA)]:
        raw, pin = snapshot(h, path, digest)
        records.append(pin)
    require(inv['catalogue']['sha256'] == CATALOGUE_SHA and
            Path(inv['catalogue']['path']) == OLD / 'catalogue/catalogue.json', 'inventory catalogue binding')
    catalogue = h.decode(h.bounded_read(OLD / 'catalogue/catalogue.json'))
    require(inv['groups'] == [{k: g[k] for k in ('topic', 'category', 'candidates')}
                              for g in catalogue['groups']], 'inventory catalogue groups')
    require(type(inv['sources']) is list and inv['sources'], 'missing inventory sources')
    for record in inv['sources']:
        require(Path(record['path']).is_absolute() and
                Path(record['path']).is_relative_to(OLD), 'inventory source outside original study')
    recheck(h, inv['sources'])
    records.extend(inv['sources'])
    old = h.decode(h.bounded_read(OLD / 'excerpts-resumed/dataset.json'))
    require(type(old) is list and len(old) == 24 and
            len({r['prefix'] for r in old}) == 24, 'old measured prefix roster')
    return inv, {r['prefix'] for r in old}, records


def complete(h, stage, records, extra):
    recheck(h, records)
    require(sha((OLD / 'collect_text.py').read_bytes()) == HELPER_SHA, 'helper changed')
    return stage.write('receipt.json', {'schema': 'jlens_natural_addon_' + stage.path.name + '_v1',
        'status': 'complete', 'completed_utc': h.utc(), 'inputs': records,
        'outputs': list(stage.outputs), **extra})


def selection(inventory_path, inventory_sha, root=OUT):
    h = helper()
    stage = h.Stage(root, 'selection')
    try:
        inv, _, records = load_inputs(h, inventory_path, inventory_sha, root)
        stage.write('bindings.json', records)
        manifest = candidate_pairs(inv)
        require(len(manifest['pairs']) >= 16, 'fewer than sixteen candidate pairs')
        stage.write('manifest.json', manifest)
        return complete(h, stage, records, {'candidate_pairs': len(manifest['pairs']),
                                           'network_requests': 0})
    except Exception as error:
        stage.fail(error)
        raise


def collect_pairs(h, stage, manifest, old_prefixes, transport, target=16, max_pairs=32):
    """Collect both members of every visited fixed pair; never re-pair survivors."""
    used_prefixes, rows, pairs, attribution = set(old_prefixes), [], [], []
    visited = requests = 0
    for candidate in manifest['pairs'][:max_pairs]:
        visited += 1
        articles, decisions = [], []
        for side in ('left', 'right'):
            pageid = candidate[side]['pageid']
            name = candidate['id'] + '-' + side
            requests += 1
            value, response = request(h, stage, name, {'pageids': pageid,
                'prop': 'extracts|revisions|info|pageprops', 'explaintext': 1, 'exintro': 1,
                'exchars': 1200, 'rvprop': 'ids|timestamp', 'rvlimit': 1, 'inprop': 'url'}, transport)
            pages = value.get('query', {}).get('pages')
            require(type(pages) is list and len(pages) == 1 and type(pages[0]) is dict,
                    'malformed candidate response')
            returned_id = pages[0].get('pageid')
            if type(returned_id) is int and returned_id > 0:
                require(returned_id == pageid, 'returned different page ID')
            article, reason = h.eligible(value, pageid, set(), set())
            decision = {'side': side, 'pageid': pageid, 'response': response,
                        'eligible': article is not None, 'technical_reason': reason}
            stage.write(name + '.eligibility.json', decision)
            articles.append(article)
            decisions.append(decision)
        reason = None
        if any(a is None for a in articles):
            reason = 'one or both articles technically ineligible'
        elif articles[0]['prefix'] == articles[1]['prefix']:
            reason = 'within-pair duplicate prefix'
        elif any(a['prefix'] in used_prefixes for a in articles):
            reason = 'old measured or accepted prefix duplicate'
        pair_decision = {'candidate_id': candidate['id'], 'topic': candidate['topic'],
                         'articles': decisions, 'accepted': reason is None, 'reason': reason}
        if reason is None:
            ident = f'N{len(pairs) + 1:02}'
            pair_decision['pair_id'] = ident
            for side, article, decision in zip(('L', 'R'), articles, decisions):
                rowid = ident + '-' + side
                used_prefixes.add(article['prefix'])
                rows.append({'id': rowid, 'topic': candidate['topic'], 'prefix': article['prefix']})
                attribution.append({'id': rowid, 'pair_id': ident, 'candidate_id': candidate['id'],
                                    'response': decision['response'], **article})
            pairs.append({'id': ident, 'left': ident + '-L', 'right': ident + '-R',
                          'topic': candidate['topic'], 'candidate_id': candidate['id']})
        stage.write(candidate['id'] + '.decision.json', pair_decision)
        if len(pairs) == target:
            break
    require(len(pairs) == target, 'fewer than required eligible pairs within fixed cap')
    stage.write('dataset.json', rows)
    stage.write('pairs.json', pairs)
    stage.write('attribution.json', {'schema': 'jlens_natural_addon_attribution_v1', 'articles': attribution,
        'snapshot_scope': 'Exact response bytes; extract and revision metadata are not atomic historical snapshots.',
        'reuse_scope': 'Source terms apply to excerpt-derived content separately from code/research.'})
    return {'selected_count': len(rows), 'selected_pairs': len(pairs), 'visited_pairs': visited,
            'article_requests': requests, 'additional_notice_check_before_publication': True}


def excerpts(inventory_path, inventory_sha, selection_sha, root=OUT, transport=fetch):
    h = helper()
    stage = h.Stage(root, 'excerpts')
    try:
        inv, old_prefixes, records = load_inputs(h, inventory_path, inventory_sha, root)
        raw, receipt_pin = snapshot(h, root / 'selection/receipt.json', selection_sha)
        receipt = h.decode(raw)
        require(receipt['schema'] == 'jlens_natural_addon_selection_v1' and
                receipt['status'] == 'complete' and receipt['inputs'] == records, 'selection receipt binding')
        records.append(receipt_pin)
        for item in receipt['outputs']:
            require(Path(item['path']).name == item['path'], 'unsafe selection output')
            raw, pin = snapshot(h, root / 'selection' / item['path'], item['sha256'])
            require(len(raw) == item['size_bytes'], 'selection output size')
            records.append(pin)
        manifest = h.decode(h.bounded_read(root / 'selection/manifest.json'))
        require(manifest == candidate_pairs(inv), 'selection manifest drift')
        stage.write('bindings.json', records)
        extra = collect_pairs(h, stage, manifest, old_prefixes, transport)
        return complete(h, stage, records, extra)
    except Exception as error:
        stage.fail(error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('selection', 'excerpts'))
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha', required=True)
    parser.add_argument('--selection-receipt-sha')
    args = parser.parse_args()
    if args.stage == 'selection':
        print(selection(args.inventory, args.inventory_sha))
    else:
        require(args.selection_receipt_sha is not None, 'selection receipt pin required')
        print(excerpts(args.inventory, args.inventory_sha, args.selection_receipt_sha))
