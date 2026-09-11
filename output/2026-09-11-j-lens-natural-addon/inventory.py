"""Pinned local metadata inventory only; no corpus acquisition or candidate pairing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

STUDY = Path('output/2026-09-11-jlens-independent-content')
TOPICS = ('astronomy', 'cooking', 'football', 'programming')
CATEGORIES = ('Category:Astronomy', 'Category:Cooking',
              'Category:Association football', 'Category:Computer programming')
PINS = {
    'catalogue/catalogue.json': 'c98cebf2222fc93c0cee48e4442af277cfa8743f091594a964e50a14ca86b251',
    'catalogue/receipt.json': 'b15afe7220a4c823454b550553c3ab8a0181712773813922bce5976ccb8a0a0d',
    'excerpts-resumed/receipt.json': '21383e456a927acb45d069cf467798e73fd777e7c388adf7663c724be6efa703',
    'excerpts/astronomy-01.request.json': '33e8cf58c71243e34b59b65f5b08eeea8e37db0b5179a1bd3fc246c2fe48ef20',
    'attribution-review.md': '1b7c4d68e5f035a254233ecabb0dc87377d511e5622addde4db13f8d0a43922a',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def finite(value):
        value = float(value)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(data, object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=lambda value: finite(value))


class Snapshot:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}

    def read(self, path, expected):
        path = Path(path)
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe source path')
        full = self.root / path
        require(not full.is_symlink() and full.resolve().is_relative_to(self.root), 'symlink escape')
        require(full.is_file() and full.stat().st_size <= 4 * 1024 * 1024, 'source missing/oversized')
        data = full.read_bytes()
        require(sha(data) == expected, f'source hash mismatch: {path}')
        require(sha(full.read_bytes()) == expected, f'source changed during read: {path}')
        self.sources[str(path)] = {'path': str(full), 'sha256': expected, 'size_bytes': len(data)}
        return data

    def finish(self):
        for path, item in self.sources.items():
            require(sha((self.root / path).read_bytes()) == item['sha256'], 'source changed after read')
        return [self.sources[path] for path in sorted(self.sources)]


def receipt_pins(value, directory):
    require(value.get('status') == 'complete', 'incomplete historical receipt')
    result = {}
    for item in value['outputs']:
        name = item['path']
        require(Path(name).name == name and name not in ('', '.', '..'), 'unsafe receipt filename')
        path = str(directory / name)
        require(path not in result, 'duplicate receipt output')
        result[path] = item
    return result


def catalogue_groups(catalogue, raw):
    require(catalogue['schema'] == 'jlens_independent_catalogue_data_v1', 'catalogue schema')
    groups = catalogue['groups']
    require([g['topic'] for g in groups] == list(TOPICS), 'original topic order')
    for group, category in zip(groups, CATEGORIES):
        require(group['category'] == category, 'category changed')
        body = raw[group['topic']]
        require(not any(key in body for key in ('continue', 'error', 'warnings')), 'incomplete raw category')
        members = body['query']['categorymembers']
        require(type(members) is list and members, 'empty category')
        ids = set()
        for row in members:
            require(set(row) == {'pageid', 'ns', 'title'}, 'raw member fields')
            require(type(row['pageid']) is int and row['pageid'] > 0 and row['pageid'] not in ids,
                    'invalid/duplicate category ID')
            require(type(row['ns']) is int and row['ns'] == 0 and type(row['title']) is str
                    and row['title'], 'invalid namespace/title')
            ids.add(row['pageid'])
        ordered = sorted(members, key=lambda row: (
            sha(f"20260914|{group['topic']}|{row['pageid']}".encode()), row['pageid']))
        require(group['candidates'] == ordered, 'raw/compiled catalogue mismatch')
    return [{key: group[key] for key in ('topic', 'category', 'candidates')} for group in groups]


def ownership(groups, excluded):
    owners, rows, counts = {}, [], []
    for group in groups:
        owned = []
        for row in group['candidates']:
            pageid = row['pageid']
            if pageid not in owners:
                owners[pageid] = group['topic']
                owned.append(pageid)
                rows.append({'pageid': pageid, 'topic': group['topic']})
        available = [pageid for pageid in owned if pageid not in excluded]
        counts.append({'topic': group['topic'], 'raw_members': len(group['candidates']),
                       'owned_unique': len(owned), 'excluded_owned': len(owned) - len(available),
                       'available_owned': len(available)})
    return rows, counts


def inspect(root, pins=PINS):
    snap = Snapshot(root)
    pinned = {str(STUDY / path): {'sha256': digest} for path, digest in pins.items()}
    def read(path, as_json=True):
        data = snap.read(path, pinned[str(path)]['sha256'])
        if 'size_bytes' in pinned[str(path)]:
            require(len(data) == pinned[str(path)]['size_bytes'], 'receipt size mismatch')
        return decode(data) if as_json else data
    for directory in ('catalogue', 'excerpts-resumed'):
        receipt = read(STUDY / directory / 'receipt.json')
        expected_schema = ('jlens_independent_catalogue_v1' if directory == 'catalogue'
                           else 'jlens_independent_excerpts_v1')
        require(receipt.get('schema') == expected_schema, 'historical receipt schema')
        for path, pin in receipt_pins(receipt, STUDY / directory).items():
            if path in pinned:
                require(pinned[path]['sha256'] == pin['sha256'], 'conflicting root pin')
            pinned[path] = pin
    read(STUDY / 'attribution-review.md', False)
    continuation = read(STUDY / 'excerpts-resumed/continuation.json')
    for name, digest in continuation['preserved_attempt'].items():
        path = STUDY / 'excerpts' / name
        require(Path(name).name == name, 'unsafe preserved filename')
        if str(path) in pinned:
            require(pinned[str(path)]['sha256'] == digest, 'preserved pin mismatch')
        pinned[str(path)] = {'sha256': digest}
    catalogue = read(STUDY / 'catalogue/catalogue.json')
    raw = {topic: read(STUDY / 'catalogue' / f'{topic}.response.json') for topic in TOPICS}
    for group in catalogue['groups']:
        response_path = STUDY / 'catalogue' / f"{group['topic']}.response.json"
        require(group['response']['path'] == response_path.name and
                group['response']['sha256'] == pinned[str(response_path)]['sha256'],
                'compiled catalogue response binding')
    groups = catalogue_groups(catalogue, raw)
    expected_requests = {path for path in pinned if path.endswith('.request.json')}
    discovered = set()
    scanned_roots = []
    for directory in sorted((snap.root / 'output').iterdir()):
        if directory.is_dir() and ('jlens' in directory.name or 'j-lens' in directory.name):
            scanned_roots.append(str(directory))
            discovered.update(str(path.relative_to(snap.root)) for path in directory.rglob('*.request.json'))
    require(discovered == expected_requests, 'new/missing scoped request records require manual review')
    records, excluded, events = [], set(), {}
    for path in sorted(discovered):
        request = read(Path(path))
        params = request['params']
        require(request['endpoint'] == 'https://en.wikipedia.org/w/api.php', 'unexpected endpoint')
        require(request['status'] == 200 and request['body_truncated_at_cap'] is False,
                'historical request incomplete')
        response = request['response']
        require(Path(response['path']).name == response['path'], 'unsafe response path')
        response_path = Path(path).parent / response['path']
        require(pinned[str(response_path)]['sha256'] == response['sha256'], 'response pin conflict')
        response_bytes = read(response_path, False)  # Article response TEXT is never parsed.
        require(len(response_bytes) == response['size_bytes'], 'response size mismatch')
        record = {'path': str(snap.root / path), 'sha256': pinned[path]['sha256'],
                  'request': request, 'response_path': str(snap.root / response_path)}
        if params.get('list') == 'categorymembers':
            require('pageids' not in params and params['cmtitle'] in CATEGORIES
                    and params['cmnamespace'] == 0 and params['cmlimit'] == 500,
                    'category request contract')
            record['kind'] = 'catalogue_listing_not_article_exposure'
        else:
            pageid = params.get('pageids')
            require(type(pageid) is int and pageid > 0 and 'extracts' in params.get('prop', '').split('|'),
                    'unclassified request requires review')
            excluded.add(pageid)
            record.update(kind='article_extract_request', pageid=pageid)
            key = (pageid, request['started_utc'], request['completed_utc'], response['sha256'])
            events.setdefault(key, []).append(record)
            decision_path = Path(path.replace('.request.json', '.decision.json'))
            if str(decision_path) in pinned:
                decision = read(decision_path)
                require(decision['pageid'] == pageid and decision['response'] == response,
                        'decision/request mismatch')
                require(type(decision['selected']) is bool, 'invalid decision')
                record['decision'] = decision
                record['decision_path'] = str(snap.root / decision_path)
        records.append(record)
    for same in events.values():
        originals = [r for r in same if r['request'].get('new_network_request', True) is True]
        require(len(originals) == 1, 'duplicate event lacks unique original request')
        original = originals[0]
        for copy in same:
            if copy is original:
                continue
            require(copy['request'].get('new_network_request') is False and
                    copy['request'].get('reused_from'), 'unproven copied request')
            reused = (Path(copy['path']).parent / copy['request']['reused_from']).resolve()
            require(reused == Path(original['response_path']).resolve(), 'copied response target mismatch')
            require(copy['request']['params'] == original['request']['params'], 'copied request params drift')
            copy['same_network_event_as'] = original['path']
    attribution = read(STUDY / 'excerpts-resumed/attribution.json')
    dataset = read(STUDY / 'excerpts-resumed/dataset.json')
    articles = attribution['articles']
    require(len({a['pageid'] for a in articles}) == len(articles), 'duplicate attribution ID')
    require({a['pageid'] for a in articles} <= excluded, 'attribution article outside exclusion union')
    require(type(dataset) is list and
            [(a['id'], a['prefix']) for a in articles] == [(r['id'], r['prefix']) for r in dataset],
            'dataset/attribution join')
    owners, counts = ownership(groups, excluded)
    require(excluded <= {r['pageid'] for r in owners}, 'requested article outside original catalogue')
    return {'schema': 'jlens_natural_addon_inventory_v1',
            'catalogue': {'path': str(snap.root / STUDY / 'catalogue/catalogue.json'),
                          'sha256': pins['catalogue/catalogue.json']},
            'excluded_pageids': sorted(excluded), 'groups': groups, 'ownership': owners,
            'counts': counts, 'request_records': records,
            'article_request_record_count': sum(r['kind'] == 'article_extract_request' for r in records),
            'distinct_article_network_events': len(events),
            'selected_pageids': sorted({r['pageid'] for r in records
                                       if r.get('decision', {}).get('selected') is True}),
            'technical_skipped_pageids': sorted({r['pageid'] for r in records
                                                if r.get('decision', {}).get('selected') is False}),
            'raw_member_count': sum(c['raw_members'] for c in counts),
            'unique_member_count': len(owners), 'available_unique_count': sum(c['available_owned'] for c in counts),
            'attribution_screen': {'article_pageids': [a['pageid'] for a in articles],
                'article_urls': [a['url'] for a in articles], 'same_article_history_and_talk_checked': True,
                'evidence': str(snap.root / STUDY / 'attribution-review.md'),
                'additional_inspected_articles_identified': [],
                'limitation': 'Scoped persisted metadata and attribution note, not a complete browser transcript audit; linked notices explicitly not opened.'},
            'scanned_request_roots': scanned_roots, 'sources': snap.finish(),
            'candidate_pairing_performed': False}


def build(root, output, pins=PINS):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), 'inventory output already exists')
    source_sha = sha(Path(__file__).read_bytes())
    with output.open('x', encoding='utf-8') as handle:  # Failure preserves the exclusive empty attempt.
        result = inspect(root, pins)
        require(sha(Path(__file__).read_bytes()) == source_sha, 'builder source changed')
        result['builder'] = {'source_sha256': source_sha, 'python': platform.python_version()}
        json.dump(result, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write('\n')
    return sha(output.read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.root, args.output))
