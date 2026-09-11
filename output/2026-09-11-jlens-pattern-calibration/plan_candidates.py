"""Metadata-only prospective pair/role manifest; never requests article content."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKER = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree')
INVENTORY = WORKER/'output/2026-09-11-j-lens-pattern-calibration/inventory.json'
INVENTORY_SHA = 'e57b194d6ed0b6b0999097d64c12dc37198f7b73d9b27bb5c2dd31e51a54f7aa'
TOPICS = ['astronomy', 'cooking', 'football', 'programming']
SEED = '20260921'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def strict_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value')
    # These metadata files contain no floating-point scientific values.
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid, parse_float=invalid)


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def validate_sources(inventory):
    sources = inventory['sources']
    require(len(sources) == len({s['path'] for s in sources}) == 146, 'source inventory changed')
    for record in sources:
        p = Path(record['path'])
        require(p.is_absolute() and (p.is_relative_to(ROOT) or p.is_relative_to(WORKER)),
                'unexpected metadata source root')
        require(p.is_file() and not p.is_symlink(), 'metadata source absent/symlink')
        require(p.stat().st_size == record['size_bytes'] <= 4_000_000, 'metadata size changed')
        require(sha(p) == record['sha256'], 'metadata content changed: '+str(p))


def available_from_raw(catalogue, requests):
    require([g['topic'] for g in catalogue['groups']] == TOPICS, 'owner ordering changed')
    excluded = set()
    for request in requests:
        value = request['params']['pageids']
        require(type(value) is int and value > 0, 'invalid old article request ID')
        excluded.add(value)
    owners, groups = {}, []
    for group in catalogue['groups']:
        rows = []
        for row in group['candidates']:
            pid = row['pageid']
            require(type(pid) is int and pid > 0 and row['ns'] == 0, 'invalid catalogue ID')
            if pid not in owners:
                owners[pid] = group['topic']
                if pid not in excluded:
                    rows.append(pid)
        groups.append({'topic': group['topic'], 'remaining_pageids': sorted(rows)})
    return groups, sorted(excluded)


def ordered_pairs(groups):
    require([g['topic'] for g in groups] == TOPICS, 'topic order')
    all_ids = [pid for g in groups for pid in g['remaining_pageids']]
    require(all(type(pid) is int and pid > 0 for pid in all_ids), 'bad remaining ID')
    require(len(all_ids) == len(set(all_ids)), 'duplicate ownership')
    pairs, tails = [], []
    for group in groups:
        topic = group['topic']
        ordered = sorted(group['remaining_pageids'], key=lambda pid: (digest(f'{SEED}|candidate|{topic}|{pid}'.encode()), pid))
        if len(ordered) % 2:
            tails.append({'topic': topic, 'pageid': ordered[-1], 'reason': 'odd unpaired tail; never a replacement'})
        for i in range(0, len(ordered)-1, 2):
            left, right = ordered[i:i+2]
            pairs.append({'topic': topic, 'left_pageid': left, 'right_pageid': right,
                          'order_hash': digest(f'{SEED}|pair|{topic}|{left}|{right}'.encode())})
    pairs.sort(key=lambda p: (p['order_hash'], TOPICS.index(p['topic']), p['left_pageid'], p['right_pageid']))
    for i, pair in enumerate(pairs):
        pair['candidate_id'] = f'K{i+1:03d}'
        pair['role'] = 'evaluation' if i % 3 == 2 else 'calibration'
    return pairs, tails


def inspect_inventory():
    require(sha(INVENTORY) == INVENTORY_SHA, 'inventory pin changed')
    inventory = strict_json(INVENTORY.read_bytes())
    require(inventory['metadata_only'] and not inventory['actual_selection_performed'] and inventory['network_calls'] == 0,
            'unexpected prior task scope')
    validate_sources(inventory)
    records = inventory['requests']
    require(len(records) == len({r['path'] for r in records}) == 63, 'article record roster')
    discovered = set()
    for root in [ROOT, WORKER]:
        for study in (root/'output').iterdir():
            if study.is_dir() and ('jlens' in study.name or 'j-lens' in study.name):
                discovered.update(str(p) for p in study.rglob('*.request.json'))
    require(discovered == {r['path'] for r in records} | set(inventory['catalogue_request_paths']),
            'new/unrecognized scoped request records; reconcile before selection')
    requests = []
    for record in records:
        p = Path(record['path'])
        require(sha(p) == record['sha256'], 'request content drift')
        request = strict_json(p.read_bytes())
        require(request['params']['pageids'] == record['pageid'], 'request ID differs from inventory')
        require(request.get('new_network_request', True) == record['new_network_request'], 'copy/event differs')
        requests.append(request)
    catalogue = strict_json(Path(inventory['catalogue']['path']).read_bytes())
    groups, excluded = available_from_raw(catalogue, requests)
    require(excluded == inventory['excluded_pageids'] and len(excluded) == 62, 'exclusion union')
    for actual, prior in zip(groups, inventory['groups'], strict=True):
        require(actual['topic'] == prior['topic'] and actual['remaining_pageids'] == prior['remaining_pageids'],
                'independent remaining-ID reconstruction differs')
    require([len(g['remaining_pageids']) for g in groups] == [40, 26, 0, 107], 'capacity counts')
    validate_sources(inventory)
    return inventory, groups


def run(protocol_sha):
    target = HERE/'selection'
    require(not target.exists() and not target.is_symlink(), 'selection stage consumed')
    target.mkdir()
    write(target/'attempt.json', {'utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
                                 'protocol_sha256': protocol_sha, 'network_calls': 0, 'automatic_retry': False})
    try:
        require(sha(HERE/'protocol.md') == protocol_sha, 'protocol changed')
        inventory, groups = inspect_inventory()
        pairs, tails = ordered_pairs(groups)
        counts = dict(Counter(p['role'] for p in pairs))
        require(len(pairs) == 86 and counts == {'calibration': 58, 'evaluation': 28} and len(tails) == 1,
                'manifest capacity mismatch')
        manifest = {'schema': 'jlens_pattern_calibration_candidate_manifest_v1',
                    'inventory_sha256': INVENTORY_SHA, 'protocol_sha256': protocol_sha,
                    'source_sha256': sha(Path(__file__)), 'seed': SEED,
                    'targets': {'calibration_pairs': 32, 'evaluation_pairs': 16},
                    'maximum_requested_pairs': 80, 'maximum_article_requests': 160,
                    'candidate_role_counts': counts, 'candidates': pairs, 'odd_tails': tails,
                    'eligibility_or_measurement_performed': False}
        validate_sources(inventory)
        require(sha(INVENTORY) == INVENTORY_SHA and sha(HERE/'protocol.md') == protocol_sha, 'input drift')
        write(target/'manifest.json', manifest)
        write(target/'receipt.json', {'status': 'METADATA_SELECTION_COMPLETE', 'network_calls': 0,
              'manifest_sha256': sha(target/'manifest.json'), 'inventory_sha256': INVENTORY_SHA,
              'protocol_sha256': protocol_sha, 'source_sha256': sha(Path(__file__)),
              'verified_source_count': len(inventory['sources']), 'candidate_pairs': len(pairs), 'role_counts': counts})
        print(json.dumps({'status': 'METADATA_SELECTION_COMPLETE', 'manifest_sha256': sha(target/'manifest.json'), 'role_counts': counts}))
    except Exception as error:
        write(target/'failure.json', {'status': 'FAILED_WITHOUT_ACQUISITION', 'error_type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True)
    run(parser.parse_args().protocol_sha)
