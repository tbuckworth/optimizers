"""Independent saved-manifest arithmetic; no collector or planner imports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-pattern-calibration/inventory.json')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(ok, label):
    if not ok:
        raise ValueError(label)


def run():
    target = HERE/'selection-audit.json'
    check(not target.exists(), 'saved-manifest audit already recorded')
    expected = {
        HERE/'selection/manifest.json': '8287106e8e3238ccf18d0c60bc0a4b59de894c7a8fd038c6a870a2cd1e94ed63',
        HERE/'protocol.md': 'a3b47578fad5c1075e7b2855d01cffe954d5d0fb6b5c4d7fd2e51262a42f0d4e',
        INVENTORY: 'e57b194d6ed0b6b0999097d64c12dc37198f7b73d9b27bb5c2dd31e51a54f7aa',
    }
    for path, pin in expected.items():
        check(digest(path.read_bytes()) == pin, 'source binding: '+str(path))
    inv = json.loads(INVENTORY.read_bytes())
    manifest = json.loads((HERE/'selection/manifest.json').read_bytes())
    receipt = json.loads((HERE/'selection/receipt.json').read_bytes())
    topics = ['astronomy', 'cooking', 'football', 'programming']
    check(manifest['seed'] == '20260921', 'seed')
    expected_pairs, expected_tails = [], []
    for group in inv['groups']:
        topic = group['topic']
        ranked = sorted(group['remaining_pageids'], key=lambda pid:
                        (digest(('20260921|candidate|'+topic+'|'+str(pid)).encode('utf-8')), pid))
        for offset in range(0, len(ranked)-1, 2):
            left, right = ranked[offset:offset+2]
            order = digest(f'20260921|pair|{topic}|{left}|{right}'.encode('utf-8'))
            expected_pairs.append((order, topics.index(topic), left, right))
        if len(ranked) % 2:
            expected_tails.append((topic, ranked[-1]))
    expected_pairs.sort()
    pairs = manifest['candidates']
    actual_pairs = [(p['order_hash'], topics.index(p['topic']), p['left_pageid'], p['right_pageid']) for p in pairs]
    check(actual_pairs == expected_pairs and len(pairs) == 86, 'complete adjacent/global order')
    check([(p['topic'], p['pageid']) for p in manifest['odd_tails']] == expected_tails, 'odd tails')
    used_ids = [p[side+'_pageid'] for p in pairs for side in ['left', 'right']]
    check(len(used_ids) == len(set(used_ids)) == 172, 'disjoint candidate articles')
    check(not set(used_ids) & set(inv['excluded_pageids']), 'old requests excluded')
    for index, pair in enumerate(pairs):
        check(pair['candidate_id'] == f'K{index+1:03d}', 'candidate ID')
        check(pair['role'] == ('evaluation' if index % 3 == 2 else 'calibration'), 'fixed role')
    roles = dict(Counter(p['role'] for p in pairs))
    check(roles == manifest['candidate_role_counts'] == receipt['role_counts'] == {'calibration': 58, 'evaluation': 28}, 'role capacity')
    check(manifest['targets'] == {'calibration_pairs': 32, 'evaluation_pairs': 16}, 'targets')
    check(manifest['maximum_requested_pairs'] == 80 and manifest['maximum_article_requests'] == 160, 'request ceilings')
    check(manifest['eligibility_or_measurement_performed'] is False, 'metadata scope')
    check(receipt['status'] == 'METADATA_SELECTION_COMPLETE' and receipt['network_calls'] == 0, 'stage status')
    for field, path in [('manifest_sha256', HERE/'selection/manifest.json'),
                        ('protocol_sha256', HERE/'protocol.md'), ('inventory_sha256', INVENTORY),
                        ('source_sha256', HERE/'plan_candidates.py')]:
        check(receipt[field] == digest(path.read_bytes()), 'receipt '+field)
    result = {'status': 'PASS', 'scope': 'independent saved metadata arithmetic, not selection replay or article eligibility',
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': digest(Path(__file__).read_bytes()),
              'inputs': [{'path': str(p), 'sha256': h} for p, h in expected.items()],
              'candidate_pairs': 86, 'unique_paired_articles': 172, 'unused_tails': expected_tails,
              'role_counts': roles, 'candidate_topic_counts': dict(Counter(p['topic'] for p in pairs)),
              'network_calls': 0, 'model_or_tokenizer_calls': 0}
    with target.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': 'PASS', 'candidate_pairs': 86, 'role_counts': roles}))


if __name__ == '__main__':
    run()
