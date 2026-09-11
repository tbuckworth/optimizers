"""Independent saved-packet/identity audit; no producer, neural key or arrays."""
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
MANIFEST_SHA = 'b4315514f62b158957aecbadce42b4689024c0eed9d51341433b36755af5ea26'


def check(ok, label):
    if not ok: raise ValueError(label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def run():
    out = HERE/'packet-audit.json'
    check(not out.exists(), 'audit already recorded')
    stage = HERE/'packets'
    check(sha(stage/'manifest.json') == MANIFEST_SHA, 'manifest identity')
    manifest = read(stage/'manifest.json')
    check(manifest['schema'] == 'jlens_pattern_calibration_packets_v1' and manifest['seed'] == 20260922
          and manifest['score_key_or_array_files_opened'] == 0, 'packet scope')
    check(manifest['source_sha256'] == sha(HERE/'reader_packets.py') ==
          'daf506324f9da867a75d10046a3a11d8ec17b960f61bec585d3371d4cc0413a3', 'source identity')
    allowed = {'protocol', 'references', 'calibration_receipt', 'dataset', 'pairs', 'preflight', 'random_source'}
    check(set(manifest['inputs']) == allowed, 'key-free input labels')
    for name, item in manifest['inputs'].items():
        path = Path(item['path'])
        check(path.suffix in ('.md', '.json', '.py') and path.name not in ('scores.json', 'gaps.json'), 'no key path')
        check(sha(path) == item['sha256'], 'input hash '+name)
    check({p.name for p in stage.iterdir()} == {'attempt.json', 'manifest.json', 'private-map.json',
          'rater1.json', 'rater2.json', 'rater3.json', 'rater4.json'}, 'exact saved roster')
    check(len(manifest['outputs']) == 5, 'output count')
    for item in manifest['outputs']:
        path = stage/item['path']
        check(sha(path) == item['sha256'] and path.stat().st_size == item['size_bytes'], 'output hash/size')
    refs = read(Path(manifest['inputs']['references']['path']))
    rows = read(Path(manifest['inputs']['dataset']['path']))
    pairs = read(Path(manifest['inputs']['pairs']['path']))
    text = {r['id']: r['prefix'] for r in rows}
    check(len(text) == 32 and len(pairs) == 16, 'panel count')
    mapping = read(stage/'private-map.json')
    check(mapping['schema'] == 'jlens_pattern_calibration_private_map_v1' and len(mapping['rows']) == 256, 'map schema/count')
    private = mapping['rows']
    check(len({r['item_id'] for r in private}) == 256 and len({r['block_id'] for r in private}) == 16, 'global unique IDs')
    axes = [f'PC{i}' for i in range(1, 5)]
    rng = random.Random(20260922)
    bits = {(a, p['id']): rng.choice((False, True)) for a in axes for p in pairs}
    bids, qids = list(range(1, 17)), list(range(1, 257))
    rng.shuffle(bids); rng.shuffle(qids)
    axis_order = axes.copy(); rng.shuffle(axis_order)
    orders = {}
    for a in axes:
        orders[a] = pairs.copy(); rng.shuffle(orders[a])
    all_prompts, identities, count = [], [], 0
    for ri in range(4):
        rater = f'rater{ri+1}'; arm = 'P' if ri % 2 == 0 else 'U'; cohort = 1 if ri < 2 else 2
        packet = read(stage/(rater+'.json'))
        check(set(packet) == {'schema', 'packet_id', 'prompt', 'blocks'} and
              packet['schema'] == 'jlens_pattern_calibration_public_v1', 'public fields')
        all_prompts.append(packet['prompt'])
        selected = [r for r in private if r['rater'] == rater]
        check(len(selected) == 64 and len(packet['blocks']) == 4, 'rater counts')
        cursor = 0
        for a, block in zip(axis_order, packet['blocks'], strict=True):
            check(set(block) == {'block_id', 'positive_reference', 'negative_reference', 'comparisons'}, 'anonymous block')
            check(block['block_id'] == f'B{bids.pop():03}', 'block draw sequence')
            expected_ref = refs['arms'][arm]['axes'][axes.index(a)]
            for pole in ('positive_reference', 'negative_reference'):
                check(block[pole] == expected_ref[pole], 'all reference strings unchanged')
            check(len(block['comparisons']) == 16, 'all pair cases')
            for pair, public in zip(orders[a], block['comparisons'], strict=True):
                meta = selected[cursor]; cursor += 1
                first, second = pair['left'], pair['right']
                if bits[a, pair['id']] != (cohort == 2): first, second = second, first
                ident = f'Q{qids.pop():03}'
                check(public == {'item_id': ident, 'first': text[first], 'second': text[second]}, 'exact public text/draw order')
                check(meta == {'item_id': ident, 'block_id': block['block_id'], 'rater': rater, 'axis': a,
                      'arm': arm, 'cohort': cohort, 'pair_id': pair['id'], 'topic': pair['topic'],
                      'candidate_id': pair['candidate_id'], 'first_id': first, 'second_id': second}, 'exact private binding')
                identities.append((rater, a, pair['id'], first, second)); count += 1
        check(packet['packet_id'] == f'R{rng.getrandbits(64):016x}', 'packet ID draw sequence')
    check(all(p == all_prompts[0] for p in all_prompts) and count == 256 and not bids and not qids, 'common task and complete random stream')
    by_case = {(r, a, p): (f, s) for r, a, p, f, s in identities}
    for a in axes:
        for p in pairs:
            c1 = by_case['rater1', a, p['id']]; c2 = by_case['rater3', a, p['id']]
            check(c1 == by_case['rater2', a, p['id']] and c2 == by_case['rater4', a, p['id']]
                  and c1 == c2[::-1], 'paired/opposite-order design')
    result = {'status': 'PASS', 'manifest_sha256': MANIFEST_SHA, 'audit_source_sha256': sha(Path(__file__)),
        'public_packets': 4, 'unique_items': 256, 'unique_blocks': 16,
        'exact_random_sequence_and_reference_texts': True, 'evaluation_key_files_opened': 0,
        'reader_calls': 0, 'scope': 'saved public metadata checks, not new packet generation or judging'}
    with out.open('x') as handle: json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': run()
