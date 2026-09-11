"""Saved-token structural and source-binding check; never initializes a tokenizer."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-pattern-calibration')
RECEIPT_SHA = 'd4cba9b4376098720a9fa8111f7f7d7dcf7f634e9e66413cb591860ef0c25ec4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(ok, label):
    if not ok:
        raise ValueError(label)


def run():
    output = HERE/'preflight-audit.json'
    check(not output.exists(), 'saved preflight audit already recorded')
    stage = WORK/'token-preflight'
    check(sha(stage/'receipt.json') == RECEIPT_SHA, 'receipt bytes')
    r = read(stage/'receipt.json')
    check(r['schema'] == 'jlens_pattern_calibration_token_preflight_receipt_v1' and r['status'] == 'complete', 'complete scope')
    check(r['count'] == 96 and r['role_counts'] == {'calibration': 64, 'evaluation': 32}
          and r['token_limit'] == 96 and r['capture_locations'] == ['prefix_end'], 'measurement plan')
    check(r['model_loaded'] is False and r['scientific_array_loaded'] is False and r['reference_decodes'] == 0, 'no neural stage')
    check(r['source_sha256'] == sha(WORK/'preflight.py') == 'f484e4b022b835f4f38d1e3ffdce4722f5a1694e50297bf05d3db7e94f100a3d', 'source')
    for path, digest in r['input_pins'].items():
        check(sha(Path(path)) == digest, 'input '+path)
    check({p.name for p in stage.iterdir()} == {'attempt.json', 'receipt.json', 'calibration-tokens.json', 'evaluation-tokens.json'}, 'stage file roster')
    check({p['path'] for p in r['outputs']} == {'calibration-tokens.json', 'evaluation-tokens.json'}, 'role outputs')
    for item in r['outputs']:
        p = stage/item['path']
        check(sha(p) == item['sha256'] and p.stat().st_size == item['size_bytes'], 'output binding')
    counts, all_ids = {}, set()
    for role, size in r['role_counts'].items():
        payload = read(stage/(role+'-tokens.json'))
        rows = read(WORK/'excerpts'/(role+'-dataset.json'))
        check(payload['schema'] == 'jlens_pattern_calibration_tokens_v1' and payload['role'] == role
              and len(payload['records']) == len(rows) == size, 'role schema')
        lengths = []
        for row, record in zip(rows, payload['records'], strict=True):
            check(all(record[k] == v for k, v in row.items()) and row['id'] not in all_ids, 'row join')
            all_ids.add(row['id'])
            ids, mask, offsets = record['input_ids'], record['attention_mask'], record['offset_mapping']
            n, text = len(ids), row['prefix']; lengths.append(n)
            check(1 <= n <= 96 and all(type(v) is int and 0 <= v < 248320 for v in ids), 'token ID/count')
            check(len(mask) == n and all(type(v) is int and v == 1 for v in mask), 'all-valid mask')
            check(len(offsets) == n and all(len(p) == 2 and all(type(v) is int for v in p)
                  and 0 <= p[0] < p[1] <= len(text) for p in offsets), 'offset bounds')
            check(offsets[0][0] == 0 and offsets[-1][1] == len(text), 'full endpoint')
            covered, previous_start = 0, -1
            for start, end in offsets:
                check(start >= previous_start and end >= covered, 'offset ordering')
                check(start <= covered or text[covered:start].isspace(), 'uncovered nonspace')
                covered, previous_start = end, start
            check(record['layer'] == 11 and record['captured_positions'] == {'prefix_end': n-1}
                  and record['selected_substrings'] == {'prefix_end': text[slice(*offsets[-1])]}, 'final-subtoken role')
        counts[role] = {'rows': size, 'minimum_tokens': min(lengths), 'maximum_tokens': max(lengths)}
    check(len(all_ids) == 96, 'all disjoint IDs')
    check(r['runtime']['python'] == r['host']['python'] == '3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]', 'actual current build')
    check(r['runtime']['packages'] == {'numpy': '1.26.4', 'torch': '2.11.0+cu128',
          'transformers': '5.5.0', 'huggingface_hub': '1.8.0'}, 'actual numerical versions')
    for path, digest in r['host']['binary_sha256s'].items():
        check(sha(Path(path)) == digest, 'runtime binary unchanged')
    result = {'status': 'PASS', 'scope': 'saved token/offset/source checks, not retokenization or model measurement',
        'receipt_sha256': RECEIPT_SHA, 'audit_source_sha256': sha(Path(__file__)),
        'roles': counts, 'elapsed_seconds': r['elapsed_seconds'], 'completed_utc': r['completed_utc'],
        'new_tokenizer_model_or_network_calls': 0}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
