"""Inert whole-96-prefix tokenizer preflight; no model, lens, arrays or old stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import sys
import time
import types

OUT = Path(__file__).resolve().parent
MAIN = Path('output/2026-09-11-jlens-pattern-calibration')
OLD = OUT.parent/'2026-09-11-j-lens-natural-addon'
BASE = OLD/'forward.py'
BASE_SHA = 'd2557036a3ebdd6ecbc61be89adf3582f16f084fda091e86f3faabe9ff39f1d0'
HOST_HELPER = OLD/'forward_runtime_amendment.py'
HOST_HELPER_SHA = '894f465e36d50f6df15229754367614d3ffc26f20301db3eb57f9535189bbc57'
PROTOCOL_SHA = 'a3b47578fad5c1075e7b2855d01cffe954d5d0fb6b5c4d7fd2e51262a42f0d4e'
MANIFEST_SHA = '8287106e8e3238ccf18d0c60bc0a4b59de894c7a8fd038c6a870a2cd1e94ed63'
COLLECTOR_SHA = '4001f31ab9d592f0e1f5a230d79e5f493a64fa9ee8d074d2a03719c1534550b1'
# Root's terminal, independently audited collection; no tokenizer run yet.
COLLECTION_SHA = 'cf86bce1a96099939434fca3c391568db3d802786aed05f04d66f6ad76d82776'
COLLECTION_OUTPUTS = {
    'calibration-dataset.json': '249f6bb36e0247a9c18388955ed932773028cc8395aaba15234011b25056d971',
    'calibration-pairs.json': '082a23f6130a5322fe6cd960bd7250ec9f547c384e718995b8501ab3b09a3ce5',
    'evaluation-dataset.json': '54e7e1051f8ec7edee336f5049426bb9d6af6eab17ce85d44b0f40c904952d5d',
    'evaluation-pairs.json': '8bc628c8a0bc8972ad6f18d67fb955a0465eba09e4faeea877612d0e31d90573',
    'attribution.json': 'd1c02cec4cf815b247cb856d99600995d795abc364913ca338a15d92e5c997ee',
    'candidate-dispositions.json': 'c5aac90dd2e388faa86ceb82124567d104d0a0effe370b09d6dc0d52010bd7fd'}
ROLE_COUNTS = {'calibration': 64, 'evaluation': 32}
CACHE_NAMES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')
LOCATIONS = ['prefix_end']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def import_pinned(path, digest):
    require(not path.is_symlink() and path.is_file(), 'helper source path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'helper source pin')
    module = types.ModuleType('pinned_'+path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(sha(path) == digest, 'helper changed during import')
    return module


def frozen_pins(f):
    require(all(type(d) is str and re.fullmatch('[0-9a-f]{64}', d)
                for d in [COLLECTION_SHA, *COLLECTION_OUTPUTS.values()]), 'audited collection pins required')
    return {BASE: BASE_SHA, HOST_HELPER: HOST_HELPER_SHA,
        MAIN/'protocol.md': PROTOCOL_SHA, MAIN/'selection/manifest.json': MANIFEST_SHA,
        OUT/'collect.py': COLLECTOR_SHA, OUT/'excerpts/receipt.json': COLLECTION_SHA,
        **{OUT/'excerpts'/name: digest for name, digest in COLLECTION_OUTPUTS.items()},
        **{f.SNAPSHOT/name: f.PINS[f.SNAPSHOT/name] for name in CACHE_NAMES}}


def verify(pins):
    for path, digest in pins.items():
        require(sha(path) == digest, 'input pin mismatch: '+str(path))


def validate_panel(datasets, pairsets, manifest, articles, dispositions):
    require(manifest['schema'] == 'jlens_pattern_calibration_candidate_manifest_v1' and
            manifest['protocol_sha256'] == PROTOCOL_SHA and manifest['seed'] == '20260921', 'manifest scope')
    candidates = manifest['candidates']
    require(len(candidates) == 86 and [p['candidate_id'] for p in candidates] == [f'K{i:03}' for i in range(1, 87)],
            'frozen candidate roster')
    require(all(p['role'] == ('evaluation' if i % 3 == 2 else 'calibration') for i, p in enumerate(candidates)),
            'immutable candidate roles')
    by_candidate = {p['candidate_id']: p for p in candidates}
    require(type(dispositions) is list and len(dispositions) == 86 and
            [d['candidate_id'] for d in dispositions] == list(by_candidate), 'complete disposition roster')
    accepted = {d['accepted_pair_id']: d for d in dispositions if d['status'] == 'accepted'}
    require(len(accepted) == sum(d['status'] == 'accepted' for d in dispositions) == 48, 'accepted pair identities')
    require(type(articles) is list and len(articles) == 96 and len({a['id'] for a in articles}) == 96,
            'exact attribution roster')
    by_id = {a['id']: a for a in articles}
    prefixes, pageids, used_candidates, all_ids = set(), set(), set(), set()
    for role, count in ROLE_COUNTS.items():
        rows, pairs = datasets[role], pairsets[role]
        letter = 'C' if role == 'calibration' else 'T'
        ids = [f'{letter}{i:02}-{side}' for i in range(1, count//2+1) for side in ('L', 'R')]
        require(type(rows) is list and len(rows) == count and [r['id'] for r in rows] == ids, 'role row count/order')
        require(type(pairs) is list and len(pairs) == count//2, 'role pair count')
        ranks = []
        for row in rows:
            require(set(row) == {'id', 'topic', 'prefix'} and row['topic'] in ('astronomy', 'cooking', 'football', 'programming'), 'row fields/topic')
            text = row['prefix']
            require(type(text) is str and len(text.split()) == 16 and text == ' '.join(text.split()), 'exact sixteen words')
            require(text not in prefixes, 'cross-role or within-role duplicate prefix')
            prefixes.add(text); all_ids.add(row['id'])
        for i, pair in enumerate(pairs):
            require(set(pair) == {'id', 'left', 'right', 'topic', 'candidate_id', 'role'} and
                    pair['id'] == f'{letter}{i+1:02}' and pair['role'] == role and
                    pair['left'] == ids[2*i] and pair['right'] == ids[2*i+1], 'fixed pair fields/role/joins')
            candidate = by_candidate[pair['candidate_id']]
            decision = accepted[pair['id']]
            require(candidate['role'] == decision['role'] == role and
                    candidate['topic'] == decision['topic'] == pair['topic'] == rows[2*i]['topic'] == rows[2*i+1]['topic'] and
                    candidate['candidate_id'] == decision['candidate_id'] and pair['candidate_id'] not in used_candidates,
                    'candidate role/topic/acceptance binding')
            used_candidates.add(pair['candidate_id']); ranks.append(int(pair['candidate_id'][1:]))
            for j, side in enumerate(('left', 'right')):
                row = rows[2*i+j]; article = by_id[row['id']]; pid = candidate[side+'_pageid']
                require(type(pid) is int and pid > 0 and pid not in pageids and article['pageid'] == pid and
                        article['prefix'] == row['prefix'] and article['role'] == role and
                        article['pair_id'] == pair['id'] and article['candidate_id'] == pair['candidate_id'],
                        'exact article/prefix split identity')
                pageids.add(pid)
        require(ranks == sorted(set(ranks)), 'within-role acceptance order')
    require(len(pageids) == len(prefixes) == len(all_ids) == 96 and set(by_id) == all_ids, 'disjoint whole panel')


def load_panel(f):
    receipt = f.read_json(OUT/'excerpts/receipt.json')
    require(receipt['schema'] == 'jlens_pattern_calibration_excerpts_v1' and receipt['status'] == 'complete' and
            receipt['source_sha256'] == COLLECTOR_SHA and receipt['protocol_sha256'] == PROTOCOL_SHA and
            receipt['manifest_sha256'] == MANIFEST_SHA and receipt['selected_count'] == 96 and
            receipt['accepted_pairs'] == {'calibration': 32, 'evaluation': 16} and
            48 <= receipt['requested_pairs'] <= 80 and receipt['article_requests'] == 2*receipt['requested_pairs'] and
            receipt['candidate_dispositions'] == 86 and receipt['tokenizer_or_model_called'] is False, 'collection receipt scope')
    outputs = receipt['outputs']
    require(len(outputs) == len({p['path'] for p in outputs}), 'duplicate receipt artifact')
    by_name = {p['path']: p for p in outputs}
    for name, digest in COLLECTION_OUTPUTS.items():
        require(by_name[name]['sha256'] == digest and by_name[name]['size_bytes'] == (OUT/'excerpts'/name).stat().st_size,
                'collection output pin/size binding')
    datasets = {r: f.read_json(OUT/'excerpts'/f'{r}-dataset.json') for r in ROLE_COUNTS}
    pairsets = {r: f.read_json(OUT/'excerpts'/f'{r}-pairs.json') for r in ROLE_COUNTS}
    manifest = f.read_json(MAIN/'selection/manifest.json')
    attribution = f.read_json(OUT/'excerpts/attribution.json')
    require(attribution['schema'] == 'jlens_pattern_calibration_attribution_v1', 'attribution schema')
    dispositions = f.read_json(OUT/'excerpts/candidate-dispositions.json')
    validate_panel(datasets, pairsets, manifest, attribution['articles'], dispositions)
    return datasets, pairsets


def validate_runtime(env, host, a):
    require(env['python'] == host['python'] == a.NEW_PYTHON, 'current approved Python build required')
    require(env['interpreter'] == host['interpreter'] == '/usr/bin/python3' and
            host['resolved_interpreter'] == '/usr/bin/python3.12', 'fixed interpreter path')
    require(env['packages'] == a.NUMERICAL and host['package_versions'] == a.PACKAGES and
            host['binary_sha256s'] == a.BINARIES, 'numerical/distro/runtime binary drift')


def load_tokenizer(f):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f.MODEL, revision=f.REV, cache_dir=str(f.HUB),
        local_files_only=True, trust_remote_code=False, use_fast=True, token=False)
    require(tokenizer.is_fast and tokenizer.bos_token_id is None, 'tokenizer/BOS configuration changed')
    return tokenizer


def run(source_sha, root=OUT):
    require(os.environ.get('JLENS_PATTERN_CALIBRATION_PREFLIGHT_RELEASE') == '1', 'root preflight admission required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1'
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'CPU-only single-thread envelope')
    require(type(source_sha) is str and re.fullmatch('[0-9a-f]{64}', source_sha) and sha(Path(__file__)) == source_sha,
            'reviewed preflight source hash required')
    target = root/'token-preflight'
    if target.exists() or target.is_symlink(): raise FileExistsError('preflight stage already consumed')
    target.mkdir(); started = time.monotonic()
    # Source-only helpers do not initialize the tokenizer, numerical runtime or old stages.
    import json
    def write(name, value):
        with (target/name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2); handle.write('\n')
    write('attempt.json', {'schema': 'jlens_pattern_calibration_token_preflight_attempt_v1',
        'source_sha256': source_sha, 'started_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False})
    try:
        f, a = import_pinned(BASE, BASE_SHA), import_pinned(HOST_HELPER, HOST_HELPER_SHA)
        expected = frozen_pins(f); verify(expected)
        datasets, pairsets = load_panel(f)
        env = f.runtime(); host = a.host_identity(); validate_runtime(env, host, a)
        tokenizer = load_tokenizer(f)
        records = {role: f.tokenize(rows, tokenizer) for role, rows in datasets.items()}
        for role in ROLE_COUNTS:
            require(len(records[role]) == ROLE_COUNTS[role], 'whole role token count')
            for record, row in zip(records[role], datasets[role], strict=True):
                require(record == f.token_record(row, record), 'exact saved token record')
        verify(expected)
        require(load_panel(f) == (datasets, pairsets), 'collection input drift')
        final_host = a.host_identity(); validate_runtime(env, final_host, a)
        require(final_host == host and {n: sys.modules[n].__version__ for n in a.NUMERICAL} == a.NUMERICAL,
                'runtime changed during preflight')
        require(sha(Path(__file__)) == source_sha, 'source changed during preflight')
        outputs = []
        for role in ROLE_COUNTS:
            name = role+'-tokens.json'
            write(name, {'schema': 'jlens_pattern_calibration_tokens_v1', 'role': role, 'records': records[role]})
            outputs.append({'path': name, 'sha256': sha(target/name), 'size_bytes': (target/name).stat().st_size})
        write('receipt.json', {'schema': 'jlens_pattern_calibration_token_preflight_receipt_v1', 'status': 'complete',
            'source_sha256': source_sha, 'base_helper_sha256': BASE_SHA, 'host_helper_sha256': HOST_HELPER_SHA,
            'input_pins': {str(p): s for p, s in expected.items()}, 'outputs': outputs, 'runtime': env, 'host': host,
            'count': 96, 'role_counts': ROLE_COUNTS, 'token_limit': 96, 'capture_locations': LOCATIONS,
            'model_loaded': False, 'scientific_array_loaded': False, 'reference_decodes': 0,
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started})
    except Exception as error:
        write('failure.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'reason': str(error)[:500],
            'automatic_retry': False, 'model_loaded': False, 'scientific_array_loaded': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    run(parser.parse_args().source_sha)
