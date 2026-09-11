"""Inert isolated 32-state evaluation; its key stays unopened until reader lock."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MAIN = Path('output/2026-09-11-jlens-pattern-calibration')
HELPER_SHA = 'c46990f72512baa3e86d261b668de53d194ea737b650f84674db00d2422868e5'
REFERENCE_COMMIT = 'ef8168b6ba441c5e3a594df66d872b159387f2f1'
CALIBRATION_RECEIPT_SHA = 'd4ab847e8239093536f2124e40507cb2b2f56a431aeb2eb1006278a7532cbf15'
REFERENCE_SHA = '42508c7272e108a8103596eeedadf836cc7803475cd64918c3e6fb1e24ce0296'
EVALUATION_TOKENS_SHA = '49936ea138f743b5b619573b6623bc9f2df81886817449b9dddbe6f772e9ea25'  # gitleaks:allow -- independently verified committed token-record file SHA256, not a credential
IDS = [f'T{i:02}-{side}' for i in range(1, 17) for side in ('L', 'R')]
LOCATIONS = ['prefix_end']


def require(ok, message):
    if not ok: raise ValueError(message)


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2); handle.write('\n')


def import_helper():
    path = OUT/'calibrate.py'
    require(path.is_file() and not path.is_symlink(), 'calibration helper source path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == HELPER_SHA, 'calibration helper source pin')
    module = types.ModuleType('pinned_calibration_helper'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(module.sha(path) == HELPER_SHA, 'helper source drift')
    return module


def input_pins(c, p, f):
    # No calibration features, moments, tokens, dataset or lens checkpoint read.
    return {OUT/'calibrate.py': HELPER_SHA, OUT/'preflight.py': c.PREFLIGHT_SOURCE_SHA,
        p.BASE: p.BASE_SHA, p.HOST_HELPER: p.HOST_HELPER_SHA,
        MAIN/'protocol.md': p.PROTOCOL_SHA, MAIN/'selection/manifest.json': p.MANIFEST_SHA,
        OUT/'excerpts/receipt.json': p.COLLECTION_SHA,
        **{OUT/'excerpts'/n: p.COLLECTION_OUTPUTS[n] for n in ('evaluation-dataset.json', 'evaluation-pairs.json')},
        OUT/'token-preflight/receipt.json': c.PREFLIGHT_RECEIPT_SHA,
        OUT/'token-preflight/evaluation-tokens.json': EVALUATION_TOKENS_SHA,
        OUT/'calibration/receipt.json': CALIBRATION_RECEIPT_SHA,
        OUT/'calibration/references.json': REFERENCE_SHA,
        OUT/'calibration/original-interpretations.json': c.ORIGINAL_SHA,
        c.DIRECTIONS: c.DIRECTIONS_SHA,
        **{f.SNAPSHOT/n: f.PINS[f.SNAPSHOT/n] for n in (*p.CACHE_NAMES, 'model.safetensors.index.json')},
        f.SNAPSHOT/'model.safetensors-00001-of-00001.safetensors': f.WEIGHT_SHA,
        **{f.UPSTREAM/n: d for n, d in c.ADAPTER_PINS.items()}, c.QWEN_SOURCE: c.QWEN_SOURCE_SHA}


def verify(c, expected):
    forbidden = {'calibration-dataset.json', 'calibration-pairs.json', 'calibration-tokens.json',
                 'features.npz', 'patterns.npz', 'decoder.npz'}
    require(not any(path.name in forbidden or path.suffix == '.pt' for path in expected), 'forbidden evaluation input')
    for path, pin in expected.items(): require(c.sha(path) == pin, 'input hash mismatch: '+str(path))


def committed_references(c, f):
    # Commit blob reads bind prior references without opening calibration arrays.
    subprocess.run(['git', 'merge-base', '--is-ancestor', REFERENCE_COMMIT, 'HEAD'],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    pins = {'receipt.json': CALIBRATION_RECEIPT_SHA, 'references.json': REFERENCE_SHA,
            'original-interpretations.json': c.ORIGINAL_SHA}
    for name, pin in pins.items():
        path = OUT/'calibration'/name
        blob = subprocess.check_output(['git', 'show', REFERENCE_COMMIT+':'+str(path.relative_to(ROOT))],
                                       cwd=ROOT, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == pin and c.sha(path) == pin, 'reference commit/file binding')
    receipt = f.read_json(OUT/'calibration/receipt.json')
    require(receipt['schema'] == 'jlens_pattern_calibration_calibration_receipt_v1' and
            receipt['status'] == 'complete' and receipt['source_sha256'] == HELPER_SHA and
            receipt['preflight_receipt_sha256'] == c.PREFLIGHT_RECEIPT_SHA and
            receipt['forward_count'] == 64 and receipt['calibration_pairs'] == 32 and
            receipt['reference_decodes'] == receipt['topk_calls'] == 8 and
            receipt['evaluation_rows_loaded'] == receipt['evaluation_forwards'] == receipt['prefix_tokenizations'] == 0 and
            receipt['parameters_unchanged'] is True and receipt['lens_parameters_unchanged'] is True and
            receipt['pca_refit'] is False, 'committed calibration scope')
    outputs = {v['path']: v for v in receipt['outputs']}
    require(len(outputs) == len(receipt['outputs']) == 7, 'calibration output inventory')
    for name in ('references.json', 'original-interpretations.json'):
        require(outputs[name]['sha256'] == pins[name] and
                outputs[name]['size_bytes'] == (OUT/'calibration'/name).stat().st_size, 'reference receipt binding')
    return receipt


def validate_rows(rows, pairs, manifest, p, f):
    require(type(rows) is list and len(rows) == 32 and [r['id'] for r in rows] == IDS and
            len({r['prefix'] for r in rows}) == 32, 'exact evaluation row roster')
    require(type(pairs) is list and len(pairs) == 16, 'sixteen evaluation pairs')
    require(manifest['schema'] == 'jlens_pattern_calibration_candidate_manifest_v1' and
            manifest['protocol_sha256'] == p.PROTOCOL_SHA and manifest['seed'] == '20260921', 'manifest binding')
    candidates = manifest['candidates']
    require(len(candidates) == 86 and [v['candidate_id'] for v in candidates] == [f'K{i:03}' for i in range(1, 87)] and
            all(v['role'] == ('evaluation' if i % 3 == 2 else 'calibration') for i, v in enumerate(candidates)),
            'fixed candidate roster/roles')
    by_candidate = {v['candidate_id']: v for v in candidates}; ranks = []
    for row in rows:
        require(set(row) == {'id', 'topic', 'prefix'} and row['topic'] in f.TOPICS, 'row fields/topic')
        require(type(row['prefix']) is str and len(row['prefix'].split()) == 16 and
                row['prefix'] == ' '.join(row['prefix'].split()), 'exact sixteen-word prefix')
    for i, pair in enumerate(pairs):
        require(set(pair) == {'id', 'left', 'right', 'topic', 'candidate_id', 'role'} and
                pair['id'] == f'T{i+1:02}' and pair['left'] == IDS[2*i] and pair['right'] == IDS[2*i+1] and
                pair['role'] == 'evaluation', 'exact evaluation pair identities')
        candidate = by_candidate[pair['candidate_id']]
        require(candidate['role'] == 'evaluation' and candidate['topic'] == pair['topic'] ==
                rows[2*i]['topic'] == rows[2*i+1]['topic'], 'evaluation candidate/topic join')
        ranks.append(int(pair['candidate_id'][1:]))
    require(ranks == sorted(set(ranks)), 'evaluation acceptance order')


def load_evaluation(c, p, f):
    preflight = f.read_json(OUT/'token-preflight/receipt.json')
    require(preflight['schema'] == 'jlens_pattern_calibration_token_preflight_receipt_v1' and
            preflight['status'] == 'complete' and preflight['source_sha256'] == c.PREFLIGHT_SOURCE_SHA and
            preflight['base_helper_sha256'] == p.BASE_SHA and preflight['host_helper_sha256'] == p.HOST_HELPER_SHA and
            preflight['input_pins'] == {str(k): v for k, v in p.frozen_pins(f).items()}, 'preflight source/input bindings')
    require(preflight['count'] == 96 and preflight['role_counts'] == {'calibration': 64, 'evaluation': 32} and
            preflight['token_limit'] == 96 and preflight['capture_locations'] == LOCATIONS and
            preflight['model_loaded'] is False and preflight['scientific_array_loaded'] is False and
            preflight['reference_decodes'] == 0, 'whole-panel preflight scope')
    outputs = preflight['outputs']
    require(len(outputs) == 2 and [r['path'] for r in outputs] == ['calibration-tokens.json', 'evaluation-tokens.json'] and
            outputs[1]['sha256'] == EVALUATION_TOKENS_SHA and
            outputs[1]['size_bytes'] == (OUT/'token-preflight/evaluation-tokens.json').stat().st_size, 'evaluation token pin')
    rows = f.read_json(OUT/'excerpts/evaluation-dataset.json')
    pairs = f.read_json(OUT/'excerpts/evaluation-pairs.json')
    validate_rows(rows, pairs, f.read_json(MAIN/'selection/manifest.json'), p, f)
    collection = f.read_json(OUT/'excerpts/receipt.json')
    require(collection['schema'] == 'jlens_pattern_calibration_excerpts_v1' and collection['status'] == 'complete' and
            collection['source_sha256'] == p.COLLECTOR_SHA and collection['protocol_sha256'] == p.PROTOCOL_SHA and
            collection['manifest_sha256'] == p.MANIFEST_SHA and collection['selected_count'] == 96 and
            collection['accepted_pairs'] == {'calibration': 32, 'evaluation': 16}, 'collection role/source binding')
    bound = {r['path']: r for r in collection['outputs']}
    require(len(bound) == len(collection['outputs']), 'duplicate collection outputs')
    for name in ('evaluation-dataset.json', 'evaluation-pairs.json'):
        require(bound[name]['sha256'] == p.COLLECTION_OUTPUTS[name] and
                bound[name]['size_bytes'] == (OUT/'excerpts'/name).stat().st_size, 'evaluation collection output pin')
    payload = f.read_json(OUT/'token-preflight/evaluation-tokens.json')
    require(payload['schema'] == 'jlens_pattern_calibration_tokens_v1' and payload['role'] == 'evaluation' and
            len(payload['records']) == 32, 'evaluation token role/roster')
    for row, record in zip(rows, payload['records'], strict=True):
        require(record == f.token_record(row, record), 'frozen IDs/masks/offset/final-position mismatch')
    return rows, pairs, payload['records'], preflight


def calculate(c, f, target, hf, model, rows, pairs, records, u, mean, device):
    import numpy as np
    require([r['id'] for r in records] == IDS and [r['id'] for r in rows] == IDS and len(pairs) == 16,
            'evaluation-only capture roster')
    baseline = f.parameter_state(hf)
    h = f.capture(model, records, device)
    require(f.parameter_state(hf) == baseline, 'model parameter mutation')
    scores, gaps, export = f.score_export(h, u, mean, rows, pairs)
    export['schema'] = 'jlens_pattern_calibration_evaluation_scores_v1'
    c.save_npz(target/'features.npz', activation_11=h, scores64=scores, gaps64=gaps,
               locations=np.array(LOCATIONS), u32=u, source_mean64=mean)
    c.write(target/'scores.json', export)
    c.write(target/'gaps.json', {'schema': 'jlens_pattern_calibration_evaluation_gaps_v1', 'locations': LOCATIONS,
        'axes': [f'PC{i}' for i in range(1, 5)], 'orientation': 'left minus right',
        'pairs': [{**p, 'gaps64': g.tolist()} for p, g in zip(pairs, gaps, strict=True)]})
    require(f.parameter_state(hf) == baseline, 'model parameter drift after scoring')


def run(source_sha):
    require(os.environ.get('JLENS_PATTERN_EVALUATION_GPU_RELEASE') == '1', 'root evaluation GPU admission required')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN')), 'offline one-thread envelope')
    require(type(source_sha) is str and re.fullmatch('[0-9a-f]{64}', source_sha) and
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_sha, 'exact reviewed evaluation source')
    target = OUT/'evaluation'
    if target.exists() or target.is_symlink(): raise FileExistsError('evaluation stage consumed')
    target.mkdir(); started = time.monotonic(); phase = 'inputs'
    write(target/'attempt.json', {'schema': 'jlens_pattern_calibration_evaluation_attempt_v1',
        'source_sha256': source_sha, 'started_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False})
    try:
        c = import_helper()
        p = c.import_pinned(OUT/'preflight.py', c.PREFLIGHT_SOURCE_SHA)
        f, a = c.import_pinned(p.BASE, p.BASE_SHA), c.import_pinned(p.HOST_HELPER, p.HOST_HELPER_SHA)
        expected = input_pins(c, p, f); verify(c, expected)
        calibration = committed_references(c, f)
        rows, pairs, records, preflight = load_evaluation(c, p, f)
        env = f.runtime(); host = a.host_identity(); p.validate_runtime(env, host, a)
        require(env['python'] == preflight['runtime']['python'] == calibration['runtime']['python'] and
                env['packages'] == preflight['runtime']['packages'] == calibration['runtime']['packages'] and
                host == preflight['host'] == calibration['host'], 'runtime differs from frozen prior stages')
        u, mean = c.load_directions(f)
        c.write(target/'inputs.json', {'schema': 'jlens_pattern_calibration_evaluation_inputs_v1',
            'role': 'evaluation', 'records': records, 'pairs': pairs, 'capture_locations': LOCATIONS,
            'mask_application': 'No padding; all-one masks implicit in unchanged adapter forward.'})
        verify(c, expected); require(a.host_identity() == host, 'runtime drift before model')
        phase = 'model_capture_score'
        hf, model = f.load_model()
        calculate(c, f, target, hf, model, rows, pairs, records, u, mean, 'cuda')
        verify(c, expected)
        require(committed_references(c, f) == calibration and
                load_evaluation(c, p, f) == (rows, pairs, records, preflight), 'metadata/reference drift')
        require(a.host_identity() == host and {n: sys.modules[n].__version__ for n in a.NUMERICAL} == a.NUMERICAL,
                'runtime drift after capture')
        require(c.sha(Path(__file__)) == source_sha, 'source changed during evaluation')
        import torch
        names = ['features.npz', 'inputs.json', 'scores.json', 'gaps.json']
        c.write(target/'receipt.json', {'schema': 'jlens_pattern_calibration_evaluation_receipt_v1', 'status': 'complete',
            'source_sha256': source_sha, 'input_pins': {str(k): v for k, v in expected.items()},
            'outputs': [{'path': n, 'sha256': c.sha(target/n), 'size_bytes': (target/n).stat().st_size} for n in names],
            'reference_commit': REFERENCE_COMMIT, 'references_sha256': REFERENCE_SHA,
            'calibration_receipt_sha256': CALIBRATION_RECEIPT_SHA, 'preflight_receipt_sha256': c.PREFLIGHT_RECEIPT_SHA,
            'runtime': env, 'host': host, 'model': f.MODEL, 'model_revision': f.REV, 'model_weight_sha256': f.WEIGHT_SHA,
            'adapter_revision': f.UPSTREAM_REV, 'forward_count': 32, 'evaluation_pairs': 16,
            'capture_locations': LOCATIONS, 'parameters_unchanged': True, 'calibration_rows_loaded': 0,
            'reference_decodes': 0, 'lens_loaded': False, 'tokenizer_loaded': False, 'prefix_tokenizations': 0,
            'pca_refit': False, 'pattern_fit': False, 'scientific_key_policy': 'Unopened until all 256 choices and validated lock are committed.',
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
            'gpu': torch.cuda.get_device_name(0), 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'memory_scope': '8 GiB PyTorch allocator bound, not total-process GPU cap'})
    except Exception as error:
        write(target/'failure.json', {'status': 'FAILED', 'phase': phase, 'error_type': type(error).__name__,
            'automatic_retry': False})  # No exception values that could disclose a scientific key.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    run(parser.parse_args().source_sha)
