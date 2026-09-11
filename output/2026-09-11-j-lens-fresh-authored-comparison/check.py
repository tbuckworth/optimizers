"""Import-inert saved-data corroboration only; never loads a tokenizer or model."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

OUT = Path(__file__).resolve().parent
FORWARD_SHA = '2493467e72b0e2cffd1488045d8e4d8674bfd12c51b60007290620120823c516'


def sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False); handle.write('\n')


def load_helper():
    path = OUT/'forward.py'
    require(sha(path) == FORWARD_SHA, 'frozen producer/helper source changed')
    spec = importlib.util.spec_from_file_location('fresh_authored_checked_helper', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper

JUDGING = Path('output/2026-09-11-jlens-fresh-authored-comparison/judging.py')
JUDGING_SHA = '6be05f391f84ce0002f855e0649050b2dd48c02d4dd70aa51e12e0bf5aa17b83'


def reader_gate(packets, manifest_sha, responses, lock_sha, lock_commit):
    """Public JSON / committed responses only. No measurement or private-key I/O."""
    require(sha(JUDGING) == JUDGING_SHA, 'frozen judging gate source changed')
    spec = importlib.util.spec_from_file_location('fresh_authored_lock_gate', JUDGING)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    verified = gate.verify_response_lock(packets=packets, manifest_sha256=manifest_sha,
                                         responses=responses, lock_sha256=lock_sha, lock_commit=lock_commit)
    inputs = verified['manifest']['inputs']
    for name, expected in gate.FROZEN.items():
        require(inputs[name]['sha256'] == expected, 'response panel/reference binding')
    require(sha(JUDGING) == JUDGING_SHA, 'judging source changed during lock check')
    return {'judging_source_sha256': JUDGING_SHA, 'packets': str(Path(packets).resolve()),
            'manifest_sha256': manifest_sha, 'responses': str(Path(responses).resolve()),
            'lock_sha256': lock_sha, 'lock_commit': lock_commit, 'choices': 256,
            'preflight_sha256': inputs['preflight']['sha256'], 'tokens_sha256': inputs['tokens']['sha256']}


def corroborate(h, u, mean, scores, gaps, locations, rows, pairs, exports):
    """Independent scalar fsum, saved-score subtraction and all JSON joins."""
    import numpy as np
    for value, shape, dtype in [(h, (32, 1, 1024), np.float32), (u, (1024, 4), np.float32),
            (mean, (1024,), np.float64), (scores, (32, 1, 4), np.float64), (gaps, (16, 1, 4), np.float64)]:
        require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
                and np.isfinite(value).all(), 'array shape/dtype/finite failure')
    require(np.allclose(np.linalg.norm(u.astype(np.float64), axis=0), 1, rtol=0, atol=1e-6), 'unit direction failure')
    roles = ['verb']; axes = ['PC1', 'PC2', 'PC3', 'PC4']
    require(locations.tolist() == roles, 'saved role order')
    require(exports['scores']['schema'] == 'jlens_fresh_authored_scores_v1'
            and list(exports['scores']['locations']) == roles, 'score schema/role order')
    require(exports['gaps']['locations'] == roles and exports['gaps']['axes'] == axes
            and exports['gaps']['orientation'] == 'observation minus provision', 'gap schema/orientation')
    max_error = 0.0
    for role, name in enumerate(roles):
        axis_rows = exports['scores']['locations'][name]
        require([a['axis'] for a in axis_rows] == axes, 'axis roster')
        for axis, exported in enumerate(axis_rows):
            require(list(exported['values']) == [r['id'] for r in rows], 'score row order')
            for i, row in enumerate(rows):
                value = math.fsum((float(h[i, role, k])-float(mean[k]))*float(u[k, axis]) for k in range(1024))
                error = abs(value-float(scores[i, role, axis])); max_error = max(max_error, error)
                require(error <= 1e-12, 'scalar projection mismatch')
                exported_value = exported['values'][row['id']]
                require(type(exported_value) is float and exported_value == float(scores[i, role, axis]), 'scalar JSON mismatch')
    index = {r['id']: i for i, r in enumerate(rows)}
    require(len(exports['gaps']['pairs']) == 16, 'gap pair count')
    for p, pair in enumerate(pairs):
        expected = scores[index[pair['observation_id']]]-scores[index[pair['provision_id']]]
        require(np.array_equal(gaps[p], expected), 'saved O-minus-P gap mismatch')
        require(exports['gaps']['pairs'][p] == {**pair, 'gaps64': gaps[p].tolist()}, 'gap metadata/JSON mismatch')
    summaries = {}
    for role, name in enumerate(roles):
        summaries[name] = []
        for axis, label in enumerate(axes):
            values = gaps[:, role, axis]
            summaries[name].append({'axis': label, 'positive': int((values > 0).sum()),
                'zero': int((values == 0).sum()), 'negative': int((values < 0).sum()),
                'both_templates_positive': sum(bool(values[i] > 0 and values[i+1] > 0) for i in range(0, 16, 2)),
                'template_sign_changes': [pairs[i]['content_pair'] for i in range(0, 16, 2)
                    if int(np.sign(values[i])) != int(np.sign(values[i+1]))]})
    return {'schema': 'jlens_fresh_authored_check_results_v1', 'scalar_checks': 128,
        'gap_checks': 64, 'max_scalar_error': max_error, 'scalar_tolerance': 1e-12,
        'descriptive_all_16_verb_PC4_positive': bool((gaps[:, 0, 3] > 0).all()),
        'locations': roles, 'axes': axes, 'rows': rows,
        'pairs': [{**pair, 'gaps64': gaps[i].tolist()} for i, pair in enumerate(pairs)], 'summaries': summaries}


def run(receipt_sha, *, packets, manifest_sha, responses, lock_sha, lock_commit, root=OUT):
    require(os.environ.get('JLENS_FRESH_AUTHORED_CHECK_RELEASE') == '1', 'main checker release required')
    target = root/'checked'
    if target.exists() or target.is_symlink(): raise FileExistsError(f'Consumed checker: {target}')
    target.mkdir(); started = time.monotonic(); source_sha = sha(Path(__file__))
    write(target/'attempt.json', {'source_sha256': source_sha, 'started_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False})
    try:
        require(type(receipt_sha) is str and re.fullmatch('[0-9a-f]{64}', receipt_sha) is not None, 'explicit receipt hash required')
        lock_binding = reader_gate(packets, manifest_sha, responses, lock_sha, lock_commit)
        f = load_helper(); expected = f.pins(f.DATASET_SHA, f.PAIRS_SHA)
        expected[Path(f.__file__)] = FORWARD_SHA
        expected[root/'forwards/receipt.json'] = receipt_sha
        f.verify(expected); rows, pairs = f.load_panel()
        receipt = f.read_json(root/'forwards/receipt.json')
        input_pins = {str(p): digest for p, digest in f.pins(f.DATASET_SHA, f.PAIRS_SHA).items()}
        require(receipt['schema'] == 'jlens_fresh_authored_forwards_receipt_v1' and receipt['status'] == 'complete'
                and receipt['source_sha256'] == FORWARD_SHA and receipt['input_pins'] == input_pins, 'forward receipt identity')
        require(receipt['forward_count'] == 32 and receipt['capture_locations'] == f.LOCATIONS
                and receipt['parameters_unchanged'] is True and receipt['reference_decodes'] == 0
                and receipt['pca_refit'] is False and receipt['tokenizer_loaded'] is False
                and receipt['model_weight_sha256'] == f.WEIGHT_SHA and receipt['adapter_revision'] == f.UPSTREAM_REV, 'forward scope/model identity')
        names = ['features.npz', 'inputs.json', 'scores.json', 'gaps.json']
        require([o['path'] for o in receipt['outputs']] == names, 'exact output inventory')
        for output in receipt['outputs']: expected[root/'forwards'/output['path']] = output['sha256']
        records, preflight = f.frozen_tokens(root, receipt['preflight_receipt_sha256'], f.pins(f.DATASET_SHA, f.PAIRS_SHA), rows)
        require(lock_binding['preflight_sha256'] == receipt['preflight_receipt_sha256']
                and lock_binding['tokens_sha256'] == preflight['outputs'][0]['sha256'], 'judged token/measurement binding')
        require(receipt['runtime']['python'] == preflight['runtime']['python']
                and receipt['runtime']['packages'] == preflight['runtime']['packages'], 'runtime mismatch')
        expected[root/'token-preflight/receipt.json'] = receipt['preflight_receipt_sha256']
        expected[root/'token-preflight/tokens.json'] = preflight['outputs'][0]['sha256']
        f.verify(expected)
        exports = {name: f.read_json(root/'forwards'/f'{name}.json') for name in ['inputs', 'scores', 'gaps']}
        require(exports['inputs']['schema'] == 'jlens_fresh_authored_inputs_v1' and exports['inputs']['records'] == records, 'captured input/position binding')
        import numpy as np
        with np.load(root/'forwards/features.npz', allow_pickle=False) as archive:
            require(set(archive.files) == {'activation_11', 'scores64', 'gaps64', 'locations'}, 'feature inventory')
            h, scores, gaps, locations = [archive[k] for k in ['activation_11', 'scores64', 'gaps64', 'locations']]
        with np.load(f.OLD/'preparation/directions.npz', allow_pickle=False) as archive:
            u, mean = archive['u32'], archive['source_mean64']
        result = corroborate(h, u, mean, scores, gaps, locations, rows, pairs, exports)
        f.verify(expected); require(sha(Path(__file__)) == source_sha, 'checker changed')
        require(reader_gate(packets, manifest_sha, responses, lock_sha, lock_commit) == lock_binding, 'response lock changed')
        write(target/'results.json', result)
        write(target/'receipt.json', {'schema': 'jlens_fresh_authored_check_receipt_v1', 'status': 'complete',
            'source_sha256': source_sha, 'helper_sha256': FORWARD_SHA, 'response_lock': lock_binding, 'input_pins': {str(p): v for p, v in expected.items()},
            'outputs': [{'path': 'results.json', 'sha256': sha(target/'results.json')}],
            'python': sys.version, 'numpy': np.__version__, 'interpreter': sys.executable,
            'elapsed_seconds': time.monotonic()-started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'model_loaded': False, 'tokenizer_loaded': False, 'pca_refit': False})
    except Exception as error:
        write(target/'failure.json', {'error_type': type(error).__name__, 'automatic_retry': False}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward-receipt-sha', required=True)
    parser.add_argument('--packets', required=True, type=Path)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--responses', required=True, type=Path)
    parser.add_argument('--lock-sha', required=True)
    parser.add_argument('--lock-commit', required=True)
    args = parser.parse_args()
    run(args.forward_receipt_sha, packets=args.packets, manifest_sha=args.manifest_sha,
        responses=args.responses, lock_sha=args.lock_sha, lock_commit=args.lock_commit)
