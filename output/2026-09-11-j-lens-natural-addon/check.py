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
FORWARD_SHA = 'd2557036a3ebdd6ecbc61be89adf3582f16f084fda091e86f3faabe9ff39f1d0'
RECOVERY_SHA = '894f465e36d50f6df15229754367614d3ffc26f20301db3eb57f9535189bbc57'
MEASUREMENT_DIR = 'forwards-runtime-amendment'
AMENDMENT_SHA = '5bcc78c413c319e1ccf329066d01699a9c56bfdbb7a8bb0aa4ecc5abb9a8a72c'
PREFLIGHT_PYTHON = '3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]'
MEASUREMENT_PYTHON = '3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]'
DISTRO = {name: '3.12.3-1ubuntu0.17' for name in ['python3.12', 'python3.12-minimal',
         'libpython3.12-minimal', 'libpython3.12-stdlib', 'libpython3.12t64']}
DISTRO.update({'libc6': '2.39-0ubuntu8.9', 'libc-bin': '2.39-0ubuntu8.9'})
BINARIES = {'/usr/bin/python3.12': 'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f',
            '/lib/x86_64-linux-gnu/libc.so.6': '3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf'}
FAILED = {'attempt': 'c313921dcdb033cd973dbbd8884d80feef1ca80e79362d02b453c8e9df2bf6ba',
          'failure': '7f81cc5fad6f71b7d8ee901a642100cce1040dae9ad8fc1c2d440db2616fe1c6'}


def sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False); handle.write('\n')


def load_helper():
    require(type(FORWARD_SHA) is str and re.fullmatch('[0-9a-f]{64}', FORWARD_SHA), 'producer pin not released')
    path = OUT/'forward.py'
    require(sha(path) == FORWARD_SHA, 'frozen producer/helper source changed')
    spec = importlib.util.spec_from_file_location('natural_addon_checked_helper', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper

JUDGING = Path('output/2026-09-11-jlens-natural-addon/judging.py')
JUDGING_SHA = '17a2d159ed7dcb7b33b6d4e8f0e36b9d51890b902561705ea4c9754b2b73d11c'


def reader_gate(packets, manifest_sha, responses, lock_sha, lock_commit):
    """Public JSON / committed responses only. No measurement or private-key I/O."""
    require(type(JUDGING_SHA) is str and re.fullmatch('[0-9a-f]{64}', JUDGING_SHA), 'judging pin not released')
    require(sha(JUDGING) == JUDGING_SHA, 'frozen judging gate source changed')
    spec = importlib.util.spec_from_file_location('natural_addon_lock_gate', JUDGING)
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
    roles = ['prefix_end']; axes = ['PC1', 'PC2', 'PC3', 'PC4']
    require(locations.tolist() == roles, 'saved role order')
    require(exports['scores']['schema'] == 'jlens_natural_addon_scores_v1'
            and list(exports['scores']['locations']) == roles, 'score schema/role order')
    require(exports['gaps']['locations'] == roles and exports['gaps']['axes'] == axes
            and exports['gaps']['orientation'] == 'left minus right', 'gap schema/orientation')
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
        expected = scores[index[pair['left']]]-scores[index[pair['right']]]
        require(np.array_equal(gaps[p], expected), 'saved left-minus-right gap mismatch')
        require(exports['gaps']['pairs'][p] == {**pair, 'gaps64': gaps[p].tolist()}, 'gap metadata/JSON mismatch')
    summaries = {}
    for role, name in enumerate(roles):
        summaries[name] = []
        for axis, label in enumerate(axes):
            values = gaps[:, role, axis]
            summaries[name].append({'axis': label, 'positive': int((values > 0).sum()),
                'zero': int((values == 0).sum()), 'negative': int((values < 0).sum())})
    return {'schema': 'jlens_natural_addon_check_results_v1', 'scalar_checks': 128,
        'gap_checks': 64, 'max_scalar_error': max_error, 'scalar_tolerance': 1e-12,
        'locations': roles, 'axes': axes, 'rows': rows,
        'pairs': [{**pair, 'gaps64': gaps[i].tolist()} for i, pair in enumerate(pairs)], 'summaries': summaries}


def run(receipt_sha, *, packets, manifest_sha, responses, lock_sha, lock_commit, root=OUT):
    require(os.environ.get('JLENS_NATURAL_ADDON_CHECK_RELEASE') == '1', 'main checker release required')
    target = root/'checked'
    if target.exists() or target.is_symlink(): raise FileExistsError(f'Consumed checker: {target}')
    target.mkdir(); started = time.monotonic(); source_sha = sha(Path(__file__))
    write(target/'attempt.json', {'source_sha256': source_sha, 'started_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False})
    try:
        require(type(receipt_sha) is str and re.fullmatch('[0-9a-f]{64}', receipt_sha) is not None, 'explicit receipt hash required')
        lock_binding = reader_gate(packets, manifest_sha, responses, lock_sha, lock_commit)
        f = load_helper(); expected = f.pins(f.DATASET_SHA, f.PAIRS_SHA)
        expected[Path(f.__file__)] = FORWARD_SHA
        require(type(RECOVERY_SHA) is str and re.fullmatch('[0-9a-f]{64}', RECOVERY_SHA), 'amended producer pin required')
        expected[OUT/'forward_runtime_amendment.py'] = RECOVERY_SHA
        expected[root/MEASUREMENT_DIR/'receipt.json'] = receipt_sha
        f.verify(expected); rows, pairs = f.load_panel()
        receipt = f.read_json(root/MEASUREMENT_DIR/'receipt.json')
        input_pins = {str(p): digest for p, digest in f.pins(f.DATASET_SHA, f.PAIRS_SHA).items()}
        require(receipt['schema'] == 'jlens_natural_addon_forwards_receipt_v1' and receipt['status'] == 'complete'
                and receipt['source_sha256'] == RECOVERY_SHA and receipt['base_producer_sha256'] == FORWARD_SHA
                and receipt['runtime_amendment_sha256'] == AMENDMENT_SHA
                and receipt['failed_admission_sha256s'] == FAILED
                and receipt['input_pins'] == input_pins, 'forward receipt identity')
        require(receipt['forward_count'] == 32 and receipt['capture_locations'] == f.LOCATIONS
                and receipt['parameters_unchanged'] is True and receipt['reference_decodes'] == 0
                and receipt['pca_refit'] is False and receipt['tokenizer_loaded'] is False
                and receipt['model_weight_sha256'] == f.WEIGHT_SHA and receipt['adapter_revision'] == f.UPSTREAM_REV, 'forward scope/model identity')
        names = ['features.npz', 'inputs.json', 'scores.json', 'gaps.json']
        require([o['path'] for o in receipt['outputs']] == names, 'exact output inventory')
        for output in receipt['outputs']: expected[root/MEASUREMENT_DIR/output['path']] = output['sha256']
        records, preflight = f.frozen_tokens(root, receipt['preflight_receipt_sha256'], f.pins(f.DATASET_SHA, f.PAIRS_SHA), rows)
        require(lock_binding['preflight_sha256'] == receipt['preflight_receipt_sha256']
                and lock_binding['tokens_sha256'] == preflight['outputs'][0]['sha256'], 'judged token/measurement binding')
        require(receipt['runtime']['python'] == MEASUREMENT_PYTHON
                and preflight['runtime']['python'] == PREFLIGHT_PYTHON
                and receipt['preflight_runtime'] == preflight['runtime']
                and receipt['runtime']['packages'] == preflight['runtime']['packages'], 'runtime mismatch')
        require(receipt['runtime_transition'] == {'preflight_python': PREFLIGHT_PYTHON,
                'new_python': MEASUREMENT_PYTHON, 'old_preflight_source_sha256': FORWARD_SHA,
                'package_versions': DISTRO, 'binary_sha256s': BINARIES}, 'runtime transition metadata')
        expected[root/'token-preflight/receipt.json'] = receipt['preflight_receipt_sha256']
        expected[root/'token-preflight/tokens.json'] = preflight['outputs'][0]['sha256']
        f.verify(expected)
        exports = {name: f.read_json(root/MEASUREMENT_DIR/f'{name}.json') for name in ['inputs', 'scores', 'gaps']}
        require(exports['inputs']['schema'] == 'jlens_natural_addon_inputs_v1' and exports['inputs']['records'] == records, 'captured input/position binding')
        import numpy as np
        with np.load(root/MEASUREMENT_DIR/'features.npz', allow_pickle=False) as archive:
            require(set(archive.files) == {'activation_11', 'scores64', 'gaps64', 'locations'}, 'feature inventory')
            h, scores, gaps, locations = [archive[k] for k in ['activation_11', 'scores64', 'gaps64', 'locations']]
        with np.load(f.OLD/'preparation/directions.npz', allow_pickle=False) as archive:
            u, mean = archive['u32'], archive['source_mean64']
        result = corroborate(h, u, mean, scores, gaps, locations, rows, pairs, exports)
        f.verify(expected); require(sha(Path(__file__)) == source_sha, 'checker changed')
        require(reader_gate(packets, manifest_sha, responses, lock_sha, lock_commit) == lock_binding, 'response lock changed')
        write(target/'results.json', result)
        write(target/'receipt.json', {'schema': 'jlens_natural_addon_check_receipt_v1', 'status': 'complete',
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
