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
FORWARD_SHA = '6beef7ce564f071a854f351eba6ad6f6363deba46d7a3da15e4af6775f87b8fe'


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
    spec = importlib.util.spec_from_file_location('new_verb_checked_helper', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper


def corroborate(h, u, mean, scores, gaps, locations, rows, pairs, exports):
    """Independent scalar fsum, saved-score subtraction and all JSON joins."""
    import numpy as np
    for value, shape, dtype in [(h, (32, 2, 1024), np.float32), (u, (1024, 4), np.float32),
            (mean, (1024,), np.float64), (scores, (32, 2, 4), np.float64), (gaps, (16, 2, 4), np.float64)]:
        require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
                and np.isfinite(value).all(), 'array shape/dtype/finite failure')
    require(np.allclose(np.linalg.norm(u.astype(np.float64), axis=0), 1, rtol=0, atol=1e-6), 'unit direction failure')
    roles = ['verb', 'sentence_end']; axes = ['PC1', 'PC2', 'PC3', 'PC4']
    require(locations.tolist() == roles, 'saved role order')
    require(exports['scores']['schema'] == 'jlens_new_verb_transfer_scores_v1'
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
    return {'schema': 'jlens_new_verb_transfer_check_results_v1', 'scalar_checks': 256,
        'gap_checks': 128, 'max_scalar_error': max_error, 'scalar_tolerance': 1e-12,
        'primary_all_16_verb_PC4_positive': bool((gaps[:, 0, 3] > 0).all()),
        'locations': roles, 'axes': axes, 'rows': rows,
        'pairs': [{**pair, 'gaps64': gaps[i].tolist()} for i, pair in enumerate(pairs)], 'summaries': summaries}


def run(receipt_sha, root=OUT):
    require(os.environ.get('JLENS_NEW_VERB_TRANSFER_CHECK_RELEASE') == '1', 'main checker release required')
    target = root/'checked'
    if target.exists() or target.is_symlink(): raise FileExistsError(f'Consumed checker: {target}')
    target.mkdir(); started = time.monotonic(); source_sha = sha(Path(__file__))
    write(target/'attempt.json', {'source_sha256': source_sha, 'started_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False})
    try:
        require(type(receipt_sha) is str and re.fullmatch('[0-9a-f]{64}', receipt_sha) is not None, 'explicit receipt hash required')
        f = load_helper(); expected = f.pins(f.DATASET_SHA, f.PAIRS_SHA)
        expected[Path(f.__file__)] = FORWARD_SHA
        expected[root/'forwards/receipt.json'] = receipt_sha
        f.verify(expected); rows, pairs = f.load_panel()
        receipt = f.read_json(root/'forwards/receipt.json')
        input_pins = {str(p): digest for p, digest in f.pins(f.DATASET_SHA, f.PAIRS_SHA).items()}
        require(receipt['schema'] == 'jlens_new_verb_transfer_forwards_receipt_v1' and receipt['status'] == 'complete'
                and receipt['source_sha256'] == FORWARD_SHA and receipt['input_pins'] == input_pins, 'forward receipt identity')
        require(receipt['forward_count'] == 32 and receipt['capture_locations'] == f.LOCATIONS
                and receipt['parameters_unchanged'] is True and receipt['reference_decodes'] == 0
                and receipt['pca_refit'] is False and receipt['tokenizer_loaded'] is False
                and receipt['model_weight_sha256'] == f.WEIGHT_SHA and receipt['adapter_revision'] == f.UPSTREAM_REV, 'forward scope/model identity')
        names = ['features.npz', 'inputs.json', 'scores.json', 'gaps.json']
        require([o['path'] for o in receipt['outputs']] == names, 'exact output inventory')
        for output in receipt['outputs']: expected[root/'forwards'/output['path']] = output['sha256']
        records, preflight = f.frozen_tokens(root, receipt['preflight_receipt_sha256'], f.pins(f.DATASET_SHA, f.PAIRS_SHA), rows)
        require(receipt['runtime']['python'] == preflight['runtime']['python']
                and receipt['runtime']['packages'] == preflight['runtime']['packages'], 'runtime mismatch')
        expected[root/'token-preflight/receipt.json'] = receipt['preflight_receipt_sha256']
        expected[root/'token-preflight/tokens.json'] = preflight['outputs'][0]['sha256']
        f.verify(expected)
        exports = {name: f.read_json(root/'forwards'/f'{name}.json') for name in ['inputs', 'scores', 'gaps']}
        require(exports['inputs']['schema'] == 'jlens_new_verb_transfer_inputs_v1' and exports['inputs']['records'] == records, 'captured input/position binding')
        import numpy as np
        with np.load(root/'forwards/features.npz', allow_pickle=False) as archive:
            require(set(archive.files) == {'activation_11', 'scores64', 'gaps64', 'locations'}, 'feature inventory')
            h, scores, gaps, locations = [archive[k] for k in ['activation_11', 'scores64', 'gaps64', 'locations']]
        with np.load(f.OLD/'preparation/directions.npz', allow_pickle=False) as archive:
            u, mean = archive['u32'], archive['source_mean64']
        result = corroborate(h, u, mean, scores, gaps, locations, rows, pairs, exports)
        f.verify(expected); require(sha(Path(__file__)) == source_sha, 'checker changed')
        write(target/'results.json', result)
        write(target/'receipt.json', {'schema': 'jlens_new_verb_transfer_check_receipt_v1', 'status': 'complete',
            'source_sha256': source_sha, 'helper_sha256': FORWARD_SHA, 'input_pins': {str(p): v for p, v in expected.items()},
            'outputs': [{'path': 'results.json', 'sha256': sha(target/'results.json')}],
            'python': sys.version, 'numpy': np.__version__, 'interpreter': sys.executable,
            'elapsed_seconds': time.monotonic()-started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'model_loaded': False, 'tokenizer_loaded': False, 'pca_refit': False})
    except Exception as error:
        write(target/'failure.json', {'error_type': type(error).__name__, 'automatic_retry': False}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward-receipt-sha', required=True)
    run(parser.parse_args().forward_receipt_sha)
