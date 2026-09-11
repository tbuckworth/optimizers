"""One independent saved-array check, admitted only after committed reader lock."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-pattern-calibration')
GATE_SHA = '55d659a991965a6d572101a0ec513978ec7b0534113e9a6e9438ffe6172d34b5'
RELEASE_SHA = '5565779ea44b67240c5254219d13e2486c3de840ac30ad9a9ffdc1e6f77160aa'
RELEASE_COMMIT = '866ab6f30feecc6e71ef5fc53ef19491d8a5247a'
DIRECTIONS_SHA = '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa'
EVALUATION_COMMIT = 'e955d2ddaad66693a5cc1aa8904126e50d5f0065'


def require(ok, label):
    if not ok: raise ValueError(label)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def check_arrays(h, u, mean, scores, gaps):
    require(h.ndim == 3 and h.shape[1] == 1 and h.shape[0] % 2 == 0 and h.shape[0] > 0, 'activation layout')
    n, _, width = h.shape
    require(u.ndim == 2 and u.shape[0] == width and u.shape[1] > 0, 'basis layout')
    k = u.shape[1]
    for value, shape, dtype in ((h, (n, 1, width), np.float32), (u, (width, k), np.float32),
                              (mean, (width,), np.float64), (scores, (n, 1, k), np.float64),
                              (gaps, (n//2, 1, k), np.float64)):
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), 'finite typed arrays')
    # Independent reduction, not producer import. Mean subtraction precedes dot.
    expected = np.einsum('nlw,wk->nlk', h.astype(np.float64)-mean, u.astype(np.float64), optimize=False)
    require(np.allclose(scores, expected, rtol=1e-10, atol=1e-11), 'original centered scores')
    # The target is subtraction of the recorded scores, not delta-h projection.
    require(np.array_equal(gaps, scores[::2]-scores[1::2]), 'exact recorded-score differences')
    return {'maximum_independent_score_error': float(np.max(np.abs(scores-expected))),
            'rows': n, 'pairs': n//2, 'axes': k, 'width': width}


def run(lock_sha, lock_commit, own_sha):
    require(sha(Path(__file__).read_bytes()) == own_sha, 'reviewed audit source')
    source = WORK/'responses.py'
    require(sha(source.read_bytes()) == GATE_SHA, 'reviewed public gate source')
    spec = importlib.util.spec_from_file_location('committed_response_gate', source)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    checked = gate.verify_response_lock(release_path=HERE/'judging-release.json', release_sha=RELEASE_SHA,
        release_commit=RELEASE_COMMIT, responses=HERE/'responses', lock_sha=lock_sha, lock_commit=lock_commit)
    # Nothing scientific is read above this line. No producer/model imports anywhere.
    target = HERE/'evaluation-audit.json'
    require(not target.exists(), 'saved-array audit already consumed')
    p = gate.helper(); gate.forward_scope(p, checked['released'])
    receipt_path = WORK/'evaluation/receipt.json'
    receipt_raw = gate.read(checked['released']['evaluation']['receipt']); receipt = p.decode(receipt_raw)
    outputs = {}
    for row in receipt['outputs']:
        pin = gate.member(receipt_path.parent, row, row['path'])
        outputs[row['path']] = gate.read(pin)
    gate.committed({receipt_path: receipt_raw, **{receipt_path.parent/n: b for n, b in outputs.items()}}, EVALUATION_COMMIT)
    original_path = WORK.parent/'2026-09-10-j-lens-fresh-content/preparation/directions.npz'
    require(sha(original_path.read_bytes()) == DIRECTIONS_SHA, 'original basis archive')
    with np.load(original_path, allow_pickle=False) as old:
        original_u, original_mean = old['u32'], old['source_mean64']
    with np.load(WORK/'evaluation/features.npz', allow_pickle=False) as archive:
        require(set(archive.files) == {'activation_11', 'scores64', 'gaps64', 'locations', 'u32', 'source_mean64'}, 'archive fields')
        h, u, mean, scores, gaps = [archive[n] for n in ('activation_11', 'u32', 'source_mean64', 'scores64', 'gaps64')]
        require(archive['locations'].tolist() == ['prefix_end'], 'fixed capture position')
    require(np.array_equal(u, original_u) and np.array_equal(mean, original_mean), 'exact original basis/mean')
    require(h.shape == (32, 1, 1024) and u.shape == (1024, 4), 'fixed full evaluation scope')
    result = check_arrays(h, u, mean, scores, gaps)
    dataset = p.decode((WORK/'excerpts/evaluation-dataset.json').read_bytes())
    pairs = p.decode((WORK/'excerpts/evaluation-pairs.json').read_bytes())
    for name, value in [('dataset', dataset), ('pairs', pairs)]:
        path, digest = p.PINS[name]; require(sha(path.read_bytes()) == digest, 'frozen '+name)
    ids = [f'T{i:02}-{side}' for i in range(1, 17) for side in ('L', 'R')]
    require([v['id'] for v in dataset] == ids, 'ordered evaluation IDs')
    inputs = p.decode(outputs['inputs.json'])
    frozen_tokens = p.decode((WORK/'token-preflight/evaluation-tokens.json').read_bytes())
    token_path = WORK/'token-preflight/evaluation-tokens.json'
    require(sha(token_path.read_bytes()) == receipt['input_pins'][str(token_path)], 'frozen token records')
    require(inputs['records'] == frozen_tokens['records'] and inputs['pairs'] == pairs and
            inputs['capture_locations'] == ['prefix_end'] and inputs['role'] == 'evaluation', 'recorded input joins')
    for row, record in zip(dataset, inputs['records'], strict=True):
        require(all(record[key] == row[key] for key in ('id', 'topic', 'prefix')), 'unchanged captured prefix')
    exported = p.decode(outputs['scores.json'])
    expected_export = {'schema': 'jlens_pattern_calibration_evaluation_scores_v1', 'locations': {'prefix_end': [
        {'axis': f'PC{a+1}', 'values': {identifier: float(scores[i, 0, a]) for i, identifier in enumerate(ids)}} for a in range(4)]}}
    require(exported == expected_export, 'every exported JSON score equals saved array')
    expected_gaps = {'schema': 'jlens_pattern_calibration_evaluation_gaps_v1', 'locations': ['prefix_end'],
        'axes': [f'PC{i}' for i in range(1, 5)], 'orientation': 'left minus right',
        'pairs': [{**pair, 'gaps64': value.tolist()} for pair, value in zip(pairs, gaps, strict=True)]}
    require(p.decode(outputs['gaps.json']) == expected_gaps, 'every exported gap and pair binding')
    require(sha((WORK/'evaluation/features.npz').read_bytes()) == receipt['outputs'][0]['sha256'], 'archive drift')
    result.update(status='PASS', source_sha256=own_sha, public_lock_verified_before_key=True,
        lock_sha256=lock_sha, lock_commit=lock_commit, evaluation_commit=EVALUATION_COMMIT,
        output_pins=receipt['outputs'], model_forwards=0, completed_utc=datetime.now(timezone.utc).isoformat())
    with target.open('x') as handle: json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lock-sha', 'lock-commit', 'source-sha'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); run(args.lock_sha, args.lock_commit, args.source_sha)
