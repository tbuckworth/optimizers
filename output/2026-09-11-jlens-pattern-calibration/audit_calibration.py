"""Independent saved-calibration arithmetic; no producer/model/tokenizer imports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-pattern-calibration')
OLD = WORK.parent/'2026-09-10-j-lens-fresh-content'
ORIGINAL_SHA = '0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c'
DIRECTIONS_SHA = '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa'
PREFLIGHT_SHA = 'd4cba9b4376098720a9fa8111f7f7d7dcf7f634e9e66413cb591860ef0c25ec4'


def check(ok, label):
    if not ok:
        raise ValueError(label)


def array(value, shape, dtype, label):
    check(isinstance(value, np.ndarray) and value.shape == shape and
          value.dtype == dtype and np.isfinite(value).all(), label)


def close(actual, expected, label):
    array(actual, expected.shape, expected.dtype, label+' layout')
    check(np.allclose(actual, expected, rtol=1e-10, atol=1e-12), label+' arithmetic')


def audit_moments(h, u, pair_indices, saved):
    """Independent matrix formulation; generic dimensions allow fabricated tests."""
    check(h.ndim == 2 and u.ndim == 2 and h.shape[1] == u.shape[0], 'feature/basis layout')
    n, m = h.shape
    k = u.shape[1]
    array(h, (n, m), np.dtype('float32'), 'features')
    array(u, (m, k), np.dtype('float32'), 'basis')
    check(n > 0 and m > 0 and k > 0 and len(pair_indices)*2 == n, 'complete pair count')
    check(all(len(p) == 2 and all(type(i) is int for i in p) for p in pair_indices), 'pair indices')
    check(sorted(i for p in pair_indices for i in p) == list(range(n)), 'each row exactly once')
    basis = u.astype(np.float64)
    check(np.allclose(np.linalg.norm(basis, axis=0), 1, rtol=0, atol=1e-6), 'unit basis')
    delta = h[[p[0] for p in pair_indices]].astype(np.float64)-h[[p[1] for p in pair_indices]].astype(np.float64)
    gaps = delta @ basis
    cross = delta.T @ gaps
    energy = np.einsum('ij,ij->j', gaps, gaps)
    trace = np.einsum('ij,ij->', delta, delta)
    check(np.isfinite(cross).all() and np.isfinite(energy).all() and (energy > 0).all()
          and np.isfinite(trace) and trace > 0, 'valid full fit')
    patterns = cross/energy
    expected = {
        'patterns64': patterns, 'cross64': cross, 'score_energy64': energy,
        'calibration_gaps64': gaps,
        'max_single_pair_energy_share64': np.max(gaps*gaps, axis=0)/energy,
        'direction_energy_fraction64': energy/trace,
        'unit_score_identity64': np.sum(basis*patterns, axis=0)}
    check(set(saved) == {*expected, 'signed_inputs32'}, 'all pattern outputs')
    for name, value in expected.items():
        close(saved[name], value, name)
    check(np.allclose(saved['unit_score_identity64'], 1, rtol=0, atol=1e-10), 'unit score identity')
    # Exact normalization check uses the archived raw fit, whose arithmetic was
    # checked independently above; no FP32 sign/reordering tolerance is needed.
    norms = np.linalg.norm(saved['patterns64'], axis=0)
    check(np.isfinite(norms).all() and (norms > 0).all(), 'positive pattern norms')
    unit = (saved['patterns64']/norms).astype(np.float32).T
    signed = np.stack([sign*unit[j] for j in range(k) for sign in (1, -1)])
    array(saved['signed_inputs32'], (2*k, m), np.dtype('float32'), 'signed layout')
    check(np.array_equal(saved['signed_inputs32'], signed), 'exact signed normalized inputs')
    return {'axes': k, 'pairs': n//2, 'width': m,
            'maximum_identity_error': float(np.max(np.abs(saved['unit_score_identity64']-1)))}


def audit_topk(logits, ids, values, rows):
    """Validate saved membership/tie accounting without repeating torch.topk."""
    check(logits.ndim == 2 and ids.ndim == 2, 'decoder layout')
    poles, vocab = logits.shape
    topn = ids.shape[1]
    check(ids.shape[0] == poles and 0 < topn <= vocab and len(rows) == poles, 'decoder counts')
    array(logits, (poles, vocab), np.dtype('float32'), 'vocabulary logits')
    array(ids, (poles, topn), np.dtype('int64'), 'selected IDs')
    array(values, (poles, topn), np.dtype('float32'), 'selected logits')
    summaries = []
    for i, (full, selected, top, row) in enumerate(zip(logits, ids, values, rows, strict=True)):
        check(len(set(selected.tolist())) == topn and ((selected >= 0) & (selected < vocab)).all(), 'unique valid IDs')
        check(np.array_equal(full[selected], top) and np.all(top[:-1] >= top[1:]), 'sorted saved top values')
        cutoff = np.partition(full, vocab-topn)[vocab-topn]
        above = int(np.count_nonzero(full > cutoff))
        tied = int(np.count_nonzero(full == cutoff))
        selected_ties = int(np.count_nonzero(top == cutoff))
        check(top[-1] == cutoff and selected_ties == topn-above, 'all strict winners and admissible ties')
        check(row['token_ids'] == selected.tolist() and row['scores'] == top.tolist(), 'JSON/array selection binding')
        check(type(row['tokens']) is list and len(row['tokens']) == topn and
              all(type(t) is str for t in row['tokens']), 'raw token strings')
        check(row['cutoff_logit'] == float(cutoff) and row['strictly_above_cutoff_count'] == above
              and row['cutoff_tie_count'] == tied and row['selected_at_cutoff_count'] == selected_ties,
              'cutoff accounting')
        summaries.append({'pole': i, 'strictly_above_cutoff_count': above,
                          'cutoff_tie_count': tied, 'selected_at_cutoff_count': selected_ties})
    return summaries


def audit_references(original, public, rows):
    axes = [f'PC{i}' for i in range(1, 5)]
    check(original['schema'] == 'jlens_fresh_references_v1' and
          [a['axis'] for a in original['axes']] == axes, 'original reference roster')
    check(set(public) == {'schema', 'arms'} and public['schema'] == 'jlens_pattern_calibration_references_v1'
          and set(public['arms']) == {'U', 'P'}, 'public arm schema')
    for arm, content in public['arms'].items():
        check(set(content) == {'axes'} and len(content['axes']) == 4, 'four axes per arm')
        for i, (old, new) in enumerate(zip(original['axes'], content['axes'], strict=True)):
            check(set(new) == {'axis', 'positive_reference', 'negative_reference'} and
                  new['axis'] == axes[i], 'same axis/order/schema')
            for j, pole in enumerate(('positive', 'negative')):
                expected = {'example_prefix': old['C'][pole],
                            'direction_tokens': old['A'][pole] if arm == 'U' else rows[2*i+j]['tokens']}
                check(type(expected['example_prefix']) is str and len(expected['direction_tokens']) == 12
                      and all(type(v) is str for v in expected['direction_tokens']), 'full reference strings')
                check(new[pole+'_reference'] == expected, 'unchanged examples/control or exact pattern readout')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    check(not path.is_symlink() and path.stat().st_size <= 1024**2, 'bounded JSON path')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def run(receipt_sha, producer_sha):
    check(all(type(s) is str and re.fullmatch('[0-9a-f]{64}', s) for s in (receipt_sha, producer_sha)), 'exact SHA arguments')
    target = HERE/'calibration-audit.json'
    check(not target.exists() and not target.is_symlink(), 'calibration audit already recorded')
    stage = WORK/'calibration'
    check(sha(stage/'receipt.json') == receipt_sha, 'completed receipt pin')
    receipt = read(stage/'receipt.json')
    check(receipt['schema'] == 'jlens_pattern_calibration_calibration_receipt_v1' and
          receipt['status'] == 'complete', 'successful calibration scope')
    check(receipt['source_sha256'] == producer_sha == sha(WORK/'calibrate.py'), 'reviewed source binding')
    check(receipt['forward_count'] == 64 and receipt['calibration_pairs'] == 32 and
          receipt['reference_decodes'] == receipt['topk_calls'] == 8 and
          receipt['evaluation_rows_loaded'] == receipt['evaluation_forwards'] == receipt['prefix_tokenizations'] == 0
          and receipt['pca_refit'] is False and receipt['parameters_unchanged'] is True and
          receipt['lens_parameters_unchanged'] is True and receipt['capture_locations'] == ['prefix_end'], 'stage boundaries')
    check(receipt['preflight_receipt_sha256'] == PREFLIGHT_SHA == sha(WORK/'token-preflight/receipt.json'), 'preflight identity')
    preflight = read(WORK/'token-preflight/receipt.json')
    check(receipt['host'] == preflight['host'] and receipt['runtime']['python'] == preflight['runtime']['python']
          and receipt['runtime']['packages'] == preflight['runtime']['packages'], 'unchanged current runtime')
    pins = receipt['input_pins']
    required = {OLD/'preparation/directions.npz': DIRECTIONS_SHA,
                OLD/'references/interpretations.json': ORIGINAL_SHA,
                WORK/'token-preflight/receipt.json': PREFLIGHT_SHA,
                WORK/'token-preflight/calibration-tokens.json': '460965950596e03661eb4bc574423780be8b1baad4346e43e79045deaea5398a',  # gitleaks:allow (sha256 file digest, not a secret)
                WORK/'excerpts/calibration-dataset.json': '249f6bb36e0247a9c18388955ed932773028cc8395aaba15234011b25056d971',
                WORK/'excerpts/calibration-pairs.json': '082a23f6130a5322fe6cd960bd7250ec9f547c384e718995b8501ab3b09a3ce5',
                HERE/'protocol.md': 'a3b47578fad5c1075e7b2855d01cffe954d5d0fb6b5c4d7fd2e51262a42f0d4e',
                HERE/'patterns.py': '524efc43e54c9549e66586f8a8dcc9db8542979637a56d840c3739f1c156a789'}
    for path, digest in required.items():
        check(pins.get(str(path)) == digest, 'required immutable input binding')
    for name, digest in pins.items():
        path = Path(name)
        check('evaluation' not in path.name and path.is_absolute(), 'calibration-only input set')
        check(sha(path) == digest, 'input bytes: '+name)
    names = {'inputs.json', 'features.npz', 'patterns.npz', 'decoder.npz', 'readouts.json',
             'original-interpretations.json', 'references.json'}
    check(len(receipt['outputs']) == len(names) and {r['path'] for r in receipt['outputs']} == names, 'output manifest')
    check({p.name for p in stage.iterdir()} == names | {'receipt.json', 'attempt.json'}, 'successful exact file roster')
    check(read(stage/'attempt.json')['source_sha256'] == producer_sha, 'attempt source')
    for item in receipt['outputs']:
        path = stage/item['path']
        check(sha(path) == item['sha256'] and path.stat().st_size == item['size_bytes'], 'output bytes/size')
    inputs = read(stage/'inputs.json')
    tokens = read(WORK/'token-preflight/calibration-tokens.json')
    pairs = read(WORK/'excerpts/calibration-pairs.json')
    rows = read(WORK/'excerpts/calibration-dataset.json')
    check(inputs['schema'] == 'jlens_pattern_calibration_calibration_inputs_v1' and inputs['role'] == 'calibration'
          and inputs['capture_locations'] == ['prefix_end'] and inputs['records'] == tokens['records']
          and inputs['pairs'] == pairs, 'exact captured records/pairs')
    ids = [f'C{i:02}-{s}' for i in range(1, 33) for s in ('L', 'R')]
    check([r['id'] for r in rows] == [r['id'] for r in inputs['records']] == ids and len(pairs) == 32, '64/32 frozen roster')
    indices = []
    for i, pair in enumerate(pairs):
        check(pair['id'] == f'C{i+1:02}' and pair['left'] == ids[2*i] and pair['right'] == ids[2*i+1]
              and pair['role'] == 'calibration', 'unchanged pair order/role')
        indices.append((2*i, 2*i+1))
    features = archive(stage/'features.npz')
    check(set(features) == {'activation_11', 'u32', 'source_mean64'}, 'feature archive schema')
    array(features['activation_11'], (64, 1, 1024), np.dtype('float32'), 'actual layer capture')
    original_arrays = archive(OLD/'preparation/directions.npz')
    for name, shape, dtype in [('u32', (1024, 4), 'float32'), ('source_mean64', (1024,), 'float64')]:
        array(features[name], shape, np.dtype(dtype), 'fixed '+name)
        check(np.array_equal(features[name], original_arrays[name]), 'unchanged canonical '+name)
    fitted = archive(stage/'patterns.npz')
    moment_summary = audit_moments(features['activation_11'][:, 0], features['u32'], indices, fitted)
    decoded = archive(stage/'decoder.npz')
    check(set(decoded) == {'inputs32', 'logits32', 'top_ids64', 'top_logits32'}, 'decoder archive schema')
    for name, shape, dtype in [('inputs32', (8, 1024), 'float32'), ('logits32', (8, 248320), 'float32'),
                               ('top_ids64', (8, 12), 'int64'), ('top_logits32', (8, 12), 'float32')]:
        array(decoded[name], shape, np.dtype(dtype), 'actual decoder '+name)
    check(np.array_equal(decoded['inputs32'], fitted['signed_inputs32']), 'actual signed decoder input bytes')
    payload = read(stage/'readouts.json')
    check(payload['schema'] == 'jlens_pattern_calibration_readouts_v1' and payload['layer'] == 11
          and payload['method'] == 'jlens', 'readout schema')
    readouts = payload['readouts']
    check([r['name'] for r in readouts] == [f'PC{i}{s}' for i in range(1, 5) for s in ('+', '-')], 'fixed signed readout order')
    for i, row in enumerate(readouts):
        check(row['axis'] == f'PC{i//2+1}' and row['pole'] == ('positive' if i % 2 == 0 else 'negative'), 'axis/pole binding')
    ties = audit_topk(decoded['logits32'], decoded['top_ids64'], decoded['top_logits32'], readouts)
    check(sha(stage/'original-interpretations.json') == ORIGINAL_SHA, 'original reference byte copy')
    audit_references(read(stage/'original-interpretations.json'), read(stage/'references.json'), readouts)
    result = {'status': 'PASS', 'scope': 'independent saved-array arithmetic and provenance; no remeasurement or redecoding',
              'audit_source_sha256': sha(Path(__file__)), 'receipt_sha256': receipt_sha,
              'producer_sha256': producer_sha, 'moments': moment_summary, 'cutoff_ties': ties,
              'max_single_pair_energy_share64': fitted['max_single_pair_energy_share64'].tolist(),
              'direction_energy_fraction64': fitted['direction_energy_fraction64'].tolist(),
              'evaluation_files_opened': 0, 'new_model_lens_tokenizer_or_network_calls': 0,
              'limitation': 'Raw decoded strings are source/receipt-bound, not independently re-decoded; reader benefit untested.'}
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-sha', required=True)
    parser.add_argument('--producer-sha', required=True)
    args = parser.parse_args()
    run(args.receipt_sha, args.producer_sha)
