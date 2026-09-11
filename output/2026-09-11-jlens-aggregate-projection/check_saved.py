"""Independent saved-artifact checks; never import producer/analyzer or run models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
WORKER = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-aggregate-projection')
PANEL_SHA = '8a034642a4946c4cf7967d28adecb470151101d3de24f10aca8faf5e89bbbe74'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def cosine(a, b):
    norm = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a, b)/norm) if norm else None


def optional_close(actual, expected):
    if expected is None:
        assert actual is None
        return 0.
    return close(actual, expected)


def close(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('comparison requires identical shape and finite arrays')
    scale = max(float(np.linalg.norm(expected)), 1e-20)
    error = float(np.linalg.norm(np.asarray(actual)-expected))
    if error > 1e-9*scale:
        raise ValueError(f'numerical disagreement: relative {error/scale}')
    return error/scale


def acquisition():
    stage = WORKER/'acquisition'
    receipt = read(stage/'receipt.json')
    assert receipt['status'] == 'complete'
    assert receipt['forward_count'] == receipt['backward_count'] == 1792
    assert receipt['parameters_unchanged'] and receipt['parameter_gradients_absent']
    assert receipt['layer'] == 11 and receipt['capture_index'] == 31
    assert not receipt['tokenizer_loaded'] and not receipt['pca_fit'] and not receipt['lens_loaded']
    for path, checksum in receipt['input_pins'].items():
        assert sha(path) == checksum
    for row in receipt['outputs']:
        assert sha(stage/row['path']) == row['sha256']
        assert (stage/row['path']).stat().st_size == row['size_bytes']
    assert sha(ROOT/'panel/panel.json') == PANEL_SHA
    panel = read(ROOT/'panel/panel.json')
    assert read(stage/'inputs.json') == panel
    with np.load(stage/'features.npz', allow_pickle=False) as archive:
        arrays = dict(archive)
    assert list(arrays['ids']) == [r['id'] for r in panel['rows']]
    expected_shapes = {'activation32': (1792, 1024), 'gradient32': (1792, 1024),
                       'losses32': (1792, 8), 'mean_losses32': (1792,)}
    for key, shape in expected_shapes.items():
        assert arrays[key].shape == shape and arrays[key].dtype == np.float32
        assert np.isfinite(arrays[key]).all()
    np.testing.assert_allclose(arrays['losses32'].mean(axis=1), arrays['mean_losses32'], rtol=3e-7, atol=1e-7)
    assert (arrays['losses32'] >= 0).all()
    artifacts = read(stage/'row-artifacts.json')
    assert len(artifacts) == 1792
    for i, entry in enumerate(artifacts):
        assert entry['index'] == i and entry['id'] == panel['rows'][i]['id']
        assert entry['path'] == f'rows/{i:04}.npz'
        assert sha(stage/entry['path']) == entry['sha256']
        with np.load(stage/entry['path'], allow_pickle=False) as row:
            assert set(row.files) == set(expected_shapes)
            for key in expected_shapes:
                np.testing.assert_array_equal(row[key], arrays[key][i])
    return {'rows_checked': 1792, 'per_row_exact': True, 'input_receipt_bindings': True,
            'all_finite_FP32': True, 'feature_sha256': sha(stage/'features.npz'),
            'receipt_sha256': sha(stage/'receipt.json'), 'no_model_reexecution': True}


def check_geometry(x, query, refs, a, family):
    """Check direct sums, eigensystem and scalar errors independently of analysis source."""
    get = lambda key: a[family+'_'+key]
    mu = np.sum(x, axis=0)/len(x)
    cov = sum(np.outer(row-mu, row-mu) for row in x)/len(x)
    maximum = max(close(get('fit_mean64'), mu), close(get('covariance64'), cov))
    raw = np.sum(query, axis=1)/query.shape[1]
    r = np.sum(refs, axis=1)/refs.shape[1]
    r = np.vstack((r, np.sum(r, axis=0)/2))
    maximum = max(maximum, close(get('references64'), r), close(get('raw'), raw))
    expected = {'raw': raw, 'zero': np.zeros_like(raw), 'fit_mean': np.broadcast_to(mu, raw.shape)}
    for prefix, arm in [('centered', 'centered'), ('moment', 'second_moment')]:
        matrix = cov if prefix == 'centered' else cov+np.outer(mu, mu)
        u = get(prefix+'_basis64')
        spectrum = get(prefix+'_spectrum64')
        maximum = max(maximum, close(u.T@u, np.eye(u.shape[1])),
                      close(matrix@u, u*spectrum[:u.shape[1]]),
                      close(spectrum, np.linalg.eigvalsh(matrix)[::-1]))
        # Use explicit P, unlike producer's two coordinate multiplications.
        projector = u@u.T
        projected = np.stack([projector@row for row in raw])
        expected[arm] = projected
        maximum = max(maximum, close(get(arm), projected),
                      close(get(prefix+'_coefficients64'), raw@u),
                      close(get(prefix+'_discarded64'), raw-projected))
    return maximum, expected, r


def analysis():
    target = ROOT/'analysis'
    receipt = read(target/'receipt.json')
    assert receipt['status'] == 'complete'
    assert receipt['source_sha256'] == sha(ROOT/'analyze.py')
    assert receipt['panel_sha256'] == PANEL_SHA == sha(ROOT/'panel/panel.json')
    assert receipt['features_sha256'] == sha(WORKER/'acquisition/features.npz')
    for name, checksum in receipt['artifacts'].items():
        assert sha(target/name) == checksum
    rows = read(ROOT/'panel/panel.json')['rows']
    with np.load(WORKER/'acquisition/features.npz', allow_pickle=False) as f:
        assert list(f['ids']) == [r['id'] for r in rows]
        for key in ('activation32', 'gradient32'):
            assert f[key].dtype == np.float32 and f[key].shape == (1792, 1024) and np.isfinite(f[key]).all()
        features = {family: f[family+'32'].astype(np.float64) for family in ('activation', 'gradient')}
    with np.load(target/'arrays.npz', allow_pickle=False) as f:
        arrays = dict(f)
    results = read(target/'results.json')
    maximum = 0.; display_expected = []
    for family, x in features.items():
        assert all(arrays[family+'_'+prefix+'_basis64'].shape == (1024, 32) for prefix in ('centered', 'moment'))
        fit = x[[r['role'] == 'fit' for r in rows]]
        query = np.stack([x[[r['role'] == 'query' and r['batch'] == i for r in rows]] for i in range(16)])
        ref = np.stack([x[[r['role'] == 'reference' and r['batch'] == i for r in rows]] for i in range(2)])
        error, expected, references = check_geometry(fit, query, ref, arrays, family)
        maximum = max(maximum, error)
        family_result = results['results'][family]
        mean = fit.mean(axis=0)
        maximum = max(maximum, close(family_result['fit_mean_norm'], np.linalg.norm(mean)),
                      close(family_result['reference_norms'], np.linalg.norm(references, axis=1)))
        for prefix, field in [('centered', 'centered'), ('moment', 'second_moment')]:
            u = arrays[family+'_'+prefix+'_basis64']
            maximum = max(maximum,
                close(family_result[prefix+'_projected_reference_norms'], np.linalg.norm(references@u, axis=1)),
                close(family_result[prefix+'_discarded_fit_mean_energy'], np.linalg.norm(mean-(mean@u)@u.T)**2),
                optional_close(family_result[field+'_fit_mean_retained_fraction_squared'],
                               float(np.linalg.norm(mean@u)**2/np.linalg.norm(mean)**2) if np.linalg.norm(mean) else None))
        for arm, values in expected.items():
            errors = np.array([[np.dot(row-r, row-r) for r in references] for row in values])
            record = results['results'][family]['arms'][arm]
            maximum = max(maximum, close(record['batch_squared_errors'], errors),
                          close(record['mean_squared_errors'], errors.mean(axis=0)),
                          close(record['norms'], np.linalg.norm(values, axis=1)))
            assert len(record['batch_cosines']) == len(values)
            for i, row in enumerate(values):
                assert len(record['batch_cosines'][i]) == 3
                for j, reference in enumerate(references):
                    maximum = max(maximum, optional_close(record['batch_cosines'][i][j], cosine(row, reference)))
                denominator = np.linalg.norm(expected['raw'][i])
                maximum = max(maximum, optional_close(record['norm_ratios_to_raw'][i],
                                  float(np.linalg.norm(row)/denominator) if denominator else None))
            if arm in ('raw', 'centered', 'second_moment'):
                pooled = values.mean(axis=0)
                pooled_result = family_result['pooled_query_descriptive'][arm]
                maximum = max(maximum,
                    close(pooled_result['squared_errors'], [np.dot(pooled-r, pooled-r) for r in references]),
                    close(pooled_result['norm'], np.linalg.norm(pooled)))
                for j, reference in enumerate(references):
                    maximum = max(maximum, optional_close(pooled_result['cosines'][j], cosine(pooled, reference)))
            if arm in ('centered', 'second_moment'):
                rawerrors = np.array([[np.dot(row-r, row-r) for r in references] for row in expected['raw']])
                delta = rawerrors-errors
                maximum = max(maximum, close(record['mean_raw_minus_arm_error'], delta.mean(axis=0)))
                assert record['positive_batches'] == (delta > 0).sum(axis=0).tolist()
                assert record['positive_both_reference_groups'] == bool((delta.mean(axis=0)[:2] > 0).all())
        assert results['primary_positive'] == results['results']['gradient']['arms']['centered']['positive_both_reference_groups']
    displays = read(target/'displays.json')
    suffixes = [f'query{i:02}/{arm}' for i in range(16) for arm in ('raw', 'centered', 'second_moment')]
    suffixes += ['query_pooled/'+arm for arm in ('raw', 'centered', 'second_moment')]
    suffixes += ['reference0', 'reference1', 'reference_pooled', 'fit_mean']
    assert [r['name'] for r in displays] == [family+'/'+name for family in ('gradient', 'activation') for name in suffixes]
    for i, d in enumerate(displays):
        family, name = d['name'].split('/', 1)
        assert d['family'] == family
        base = arrays[family+'_raw']
        if name.startswith('query_pooled/'):
            arm = name.split('/')[1]; expected = arrays[family+'_'+arm].mean(axis=0)
        elif name.startswith('query'):
            batch, arm = name.split('/'); expected = arrays[family+'_'+arm][int(batch[5:])]
        elif name == 'fit_mean': expected = arrays[family+'_fit_mean64']
        else: expected = arrays[family+'_references64'][{'reference0': 0, 'reference1': 1, 'reference_pooled': 2}[name]]
        maximum = max(maximum, close(arrays['display_raw64'][i], expected))
        norm = float(np.linalg.norm(expected)); sign = -1 if family == 'gradient' else 1
        assert d['norm'] == norm and d['sign'] == sign and d['defined'] == bool(norm)
        actual = (sign*expected/norm).astype(np.float32) if norm else np.zeros(1024, dtype=np.float32)
        np.testing.assert_array_equal(arrays['display_inputs32'][i], actual)
    return {'maximum_relative_numeric_disagreement': maximum, 'display_count': 110,
            'direct_error_and_eigensystem_checks': True, 'no_model_or_analyzer_import': True,
            'analysis_receipt_sha256': sha(target/'receipt.json')}


def decoder():
    target = WORKER/'decoding'
    receipt = read(target/'receipt.json')
    assert receipt['status'] == 'complete' and receipt['display_count'] == 110
    assert receipt['model_forward_count'] == receipt['tokenizer_encode_calls'] == 0
    assert receipt['parameters_unchanged'] and receipt['lens_unchanged']
    for path, checksum in receipt['input_pins'].items(): assert sha(path) == checksum
    for name, checksum in receipt['artifacts'].items(): assert sha(target/name) == checksum
    with np.load(target/'decoder.npz', allow_pickle=False) as f: arrays = dict(f)
    with np.load(ROOT/'analysis/arrays.npz', allow_pickle=False) as f: inputs = f['display_inputs32']
    displays = read(ROOT/'analysis/displays.json')
    rows = read(target/'readouts.json')['readouts']
    assert len(rows) == 110 and [r['name'] for r in rows] == [r['name'] for r in displays]
    np.testing.assert_array_equal(arrays['inputs32'], inputs)
    assert arrays['logits32'].shape == (110, 248320) and arrays['logits32'].dtype == np.float32
    assert arrays['transported32'].shape == (110, 1024) and arrays['transported32'].dtype == np.float32
    assert np.isfinite(arrays['logits32']).all() and np.isfinite(arrays['transported32']).all()
    assert arrays['top_ids64'].shape == arrays['top_logits32'].shape == (110, 12)
    assert arrays['top_ids64'].dtype == np.int64 and arrays['top_logits32'].dtype == np.float32
    assert arrays['defined'].dtype == np.bool_ and arrays['defined'].shape == (110,)
    index = {r['name']: i for i, r in enumerate(rows)}; maximum = 0.
    for i, row in enumerate(rows):
        assert {k: row[k] for k in displays[i]} == displays[i]
        assert arrays['defined'][i] == row['defined']
        if row['defined']:
            ids = row['token_ids']; scores = row['scores']; logits = arrays['logits32'][i]
            assert len(ids) == len(set(ids)) == len(scores) == len(row['tokens']) == 12
            assert all(type(t) is str for t in row['tokens'])
            assert all(type(t) is int and 0 <= t < 248320 for t in ids)
            np.testing.assert_array_equal(ids, arrays['top_ids64'][i])
            np.testing.assert_array_equal(scores, arrays['top_logits32'][i])
            np.testing.assert_array_equal(logits[ids], scores)
            assert all(scores[j] >= scores[j+1] for j in range(11))
            cutoff = min(scores)
            assert np.count_nonzero(logits > cutoff) < 12 <= np.count_nonzero(logits >= cutoff)
            assert row['ties'] == {'cutoff_logit': cutoff,
                'strictly_above_cutoff_count': int((logits > cutoff).sum()),
                'cutoff_tie_count': int((logits == cutoff).sum()),
                'selected_at_cutoff_count': scores.count(cutoff)}
        else:
            assert row['token_ids'] == row['tokens'] == row['scores'] == [] and row['ties'] is None
            assert (arrays['top_ids64'][i] == -1).all()
            assert all((arrays[key][i] == 0).all() for key in ('transported32', 'logits32', 'top_logits32'))
        for reference in ('reference0', 'reference1', 'reference_pooled'):
            j = index[row['family']+'/'+reference]
            record = row['reference_fidelity'][reference]
            if row['defined'] and rows[j]['defined']:
                expected_cos = cosine(arrays['logits32'][i].astype(np.float64), arrays['logits32'][j].astype(np.float64))
                overlap = len(set(row['token_ids']).intersection(rows[j]['token_ids']))
                maximum = max(maximum, optional_close(record['full_logit_cosine'], expected_cos))
                assert record['top12_overlap_count'] == overlap and record['top12_overlap_fraction'] == overlap/12
            else: assert all(v is None for v in record.values())
    assert receipt['decode_count'] == receipt['topk_calls'] == int(arrays['defined'].sum())
    return {'display_count': 110, 'stored_topk_and_fidelity_checked': True,
            'maximum_fidelity_disagreement': maximum, 'actual_model_or_decoder_rerun': False,
            'token_text_not_independently_redecoded': True,
            'receipt_sha256': sha(target/'receipt.json')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('acquisition', 'analysis', 'decoder'))
    args = p.parse_args()
    output = ROOT/(args.stage+'-audit.json')
    if output.exists(): raise FileExistsError(output)
    checks = {'acquisition': acquisition, 'analysis': analysis, 'decoder': decoder}[args.stage]()
    with output.open('x') as f:
        json.dump({'status': 'PASS', 'source_sha256': sha(__file__), **checks}, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'stage': args.stage, 'status': 'PASS', **checks}))
