"""Frozen aggregate-estimation analysis. No model, tokenizer or decoder calls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
PROTOCOL_SHA = 'c933bf833b2e6bda0eacd0b3a52597adf1eda2ea05b7e69122064892a8d75b8e'
ADDENDUM_SHA = 'a6bd2ccc029be981b99604616a0055695c94434158bd478f07cbeaf5b329bb08'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def cosine(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/denominator) if denominator else None


def top_basis(matrix, k):
    spectrum, vectors = np.linalg.eigh(matrix)
    order = np.arange(len(spectrum)-1, -1, -1)
    spectrum, vectors = spectrum[order], vectors[:, order]
    if spectrum[-1] < -1e-10*max(float(spectrum[0]), 1e-30):
        raise ValueError('not PSD within numerical precision')
    u = vectors[:, :k]
    np.testing.assert_allclose(u.T@u, np.eye(k), atol=1e-10, rtol=0)
    return spectrum, u


def estimate(fit, query, references, k):
    """Query has [batches, observations, width], references [2, observations, width]."""
    mean = fit.mean(axis=0)
    centered = fit-mean
    covariance = centered.T@centered/len(fit)
    moment = covariance+np.outer(mean, mean)
    ce, u = top_basis(covariance, k)
    me, v = top_basis(moment, k)
    raw = query.mean(axis=1)
    refs = references.mean(axis=1)
    refs = np.concatenate([refs, refs.mean(axis=0)[None]], axis=0)
    projections = {'raw': raw, 'centered': (raw@u)@u.T,
                   'second_moment': (raw@v)@v.T, 'zero': np.zeros_like(raw),
                   'fit_mean': np.broadcast_to(mean, raw.shape).copy()}
    result = {'k': k, 'fit_count': len(fit), 'query_batches': len(query),
              'query_batch_size': query.shape[1], 'reference_group_size': references.shape[1],
              'fit_mean_norm': float(np.linalg.norm(mean)),
              'reference_norms': np.linalg.norm(refs, axis=1).tolist(),
              'centered_projected_reference_norms': np.linalg.norm(refs@u, axis=1).tolist(),
              'moment_projected_reference_norms': np.linalg.norm(refs@v, axis=1).tolist(),
              'centered_discarded_fit_mean_energy': float(np.linalg.norm(mean-(mean@u)@u.T)**2),
              'moment_discarded_fit_mean_energy': float(np.linalg.norm(mean-(mean@v)@v.T)**2),
              'centered_fit_mean_retained_fraction_squared': float(np.linalg.norm(mean@u)**2/np.linalg.norm(mean)**2) if np.linalg.norm(mean) else None,
              'second_moment_fit_mean_retained_fraction_squared': float(np.linalg.norm(mean@v)**2/np.linalg.norm(mean)**2) if np.linalg.norm(mean) else None,
              'arms': {}}
    for arm, values in projections.items():
        error = np.sum((values[:, None, :]-refs[None, :, :])**2, axis=-1)
        result['arms'][arm] = {'batch_squared_errors': error.tolist(),
            'mean_squared_errors': error.mean(axis=0).tolist(),
            'batch_cosines': [[cosine(row, ref) for ref in refs] for row in values],
            'norms': np.linalg.norm(values, axis=1).tolist(),
            'norm_ratios_to_raw': [float(np.linalg.norm(row)/np.linalg.norm(base)) if np.linalg.norm(base) else None
                                   for row, base in zip(values, raw, strict=True)]}
    raw_errors = np.array(result['arms']['raw']['batch_squared_errors'])
    for arm in ('centered', 'second_moment'):
        effect = raw_errors-np.array(result['arms'][arm]['batch_squared_errors'])
        result['arms'][arm]['mean_raw_minus_arm_error'] = effect.mean(axis=0).tolist()
        result['arms'][arm]['positive_batches'] = (effect > 0).sum(axis=0).tolist()
        result['arms'][arm]['positive_both_reference_groups'] = bool((effect.mean(axis=0)[:2] > 0).all())
    arrays = {'fit_mean64': mean, 'covariance64': covariance, 'centered_spectrum64': ce,
              'moment_spectrum64': me, 'centered_basis64': u, 'moment_basis64': v,
              'references64': refs, 'centered_coefficients64': raw@u, 'moment_coefficients64': raw@v,
              'centered_discarded64': raw-projections['centered'],
              'moment_discarded64': raw-projections['second_moment'], **projections}
    return result, arrays


def run(args):
    if (sha(__file__) != args.source_sha or sha(OUT/'protocol.md') != PROTOCOL_SHA
            or sha(OUT/'readout-addendum.md') != ADDENDUM_SHA):
        raise ValueError('source/protocol checksum mismatch')
    panel_path = OUT/'panel/panel.json'
    if sha(panel_path) != args.panel_sha or sha(args.features) != args.features_sha:
        raise ValueError('panel/features checksum mismatch')
    panel = json.loads(panel_path.read_text())
    rows = panel['rows']
    if panel['protocol_sha256'] != PROTOCOL_SHA or len(rows) != 1792:
        raise ValueError('panel scope mismatch')
    target = OUT/'analysis'
    target.mkdir()
    write(target/'attempt.json', vars(args))
    with np.load(args.features, allow_pickle=False) as archive:
        if list(archive['ids']) != [r['id'] for r in rows]:
            raise ValueError('feature row order mismatch')
        if any(archive[key].dtype != np.float32 for key in ('gradient32', 'activation32')):
            raise ValueError('original feature precision must be FP32')
        features = {family: archive[key].astype(np.float64) for family, key in
                    (('gradient', 'gradient32'), ('activation', 'activation32'))}
    results, arrays, displays, raw_displays = {}, {}, [], []
    for family, data in features.items():
        if data.shape != (1792, 1024) or not np.isfinite(data).all():
            raise ValueError('invalid feature array')
        fit = data[[r['role'] == 'fit' for r in rows]]
        query = np.stack([data[[r['role'] == 'query' and r['batch'] == b for r in rows]] for b in range(16)])
        ref = np.stack([data[[r['role'] == 'reference' and r['batch'] == b for r in rows]] for b in range(2)])
        result, values = estimate(fit, query, ref, 32)
        result['pooled_query_descriptive'] = {}
        for arm in ('raw', 'centered', 'second_moment'):
            pooled = values[arm].mean(axis=0)
            values['pooled_query_'+arm] = pooled
            result['pooled_query_descriptive'][arm] = {
                'squared_errors': np.sum((pooled[None, :]-values['references64'])**2, axis=1).tolist(),
                'cosines': [cosine(pooled, reference) for reference in values['references64']],
                'norm': float(np.linalg.norm(pooled))}
        results[family] = result
        arrays.update({family+'_'+key: value for key, value in values.items()})
        records = [(f'query{b:02}/{arm}', values[arm][b]) for b in range(16)
                   for arm in ('raw', 'centered', 'second_moment')]
        records += [('query_pooled/'+arm, values['pooled_query_'+arm])
                    for arm in ('raw', 'centered', 'second_moment')]
        records += [(name, values['references64'][i]) for i, name in enumerate(('reference0', 'reference1', 'reference_pooled'))]
        records += [('fit_mean', values['fit_mean64'])]
        for name, value in records:
            sign = -1 if family == 'gradient' else 1
            norm = float(np.linalg.norm(value))
            raw_displays.append(value.copy())
            displays.append({'name': family+'/'+name, 'family': family, 'sign': sign,
                             'norm': norm, 'defined': norm > 0})
    raw_displays = np.stack(raw_displays)
    if len(displays) != 110 or len({r['name'] for r in displays}) != 110:
        raise ValueError('exact110 display roster required')
    inputs = np.stack([row['sign']*value/row['norm'] if row['defined'] else np.zeros_like(value)
                       for row, value in zip(displays, raw_displays, strict=True)]).astype(np.float32)
    with (target/'arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays, display_raw64=raw_displays, display_inputs32=inputs)
    write(target/'displays.json', displays)
    write(target/'results.json', {'schema': 'jlens_aggregate_estimation_v1', 'results': results,
          'primary_positive': results['gradient']['arms']['centered']['positive_both_reference_groups'],
          'independent_replications': False, 'no_semantic_quality_claim': True})
    if (sha(__file__) != args.source_sha or sha(OUT/'protocol.md') != PROTOCOL_SHA
            or sha(OUT/'readout-addendum.md') != ADDENDUM_SHA
            or sha(panel_path) != args.panel_sha or sha(args.features) != args.features_sha):
        raise ValueError('source/input changed during analysis; no complete receipt')
    write(target/'receipt.json', {'source_sha256': args.source_sha, 'protocol_sha256': PROTOCOL_SHA,
          'readout_addendum_sha256': ADDENDUM_SHA,
          'panel_sha256': args.panel_sha, 'features_sha256': args.features_sha,
          'artifacts': {name: sha(target/name) for name in ('arrays.npz', 'displays.json', 'results.json')},
          'status': 'complete', 'numpy': np.__version__, 'model_calls': 0, 'decoder_calls': 0})
    print(json.dumps({'status': 'complete', 'primary_positive': results['gradient']['arms']['centered']['positive_both_reference_groups']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('features', 'features-sha', 'panel-sha', 'source-sha'):
        parser.add_argument('--'+arg, required=True)
    run(parser.parse_args())
