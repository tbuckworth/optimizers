"""New saved-panel covariance diagnostic. Import and --self-test do no study I/O."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output')
OLD = WORK / '2026-09-10-j-lens-fresh-content'
PINS = {
    'directions': (OLD/'preparation/directions.npz', '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa'),
    'selection': (OLD/'preparation/selection.json', 'aac75625b67d9a90ba351d979f3e1a77cd75f69706a8b3113e071630de023f4d'),
    'fit_features': (WORK/'2026-09-10-j-lens-completions/features.npz', '96e936942fd5cde63c9c482ef19a004ba3fee5adbc46676373772fa5e4123b61'),
    'fit_dataset': (WORK/'2026-09-10-j-lens-completions/dataset.json', 'b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6'),
    'authored_features': (OLD/'fresh-features/features.npz', '53547cd61e81903698971debbb82ddcec8f9d8637ec9485da9331c40373b67ab'),
    'authored_dataset': (MAIN/'output/2026-09-10-jlens-fresh-content/dataset.json', 'c655a5531af3215a29e332ddc6b23ac4854b0bffdf91b8ad5e8a045748ff7cd9'),
    'wiki_features': (WORK/'2026-09-11-j-lens-independent-content/forwards/features.npz', 'd6ed700ce1b4f74001097880b5ddc0b41481a31c138395156dfda55544c434b1'),
    'wiki_dataset': (MAIN/'output/2026-09-11-jlens-independent-content/excerpts-resumed/dataset.json', '7391a1fa6c02872cdeabb2e8dffc1e63181b6379d80e2042f33b38a61308ff02'),
}
TOPICS = ['astronomy', 'cooking', 'football', 'programming']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify_pins():
    for path, expected in PINS.values():
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16*1024**2,
                'bounded regular input')
        require(sha(path) == expected, 'input digest: ' + str(path))


def covariance_metrics(h, u, groups):
    import numpy as np
    h, u = np.asarray(h, dtype=np.float64), np.asarray(u, dtype=np.float64)
    require(h.ndim == u.ndim == 2 and h.shape[1] == u.shape[0] and
            h.shape[0] == len(groups) >= 2, 'matrix/label shape')
    require(np.isfinite(h).all() and np.isfinite(u).all(), 'finite inputs')
    mean = h.mean(axis=0)
    x = h - mean
    z = x @ u
    cu = x.T @ z / len(h)
    covariance = z.T @ z / len(h)
    variance = np.diag(covariance).copy()
    q = (u*u).sum(axis=0)
    trace = float((x*x).sum()/len(h))
    require((variance > 0).all() and (q > 0).all() and trace > 0, 'positive direction variance/norm')
    slope = cu / variance
    reference = u / q
    cosine = variance / (np.sqrt(q) * np.linalg.norm(cu, axis=0))
    relative = np.linalg.norm(slope-reference, axis=0) / np.linalg.norm(reference, axis=0)
    between = np.zeros(u.shape[1], dtype=np.float64)
    within = np.zeros_like(between)
    group_values = []
    for group in sorted(set(groups)):
        index = [i for i, g in enumerate(groups) if g == group]
        subset = z[index]
        center = subset.mean(axis=0)
        between += len(index)/len(h) * center**2
        within += ((subset-center)**2).sum(axis=0)/len(h)
        group_values.append({'group': group, 'rows': len(index), 'centered_mean_scores': center.tolist()})
    require(np.allclose(between+within, variance, rtol=2e-11, atol=2e-12), 'variance decomposition')
    require(np.allclose((u*cu).sum(axis=0), variance, rtol=2e-11, atol=2e-12), 'quadratic form')
    require(np.allclose((u*slope).sum(axis=0), 1, rtol=2e-11, atol=2e-12), 'regression slope normalization')
    require(np.allclose(relative**2, 1/cosine**2-1, rtol=2e-10, atol=2e-12), 'off-axis identity')
    correlation = covariance / np.sqrt(variance[:, None]*variance[None, :])
    arrays = {'mean': mean, 'scores_centered': z, 'covariance_times_u': cu, 'regression_slopes': slope}
    axes = [{'axis': 'PC'+str(a+1), 'variance': float(variance[a]), 'u_squared_norm': float(q[a]),
             'cosine_u_Cu': float(cosine[a]), 'relative_off_axis': float(relative[a]),
             'normalized_direction_variance_fraction': float(variance[a]/q[a]/trace),
             'between_group_fraction': float(between[a]/variance[a]),
             'within_group_fraction': float(within[a]/variance[a])} for a in range(u.shape[1])]
    return {'rows': len(h), 'width': h.shape[1], 'covariance_divisor': len(h), 'trace_covariance': trace,
            'axes': axes, 'score_correlation': correlation.tolist(), 'groups': group_values}, arrays


def scalar_check(h, u, groups, summary, arrays):
    """Independent scalar fsum ordering/centering, not another model measurement."""
    import numpy as np
    h, u = h.astype(np.float64), u.astype(np.float64)
    n, width = h.shape
    k = u.shape[1]
    mean = [math.fsum(float(h[i,j]) for i in range(n))/n for j in range(width)]
    x = [[float(h[i,j])-mean[j] for j in range(width)] for i in range(n)]
    z = [[math.fsum(x[i][j]*float(u[j,a]) for j in range(width)) for a in range(k)] for i in range(n)]
    cu = [[math.fsum(x[i][j]*z[i][a] for i in range(n))/n for a in range(k)] for j in range(width)]
    count, largest = 0, 0.0
    def compare(a, b):
        nonlocal count, largest
        count += 1
        largest = max(largest, abs(a-b))
        require(math.isclose(a, b, rel_tol=2e-11, abs_tol=2e-12), 'scalar fsum check')
    for j in range(width):
        for a in range(k):
            compare(cu[j][a], float(arrays['covariance_times_u'][j,a]))
    for a in range(k):
        variance = math.fsum(row[a]**2 for row in z)/n
        between = math.fsum(len(ix)/n*(math.fsum(z[i][a] for i in ix)/len(ix))**2
                            for g in sorted(set(groups))
                            for ix in [[i for i,t in enumerate(groups) if t == g]])
        compare(variance, summary['axes'][a]['variance'])
        compare(between, summary['axes'][a]['between_group_fraction']*variance)
    return {'checks': count, 'largest_absolute_difference': largest, 'relative_tolerance': 2e-11,
            'absolute_tolerance': 2e-12}


def run():
    import numpy as np
    target = HERE/'analysis'
    target.mkdir(exist_ok=False)
    source_hash = sha(Path(__file__))
    protocol_hash = sha(HERE/'protocol.md')
    write(target/'attempt.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                                'source_sha256': source_hash, 'protocol_sha256': protocol_hash})
    try:
        verify_pins()
        read = lambda key: json.loads(PINS[key][0].read_text())
        fit_rows, selection = read('fit_dataset'), read('selection')
        expected = [f'{t}-{i}-{s}' for t in TOPICS for i in range(6) for s in ['plain','note']]
        fit_ids = [f'{t}-{i}-{s}' for t in TOPICS for i in range(4) for s in ['plain','note']]
        require([r['id'] for r in fit_rows] == expected and selection['fit_ids'] == fit_ids, 'fixed fit roster')
        mask = [i for i,r in enumerate(fit_rows) if r['id'] in fit_ids]
        require(len(mask) == 32 and all(fit_rows[i]['split'] == 'fit' for i in mask), 'fit split')
        with np.load(PINS['directions'][0], allow_pickle=False) as f:
            u, original_mean, fit_scores = f['u32'], f['source_mean64'], f['fit_scores64']
        require(u.shape == (1024,4) and u.dtype == np.float32 and original_mean.shape == (1024,)
                and original_mean.dtype == np.float64 and fit_scores.shape == (32,4), 'canonical arrays')
        require(np.allclose(u.astype(np.float64).T@u.astype(np.float64), np.eye(4), rtol=0, atol=1e-6), 'canonical near orthonormality')
        result, saved = {}, {}
        for name in ['fit','authored','wiki']:
            with np.load(PINS[name+'_features'][0], allow_pickle=False) as f:
                h = f['activation_11']
                stored = fit_scores if name == 'fit' else f['scores64']
            require(h.shape == (48 if name == 'fit' else 24, 1024) and h.dtype == np.float32,
                    'exact source activation shape/dtype')
            if name == 'fit':
                h = h[mask]
                rows = [fit_rows[i] for i in mask]
                require(np.allclose(h.astype(np.float64).mean(0), original_mean, rtol=0, atol=1e-12), 'fit mean')
                groups = [r['group'] for r in rows]
            else:
                rows = read(name+'_dataset')
                suffix = 'new' if name == 'authored' else 'wiki'
                require([r['id'] for r in rows] == [f'{t}-{suffix}-{i}' for t in TOPICS for i in range(6)], 'fresh roster')
                groups = [r['topic'] for r in rows]
            expected_groups = [t for t in TOPICS for _ in range(8 if name == 'fit' else 6)]
            require(groups == expected_groups, 'unchanged group roster')
            projection = (h.astype(np.float64)-original_mean)@u.astype(np.float64)
            require(stored.shape == (len(h),4) and stored.dtype == np.float64 and
                    np.allclose(projection, stored, rtol=2e-13, atol=2e-12), 'canonical saved score identity')
            summary, arrays = covariance_metrics(h, u, groups)
            summary['ids'] = [r['id'] for r in rows]
            summary['scalar_check'] = scalar_check(h, u, groups, summary, arrays)
            summary['max_saved_score_difference'] = float(np.max(np.abs(projection-stored)))
            result[name] = summary
            saved.update({name+'_'+key: value for key,value in arrays.items()})
        for panel in result.values():
            for a,row in enumerate(panel['axes']):
                row['variance_ratio_to_fit'] = row['variance']/result['fit']['axes'][a]['variance']
        verify_pins()
        require(sha(Path(__file__)) == source_hash and sha(HERE/'protocol.md') == protocol_hash, 'unchanged source/protocol')
        with (target/'arrays.npz').open('xb') as f:
            np.savez_compressed(f, **saved)
        write(target/'results.json', {'schema':'jlens_transfer_geometry_v1', 'panels':result})
        write(target/'receipt.json', {'status':'PASS', 'completed_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':source_hash, 'protocol_sha256':protocol_hash, 'numpy_version':np.__version__,
            'python_version':sys.version, 'inputs':{k:{'path':str(p),'sha256':v} for k,(p,v) in PINS.items()},
            'outputs':{name:sha(target/name) for name in ['results.json','arrays.npz']},
            'new_model_calls':0, 'new_judgments':0, 'new_pca':False})
        print(json.dumps({'status':'PASS','panels':{k:{'cosines':[a['cosine_u_Cu'] for a in v['axes']],
             'variance_ratios':[a['variance_ratio_to_fit'] for a in v['axes']], 'checks':v['scalar_check']} for k,v in result.items()}}, indent=2))
    except Exception as e:
        write(target/'failure.json', {'error_type':type(e).__name__, 'message':str(e)})
        raise


def self_test():
    import numpy as np
    # Orthogonal centered columns, variances 4 and 1.
    h = np.array([[2,1],[2,-1],[-2,1],[-2,-1]], dtype=np.float64)
    u = np.eye(2)
    group = ['a','a','b','b']
    r, a = covariance_metrics(h, u, group)
    np.testing.assert_allclose([v['cosine_u_Cu'] for v in r['axes']], [1,1])
    np.testing.assert_allclose([v['between_group_fraction'] for v in r['axes']], [1,0])
    np.testing.assert_allclose(a['regression_slopes'], u)
    r2, a2 = covariance_metrics(h+np.array([50,-23]), u, group)
    assert r2 == r
    r3, a3 = covariance_metrics(3*h, u, group)
    np.testing.assert_allclose(a3['regression_slopes'], a['regression_slopes'])
    np.testing.assert_allclose([v['variance'] for v in r3['axes']], [36,9])
    r4, a4 = covariance_metrics(h, 2*u, group)
    np.testing.assert_allclose(a4['regression_slopes'], .5*u)
    # C = [[4,4],[4,5]], u=e1: slope [1,1], cosine 1/sqrt(2).
    rotated = h@np.array([[1,1],[0,1]])
    rr, aa = covariance_metrics(rotated, u, group)
    np.testing.assert_allclose(aa['regression_slopes'][:,0], [1,1])
    np.testing.assert_allclose(rr['axes'][0]['cosine_u_Cu'], 1/math.sqrt(2))
    np.testing.assert_allclose(rr['axes'][0]['relative_off_axis'], 1)
    scalar_check(rotated, u, group, rr, aa)
    # Unequal group counts: between variance 49/72, total variance 8/9.
    unbalanced = np.array([[0.],[1.],[2.],[2.],[2.],[3.]])
    rb, _ = covariance_metrics(unbalanced, np.ones((1,1)), ['a','a','b','b','b','b'])
    np.testing.assert_allclose(rb['axes'][0]['variance'], 8/9)
    np.testing.assert_allclose(rb['axes'][0]['between_group_fraction'], 49/64)
    rejected = 0
    for bad in [np.zeros((4,2)), np.full((4,2), np.nan), np.ones((3,2))]:
        try: covariance_metrics(bad, u, group)
        except ValueError: rejected += 1
    require(rejected == 3, 'invalid fixtures rejected')
    print('PASS: fabricated covariance, nonunit, off-axis, translation, scaling, weighted groups, scalar cross-check and three failure fixtures; no study I/O')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['self-test','run'])
    args = parser.parse_args()
    self_test() if args.stage == 'self-test' else run()
