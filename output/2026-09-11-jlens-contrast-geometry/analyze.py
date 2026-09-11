"""New saved-data geometry only. Import is inert; no model/network dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output')
NEW = WORK / '2026-09-11-j-lens-endpoint-location'
OLD = WORK / '2026-09-11-j-lens-content-template'
NM = MAIN / '2026-09-11-jlens-endpoint-location'
OM = MAIN / '2026-09-11-jlens-content-template'
UFILE = WORK / '2026-09-10-j-lens-fresh-content/preparation/directions.npz'
PINS = {
    NEW/'forwards/receipt.json': '4d76cce26a352fc64cace96a01cf2f8db29e2761368962ba0cae875396c51103',
    OLD/'forwards/receipt.json': '6308dd68c456b1cb0fc75df576c58d17fcee98834e23642f55b1e2c0dba21f02',
    NM/'analysis/results.json': 'fe0867a5f76773c11de5eeec973d6193f6ebb6d4b02e59b4b49a92e4869254fd',
    UFILE: '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa',
}
ROLES = ['verb', 'sentence_end', 'old_full_end']
TRANSITIONS = [(0, 1), (1, 2), (0, 2)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def vector_stats(delta, u):
    delta, u = np.asarray(delta, dtype=np.float64), np.asarray(u, dtype=np.float64)
    require(delta.ndim == 1 and u.ndim == 2 and u.shape[0] == delta.size, 'dimensions')
    require(np.isfinite(delta).all() and np.isfinite(u).all(), 'finite vectors')
    rho = float(np.linalg.norm(delta, ord=2))
    n = np.linalg.norm(u, ord=2, axis=0)
    require((n > 0).all(), 'zero direction')
    dot = delta @ u
    scalar_rho = math.sqrt(math.fsum(float(x)*float(x) for x in delta))
    scalar_n = np.array([math.sqrt(math.fsum(float(x)*float(x) for x in col)) for col in u.T])
    scalar_dot = np.array([math.fsum(float(x)*float(y) for x, y in zip(delta, col, strict=True)) for col in u.T])
    np.testing.assert_allclose(rho, scalar_rho, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(n, scalar_n, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(dot, scalar_dot, atol=1e-12, rtol=1e-12)
    q = dot/n
    cosine = q/rho if rho > 0 else None
    require(cosine is None or (abs(cosine) <= 1+1e-12).all(), 'cosine bounds')
    return {'norm': rho, 'direction_norms': n.tolist(), 'gap': dot.tolist(),
            'unit_projection': q.tolist(), 'alignment': None if cosine is None else cosine.tolist()}


def transition(a, b, da, db):
    ra, rb = a['norm'], b['norm']
    out = {'norm_ratio': rb/ra if ra else None, 'contrast_cosine': None,
           'magnitude_term': None, 'alignment_term': None,
           'unit_projection_change': (np.array(b['unit_projection'])-a['unit_projection']).tolist()}
    if ra > 0 and rb > 0:
        dot = math.fsum(float(x)*float(y) for x, y in zip(da, db, strict=True))
        out['contrast_cosine'] = dot/(ra*rb)
        require(abs(out['contrast_cosine']) <= 1+1e-12, 'transition cosine')
        ca, cb = np.array(a['alignment']), np.array(b['alignment'])
        magnitude = (rb-ra)*(ca+cb)/2
        alignment = (cb-ca)*(ra+rb)/2
        np.testing.assert_allclose(magnitude+alignment, out['unit_projection_change'], atol=1e-12, rtol=1e-12)
        out['magnitude_term'], out['alignment_term'] = magnitude.tolist(), alignment.tolist()
    return out


def fixtures():
    u = np.eye(2)
    cases = [([2, 0], [1, 0], [-1, 0], [0, 0]),
             ([1, 0], [0, 1], [0, 0], [-1, 1]),
             ([1, 0], [0, 2], [.5, .5], [-1.5, 1.5]),
             ([1, 0], [-1, 0], [0, 0], [-2, 0])]
    for da, db, m, al in cases:
        result = transition(vector_stats(da, u), vector_stats(db, u), da, db)
        np.testing.assert_allclose(result['magnitude_term'], m, atol=1e-12)
        np.testing.assert_allclose(result['alignment_term'], al, atol=1e-12)
    zero = vector_stats([0, 0], u)
    require(zero['alignment'] is None, 'undefined zero direction')
    require(transition(zero, vector_stats([1, 0], u), [0, 0], [1, 0])['norm_ratio'] is None, 'zero ratio')
    for invalid in [np.zeros((2, 2)), np.array([[1, math.nan], [0, 1]])]:
        try:
            vector_stats([1, 2], invalid)
        except ValueError:
            continue
        raise AssertionError('invalid direction accepted')
    print('PASS: seven fabricated cases; no scientific arrays loaded')


def run():
    target = HERE/'analysis'
    target.mkdir(exist_ok=False)
    pins = {**PINS, HERE/'protocol.md': sha(HERE/'protocol.md'), Path(__file__): sha(Path(__file__))}
    save(target/'attempt.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                              'inputs': {str(p): d for p, d in pins.items()}, 'automatic_retry': False})
    for path, digest in pins.items():
        require(sha(path) == digest, 'pin '+str(path))
    arrays, datasets, records, receipts = [], [], [], []
    for root, main, shape in [(NEW, NM, (16, 2, 1024)), (OLD, OM, (16, 1024))]:
        receipt = read(root/'forwards/receipt.json'); receipts.append(receipt)
        require(receipt['status'] == 'complete' and receipt['parameters_unchanged'], 'producer incomplete')
        for item in receipt['outputs']:
            path = root/'forwards'/item['path']
            require(sha(path) == item['sha256'], 'producer output changed')
            pins[path] = item['sha256']
        for name in ['dataset.json', 'pairs.json']:
            path = main/name
            matched = [d for p, d in receipt['input_pins'].items() if p.endswith('/'+main.name+'/'+name)]
            require(len(matched) == 1 and sha(path) == matched[0], 'dataset binding')
            pins[path] = matched[0]
        rows = read(main/'dataset.json')['rows']; datasets.append(rows)
        recs = read(root/'forwards/inputs.json')['records']; records.append(recs)
        require(len(rows) == len(recs) == 16 and len({r['id'] for r in rows}) == 16, 'row count')
        for row, rec in zip(rows, recs, strict=True):
            require(all(rec[k] == v for k, v in row.items()) and rec['layer'] == 11, 'record binding')
        with np.load(root/'forwards/features.npz', allow_pickle=False) as archive:
            h = archive['activation_11']
            if root == NEW:
                require(archive['locations'].tolist() == ROLES[:2], 'saved roles')
        require(h.shape == shape and h.dtype == np.float32 and np.isfinite(h).all(), 'activation schema')
        arrays.append(h.astype(np.float64))
    for key in ['model_weight_sha256', 'adapter_revision']:
        require(receipts[0][key] == receipts[1][key], 'different '+key)
    require(receipts[0]['runtime']['revision'] == receipts[1]['runtime']['revision'], 'model revision')
    for short, old in zip(records[0], records[1], strict=True):
        require(all(short[k] == old[k] for k in ['id', 'pair_id', 'template', 'pole']), 'old/new row binding')
        require(old['text'] == short['text']+' This happened yesterday.', 'suffix')
        require(old['input_ids'][:len(short['input_ids'])] == short['input_ids'], 'token prefix')
        require(list(short['captured_positions']) == ROLES[:2], 'record roles')
        require(short['captured_positions']['sentence_end'] == len(short['input_ids'])-1, 'new final role')
        require(old['captured_position'] == len(old['input_ids'])-1, 'old final role')
    pairs = read(NM/'pairs.json')['pairs']
    require(pairs == read(OM/'pairs.json')['pairs'] and len(pairs) == 8, 'pair binding')
    prior = read(NM/'analysis/results.json')
    require(prior['status'] == 'PASS' and prior['locations'] == ROLES, 'prior comparison')
    with np.load(UFILE, allow_pickle=False) as archive:
        u = archive['u32']
    require(u.shape == (1024, 4) and u.dtype == np.float32, 'directions schema')
    states = np.concatenate([arrays[0], arrays[1][:, None, :]], axis=1)
    idx = {row['id']: i for i, row in enumerate(datasets[0])}
    out, max_gap_error = [], 0.0
    for pair, saved in zip(pairs, prior['pairs'], strict=True):
        require(all(saved[k] == v for k, v in pair.items()), 'prior pair binding')
        delta = states[idx[pair['observation_id']]]-states[idx[pair['provision_id']]]
        stats = [vector_stats(d, u) for d in delta]
        gaps = np.array([s['gap'] for s in stats])
        np.testing.assert_allclose(gaps, saved['gaps_by_location'], atol=2e-12, rtol=0)
        np.testing.assert_array_equal(np.sign(gaps), np.sign(saved['gaps_by_location']))
        max_gap_error = max(max_gap_error, float(abs(gaps-np.array(saved['gaps_by_location'])).max()))
        out.append({**pair, 'locations': dict(zip(ROLES, stats, strict=True)),
                    'transitions': {ROLES[a]+'->'+ROLES[b]: transition(stats[a], stats[b], delta[a], delta[b]) for a, b in TRANSITIONS}})
    for path, digest in pins.items():
        require(sha(path) == digest, 'input changed during calculation')
    save(target/'results.json', {'status': 'PASS', 'completed_utc': datetime.now(timezone.utc).isoformat(),
         'inputs': {str(p): d for p, d in pins.items()}, 'numpy': np.__version__,
         'locations': ROLES, 'axes': ['PC1', 'PC2', 'PC3', 'PC4'], 'pairs': out,
         'max_prior_gap_error': max_gap_error, 'scope': 'Saved coordinate geometry, not semantics or causal mediation.'})
    print('PASS: 24 saved contrast vectors, 96 axis projections, 24 transitions; no model or network')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['fixtures', 'run'])
    args = parser.parse_args()
    fixtures() if args.mode == 'fixtures' else run()
