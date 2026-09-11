"""Read-only accounting of saved judgments; an illustrative four-state algebra check.

No model, numerical archive, old grader, reader, network or experiment producer.
The toy is a mathematical counterexample, not data from the language model.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '2026-09-11-jlens-natural-addon'
PINS = {
    'graded/grades.json': '5f9856d2c4c470923497c38429ff3385bdebb5140b9047d356e9fbbdbcce3efd',
    'excerpts/dataset.json': 'd57300cb8511e409ff78e6ddc9fb6e1cf11b91bc629bf5492db8932cf585b129',
    'excerpts/pairs.json': '0ae85708e0b5da2942e31aa96af4ffa642c1294ff4c8b78612ec2dfc0ee54725',
}
AXES = ['PC1', 'PC2', 'PC3', 'PC4']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Non-finite JSON value: ' + value)


def read_pinned(name):
    path = OLD / name
    if path.stat().st_size > 2_000_000 or sha(path) != PINS[name]:
        raise ValueError('Input size/hash mismatch: ' + name)
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=reject_constant)


def classify(delta):
    if len(delta) != 2 or any(x not in (-1, 0, 1) for x in delta):
        raise ValueError('Expected two binary-credit differences')
    if delta == [1, 1]:
        return 'gain_both'
    if delta == [-1, -1]:
        return 'harm_both'
    if delta == [0, 0]:
        return 'unchanged'
    if 0 in delta:
        return 'one_group_only'
    return 'opposed'


def covariance(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys, strict=True)) / len(xs)


def toy():
    rows = []
    for g, z0 in itertools.product([-1, 1], repeat=2):
        h1, h2 = math.sqrt(3)*g+z0, g*z0
        rows.append({'group': g, 'Z': z0, 'h1': h1, 'h2': h2, 'score': h1, 'readout': h1+2*h2})
    h1, h2 = [[r[k] for r in rows] for k in ['h1', 'h2']]
    matrix = [[covariance(a, b) for b in [h1, h2]] for a in [h1, h2]]
    slopes = {}
    for name, group in [('pooled', rows), ('group_minus', rows[:2]), ('group_plus', rows[2:])]:
        x, y = [[r[k] for r in group] for k in ['score', 'readout']]
        slopes[name] = covariance(x, y)/covariance(x, x)
    return {'label': 'ILLUSTRATIVE ALGEBRA, NOT MODEL DATA', 'equal_probability': 0.25,
            'states': rows, 'covariance': matrix, 'slopes': slopes}


def stream_pattern(rows):
    """Toy-only verification of one fixed-score cross-scatter recurrence."""
    mh, mz, cross, zz = [0., 0.], 0., [0., 0.], 0.
    for n, row in enumerate(rows, 1):
        h, z = [row['h1'], row['h2']], row['score']
        dh, dz = [h[i]-mh[i] for i in range(2)], z-mz
        mh = [mh[i]+dh[i]/n for i in range(2)]
        mz += dz/n
        cross = [cross[i]+(n-1)/n*dh[i]*dz for i in range(2)]
        zz += (n-1)/n*dz*dz
    if zz <= 0:
        raise ValueError('No score variance')
    return [c/zz for c in cross]


def test():
    cases = Counter(classify(list(x)) for x in itertools.product([-1, 0, 1], repeat=2))
    assert cases == {'gain_both': 1, 'harm_both': 1, 'unchanged': 1, 'one_group_only': 4, 'opposed': 2}
    t = toy()
    for actual, expected in zip(sum(t['covariance'], []), [4, 0, 0, 1], strict=True):
        assert math.isclose(actual, expected, abs_tol=1e-12)
    for key, expected in [('pooled', 1), ('group_minus', -1), ('group_plus', 3)]:
        assert math.isclose(t['slopes'][key], expected, abs_tol=1e-12)
    for order in itertools.permutations(t['states']):
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(stream_pattern(order), [1, 0]))
    for group, expected in [(t['states'][:2], [1, -1]), (t['states'][2:], [1, 1])]:
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(stream_pattern(group), expected))
    for text in ['{"x":1,"x":2}', '{"x":NaN}']:
        try:
            json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid JSON accepted')
    try:
        stream_pattern([t['states'][0]]*2)
    except ValueError:
        pass
    else:
        raise AssertionError('Zero variance accepted')
    print('PASS: nine classification cases; toy moments/slopes; 24 stream orders; group slopes; invalid/zero-variance guards')


def run():
    grade, data, pairs = [read_pinned(n) for n in PINS]
    items = grade['items']
    lookup = {(r['axis'], r['pair_id'], r['cohort'], r['arm']): r for r in items}
    assert len(items) == len(lookup) == 256
    assert len(data) == len({r['id'] for r in data}) == 32
    assert len(pairs) == len({r['id'] for r in pairs}) == 16
    rows = []
    for pair in pairs:
        for axis in AXES:
            source = [lookup[(axis, pair['id'], cohort, arm)] for cohort in [1, 2] for arm in ['EA', 'E']]
            credits = [int(r['credit']) for r in source]
            assert all(r['credit'] in [0, 1] for r in source)
            truths = {r['first_id'] if r['truth'] == 'FIRST' else r['second_id'] for r in source}
            assert len(truths) == 1
            delta = [credits[0]-credits[1], credits[2]-credits[3]]
            rows.append({'axis': axis, 'pair_id': pair['id'], 'topic': pair['topic'],
                         'delta': delta, 'class': classify(delta),
                         'credits_EA1_E1_EA2_E2': credits,
                         'choices_EA1_E1_EA2_E2': [r['chosen_id'] for r in source],
                         'truth_id': next(iter(truths))})
    counts = dict(Counter(r['class'] for r in rows))
    matched = dict(Counter(d for r in rows for d in r['delta']))
    unchanged_bits = dict(Counter(''.join(map(str, r['credits_EA1_E1_EA2_E2'])) for r in rows if r['class'] == 'unchanged'))
    assert counts == {'gain_both': 5, 'harm_both': 6, 'opposed': 1, 'one_group_only': 7, 'unchanged': 45}
    assert matched == {1: 15, -1: 16, 0: 97}
    assert unchanged_bits == {'1111': 27, '0000': 16, '1100': 1, '0011': 1}
    for old in grade['gains_harm']:
        row = next(r for r in rows if r['pair_id'] == old['pair_id'] and r['axis'] == old['axis'])
        assert row['delta'][old['cohort']-1] == old['EA_minus_E_credit']
    result = {'status': 'SAVED_CASE_ACCOUNTING_COMPLETE', 'new_model_or_reader_calls': 0,
              'source_sha256': sha(Path(__file__)), 'inputs': PINS, 'counts_out_of_64': counts,
              'matched_delta_counts_out_of_128': matched, 'unchanged_credit_bits': unchanged_bits,
              'classification_is_posthoc_not_replication': True, 'cases': rows, 'toy': toy()}
    with (HERE/'analysis.json').open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'status': result['status'], 'counts': counts, 'sha256': sha(HERE/'analysis.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['test', 'run'])
    args = parser.parse_args()
    test() if args.mode == 'test' else run()
