"""Read-only independent scalar/response audit; never imports the producer/grader."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

STUDY = Path(__file__).resolve().parent
WORKER = STUDY.parents[1]
DATASET_SHA = '80749619a5bdbfd9506b8d453cc7f54d8f50c6eff73729f6a148152c0c0b35bf'
PAIRS_SHA = 'ec361bee2a15ec796230824c2e9125f46cebe76316a6b50ee45904cb2be02912'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    assert p.stat().st_size < 1024*1024
    return json.loads(p.read_text())


def run(grade_path, expected):
    assert digest(grade_path) == expected
    grade = read(grade_path)
    assert grade['schema'] == 'jlens_fresh_authored_grades_v1'
    provenance = grade['provenance']
    lock_path = STUDY/'responses/lock.json'
    assert digest(lock_path) == provenance['lock_sha256']
    lock = read(lock_path)
    for name in ['lock.json', 'rater1.json', 'rater2.json', 'rater3.json', 'rater4.json']:
        p = STUDY/'responses'/name
        blob = subprocess.check_output(['git', '-C', str(WORKER), 'show',
            provenance['lock_commit']+':'+p.relative_to(WORKER).as_posix()], timeout=10)
        assert blob == p.read_bytes()
    # Independently bind the prospective committed metadata release before key I/O.
    binding = provenance['release']
    release_path = Path(binding['receipt']['path']).resolve()
    assert digest(release_path) == binding['receipt']['sha256']
    assert release_path.stat().st_size == binding['receipt']['size_bytes']
    blob = subprocess.check_output(['git', '-C', str(WORKER), 'show',
        binding['commit']+':'+release_path.relative_to(WORKER).as_posix()], timeout=10)
    assert blob == release_path.read_bytes()
    release = read(release_path)
    assert release['schema'] == 'jlens_fresh_authored_release_v1'
    assert release['artifacts']['scores'] == provenance['scores']
    assert release['artifacts']['producer'] == provenance['producer']
    # Only after the committed responses have been verified, inspect the key.
    score_path = Path(provenance['scores']['path'])
    assert digest(score_path) == provenance['scores']['sha256']
    score_export = read(score_path)
    assert score_export['schema'] == 'jlens_fresh_authored_scores_v1'
    assert list(score_export['locations']) == ['verb']
    scores = {x['axis']: x['values'] for x in score_export['locations']['verb']}
    assert list(scores) == ['PC1','PC2','PC3','PC4']
    assert provenance['graded_location'] == 'verb'
    manifest_path = STUDY/'packets/manifest.json'
    assert digest(manifest_path) == provenance['packet_manifest_sha256'] == lock['packet_manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['release'] == binding
    mapping_path = STUDY/'packets/private-map.json'
    assert digest(mapping_path) == manifest['private_map']['sha256']
    mapping = read(mapping_path)['rows']
    inputs = manifest['inputs']
    assert inputs == provenance['inputs']
    assert all(release['artifacts'][k] == v for k,v in inputs.items())
    assert inputs['dataset']['sha256'] == DATASET_SHA
    assert inputs['pairs']['sha256'] == PAIRS_SHA
    for spec in inputs.values():
        assert digest(Path(spec['path'])) == spec['sha256']
    source_rows = {r['id']: r for r in read(Path(inputs['dataset']['path']))['rows']}
    pairs = {r['id']:r for r in read(Path(inputs['pairs']['path']))['pairs']}
    refs = {r['axis']: r for r in read(Path(inputs['references']['path']))['axes']}
    tokens = {r['id']: r for r in read(Path(inputs['tokens']['path']))['records']}
    choices, public_items = {}, {}
    for n in range(1, 5):
        rater = f'rater{n}'
        packet_path = STUDY/'packets/public'/f'{rater}.json'
        response_path = STUDY/'responses'/f'{rater}.json'
        assert digest(packet_path) == manifest['public'][rater]['sha256']
        assert digest(response_path) == lock['responses'][rater]['sha256']
        packet, response = read(packet_path), read(response_path)
        assert response['packet_id'] == packet['packet_id']
        choices[rater] = {r['item_id']: r['choice'] for r in response['responses']}
        assert len(response['responses']) == len(choices[rater]) == 64
        assert set(choices[rater].values()) <= {'FIRST','SECOND'}
        for block in packet['blocks']:
            for item in block['comparisons']:
                assert item['item_id'] not in public_items
                public_items[item['item_id']] = (rater, block, item)
    assert len(public_items) == len(mapping) == len(grade['items']) == 256
    actual = {r['item_id']: r for r in grade['items']}
    assert len(actual) == 256
    assert len({r['item_id'] for r in mapping}) == 256
    assert {r['item_id'] for r in mapping} == set(public_items) == set(actual)
    assert {(r['rater'],r['axis'],r['pair_id']) for r in mapping} == {
        (f'rater{n}',axis,pair) for n in range(1,5) for axis in scores for pair in pairs}
    orientations = {}
    calculated = []
    for row in mapping:
        rater, block, item = public_items[row['item_id']]
        assert rater == row['rater']
        assert row['arm'] == ('A' if rater in ('rater1', 'rater3') else 'C')
        assert row['cohort'] == (1 if rater in ('rater1','rater2') else 2)
        pair = pairs[row['pair_id']]
        assert row['content_pair'] == pair['content_pair'] and row['template'] == pair['template']
        assert {row['first_id'],row['second_id']} == {pair['observation_id'],pair['provision_id']}
        key = (row['axis'],row['pair_id'],row['cohort'])
        orientation = (row['first_id'],row['second_id'])
        assert orientations.setdefault(key,orientation) == orientation
        ref = refs[row['axis']][row['arm']]
        assert block['positive_reference'] == ref['positive']
        assert block['negative_reference'] == ref['negative']
        for side in ('first', 'second'):
            source = source_rows[row[side+'_id']]
            token = tokens[source['id']]
            endpoint = source['verb_span'][1]
            assert token['offset_mapping'][token['captured_positions']['verb']][1] == endpoint
            assert item[side] == source['text'][:endpoint]
        first, second = (scores[row['axis']][row[side+'_id']] for side in ('first', 'second'))
        gap = first-second
        choice = choices[rater][row['item_id']]
        truth = 'FIRST' if gap > 0 else 'SECOND' if gap < 0 else 'TIE'
        expected_row = dict(row, choice=choice, truth=truth,
            chosen_id=row['first_id' if choice == 'FIRST' else 'second_id'],
            score_first=first, score_second=second, gap=gap, absolute_gap=abs(gap),
            credit=0.5 if gap == 0 else float(choice == truth))
        assert actual[row['item_id']] == expected_row
        calculated.append(expected_row)
    for (axis,pair,cohort), orientation in orientations.items():
        assert orientation == orientations[axis,pair,3-cohort][::-1]

    def summary(rows):
        return dict(credit=sum(r['credit'] for r in rows), total=len(rows),
            correct=sum(r['credit'] == 1 for r in rows), incorrect=sum(r['credit'] == 0 for r in rows),
            exact_ties=sum(r['truth'] == 'TIE' for r in rows),
            constant_FIRST_credit=sum(0.5 if r['truth'] == 'TIE' else r['truth'] == 'FIRST' for r in rows),
            constant_SECOND_credit=sum(0.5 if r['truth'] == 'TIE' else r['truth'] == 'SECOND' for r in rows))

    def subset(**conditions):
        return [r for r in calculated if all(r[k] == v for k, v in conditions.items())]

    axes, arms = list(scores), ['A', 'C']
    for arm in arms:
        assert grade['arms'][arm] == summary(subset(arm=arm))
        for axis in axes:
            assert grade['per_axis'][axis][arm] == summary(subset(axis=axis, arm=arm))
        for field, key, values in [('cohort', 'per_cohort', [1, 2]), ('template', 'per_template', ['active', 'passive'])]:
            for value in values:
                assert grade[key][str(value)][arm] == summary(subset(arm=arm, **{field:value}))
                nested = grade['cohort_axis' if field == 'cohort' else 'template_axis'][str(value)]
                for axis in axes:
                    assert nested[axis][arm] == summary(subset(arm=arm, axis=axis, **{field:value}))
    for rater, result in grade['per_reader'].items():
        assert result['summary'] == summary(subset(rater=rater))
        for axis in axes:
            part = result['per_axis'][axis]
            assert {k:v for k,v in part.items() if k != 'arm'} == summary(subset(rater=rater, axis=axis))
    differences = grade['paired_differences']['A_minus_C']
    def difference(**conditions):
        return sum(r['credit'] for r in subset(arm='A', **conditions))-sum(r['credit'] for r in subset(arm='C', **conditions))
    assert differences['total_credit'] == difference()
    assert differences['per_axis_credit'] == {axis:difference(axis=axis) for axis in axes}
    for field, key, values in [('cohort', 'per_cohort', [1, 2]), ('template', 'per_template', ['active', 'passive'])]:
        for value in values:
            assert differences[key][str(value)] == {'total_credit':difference(**{field:value}),
                'per_axis_credit':{axis:difference(axis=axis, **{field:value}) for axis in axes}}
    arm_agreement = dict.fromkeys(arms, 0)
    for axis in axes:
        for arm in arms:
            cells = defaultdict(list)
            for row in subset(axis=axis, arm=arm):
                cells[row['pair_id']].append(row['chosen_id'])
            assert len(cells) == 16 and all(len(v) == 2 for v in cells.values())
            agree = sum(v[0] == v[1] for v in cells.values())
            reported = grade['agreement']['per_axis'][axis][arm]
            assert {k:v for k,v in reported.items() if k != 'items'} == {'agree':agree, 'disagree':16-agree, 'total':16}
            assert len(reported['items']) == 16
            for item in reported['items']:
                pair = item['pair_id']
                left = subset(axis=axis, arm=arm, pair_id=pair, cohort=1)[0]
                right = subset(axis=axis, arm=arm, pair_id=pair, cohort=2)[0]
                assert item == dict(pair_id=pair, cohort1_rater=left['rater'], cohort2_rater=right['rater'],
                    cohort1_chosen_id=left['chosen_id'], cohort2_chosen_id=right['chosen_id'],
                    agree=left['chosen_id'] == right['chosen_id'])
            arm_agreement[arm] += agree
    assert grade['agreement']['arms'] == {arm:{'agree':v, 'disagree':64-v, 'total':64} for arm,v in arm_agreement.items()}
    primary = grade['primary']
    assert primary['A_minus_C_credit_out_of_32'] == difference(axis='PC4')
    assert primary['A_reader_credit_out_of_16'] == {r:summary(subset(rater=r, axis='PC4'))['credit'] for r in ('rater1', 'rater3')}
    assert primary['A_minus_C_by_cohort_out_of_16'] == {str(c):difference(axis='PC4', cohort=c) for c in (1, 2)}
    assert primary['each_A_reader_exceeds_8'] == all(v > 8 for v in primary['A_reader_credit_out_of_16'].values())
    assert primary['A_minus_C_positive_each_cohort'] == all(v > 0 for v in primary['A_minus_C_by_cohort_out_of_16'].values())
    return {'status':'PASS', 'grade_sha256':expected, 'auditor_sha256':digest(Path(__file__)),
            'exact_item_rows':256, 'committed_response_blobs':5,
            'all_reference_and_target_bindings':True, 'all_summary_counts':True,
            'primary':primary, 'arms':grade['arms'], 'per_axis':grade['per_axis'],
            'model_calls':0, 'grader_imported':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grades', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.grades, args.sha256), indent=2, allow_nan=False))
