"""One-shot saved-JSON corroboration; never imports the judging implementation.

Stdlib only. Consumes the frozen completed result, public/private packets,
committed responses and old scalar key. Writes a separate exclusive receipt.
This is not a grading stage, rater invocation, or source/acquisition audit.
"""

import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path('/private-artifacts/repositories/optimizers-launch-investigation')
STUDY = ROOT / 'output/2026-09-10-jlens-within-topic'
LOCK_COMMIT = 'f92d70d1dc6eefce7720b40c669701e8ded7df2f'
GRADE_COMMIT = '33279f22bb57f22e0c50d672f3706922ffeb8b06'
PINS = {
    'packets/manifest.json': '8fb30df285f757005193e1af27b213049f046e64baad65c13edf99bbca129ab3',
    'responses/lock.json': 'd7227b18e4a0e079860e0415ebff833361c0240e2df6cddb10bcad6ad576e1ea',
    'graded/grades.json': '369bbe6e077d09bcbe729f6dee00211419041d3cd2c6e35248c846d149795356',
}
SCORE_PIN = 'f1e4d2aa1ace39d9d482681a3e672f60c279348cf62884042a9ee93409a19a46'
AXES = ('PC1', 'PC2', 'PC3', 'PC4')
ARMS = ('A', 'B', 'C')
TOPICS = ('astronomy', 'cooking', 'football', 'programming')
ALLOCATION = {'rater1': 'BCAB', 'rater2': 'CABC', 'rater3': 'ABCA'}
PROMPT = ('Using only the two reference descriptions, which of the two prefixes '
          'should have the higher value on this direction? Choose FIRST or SECOND '
          'even if uncertain.')
COUNT = 0
READ = {}


def check(condition, label):
    global COUNT
    COUNT += 1
    if not condition:
        raise AssertionError(label)


def object_pairs(pairs):
    result = dict(pairs)
    check(len(result) == len(pairs), 'duplicate JSON key')
    return result


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read(path, pin=None, size=None, parse=True):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if pin is not None:
        check(digest == pin, str(path) + ' hash')
    if size is not None:
        check(len(data) == size, str(path) + ' size')
    READ[str(path)] = {'sha256': digest, 'size_bytes': len(data)}
    return (json.loads(data, object_pairs_hook=object_pairs,
                       parse_constant=reject_constant) if parse else data)


def receipt(base, spec, parse=True):
    check(set(spec) == {'path', 'size_bytes', 'sha256'}, 'receipt schema')
    return read(Path(base) / spec['path'], spec['sha256'], spec['size_bytes'], parse)


def committed(path, commit):
    value = subprocess.check_output(
        ['git', 'cat-file', 'blob', commit + ':' + str(path.relative_to(ROOT))],
        cwd=ROOT)
    check(value == path.read_bytes(), str(path) + ' committed bytes')


def summary(rows):
    return {
        'credit': math.fsum(r['credit'] for r in rows), 'total': len(rows),
        'correct': sum(r['credit'] == 1 for r in rows),
        'incorrect': sum(r['credit'] == 0 for r in rows),
        'exact_ties': sum(r['truth'] == 'TIE' for r in rows),
        'constant_FIRST_credit': math.fsum(.5 if r['truth'] == 'TIE' else
                                         float(r['truth'] == 'FIRST') for r in rows),
        'constant_SECOND_credit': math.fsum(.5 if r['truth'] == 'TIE' else
                                          float(r['truth'] == 'SECOND') for r in rows),
    }


def main():
    output = STUDY / 'result-audit-checks.json'
    check(not output.exists(), 'exclusive audit output')
    manifest = read(STUDY / 'packets/manifest.json', PINS['packets/manifest.json'])
    lock = read(STUDY / 'responses/lock.json', PINS['responses/lock.json'])
    grade = read(STUDY / 'graded/grades.json', PINS['graded/grades.json'])
    check(manifest['schema'] == 'jlens_within_topic_packets_v1', 'manifest schema')
    check(manifest['seed'] == 20260913, 'seed')
    source = read(STUDY / 'judging.py', manifest['source_sha256'], parse=False)
    check(lock['source_sha256'] == manifest['source_sha256'] ==
          grade['provenance']['source_sha256'], 'common source pin')
    inputs = manifest['inputs']
    check(inputs == grade['provenance']['inputs'], 'grade input bindings')
    protocol = receipt('/', inputs['protocol'], parse=False).decode()
    check(SCORE_PIN in protocol, 'protocol key pin')
    data = receipt('/', inputs['dataset'])
    pairs = receipt('/', inputs['pairs'])
    refs = receipt('/', inputs['references'])
    check(refs['schema'] == 'jlens_fresh_references_v1', 'reference schema')
    reference = {v['axis']: v for v in refs['axes']}
    check(len(reference) == len(refs['axes']) == 4 and tuple(reference) == AXES,
          'reference axes')
    ids = {row['id']: row for row in data}
    expected_ids = [f'{topic}-new-{i}' for topic in TOPICS for i in range(6)]
    check(list(ids) == expected_ids and len(data) == 24, 'all 24 original IDs')
    check(all(row['topic'] == TOPICS[i // 6] for i, row in enumerate(data)), 'topics')
    expected_pairs = [dict(id=f'P{i * 3 + j + 1:02d}',
                           left=f'{topic}-new-{2*j}', right=f'{topic}-new-{2*j+1}')
                      for i, topic in enumerate(TOPICS) for j in range(3)]
    check(pairs == expected_pairs, 'exact 12 adjacent within-topic pairs')
    pair_by_id = {row['id']: row for row in pairs}
    private = receipt(STUDY / 'packets', manifest['private_map'])
    check(private['schema'] == 'jlens_within_topic_private_map_v1', 'private schema')
    mapping = {row['item_id']: row for row in private['rows']}
    check(len(mapping) == len(private['rows']) == 144, '144 unique private IDs')
    check(set(mapping) == {f'Q{i:03d}' for i in range(1, 145)}, 'exact private IDs')
    check(Counter((r['axis'], r['arm'], r['pair_id']) for r in mapping.values()) ==
          Counter((a, b, p['id']) for a in AXES for b in ARMS for p in pairs),
          'complete axis-arm-pair Cartesian product')
    swaps, choices, public_seen = {}, {}, set()
    for rater, allocation in ALLOCATION.items():
        public = receipt(STUDY / 'packets', manifest['public'][rater])
        response = receipt(STUDY / 'responses', lock['responses'][rater])
        check(set(public) == {'schema', 'packet_id', 'prompt', 'blocks'}, 'public schema')
        check(public['schema'] == 'jlens_within_topic_public_v1' and
              public['prompt'] == PROMPT and len(public['blocks']) == 4, 'public contract')
        check(set(response) == {'schema', 'packet_id', 'responses'} and
              response['schema'] == 'jlens_within_topic_responses_v1' and
              response['packet_id'] == public['packet_id'], 'response contract')
        check(len(response['responses']) == 48, '48 responses per reader')
        local_choices = {}
        for entry in response['responses']:
            check(set(entry) == {'item_id', 'choice'} and
                  entry['choice'] in ('FIRST', 'SECOND'), 'strict response entry')
            check(entry['item_id'] not in local_choices, 'no duplicate response')
            local_choices[entry['item_id']] = entry['choice']
        local_seen, axes_seen = set(), set()
        for block in public['blocks']:
            check(set(block) == {'block_id', 'positive_reference', 'negative_reference',
                                 'comparisons'} and len(block['comparisons']) == 12,
                  'public block contract')
            axis_arm = set()
            for item in block['comparisons']:
                check(set(item) == {'item_id', 'first', 'second'}, 'redacted item schema')
                item_id = item['item_id']
                check(item_id not in public_seen and item_id in mapping, 'unique public join')
                public_seen.add(item_id)
                local_seen.add(item_id)
                m = mapping[item_id]
                check(set(m) == {'item_id', 'block_id', 'rater', 'axis', 'arm',
                                 'pair_id', 'first_id', 'second_id'}, 'map schema')
                check(m['rater'] == rater and m['block_id'] == block['block_id'] and
                      m['arm'] == allocation[AXES.index(m['axis'])], 'reader allocation')
                axis_arm.add((m['axis'], m['arm']))
                p = pair_by_id[m['pair_id']]
                check({m['first_id'], m['second_id']} == {p['left'], p['right']}, 'pair join')
                check(item['first'] == ids[m['first_id']]['prefix'] and
                      item['second'] == ids[m['second_id']]['prefix'], 'unedited public prefixes')
                target = (m['axis'], m['pair_id'])
                swap = (m['first_id'], m['second_id'])
                check(swaps.setdefault(target, swap) == swap, 'common swaps across arms')
            check(len(axis_arm) == 1, 'one axis/arm per block')
            axis, arm = next(iter(axis_arm))
            check(axis not in axes_seen, 'one block per axis per reader')
            axes_seen.add(axis)
            check(block['positive_reference'] == reference[axis][arm]['positive'] and
                  block['negative_reference'] == reference[axis][arm]['negative'],
                  'unedited exact reference strings')
        check(local_seen == set(local_choices), 'exact public-response roster')
        choices.update(local_choices)
        response_path = STUDY / 'responses' / (rater + '.json')
        check(response_path.read_bytes() == (STUDY / 'returned' / (rater + '.json')).read_bytes(),
              'returned and sealed bytes')
        committed(response_path, LOCK_COMMIT)
    check(public_seen == set(mapping) == set(choices), 'complete all144 joins')
    check(lock['count'] == 144 and lock['packet_manifest_sha256'] ==
          PINS['packets/manifest.json'], 'lock roster and manifest')
    committed(STUDY / 'responses/lock.json', LOCK_COMMIT)
    committed(STUDY / 'graded/grades.json', GRADE_COMMIT)
    subprocess.run(['git', 'merge-base', '--is-ancestor', LOCK_COMMIT, GRADE_COMMIT],
                   cwd=ROOT, check=True)
    attempt = read(STUDY / 'graded/attempt.json')
    check(datetime.fromisoformat(attempt['started_utc']) >
          datetime.fromisoformat(lock['sealed_utc']), 'grade starts after seal')
    check(grade['provenance']['lock_commit'] == LOCK_COMMIT and
          grade['provenance']['lock_sha256'] == PINS['responses/lock.json'] and
          grade['provenance']['packet_manifest_sha256'] == PINS['packets/manifest.json'] and
          grade['provenance']['responses'] == lock['responses'], 'grade response provenance')
    check(grade['provenance']['scores']['sha256'] == SCORE_PIN, 'grade score pin')
    key = receipt('/', grade['provenance']['scores'])
    check(key['schema'] == 'jlens_fresh_scores_v1', 'key schema')
    scores = {row['axis']: row['values'] for row in key['scores']}
    check(tuple(scores) == AXES and len(key['scores']) == 4, 'key axis roster')
    for axis in AXES:
        check(set(scores[axis]) == set(ids), 'key text roster')
        check(all(type(v) is float and math.isfinite(v) for v in scores[axis].values()),
              '96 finite original float scores')
    derived = []
    for m in private['rows']:
        first = scores[m['axis']][m['first_id']]
        second = scores[m['axis']][m['second_id']]
        gap = first - second
        truth = 'TIE' if gap == 0 else ('FIRST' if gap > 0 else 'SECOND')
        choice = choices[m['item_id']]
        derived.append(dict(m, choice=choice, truth=truth, score_first=first,
                            score_second=second, gap=gap, absolute_gap=abs(gap),
                            credit=.5 if truth == 'TIE' else float(choice == truth)))
    check(len(grade['items']) == 144, '144 persisted grade items')
    for expected, actual in zip(derived, grade['items'], strict=True):
        check(set(expected) == set(actual), 'exact item fields')
        for field, value in expected.items():
            check(type(actual[field]) is type(value) and actual[field] == value,
                  f"item {expected['item_id']} {field}")
            if type(value) is float:
                check(actual[field].hex() == value.hex(), 'exact binary64 value')
    arm_totals = {a: summary([r for r in derived if r['arm'] == a]) for a in ARMS}
    per_axis = {a: {b: summary([r for r in derived if r['axis'] == a and r['arm'] == b])
                    for b in ARMS} for a in AXES}
    check(arm_totals == grade['arms'], 'all arm summaries')
    check(per_axis == grade['per_axis'], 'all axis summaries')
    differences = {f'{a}_minus_{b}': {
        'total_credit': arm_totals[a]['credit'] - arm_totals[b]['credit'],
        'per_axis_credit': {axis: per_axis[axis][a]['credit'] - per_axis[axis][b]['credit']
                            for axis in AXES}} for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C'))}
    check(differences == grade['paired_differences'], 'all paired differences')
    by_target = {(r['axis'], r['pair_id'], r['arm']): r for r in derived}
    support, comparisons = {}, {}
    for axis in AXES:
        support[axis] = []
        comparisons[axis] = {}
        for p in pairs:
            row = by_target[axis, p['id'], 'A']
            support[axis].append({
                'pair_id': p['id'], 'left': ids[p['left']]['prefix'],
                'right': ids[p['right']]['prefix'],
                'higher_id': row['first_id'] if row['truth'] == 'FIRST' else row['second_id'],
                'absolute_gap': row['absolute_gap'],
                'credit': {arm: by_target[axis, p['id'], arm]['credit'] for arm in ARMS},
            })
        for other in ('B', 'C'):
            categories = {'both_correct': [], 'A_only_correct': [],
                          'other_only_correct': [], 'both_incorrect': []}
            for row in support[axis]:
                a, b = row['credit']['A'], row['credit'][other]
                label = ('both_correct' if a == b == 1 else 'A_only_correct' if a == 1
                         else 'other_only_correct' if b == 1 else 'both_incorrect')
                categories[label].append(row['pair_id'])
            comparisons[axis]['A_vs_' + other] = categories
    check(all(row['truth'] != 'TIE' for row in derived), 'zero actual exact ties')
    report = {'schema': 'jlens_within_topic_independent_scalar_audit_v1', 'status': 'PASS',
              'checks': COUNT, 'scope': 'Saved JSON only; no grader import or stage rerun.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'graded_attempt_utc': attempt['started_utc'], 'sealed_utc': lock['sealed_utc'],
              'lock_commit': LOCK_COMMIT, 'grade_commit': GRADE_COMMIT,
              'inputs': READ, 'arm_summaries': arm_totals, 'per_axis': per_axis,
              'paired_differences': differences, 'paired_item_comparisons': comparisons,
              'all_48_target_support': support}
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'checks': COUNT,
                      'arms': arm_totals, 'paired_item_comparisons': comparisons,
                      'all_48_target_support': support}, indent=2))


if __name__ == '__main__':
    main()
