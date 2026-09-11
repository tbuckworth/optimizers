"""One saved-scalar corroboration; no grading-source import or stage rerun.

Default import is inert. --self-test uses only in-memory fabricated fixtures;
--audit-once claims the one exclusive receipt before reading frozen study JSON.
The author also authored judging.py: independent arithmetic, not an
independent-implementation-author review or verification of reader isolation.

Python 3.12 contracts checked: https://docs.python.org/3.12/library/json.html,
https://docs.python.org/3.12/library/functions.html#open,
https://docs.python.org/3.12/library/math.html#math.fsum.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess


ROOT = Path('/private-artifacts/repositories/optimizers-launch-investigation')
STUDY = ROOT / 'output/2026-09-11-jlens-independent-content'
WORKER = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree')
SCORES = WORKER / 'output/2026-09-11-j-lens-independent-content/forwards/scores.json'
LOCK_COMMIT = 'd1abdde61a5448338d1d75788fcfc29f9197851f'
GRADE_COMMIT = 'cbe9c407c860d7b8e1d3a3816f03a3a061ff4b45'
PINS = {
    'packets/manifest.json': 'e2f48664da9dd8945ffa1576873719fa87f6cd364f93235189915d8dae5d611e',
    'responses/lock.json': '6851e689784ac2b752b195e683a553ae8e41f6cc29d87015821e0ff876fd107e',
    'graded/grades.json': 'a08ba8cca7a23c9b985cd0951eab81fcbedcc52a1ec95d322342660b356d4744',
}
SCORE_PIN = 'e03efc7d22d480b2b7f12db89021e20653db2898a7317b63b6e2b6e9a3e5e704'
SOURCE_PIN = '2d6644718d64df6230ca7ca33881706ff88dc049302ad61f1b1e3663b74e57a1'
PROTOCOL_PIN = 'f7314607d2846707a2ee2e8b64c8f2f1fb3098a2366bb1f0d5b0bc0afdcbea3b'
REFERENCE_PIN = '0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c'
DATASET_PIN = '7391a1fa6c02872cdeabb2e8dffc1e63181b6379d80e2042f33b38a61308ff02'
AXES = ('PC1', 'PC2', 'PC3', 'PC4')
ARMS = ('A', 'B', 'C')
TOPICS = ('astronomy', 'cooking', 'football', 'programming')
RATERS = tuple('rater' + str(i) for i in range(1, 7))
ALLOCATION = dict(zip(RATERS, ('ABCA', 'BCAB', 'CABC') * 2))
COHORT = {r: 1 + (i // 3) for i, r in enumerate(RATERS)}
IDS = tuple(f'{t}-wiki-{i}' for t in TOPICS for i in range(6))
PAIR_IDS = tuple(f'P{i:02}' for i in range(1, 13))
PROMPT = ('Using only the two reference descriptions, which of the two prefixes '
          'should have the higher value on this direction? Choose FIRST or SECOND '
          'even if uncertain.')
MAX_FILE = 1024 * 1024
COUNT = 0
READ = {}


def check(condition, message):
    global COUNT
    COUNT += 1
    if not condition:
        raise AssertionError(message)


def fields(value, names, label):
    check(type(value) is dict and set(value) == set(names), label + ' fields')


def exact(actual, expected, label):
    """Check JSON types, complete structure/order and float bits, including -0."""
    check(type(actual) is type(expected), label + ' type')
    if type(expected) is dict:
        check(set(actual) == set(expected), label + ' keys')
        for key in expected:
            exact(actual[key], expected[key], label + '/' + key)
    elif type(expected) is list:
        check(len(actual) == len(expected), label + ' length')
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            exact(a, e, label + '/' + str(i))
    elif type(expected) is float:
        check(math.isfinite(actual) and actual.hex() == expected.hex(), label + ' binary64')
    else:
        check(actual == expected, label + ' value')


def decode(raw):
    check(len(raw) <= MAX_FILE, 'bounded JSON')
    def pairs(entries):
        obj = dict(entries)
        check(len(obj) == len(entries), 'duplicate JSON key')
        return obj
    def invalid(_):
        raise ValueError('nonfinite JSON constant')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    def finite(v):
        if type(v) is float:
            check(math.isfinite(v), 'finite JSON float')
        elif type(v) is dict:
            for child in v.values():
                finite(child)
        elif type(v) is list:
            for child in v:
                finite(child)
    finite(value)
    return value


def read(path, pin=None, size=None, parse=True):
    p = Path(path)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        check(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE, 'bounded regular input')
        raw = stream.read(MAX_FILE + 1)
        after = os.fstat(stream.fileno())
    check(len(raw) == before.st_size == after.st_size and
          before.st_mtime_ns == after.st_mtime_ns, 'unchanged input during read')
    digest = hashlib.sha256(raw).hexdigest()
    if pin is not None:
        check(digest == pin, str(p) + ' SHA256')
    if size is not None:
        check(len(raw) == size, str(p) + ' byte size')
    READ[str(p)] = {'sha256': digest, 'size_bytes': len(raw)}
    return decode(raw) if parse else raw


def member(base, receipt, name, parse=True):
    fields(receipt, ('path', 'sha256', 'size_bytes'), 'receipt')
    check(receipt['path'] == name, 'exact receipt member')
    return read(Path(base) / name, receipt['sha256'], receipt['size_bytes'], parse)


def committed(path, commit):
    spec = commit + ':' + Path(path).relative_to(ROOT).as_posix()
    result = subprocess.run(['git', 'cat-file', 'blob', spec], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True, timeout=10).stdout
    check(result == read(path, parse=False), str(path) + ' committed bytes')


def verify_packets(data, pairs, refs, private, public, responses):
    check(type(data) is list and len(data) == 24, '24 texts')
    for i, row in enumerate(data):
        fields(row, ('id', 'topic', 'prefix'), 'text')
        check(row['id'] == IDS[i] and row['topic'] == TOPICS[i // 6], 'text roster/order')
        check(type(row['prefix']) is str and len(row['prefix'].split()) == 16 and
              ' '.join(row['prefix'].split()) == row['prefix'], '16 normalized words')
    check(len({r['prefix'] for r in data}) == 24, 'distinct prefixes')
    expected_pairs = [dict(id=p, left=IDS[2*i], right=IDS[2*i+1])
                      for i, p in enumerate(PAIR_IDS)]
    exact(pairs, expected_pairs, '12 fixed adjacent pairs')
    texts, pair_map = {r['id']: r['prefix'] for r in data}, {r['id']: r for r in pairs}
    fields(refs, ('schema', 'axes'), 'references')
    check(refs['schema'] == 'jlens_fresh_references_v1' and len(refs['axes']) == 4,
          'reference schema/count')
    reference = {}
    for axis, row in zip(AXES, refs['axes'], strict=True):
        fields(row, ('axis', *ARMS), 'reference axis')
        check(row['axis'] == axis, 'reference axis order')
        reference[axis] = row
        for arm in ARMS:
            fields(row[arm], ('positive', 'negative'), 'reference poles')
            for v in row[arm].values():
                check((type(v) is str if arm == 'C' else
                       type(v) is list and len(v) == 12 and all(type(t) is str for t in v)),
                      'reference string/list')
    fields(private, ('schema', 'rows'), 'private map')
    check(private['schema'] == 'jlens_independent_content_private_map_v1' and
          len(private['rows']) == 288, 'private count/schema')
    mapping = {r['item_id']: r for r in private['rows']}
    check(len(mapping) == 288 and set(mapping) == {f'Q{i:03}' for i in range(1, 289)},
          'all 288 distinct anonymous item IDs')
    check(Counter((r['axis'], r['arm'], r['cohort'], r['pair_id']) for r in mapping.values()) ==
          Counter((a, b, c, p) for a in AXES for b in ARMS for c in (1, 2) for p in PAIR_IDS),
          'complete 288-cell Cartesian roster')
    fields(public, RATERS, 'public readers')
    fields(responses, RATERS, 'response readers')
    choices, orientations, seen, blocks_seen, packet_ids = {}, {}, set(), set(), set()
    for rater in RATERS:
        packet, response = public[rater], responses[rater]
        fields(packet, ('schema', 'packet_id', 'prompt', 'blocks'), 'public packet')
        check(packet['schema'] == 'jlens_independent_content_public_v1' and
              packet['prompt'] == PROMPT and len(packet['blocks']) == 4, 'public contract')
        packet_ids.add(packet['packet_id'])
        fields(response, ('schema', 'packet_id', 'responses'), 'response')
        check(response['schema'] == 'jlens_independent_content_responses_v1' and
              response['packet_id'] == packet['packet_id'] and len(response['responses']) == 48,
              'response contract')
        local_choices = {}
        for row in response['responses']:
            fields(row, ('item_id', 'choice'), 'response row')
            check(row['item_id'] not in local_choices and row['choice'] in ('FIRST', 'SECOND'),
                  'unique valid response')
            local_choices[row['item_id']] = row['choice']
        local_seen, axes_seen = set(), set()
        for block in packet['blocks']:
            fields(block, ('block_id', 'positive_reference', 'negative_reference', 'comparisons'),
                   'anonymous block')
            check(block['block_id'] not in blocks_seen and len(block['comparisons']) == 12,
                  'distinct block and 12 pairs')
            blocks_seen.add(block['block_id'])
            axis_arm = set()
            for item in block['comparisons']:
                fields(item, ('item_id', 'first', 'second'), 'anonymous comparison')
                q = item['item_id']
                check(q in mapping and q not in seen, 'unique public/private item join')
                seen.add(q)
                local_seen.add(q)
                m = mapping[q]
                fields(m, ('item_id', 'block_id', 'rater', 'axis', 'arm', 'cohort',
                           'pair_id', 'first_id', 'second_id'), 'private row')
                check(m['rater'] == rater and m['block_id'] == block['block_id'] and
                      m['arm'] == ALLOCATION[rater][AXES.index(m['axis'])] and
                      type(m['cohort']) is int and m['cohort'] == COHORT[rater], 'exact allocation')
                axis_arm.add((m['axis'], m['arm']))
                p = pair_map[m['pair_id']]
                check({m['first_id'], m['second_id']} == {p['left'], p['right']}, 'physical pair')
                check(item['first'] == texts[m['first_id']] and item['second'] == texts[m['second_id']],
                      'unchanged public prefixes')
                key = (m['axis'], m['cohort'], m['pair_id'])
                orientation = (m['first_id'], m['second_id'])
                check(orientations.setdefault(key, orientation) == orientation, 'common arm orientation')
            check(len(axis_arm) == 1, 'single axis/arm per block')
            axis, arm = next(iter(axis_arm))
            check(axis not in axes_seen, 'one block per axis')
            axes_seen.add(axis)
            exact(block['positive_reference'], reference[axis][arm]['positive'], 'positive reference')
            exact(block['negative_reference'], reference[axis][arm]['negative'], 'negative reference')
        check(local_seen == set(local_choices), 'complete public/response local roster')
        choices.update(local_choices)
    check(seen == set(mapping) == set(choices) and len(packet_ids) == 6 and
          blocks_seen == {f'B{i:03}' for i in range(1, 25)}, 'all public IDs retained')
    for axis in AXES:
        for pair in PAIR_IDS:
            check(orientations[axis, 1, pair] == orientations[axis, 2, pair][::-1],
                  'cohort 2 exact opposite')
    return choices


def tally(rows):
    return dict(credit=math.fsum(r['credit'] for r in rows), total=len(rows),
                correct=sum(r['credit'] == 1 for r in rows),
                incorrect=sum(r['credit'] == 0 for r in rows),
                exact_ties=sum(r['truth'] == 'TIE' for r in rows),
                constant_FIRST_credit=math.fsum(.5 if r['truth'] == 'TIE' else
                                               float(r['truth'] == 'FIRST') for r in rows),
                constant_SECOND_credit=math.fsum(.5 if r['truth'] == 'TIE' else
                                                float(r['truth'] == 'SECOND') for r in rows))


def calculate(private, choices, export):
    fields(export, ('schema', 'scores'), 'scores')
    check(export['schema'] == 'jlens_independent_content_scores_v1' and len(export['scores']) == 4,
          'score schema/count')
    scores = {}
    for axis, row in zip(AXES, export['scores'], strict=True):
        fields(row, ('axis', 'values'), 'score axis')
        check(row['axis'] == axis and set(row['values']) == set(IDS), 'score axis/text roster')
        check(all(type(v) is float and math.isfinite(v) for v in row['values'].values()),
              'exact finite binary64 exports')
        scores[axis] = row['values']
    derived = []
    for m in private['rows']:
        first, second = scores[m['axis']][m['first_id']], scores[m['axis']][m['second_id']]
        delta = first - second
        check(math.isfinite(delta), 'finite difference')
        selected = m['first_id'] if choices[m['item_id']] == 'FIRST' else m['second_id']
        # Derive correctness by the selected underlying score, not the grader's choice==truth expression.
        credit = .5 if first == second else float(scores[m['axis']][selected] == max(first, second))
        truth = 'TIE' if first == second else 'FIRST' if first > second else 'SECOND'
        derived.append(dict(m, choice=choices[m['item_id']], chosen_id=selected, truth=truth,
                            score_first=first, score_second=second, gap=delta,
                            absolute_gap=abs(delta), credit=credit))
    arm = {a: tally([r for r in derived if r['arm'] == a]) for a in ARMS}
    axes = {a: {b: tally([r for r in derived if r['axis'] == a and r['arm'] == b])
                for b in ARMS} for a in AXES}
    readers = {r: dict(cohort=COHORT[r], summary=tally([v for v in derived if v['rater'] == r]),
                      per_axis={a: dict(arm=ALLOCATION[r][ai],
                                        **tally([v for v in derived if v['rater'] == r and v['axis'] == a]))
                                for ai, a in enumerate(AXES)}) for r in RATERS}
    differences = {a + '_minus_' + b: {
        'total_credit': math.fsum(r['credit'] * (1 if r['arm'] == a else -1)
                                  for r in derived if r['arm'] in (a, b)),
        'per_axis_credit': {x: math.fsum(r['credit'] * (1 if r['arm'] == a else -1)
                                         for r in derived if r['axis'] == x and r['arm'] in (a, b))
                           for x in AXES}}
        for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C'))}
    cells = {(r['axis'], r['arm'], r['cohort'], r['pair_id']): r for r in derived}
    agreement = {'per_axis': {}, 'arms': {}}
    support = {}
    for axis in AXES:
        agreement['per_axis'][axis] = {}
        for a in ARMS:
            rows = []
            for p in PAIR_IDS:
                one, two = cells[axis, a, 1, p], cells[axis, a, 2, p]
                rows.append(dict(pair_id=p, cohort1_rater=one['rater'], cohort2_rater=two['rater'],
                                 cohort1_chosen_id=one['chosen_id'], cohort2_chosen_id=two['chosen_id'],
                                 agree=one['chosen_id'] == two['chosen_id']))
            agreed = sum(r['agree'] for r in rows)
            agreement['per_axis'][axis][a] = dict(agree=agreed, disagree=12-agreed, total=12, items=rows)
            check(axes[axis][a]['total'] == 24 and axes[axis][a]['constant_FIRST_credit'] ==
                  axes[axis][a]['constant_SECOND_credit'] == 12.0, '24-choice axis position balance')
        support[axis] = []
        for pi, p in enumerate(PAIR_IDS):
            left, right = IDS[2*pi], IDS[2*pi+1]
            gap = scores[axis][left] - scores[axis][right]
            credits = {a: [cells[axis, a, c, p]['credit'] for c in (1, 2)] for a in ARMS}
            a1, a2 = credits['A']
            category = ('tied' if gap == 0 else 'both_correct' if a1 == a2 == 1 else
                        'both_wrong' if a1 == a2 == 0 else 'mixed')
            support[axis].append(dict(pair_id=p, left_id=left, right_id=right,
                higher_id=None if gap == 0 else left if gap > 0 else right,
                absolute_gap=abs(gap), credits_by_arm=credits, A_reader_outcome=category,
                A_minus_B_pair_credit=math.fsum(credits['A']) - math.fsum(credits['B']),
                A_minus_C_pair_credit=math.fsum(credits['A']) - math.fsum(credits['C'])))
    for a in ARMS:
        agreed = sum(agreement['per_axis'][x][a]['agree'] for x in AXES)
        agreement['arms'][a] = dict(agree=agreed, disagree=48-agreed, total=48)
        check(arm[a]['total'] == 96 and arm[a]['constant_FIRST_credit'] ==
              arm[a]['constant_SECOND_credit'] == 48.0, '96-choice arm position balance')
    values = dict(schema='jlens_independent_content_grades_v1', arms=arm, per_axis=axes,
                  per_reader=readers, paired_differences=differences, agreement=agreement, items=derived)
    return values, support


def audit_once():
    output = STUDY / 'result-audit-checks.json'
    # Claim before input reads: even an interrupted/failed attempt cannot be silently rerun.
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as handle:
        started = datetime.now(timezone.utc).isoformat()
        try:
            manifest = read(STUDY / 'packets/manifest.json', PINS['packets/manifest.json'])
            lock = read(STUDY / 'responses/lock.json', PINS['responses/lock.json'])
            grade = read(STUDY / 'graded/grades.json', PINS['graded/grades.json'])
            check(manifest['schema'] == 'jlens_independent_content_packets_v1' and manifest['seed'] == 20260915,
                  'manifest schema/seed')
            check(lock['schema'] == 'jlens_independent_content_response_lock_v1' and lock['count'] == 288,
                  'lock schema/count')
            check(manifest['source_sha256'] == lock['source_sha256'] == SOURCE_PIN ==
                  grade['provenance']['source_sha256'], 'same frozen grading source')
            read(STUDY / 'judging.py', SOURCE_PIN, parse=False)  # bytes only; never imported/executed
            exact(manifest['inputs'], grade['provenance']['inputs'], 'same source inputs')
            inputs = manifest['inputs']
            check(inputs['protocol']['sha256'] == PROTOCOL_PIN and
                  inputs['references']['sha256'] == REFERENCE_PIN and
                  inputs['dataset']['sha256'] == DATASET_PIN, 'frozen input hashes')
            loaded = {name: member('/', inputs[name], inputs[name]['path'], name != 'protocol')
                      for name in ('dataset', 'pairs', 'references', 'protocol')}
            private = member(STUDY / 'packets', manifest['private_map'], 'private-map.json')
            public, responses = {}, {}
            for r in RATERS:
                public[r] = member(STUDY / 'packets', manifest['public'][r], 'public/' + r + '.json')
                responses[r] = member(STUDY / 'responses', lock['responses'][r], r + '.json')
                check(read(STUDY / 'responses' / (r + '.json'), parse=False) ==
                      read(STUDY / 'raw-responses' / (r + '.json'), parse=False), 'raw/sealed exact bytes')
                committed(STUDY / 'responses' / (r + '.json'), LOCK_COMMIT)
            committed(STUDY / 'responses/lock.json', LOCK_COMMIT)
            committed(STUDY / 'graded/grades.json', GRADE_COMMIT)
            subprocess.run(['git', 'merge-base', '--is-ancestor', LOCK_COMMIT, GRADE_COMMIT],
                           cwd=ROOT, check=True, timeout=10)
            check(lock['packet_manifest_sha256'] == PINS['packets/manifest.json'], 'lock/packet binding')
            provenance = grade['provenance']
            check(provenance['lock_commit'] == LOCK_COMMIT and
                  provenance['lock_sha256'] == PINS['responses/lock.json'] and
                  provenance['packet_manifest_sha256'] == PINS['packets/manifest.json'], 'grade lock/packet pins')
            exact(provenance['responses'], lock['responses'], 'grade response receipts')
            attempt = read(STUDY / 'graded/attempt.json')
            check(attempt['stage'] == 'grade' and attempt['source_sha256'] == SOURCE_PIN and
                  datetime.fromisoformat(attempt['started_utc']) > datetime.fromisoformat(lock['sealed_utc']),
                  'grade starts after saved seal time')
            choices = verify_packets(loaded['dataset'], loaded['pairs'], loaded['references'],
                                     private, public, responses)
            check(provenance['scores']['path'] == str(SCORES) and
                  provenance['scores']['sha256'] == SCORE_PIN, 'frozen scalar-export path/hash')
            export = member('/', provenance['scores'], str(SCORES))
            computed, support = calculate(private, choices, export)
            check(set(grade) == set(computed) | {'provenance'}, 'complete grade top-level fields')
            for field in computed:
                exact(grade[field], computed[field], 'saved grade/' + field)
            result = dict(schema='jlens_independent_content_scalar_audit_v1', status='PASS',
                          started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                          checks=COUNT, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          lock_commit=LOCK_COMMIT, grade_commit=GRADE_COMMIT,
                          sealed_utc=lock['sealed_utc'], graded_started_utc=attempt['started_utc'],
                          scope='Independent saved-scalar arithmetic; grader author is the same agent. No projection audit or reader-isolation verification.',
                          inputs=READ, arms=computed['arms'], per_axis=computed['per_axis'],
                          per_reader=computed['per_reader'], paired_differences=computed['paired_differences'],
                          agreement=computed['agreement'], all_48_target_support=support)
        except Exception as error:
            json.dump(dict(status='FAILED', started_utc=started, checks=COUNT,
                           error_type=type(error).__name__, error=str(error)), handle, indent=2)
            handle.write('\n')
            raise
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({k: result[k] for k in ('status', 'checks', 'arms', 'per_axis')}, indent=2))


def fabricated():
    """No filesystem, scientific input, grading import, RNG, or external calls."""
    data = [dict(id=x, topic=x.split('-')[0], prefix=' '.join([x] + ['word'] * 15)) for x in IDS]
    pairs = [dict(id=p, left=IDS[2*i], right=IDS[2*i+1]) for i, p in enumerate(PAIR_IDS)]
    refs = dict(schema='jlens_fresh_references_v1', axes=[dict(axis=a, **{
        b: dict(positive='example high' if b == 'C' else [a + b + '+'] * 12,
                negative='example low' if b == 'C' else [a + b + '-'] * 12) for b in ARMS}) for a in AXES])
    public, responses, mapping = {}, {}, []
    for ri, r in enumerate(RATERS):
        blocks, answers = [], []
        for ai, a in enumerate(AXES):
            arm, bid = ALLOCATION[r][ai], f'B{ri*4+ai+1:03}'
            block = dict(block_id=bid, positive_reference=refs['axes'][ai][arm]['positive'],
                         negative_reference=refs['axes'][ai][arm]['negative'], comparisons=[])
            for pi, p in enumerate(PAIR_IDS):
                left, right = IDS[pi*2], IDS[pi*2+1]
                first, second = (left, right) if (pi % 2 == 0) != (COHORT[r] == 2) else (right, left)
                q = f'Q{ri*48+ai*12+pi+1:03}'
                mapping.append(dict(item_id=q, block_id=bid, rater=r, axis=a, arm=arm,
                                    cohort=COHORT[r], pair_id=p, first_id=first, second_id=second))
                text = {row['id']: row['prefix'] for row in data}
                block['comparisons'].append(dict(item_id=q, first=text[first], second=text[second]))
                selected = right if arm == 'A' or (arm == 'B' and COHORT[r] == 2) else left
                choice = 'FIRST' if arm == 'C' or first == selected else 'SECOND'
                answers.append(dict(item_id=q, choice=choice))
            blocks.append(block)
        public[r] = dict(schema='jlens_independent_content_public_v1', packet_id=f'R{ri:016x}',
                         prompt=PROMPT, blocks=blocks)
        responses[r] = dict(schema='jlens_independent_content_responses_v1', packet_id=f'R{ri:016x}', responses=answers)
    values = []
    for ai, a in enumerate(AXES):
        scores = {x: float(ai*24+i) for i, x in enumerate(IDS)}
        if ai == 0:
            scores[IDS[0]] = scores[IDS[1]] = 0.0
        if ai == 1:
            scores[IDS[0]], scores[IDS[1]] = -0.0, 1e-300
        values.append(dict(axis=a, values=scores))
    private = dict(schema='jlens_independent_content_private_map_v1', rows=mapping)
    return data, pairs, refs, private, public, responses, dict(schema='jlens_independent_content_scores_v1', scores=values)


def self_test():
    fixture = fabricated()
    choices = verify_packets(*fixture[:6])
    result, support = calculate(fixture[3], choices, fixture[6])
    check([result['arms'][a]['credit'] for a in ARMS] == [95.0, 48.0, 48.0], 'hand-derived arm credits')
    check(result['per_axis']['PC1']['A']['credit'] == 23.0 and
          all(result['per_axis'][a]['A']['credit'] == 24.0 for a in AXES[1:]), 'hand-derived axes')
    check(all(result['agreement']['per_axis'][a]['A']['agree'] == 12 and
              result['agreement']['per_axis'][a]['B']['agree'] ==
              result['agreement']['per_axis'][a]['C']['agree'] == 0 for a in AXES),
          'underlying-prefix agreement, including opposite literals')
    check(support['PC1'][0]['higher_id'] is None and support['PC1'][0]['A_reader_outcome'] == 'tied',
          'half-credit tie support')
    row = next(r for r in result['items'] if r['axis'] == 'PC2' and r['pair_id'] == 'P01' and r['cohort'] == 1)
    check(row['absolute_gap'] == 1e-300 and row['score_first'].hex() == '-0x0.0p+0', 'tiny gap/signed zero retained')
    mutation_cases = 0
    def rejects(callback):
        nonlocal mutation_cases
        try:
            callback()
        except (AssertionError, ValueError, KeyError):
            mutation_cases += 1
        else:
            raise AssertionError('fabricated mutation was accepted')
    bad = copy.deepcopy(fixture)
    bad[3]['rows'][0]['cohort'] = 2
    rejects(lambda: verify_packets(*bad[:6]))
    bad = copy.deepcopy(fixture)
    bad[5]['rater1']['responses'].pop()
    rejects(lambda: verify_packets(*bad[:6]))
    bad = copy.deepcopy(fixture)
    bad[4]['rater1']['blocks'][0]['comparisons'][0]['first'] += ' edited'
    rejects(lambda: verify_packets(*bad[:6]))
    bad = copy.deepcopy(fixture)
    bad[3]['rows'][0]['pair_id'] = 'P02'
    rejects(lambda: verify_packets(*bad[:6]))
    bad = copy.deepcopy(result)
    bad['agreement']['per_axis']['PC1']['A']['items'][0]['agree'] = False
    rejects(lambda: exact(bad, result, 'agreement mutation'))
    bad = copy.deepcopy(result)
    bad['items'][0]['gap'] = -0.0
    rejects(lambda: exact(bad, result, 'signed-zero mutation'))
    bad = copy.deepcopy(result)
    bad['per_reader']['rater1']['summary']['credit'] += 1.0
    rejects(lambda: exact(bad, result, 'reader tally mutation'))
    rejects(lambda: decode(b'{"x":1,"x":2}'))
    rejects(lambda: decode(b'{"x":NaN}'))
    rejects(lambda: decode(b'{"x":1e9999}'))
    changed = dict(choices)
    m = next(r for r in fixture[3]['rows'] if r['axis'] == 'PC4' and r['arm'] == 'A' and
             r['cohort'] == 2 and r['pair_id'] == 'P02')
    changed[m['item_id']] = 'SECOND' if changed[m['item_id']] == 'FIRST' else 'FIRST'
    alt, evidence = calculate(fixture[3], changed, fixture[6])
    check(alt['per_axis']['PC4']['A']['credit'] == 23.0 and
          alt['agreement']['per_axis']['PC4']['A']['agree'] == 11 and
          evidence['PC4'][1]['A_reader_outcome'] == 'mixed', 'single reader change accounted')
    print(json.dumps(dict(status='PASS', mode='fabricated_only', mutation_cases_rejected=mutation_cases,
                          checks=COUNT, actual_study_inputs_read=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--audit-once', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else audit_once()
