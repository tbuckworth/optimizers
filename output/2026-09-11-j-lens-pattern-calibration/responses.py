"""Inert JSON-only committed-response seal, public lock gate and one-shot grade."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

MAIN = Path('output/2026-09-11-jlens-pattern-calibration')
PACKET_SOURCE = MAIN/'reader_packets.py'
PACKET_SHA = 'daf506324f9da867a75d10046a3a11d8ec17b960f61bec585d3371d4cc0413a3'
EVALUATOR_SHA = '2a3a288dc69321b22f8af9c886a905def0ddbc76c865b72d4eb2f118ca656569'
PYTHON = '3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]'
REFERENCE_COMMIT = 'ef8168b6ba441c5e3a594df66d872b159387f2f1'
RATERS = [f'rater{i}' for i in range(1, 5)]
AXES = [f'PC{i}' for i in range(1, 5)]
ARMS = ['P', 'U']
PAIRS = [f'T{i:02}' for i in range(1, 17)]
IDS = [p+'-'+s for p in PAIRS for s in ('L', 'R')]
TOPICS = ['astronomy', 'cooking', 'football', 'programming']
CAP = 1024**2


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def source_sha(): return sha(Path(__file__).read_bytes())
def utc(): return datetime.now(timezone.utc).isoformat()


def keys(value, names):
    require(type(value) is dict and set(value) == set(names), 'exact JSON fields')


def descriptor(value):
    keys(value, ('path', 'sha256', 'size_bytes'))
    require(type(value['path']) is str and Path(value['path']).is_absolute() and
            '..' not in Path(value['path']).parts, 'absolute artifact path')
    require(type(value['size_bytes']) is int and 0 < value['size_bytes'] <= CAP and
            type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'bounded artifact pin')


def read(value):
    descriptor(value); path = Path(value['path'])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == value['size_bytes'], 'regular bounded input')
        raw = handle.read(CAP+1); after = os.fstat(handle.fileno())
    require(len(raw) == before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns and
            sha(raw) == value['sha256'], 'input hash/size/drift')
    return raw


def helper():
    require(not PACKET_SOURCE.is_symlink(), 'packet source symlink')
    raw = PACKET_SOURCE.read_bytes(); require(sha(raw) == PACKET_SHA, 'packet source pin')
    p = types.ModuleType('pinned_reader_packets'); p.__file__ = str(PACKET_SOURCE)
    exec(compile(raw, str(PACKET_SOURCE), 'exec'), p.__dict__)
    return p


def git(cwd, *args):
    return subprocess.check_output(['git', '-C', str(cwd), *args], stderr=subprocess.PIPE, timeout=10)


def committed(files, commit):
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit) and files, 'exact committed files required')
    root = Path(git(next(iter(files)).parent, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    require(git(root, 'cat-file', '-t', commit).strip() == b'commit', 'not a commit object')
    git(root, 'merge-base', '--is-ancestor', commit, 'HEAD')
    for path, raw in files.items():
        spec = commit+':'+path.resolve(strict=True).relative_to(root).as_posix()
        require(int(git(root, 'cat-file', '-s', spec)) == len(raw) <= CAP and
                git(root, 'cat-file', 'blob', spec) == raw, 'committed blob differs')


def member(root, item, name):
    keys(item, ('path', 'size_bytes', 'sha256')); require(item['path'] == name, 'fixed member name')
    return {'path': str(Path(root)/name), 'size_bytes': item['size_bytes'], 'sha256': item['sha256']}


def ancestor(directory, older, newer):
    require(all(type(v) is str and re.fullmatch('[0-9a-f]{40}', v) for v in (older, newer)), 'fixed history commits')
    git(directory, 'merge-base', '--is-ancestor', older, newer)


class Stage:
    def __init__(self, target, name, p):
        require(os.environ.get('JLENS_PATTERN_'+name.upper()+'_RELEASE') == '1', 'explicit stage release required')
        self.p, self.root, self.used = p, Path(target), 0
        require(self.root.is_absolute(), 'absolute stage target')
        if self.root.exists() or self.root.is_symlink(): raise FileExistsError('stage consumed')
        self.root.mkdir(mode=0o700)
        self.write('attempt.json', {'stage': name, 'source_sha256': source_sha(), 'started_utc': utc(), 'automatic_retry': False})
    def raw(self, name, raw):
        require(name in ['attempt.json', 'failure.json', 'lock.json', 'grades.json', *[r+'.json' for r in RATERS]], 'output name')
        require(len(raw) <= CAP and self.used+len(raw) <= 8*CAP-(0 if name == 'failure.json' else 4096), 'output cap')
        fd = os.open(self.root/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        self.used += len(raw)
        return {'path': name, 'size_bytes': len(raw), 'sha256': sha(raw)}
    def write(self, name, value): return self.raw(name, self.p.encode(value))
    def fail(self, error): self.write('failure.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'automatic_retry': False})


def release(p, filename, digest, commit):
    path = Path(filename)
    binding = {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}
    raw = read(binding); committed({path: raw}, commit)
    value = p.decode(raw)
    keys(value, ('schema', 'packet_commit', 'reference_commit', 'packet_manifest', 'evaluation'))
    require(value['schema'] == 'jlens_pattern_calibration_judging_release_v1' and
            value['reference_commit'] == REFERENCE_COMMIT, 'release schema/reference')
    require(type(value['packet_commit']) is str and re.fullmatch('[0-9a-f]{40}', value['packet_commit']), 'packet commit')
    descriptor(value['packet_manifest']); keys(value['evaluation'], ('source', 'receipt', 'scores'))
    for pin in value['evaluation'].values(): descriptor(pin)
    require(value['evaluation']['source']['sha256'] == EVALUATOR_SHA, 'fixed evaluator source')
    require(len({v['path'] for v in [value['packet_manifest'], *value['evaluation'].values()]}) == 4, 'distinct release paths')
    return value, {'artifact': binding, 'commit': commit}


def public_bundle(p, released):
    entry = released['packet_manifest']; root = Path(entry['path']).parent
    require(Path(entry['path']).name == 'manifest.json', 'manifest name')
    raw = read(entry); manifest = p.decode(raw)
    keys(manifest, ('schema', 'seed', 'source_sha256', 'python', 'reference_commit', 'inputs', 'outputs',
                    'score_key_or_array_files_opened', 'completed_utc'))
    require(manifest['schema'] == 'jlens_pattern_calibration_packets_v1' and manifest['seed'] == 20260922 and
            manifest['source_sha256'] == PACKET_SHA and manifest['python'] == PYTHON == sys.version and
            manifest['reference_commit'] == REFERENCE_COMMIT and manifest['score_key_or_array_files_opened'] == 0,
            'packet scope/runtime/source')
    require(manifest['inputs'] == {n: {'path': str(path), 'sha256': pin} for n, (path, pin) in p.PINS.items()}, 'packet frozen inputs')
    require(type(manifest['outputs']) is list and len(manifest['outputs']) == 5 and
            [v['path'] for v in manifest['outputs']] == [r+'.json' for r in RATERS]+['private-map.json'], 'packet output roster')
    all_bytes = {Path(entry['path']): raw}; public = {}
    for r, item in zip(RATERS, manifest['outputs'][:4], strict=True):
        pin = member(root, item, r+'.json'); raw = read(pin); all_bytes[Path(pin['path'])] = raw
        public[r] = p.decode(raw); p.public_ids(public[r])
    # Validate the private map's metadata, never open it before response lock.
    descriptor(member(root, manifest['outputs'][4], 'private-map.json'))
    require(len(set.union(*[p.public_ids(v) for v in public.values()])) == 256 and
            len({v['packet_id'] for v in public.values()}) == 4 and
            len({b['block_id'] for v in public.values() for b in v['blocks']}) == 16, 'cross-packet IDs')
    committed(all_bytes, released['packet_commit'])
    return manifest, public


def seal(*, release_path, release_sha, release_commit, raw_commit, raw_paths, target):
    p = helper(); stage = Stage(target, 'seal', p); initial = source_sha()
    try:
        released, binding = release(p, release_path, release_sha, release_commit)
        manifest, public = public_bundle(p, released); keys(raw_paths, RATERS)
        require(len({v['path'] for v in raw_paths.values()}) == 4, 'four distinct raw paths')
        raws = {r: read(raw_paths[r]) for r in RATERS}
        committed({Path(raw_paths[r]['path']): raws[r] for r in RATERS}, raw_commit)
        ancestor(Path(release_path).parent, release_commit, raw_commit)
        ancestor(Path(release_path).parent, released['packet_commit'], raw_commit)
        for r in RATERS: p.validate_response(p.decode(raws[r]), public[r])
        copies = {r: stage.raw(r+'.json', raws[r]) for r in RATERS}
        require(source_sha() == initial and sha(PACKET_SOURCE.read_bytes()) == PACKET_SHA, 'source drift')
        return stage.write('lock.json', {'schema': 'jlens_pattern_calibration_response_lock_v1', 'count': 256,
            'source_sha256': initial, 'packet_source_sha256': PACKET_SHA, 'sealed_utc': utc(),
            'release': binding, 'packet_manifest': released['packet_manifest'], 'packet_commit': released['packet_commit'],
            'packet_outputs': manifest['outputs'], 'raw_commit': raw_commit, 'raw_responses': raw_paths, 'responses': copies})
    except Exception as error: stage.fail(error); raise


def verify_response_lock(*, release_path, release_sha, release_commit, responses, lock_sha, lock_commit):
    """Public-only gate; no private map, evaluation receipt, score or array I/O."""
    p = helper(); released, binding = release(p, release_path, release_sha, release_commit)
    manifest, public = public_bundle(p, released)
    path = Path(responses)/'lock.json'
    raw_lock = read({'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': lock_sha})
    lock = p.decode(raw_lock)
    keys(lock, ('schema', 'count', 'source_sha256', 'packet_source_sha256', 'sealed_utc', 'release', 'packet_manifest',
                'packet_commit', 'packet_outputs', 'raw_commit', 'raw_responses', 'responses'))
    require(lock['schema'] == 'jlens_pattern_calibration_response_lock_v1' and lock['count'] == 256 and
            lock['source_sha256'] == source_sha() and lock['packet_source_sha256'] == PACKET_SHA and
            lock['release'] == binding and lock['packet_manifest'] == released['packet_manifest'] and
            lock['packet_commit'] == released['packet_commit'] and lock['packet_outputs'] == manifest['outputs'], 'lock bindings')
    keys(lock['responses'], RATERS); keys(lock['raw_responses'], RATERS)
    copies, raws, choices = {path: raw_lock}, {}, {}
    for r in RATERS:
        pin = member(responses, lock['responses'][r], r+'.json'); copy = read(pin)
        raw = read(lock['raw_responses'][r]); require(copy == raw, 'sealed first response changed')
        copies[Path(pin['path'])] = copy; raws[Path(lock['raw_responses'][r]['path'])] = raw
        choices[r] = p.validate_response(p.decode(copy), public[r])
    require(len(copies) == 5 and len(raws) == 4, 'five locked and four raw blobs')
    committed(copies, lock_commit); committed(raws, lock['raw_commit'])
    require(lock_commit != lock['raw_commit'], 'raw first responses must precede lock commit')
    ancestor(Path(responses), lock['raw_commit'], lock_commit)
    ancestor(Path(release_path).parent, release_commit, lock['raw_commit'])
    ancestor(Path(release_path).parent, released['packet_commit'], lock['raw_commit'])
    return {'released': released, 'release': binding, 'manifest': manifest, 'public': public, 'lock': lock, 'choices': choices}


def reconstruct(p, checked):
    # This function is called only after the public gate has verified all commits.
    manifest = checked['manifest']; data = {}
    for name in ('references', 'dataset', 'pairs'):
        path, pin = p.PINS[name]
        data[name] = p.decode(read({'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': pin}))
    expected, mapping = p.make_packets(data['references'], data['dataset'], data['pairs'])
    require(expected == checked['public'], 'public reconstruction mismatch')
    root = Path(checked['released']['packet_manifest']['path']).parent
    pin = member(root, manifest['outputs'][4], 'private-map.json'); raw = read(pin)
    committed({Path(pin['path']): raw}, checked['released']['packet_commit'])
    require(p.decode(raw) == mapping, 'private map mismatch')
    return mapping


def forward_scope(p, released):
    e = released['evaluation']; raw = read(e['receipt']); receipt = p.decode(raw)
    require(receipt['schema'] == 'jlens_pattern_calibration_evaluation_receipt_v1' and receipt['status'] == 'complete' and
            receipt['source_sha256'] == EVALUATOR_SHA and receipt['reference_commit'] == REFERENCE_COMMIT and
            receipt['references_sha256'] == p.PINS['references'][1] and
            receipt['calibration_receipt_sha256'] == p.PINS['calibration_receipt'][1] and
            receipt['preflight_receipt_sha256'] == p.PINS['preflight'][1], 'evaluation source/reference binding')
    require(receipt['forward_count'] == 32 and receipt['evaluation_pairs'] == 16 and receipt['capture_locations'] == ['prefix_end'] and
            receipt['parameters_unchanged'] is True and receipt['calibration_rows_loaded'] == receipt['reference_decodes'] ==
            receipt['prefix_tokenizations'] == 0 and all(receipt[n] is False for n in ('lens_loaded', 'tokenizer_loaded', 'pca_refit', 'pattern_fit')),
            'evaluation capture scope')
    require(receipt['runtime']['python'] == PYTHON and receipt['runtime']['packages'] ==
            {'numpy': '1.26.4', 'torch': '2.11.0+cu128', 'transformers': '5.5.0', 'huggingface_hub': '1.8.0'}, 'evaluation runtime')
    for name in ('protocol', 'references', 'calibration_receipt', 'dataset', 'pairs', 'preflight'):
        path, pin = p.PINS[name]; require(receipt['input_pins'].get(str(path)) == pin, 'evaluation frozen input binding')
    outputs = receipt['outputs']
    require(len(outputs) == 4 and [v['path'] for v in outputs] == ['features.npz', 'inputs.json', 'scores.json', 'gaps.json'], 'evaluation outputs')
    scorepin = member(Path(e['receipt']['path']).parent, outputs[2], 'scores.json')
    require(scorepin == e['scores'], 'exact evaluation score output binding')
    require(read(e['source']) and e['source']['sha256'] == EVALUATOR_SHA, 'evaluation source file changed')


def grade_values(p, mapping, choices, exported):
    keys(exported, ('schema', 'locations'))
    require(exported['schema'] == 'jlens_pattern_calibration_evaluation_scores_v1', 'score schema')
    keys(exported['locations'], ('prefix_end',)); axes = exported['locations']['prefix_end']
    require(type(axes) is list and len(axes) == 4, 'four score axes'); scores = {}
    for axis, row in zip(AXES, axes, strict=True):
        keys(row, ('axis', 'values')); keys(row['values'], IDS)
        require(row['axis'] == axis and list(row['values']) == IDS and
                all(type(v) in (float, int) and math.isfinite(v) for v in row['values'].values()), 'finite ordered scores')
        scores[axis] = row['values']
    keys(mapping, ('schema', 'rows')); keys(choices, RATERS)
    require(mapping['schema'] == 'jlens_pattern_calibration_private_map_v1' and len(mapping['rows']) == 256, 'map roster')
    items, cells, seen = [], set(), set()
    for row in mapping['rows']:
        r, axis, pair = row['rater'], row['axis'], row['pair_id']
        require(r in RATERS and axis in AXES and pair in PAIRS, 'map cell')
        ri = RATERS.index(r); arm, cohort = ('P' if ri % 2 == 0 else 'U'), (1 if ri < 2 else 2)
        require(row['arm'] == arm and row['cohort'] == cohort and row['topic'] in TOPICS and
                {row['first_id'], row['second_id']} == {pair+'-L', pair+'-R'}, 'map allocation/targets')
        cell = (axis, pair, arm, cohort)
        require(cell not in cells and row['item_id'] not in seen, 'duplicate map cell/item')
        cells.add(cell); seen.add(row['item_id'])
        choice = choices[r][row['item_id']]; require(choice in ('FIRST', 'SECOND'), 'forced choice')
        first, second = (scores[axis][row[n]] for n in ('first_id', 'second_id'))
        gap = float(first)-float(second); require(math.isfinite(gap), 'finite difference')
        truth = 'FIRST' if gap > 0 else 'SECOND' if gap < 0 else 'TIE'
        items.append({**row, 'choice': choice, 'truth': truth, 'score_first': first, 'score_second': second,
            'gap': gap, 'absolute_gap': abs(gap), 'chosen_id': row['first_id'] if choice == 'FIRST' else row['second_id'],
            'credit': .5 if gap == 0 else float(choice == truth)})
    require(seen == {f'Q{i:03}' for i in range(1, 257)}, 'all256 item IDs')
    for r in RATERS: keys(choices[r], [v['item_id'] for v in items if v['rater'] == r])
    def summary(rows):
        return {'credit': sum(v['credit'] for v in rows), 'total': len(rows),
            'correct': sum(v['credit'] == 1 for v in rows), 'incorrect': sum(v['credit'] == 0 for v in rows),
            'exact_ties': sum(v['truth'] == 'TIE' for v in rows),
            **{'constant_'+s+'_credit': sum(.5 if v['truth'] == 'TIE' else float(v['truth'] == s) for v in rows) for s in ('FIRST', 'SECOND')}}
    def breakdown(field, values):
        return {str(v): {a: summary([r for r in items if r[field] == v and r['arm'] == a]) for a in ARMS} for v in values}
    arms = {a: summary([r for r in items if r['arm'] == a]) for a in ARMS}
    cohorts, per_axis = breakdown('cohort', (1, 2)), breakdown('axis', AXES)
    readers = {r: {'arm': 'P' if i % 2 == 0 else 'U', 'cohort': 1 if i < 2 else 2,
        'summary': summary([v for v in items if v['rater'] == r]),
        'per_axis': {a: summary([v for v in items if v['rater'] == r and v['axis'] == a]) for a in AXES}}
        for i, r in enumerate(RATERS)}
    controls = {}
    for r, v in readers.items():
        s = v['summary']; stronger = max(s['constant_FIRST_credit'], s['constant_SECOND_credit'])
        controls[r] = {'arm': v['arm'], 'credit': s['credit'], 'stronger_constant_credit': stronger,
                      'exceeds_stronger_constant': s['credit'] > stronger}
    primary = {'scope': 'all_four_axes', 'P_minus_U_credit_out_of_128': arms['P']['credit']-arms['U']['credit'],
        'P_minus_U_by_cohort_out_of_64': {c: v['P']['credit']-v['U']['credit'] for c, v in cohorts.items()}, 'reader_controls': controls}
    primary['positive_each_cohort'] = all(v > 0 for v in primary['P_minus_U_by_cohort_out_of_64'].values())
    primary['each_P_reader_exceeds_stronger_constant'] = all(v['exceeds_stronger_constant'] for v in controls.values() if v['arm'] == 'P')
    primary['pilot_criterion_met'] = primary['positive_each_cohort'] and primary['each_P_reader_exceeds_stronger_constant']
    by = {(v['axis'], v['pair_id'], v['arm'], v['cohort']): v for v in items}
    changes, agreement = [], []
    for axis in AXES:
        for pair in PAIRS:
            for cohort in (1, 2):
                a, b = by[axis, pair, 'P', cohort], by[axis, pair, 'U', cohort]
                require((a['first_id'], a['second_id']) == (b['first_id'], b['second_id']), 'within-cohort orientation')
                delta = a['credit']-b['credit']
                changes.append({'axis': axis, 'pair_id': pair, 'topic': a['topic'], 'cohort': cohort,
                    'P_item_id': a['item_id'], 'U_item_id': b['item_id'], 'P_credit': a['credit'], 'U_credit': b['credit'],
                    'P_minus_U_credit': delta, 'outcome': 'gain' if delta > 0 else 'harm' if delta < 0 else 'unchanged'})
            for arm in ARMS:
                a, b = by[axis, pair, arm, 1], by[axis, pair, arm, 2]
                require((a['first_id'], a['second_id']) == (b['second_id'], b['first_id']), 'opposite cohort orientation')
                agreement.append({'axis': axis, 'pair_id': pair, 'arm': arm, 'cohort1_chosen_id': a['chosen_id'],
                    'cohort2_chosen_id': b['chosen_id'], 'agree': a['chosen_id'] == b['chosen_id']})
    for s in [*arms.values(), *[v for x in per_axis.values() for v in x.values()]]:
        require(s['constant_FIRST_credit'] == s['constant_SECOND_credit'] == s['total']/2, 'pooled opposite-order control')
    return {'schema': 'jlens_pattern_calibration_grades_v1', 'primary': primary, 'arms': arms, 'per_axis': per_axis,
        'per_cohort': cohorts, 'per_topic': breakdown('topic', TOPICS), 'per_pair': breakdown('pair_id', PAIRS), 'per_reader': readers,
        'gains_harm': changes, 'agreement': {'items': agreement,
            'arms': {a: {'agree': sum(v['agree'] for v in agreement if v['arm'] == a), 'total': 64} for a in ARMS},
            'per_axis': {axis: {a: {'agree': sum(v['agree'] for v in agreement if v['arm'] == a and v['axis'] == axis),
                                 'total': 16} for a in ARMS} for axis in AXES}}, 'items': items}


def grade(*, release_path, release_sha, release_commit, responses, lock_sha, lock_commit, target):
    p = helper(); stage = Stage(target, 'grade', p); initial = source_sha()
    try:
        checked = verify_response_lock(release_path=release_path, release_sha=release_sha, release_commit=release_commit,
            responses=responses, lock_sha=lock_sha, lock_commit=lock_commit)
        mapping = reconstruct(p, checked)
        forward_scope(p, checked['released'])
        exported = p.decode(read(checked['released']['evaluation']['scores']))  # First scientific key read, after locks.
        result = grade_values(p, mapping, checked['choices'], exported)
        require(source_sha() == initial and sha(PACKET_SOURCE.read_bytes()) == PACKET_SHA, 'source drift')
        result['provenance'] = {'source_sha256': initial, 'release': checked['release'], 'lock_sha256': lock_sha,
            'lock_commit': lock_commit, 'raw_commit': checked['lock']['raw_commit'], 'responses': checked['lock']['responses'],
            'evaluation': checked['released']['evaluation'], 'packet_manifest': checked['released']['packet_manifest']}
        return stage.write('grades.json', result)
    except Exception as error: stage.fail(error); raise
