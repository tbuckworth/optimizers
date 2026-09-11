#!/usr/bin/env python3
"""One exclusive bounded I20 counterfactual; never executes I19 or redraws inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = Path('/tmp/spectral-experiment-artifacts/spectral-i19-001.wnMmy7')
I19 = HERE.parent / 'iteration-019'
SEEDS = tuple(range(19000, 19032))
PROCESS_VARIANCES = (0.0, .01, .1)
ROTATIONS = (0.0, math.pi / 4)
HORIZON, SECONDS = 4000, 540
ARRAY_LIMIT, ROOT_LIMIT = 2 * 1024**3, 3 * 1024**3
SOURCE_NAMES = ('response_core.py', 'test_response_core.py', 'run_response.py',
                'test_run_response.py', 'protocol.md', 'best-practices-check.md')
PINS = (
    (PARENT / 'attempt.json', '3ed5bfe9d66e94a0193e7881d7aa6bd11b61e80eb0d477aeb9cf89c3cfb2a1c1'),
    (PARENT / 'completion.json', '6a69b9f96f81dcd1066954f53f703280794c88ea29c32beed0c01fa329b4740e'),
    (I19 / 'analysis-001/audit.json', 'eb35695b43bb26ff2c68a4be4ed4cb5fa6ddb61b999a95e0e5d060d72a077e6c'),
    (I19 / 'analysis-001/summary.json', '5d70c8297708af99903be61e554b43d19c0447eb2c97c48bde746712edbef1b7'),
    (I19 / 'report-check-001/report-audit.json', 'dcc36b3b230d0f08c73852a6fc78062c1c20208ed952589e8a5eede5a0ce1304'),
)
INPUT_NAMES = ('g', 'mu', 'A', 'basis_present', 'full_action', 'full_basis_present', 's', 'output')
INITIALIZATION = ('all outputs g1; zero weighted moments/masses before t1; parent post-ingest mu held fixed; '
                  'unavailable full/weighted directions use identity')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and path.is_file()
         and not path.is_symlink(), 'Expected resolved regular file: ' + str(path))


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'Duplicate JSON key: ' + key)
            value[key] = item
        return value
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=lambda value:
                         (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def json_once(path, value):
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(content)


def file_record(path):
    regular(path)
    return {'path': str(path), 'size': path.stat().st_size, 'sha256': sha(path)}


def source_manifest(commit):
    need(re.fullmatch(r'[0-9a-f]{40}', commit or '') is not None, 'Full frozen commit required')
    rows = []
    for name in SOURCE_NAMES:
        path = HERE / name
        regular(path)
        relative = str(path.relative_to(REPO))
        frozen = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=REPO)
        digest = sha(path)
        need(hashlib.sha256(frozen).hexdigest() == digest, 'Changed source: ' + relative)
        rows.append({'path': relative, 'size': path.stat().st_size, 'sha256': digest})
    return rows


def verify_sources(commit, sources):
    need(re.fullmatch(r'[0-9a-f]{40}', commit or '') is not None, 'Invalid parent source commit')
    need(len({row['path'] for row in sources}) == len(sources), 'Duplicate source')
    for row in sources:
        relative = Path(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path')
        path = REPO / relative
        regular(path)
        need(path.stat().st_size == row['size'] and sha(path) == row['sha256'], 'Parent source changed')
        frozen = subprocess.check_output(['git', 'show', commit + ':' + row['path']], cwd=REPO)
        need(hashlib.sha256(frozen).hexdigest() == row['sha256'], 'Parent source freeze differs')


def expected_cells():
    return [(seed, pi, ri) for seed in SEEDS for pi in range(3) for ri in range(2)]


def cell_id(seed, pi, ri):
    return f'{seed}-p{pi}-r{ri}'


def inventory_files(root, rows):
    """Validate ordered roster, physical membership and every recorded byte hash."""
    expected = [cell_id(*cell) for cell in expected_cells()]
    need([row['id'] for row in rows] == expected, 'Incomplete/reordered/duplicated stream roster')
    names = set()
    for row in rows:
        for key, extension in (('array', '.npz'), ('metadata', '.json')):
            record = row[key]
            name = 'streams/' + row['id'] + extension
            need(record['path'] == name, 'Unsafe/incorrect inventory path')
            path = root / name
            regular(path)
            need(path.stat().st_size == record['size'] and sha(path) == record['sha256'],
                 'Inventory hash/size differs: ' + name)
            names.add(path.name)
    need({path.name for path in (root / 'streams').iterdir()} == names, 'Extra/missing physical stream file')


def verify_parent():
    files = []
    for path, digest in PINS:
        record = file_record(path)
        need(record['sha256'] == digest, 'Accepted parent pin changed: ' + str(path))
        files.append(record)
    attempt, completion, audit, summary, report = [read_json(path) for path, _ in PINS]
    need(completion['status'] == 'complete' and completion['failure'] is None
         and completion['completed_streams'] == 192 and completion['completed_observations'] == 768000,
         'Parent incomplete')
    need(completion['attempt_sha256'] == PINS[0][1], 'Parent attempt binding differs')
    need(audit['status'] == 'pass' and not audit['errors'] and report['status'] == 'pass'
         and summary['audit_status'] == 'pass', 'Parent not accepted')
    need(set(path.name for path in PARENT.iterdir()) == {'attempt.json', 'completion.json', 'streams'},
         'Parent physical root membership changed')
    inventory_files(PARENT, completion['streams'])
    provenance = audit['input_provenance']
    groups = [(attempt['frozen_commit'], attempt['sources']),
              (provenance['analysis_commit'], provenance['analysis_sources']),
              (report['input_provenance']['report_commit'], report['input_provenance']['report_sources'])]
    sources = []
    for commit, records in groups:
        verify_sources(commit, records)
        sources.extend(dict(row, frozen_commit=commit) for row in records)
    return {'root': str(PARENT), 'files': files, 'sources': sources}, completion['streams']


def load_parent(row):
    meta_path, array_path = PARENT / row['metadata']['path'], PARENT / row['array']['path']
    for path, key in ((meta_path, 'metadata'), (array_path, 'array')):
        regular(path)
        need(sha(path) == row[key]['sha256'], 'Parent input changed before load')
    metadata = read_json(meta_path)
    core = metadata['core']
    with zipfile.ZipFile(array_path) as archive:
        need(archive.namelist() == [name + '.npy' for name in core['array_order']], 'Parent ZIP membership/order differs')
        need(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()), 'Unexpected compressed parent')
        for name, info in zip(core['array_order'], archive.infolist(), strict=True):
            raw_bytes = math.prod(core['array_shapes'][name]) * np.dtype(core['array_dtypes'][name]).itemsize
            need(raw_bytes <= info.file_size <= raw_bytes + 65536, 'Parent expanded member size differs')
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in INPUT_NAMES}
    for name, value in arrays.items():
        need(value.shape == tuple(core['array_shapes'][name])
             and value.dtype == np.dtype(core['array_dtypes'][name])
             and np.isfinite(value).all(), 'Invalid parent selected array: ' + name)
    need(metadata['id'] == row['id'] and metadata['horizon'] == HORIZON, 'Parent metadata identity differs')
    return arrays, metadata


def check_parent_closures(arrays, inputs, parent_policies, policies, estimators):
    """These are algebra controls, not performance-based acceptance gates."""
    pairs = [('native/r0p9/cp', 'native_cp'), ('native/rstar/cp', 'native_cp_star'),
             ('oracle_useful/r0p9/cp', 'oracle_useful'),
             ('oracle_useful/rstar/cp', 'oracle_useful_star'),
             ('oracle_nuisance/r0p9/cp', 'oracle_nuisance')]
    pairs.extend((name + '/r0p9/rec9', 'ema_q0p9') for name in estimators)
    for target, parent_name in pairs:
        observed = arrays['output'][:, policies.index(target)]
        expected = inputs['output'][:, parent_policies.index(parent_name)]
        need(observed.shape == expected.shape and np.isfinite(observed).all()
             and np.allclose(observed, expected, rtol=2e-11, atol=5e-12),
             'Parent algebra closure differs: ' + target + ' / ' + parent_name)


def write_stream(root, identity, arrays, metadata, core):
    need(re.fullmatch(r'190[0-3][0-9]-p[0-2]-r[0-1]', identity) is not None, 'Invalid stream ID')
    shapes = core.expected_shapes(HORIZON)
    need(list(arrays) == list(core.ARRAY_ORDER), 'Output array order differs')
    for name, value in arrays.items():
        dtype = np.dtype('bool' if name == 'direction_present' else 'float64')
        need(type(value) is np.ndarray and value.shape == shapes[name] and value.dtype == dtype
             and np.isfinite(value).all(), 'Invalid new numeric array: ' + name)
    npz, meta = root / 'streams' / (identity + '.npz'), root / 'streams' / (identity + '.json')
    need(not npz.exists() and not meta.exists(), 'Consumed stream identity')
    json.dumps(metadata, allow_nan=False)
    with npz.open('xb') as handle:
        # NumPy1.26: no allow_pickle kwarg to savez; strict numeric validation above.
        np.savez(handle, **arrays)
    json_once(meta, metadata)
    return {'id': identity,
            'array': {'path': str(npz.relative_to(root)), 'size': npz.stat().st_size, 'sha256': sha(npz)},
            'metadata': {'path': str(meta.relative_to(root)), 'size': meta.stat().st_size, 'sha256': sha(meta)}}


def configure():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        need(os.environ.get(name) == '1', 'One numerical thread required: ' + name)


def preflight_root(root):
    need(root.is_absolute() and root.resolve() == root and not root.is_symlink(), 'Resolved root required')
    need(root.parent == Path('/tmp/spectral-experiment-artifacts') and root.name.startswith('spectral-i20-001.'), 'Unexpected root')
    need(os.path.ismount('/private-artifacts/storage'), 'Large volume not mounted')
    need(root.is_dir() and not any(root.iterdir()), 'Root must exist and be empty')
    need(root.stat().st_dev == Path('/private-artifacts/storage').stat().st_dev, 'Root on wrong device')
    need(shutil.disk_usage(root).free >= 10 * 1024**3, 'Less than10GiB free')


def acquire(root, commit):
    started = time.monotonic()
    configure()
    sources = source_manifest(commit)
    preflight_root(root)
    parent, parent_rows = verify_parent()
    from response_core import reconstruct, expected_shapes, POLICIES, ESTIMATORS, ARRAY_ORDER
    import response_core as core
    need(Path(core.__file__).resolve() == HERE / 'response_core.py', 'Unexpected core import')
    attempt = {'schema': 'i20_response_attempt_v1', 'created_utc': utc(), 'root': str(root),
               'frozen_commit': commit, 'sources': sources, 'parent': parent,
               'seeds': list(SEEDS), 'process_variances': list(PROCESS_VARIANCES), 'rotations': list(ROTATIONS),
               'horizon': HORIZON, 'cooperative_seconds': SECONDS, 'array_limit_bytes': ARRAY_LIMIT,
               'root_limit_bytes': ROOT_LIMIT, 'pid': os.getpid(), 'service_invocation_id': os.environ.get('INVOCATION_ID'),
               'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(),
               'cuda_visible_devices': '', 'paid_spend_usd': 0, 'paid_reserved_usd': 0}
    json_once(root / 'attempt.json', attempt)
    inventory, failure, array_bytes = [], None, 0
    total_bytes = (root / 'attempt.json').stat().st_size
    try:
        (root / 'streams').mkdir()
        for (seed, pi, ri), parent_row in zip(expected_cells(), parent_rows, strict=True):
            need(time.monotonic() - started < SECONDS, 'Cooperative time limit reached')
            cell_start = time.monotonic()
            identity = cell_id(seed, pi, ri)
            inputs, metadata = load_parent(parent_row)
            need(metadata['seed'] == seed and metadata['process_index'] == pi and metadata['rotation_index'] == ri
                 and metadata['process_variance'] == PROCESS_VARIANCES[pi] and metadata['rotation'] == ROTATIONS[ri],
                 'Parent cell mismatch')
            axis = np.array([math.cos(ROTATIONS[ri]), math.sin(ROTATIONS[ri])], dtype=np.float64)
            # No signal/variance is passed to the non-oracle numerical core.
            arrays = reconstruct(inputs['g'], inputs['mu'], inputs['A'], inputs['basis_present'],
                                 inputs['full_action'], inputs['full_basis_present'], axis)
            check_parent_closures(arrays, inputs, metadata['core']['policy_names'], POLICIES, ESTIMATORS)
            need(time.monotonic() - started < SECONDS, 'Cooperative time limit reached')
            predicted = sum(value.nbytes for value in arrays.values())
            need(array_bytes + predicted + 1024**2 < ARRAY_LIMIT, 'Prospective array cap reached')
            need(total_bytes + predicted + 1024**2 < ROOT_LIMIT, 'Prospective root cap reached')
            stream_metadata = {'schema': 'i20_response_stream_v1', 'id': identity, 'seed': seed,
                'process_index': pi, 'rotation_index': ri, 'process_variance': PROCESS_VARIANCES[pi],
                'rotation': ROTATIONS[ri], 'horizon': HORIZON, 'frozen_commit': commit,
                'parent_array_sha256': parent_row['array']['sha256'], 'parent_metadata_sha256': parent_row['metadata']['sha256'],
                'policy_names': list(POLICIES), 'estimator_names': list(ESTIMATORS), 'array_order': list(ARRAY_ORDER),
                'array_shapes': {name: list(shape) for name, shape in expected_shapes(HORIZON).items()},
                'array_dtypes': {name: str(value.dtype) for name, value in arrays.items()},
                'initialization': INITIALIZATION, 'elapsed_seconds': time.monotonic() - cell_start}
            row = write_stream(root, identity, arrays, stream_metadata, core)
            inventory.append(row)
            array_bytes += row['array']['size']
            total_bytes += row['array']['size'] + row['metadata']['size']
            need(array_bytes <= ARRAY_LIMIT and total_bytes < ROOT_LIMIT - 1024**2, 'Actual storage cap reached')
            need(time.monotonic() - started < SECONDS, 'Cooperative time limit reached after write')
            print(json.dumps({'event': 'stream_complete', 'id': identity, 'completed': len(inventory),
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
        need(source_manifest(commit) == sources, 'Source changed during counterfactual')
        need(verify_parent()[0] == parent, 'Parent changed during counterfactual')
        need(time.monotonic() - started < SECONDS, 'Cooperative time limit reached after final validation')
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    completion = {'schema': 'i20_response_completion_v1', 'created_utc': utc(),
        'status': 'complete' if failure is None else 'failed', 'failure': failure, 'frozen_commit': commit,
        'attempt_sha256': sha(root / 'attempt.json'), 'expected_streams': 192, 'completed_streams': len(inventory),
        'completed_observations': len(inventory) * HORIZON, 'elapsed_seconds': time.monotonic() - started,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'array_bytes': array_bytes,
        'root_bytes_before_completion': total_bytes, 'streams': inventory}
    json_once(root / 'completion.json', completion)
    print(json.dumps({key: value for key, value in completion.items() if key != 'streams'}), flush=True)
    return 0 if failure is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frozen-commit', required=True)
    args = parser.parse_args()
    return acquire(args.root, args.frozen_commit)


if __name__ == '__main__':
    raise SystemExit(main())
