"""Audit-only IO, provenance and resource checks; no producer imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BATCH = Path('/tmp/spectral-experiment-artifacts/spectral-grokking-raw-direction-20260909.KzGtkl')
OLD = Path('/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD')
OLD_SHA = '22bb866319fa2fa85771bd156eb406523a66abee4d0bc789cc6104d40ec86e45'
POLICY = 'raw_norm_matched'
SEEDS = tuple(range(100, 105))
STEPS = (1501, 2000, 2500)
FLOOR = 1e-30


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class Audit:
    def __init__(self):
        self.started = time.monotonic()
        self.checks, self.errors, self.inputs, self.max_error = 0, [], {}, {}
        self.out = None

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.errors.append(label)
        if time.monotonic() - self.started > 580:
            raise TimeoutError('580-second audit deadline')
        if len(self.errors) > 1000:
            raise ValueError('More than1000 audit errors; preserved as failure')

    def require(self, condition, label):
        self.check(condition, label)
        if not condition:
            raise ValueError(label)

    def close(self, left, right, label, rtol=1e-9, atol=1e-12):
        if left is None or right is None:
            self.check(left is right, label + ': undefined handling')
            return
        self.require(numeric(left) and numeric(right), label + ': finite scalars')
        err = abs(left-right)
        family = label.split(':')[0]
        self.max_error[family] = max(self.max_error.get(family, 0), err)
        self.check(math.isclose(left, right, rel_tol=rtol, abs_tol=atol), f'{label}: {left} != {right}')

    def receipt(self, rec, expected=None):
        path = Path(rec['path'])
        self.require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1, 'Regular singly-linked file '+str(path))
        if expected is not None:
            self.require(path.resolve() == Path(expected).resolve(), 'Exact receipt path '+str(path))
        self.require(re.fullmatch('[0-9a-f]{64}', rec['sha256']) is not None, 'SHA format')
        self.require(digest(path) == rec['sha256'], 'Receipt SHA '+str(path))
        self.require('size_bytes' not in rec or rec['size_bytes'] == path.stat().st_size, 'Receipt size')
        self.inputs[str(path)] = {'path': str(path), 'sha256': rec['sha256'], 'size_bytes': path.stat().st_size}
        return path

    def read(self, path, expected_sha=None):
        path = Path(path)
        self.receipt({'path': str(path), 'sha256': expected_sha or digest(path)}, path)
        self.require(path.stat().st_size < 100 * 1024**2, 'JSON input size cap')
        value = json.loads(path.read_text())
        self.require(isinstance(value, dict) and finite_tree(value), 'Finite JSON object')
        return value

    def sources(self, source_map):
        self.require(isinstance(source_map, dict) and bool(source_map), 'Nonempty source map')
        for name, sha in source_map.items():
            self.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Relative source path')
            self.receipt({'path': str(REPO/name), 'sha256': sha}, REPO/name)

    def start(self, name):
        self.out = BATCH/name
        self.require(not self.out.exists() and not self.out.is_symlink(), 'Exclusive audit output')
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.require(os.environ.get(var) == '1', var+'=1')
        group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2].lstrip('/')
        self.require(int((group/'memory.max').read_text()) == 16*1024**3, '16GiB memory cap')
        self.require(int((group/'memory.swap.max').read_text()) == 0, 'Zero swap cap')
        quota, period = map(int, (group/'cpu.max').read_text().split())
        self.require(quota == period, 'One CPU quota')
        own_unit=next((part for part in reversed(group.parts) if part.endswith('.service')),None)
        self.require(own_unit is not None,'Bounded audit service cgroup')
        own=subprocess.check_output(['systemctl','--user','show',own_unit,'-p','Type','-p','RuntimeMaxUSec','-p','Restart','-p','KillMode'],text=True)
        own=dict(line.split('=',1) for line in own.splitlines())
        self.require(own=={'Type':'exec','RuntimeMaxUSec':'10min','Restart':'no','KillMode':'control-group'},'Ten-minute exec service/no restart')
        props = subprocess.check_output(['systemctl', '--user', 'show', 'spectral-base-grokking-raw-direction-001.service', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result'], text=True)
        props = dict(line.split('=', 1) for line in props.splitlines())
        self.require(props == {'MainPID':'0', 'ActiveState':'inactive', 'Result':'success'}, 'Training service terminal/successful')
        self.require(shutil.disk_usage(BATCH).free >= 1024**3 + 100*1024**2, 'Output free reserve')
        self.out.mkdir(mode=0o700)
        for path in sorted(HERE.glob('*.py')) + [HERE/'plan.md']:
            self.receipt({'path':str(path), 'sha256':digest(path)}, path)

    def finish(self, payload):
        for rec in list(self.inputs.values()):
            try:
                self.receipt(rec)
            except Exception as error:
                self.errors.append(f'Completion recheck: {type(error).__name__}: {error}')
        result = {'status':'FAIL' if self.errors else 'PASS', 'checks':self.checks,
                  'errors':self.errors, 'input_receipts':list(self.inputs.values()),
                  'maximum_absolute_errors':self.max_error,
                  'elapsed_seconds':time.monotonic()-self.started,
                  'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  **payload}
        raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
        self.require(len(raw)<100*1024**2 and shutil.disk_usage(self.out).free>=len(raw)+1024**3, 'Bounded result output')
        with (self.out/'result.json').open('xb') as handle:
            handle.write(raw)
        print(json.dumps({'status':result['status'], 'checks':self.checks, 'result_sha256':digest(self.out/'result.json')}), flush=True)
        return not self.errors


def batch_contract(audit, expected_sha):
    audit.require(re.fullmatch('[0-9a-f]{64}', expected_sha) is not None, 'Main-supplied batch hash')
    batch = audit.read(BATCH/'batch-complete.json', expected_sha)
    manifest = audit.read(audit.receipt(batch['batch_manifest'], BATCH/'batch-manifest.json'))
    audit.require(batch['status']=='complete' and batch['schema']=='grokking_raw_direction_batch_v1', 'Completed raw batch')
    audit.require(batch['seeds']==list(SEEDS) and [x['seed'] for x in batch['accepted']]==list(SEEDS), 'Exact five seeds')
    audit.require(not (BATCH/'batch-failure.json').exists(), 'No batch failure')
    for key in manifest:
        audit.require(batch[key]==manifest[key], 'Batch manifest identity '+key)
    audit.sources(batch['source_sha256'])
    old = audit.read(OLD/'batch-complete.json', OLD_SHA)
    audit.require(old['status']=='complete' and old['seeds']==list(SEEDS) and [r['seed'] for r in old['accepted']]==list(SEEDS),'Pinned archived five-seed roster')
    completed = []
    for seed, rec, old_rec in zip(SEEDS, batch['accepted'], old['accepted'], strict=True):
        directory = BATCH/f'seed{seed}'
        value = audit.read(audit.receipt(rec, directory/'complete.json'))
        audit.require(value['seed']==seed and value['policy']==POLICY and value['status']=='complete', 'Seed identity')
        audit.require(value['source_sha256']==batch['source_sha256'], 'Seed source binding')
        audit.require(value['environment']==batch['environment'],'Shared acquisition environment')
        audit.require(value['accepted_roster']==[[POLICY,t] for t in STEPS] and value['history_steps']==list(range(1501,2501)), 'Exact seed step roster')
        audit.require(value['completed_updates']==1000 and not (directory/'failure.json').exists(), 'Complete seed/no failure')
        expected_admission=None if seed==100 else {key:batch['accepted'][0][key] for key in ('path','sha256','size_bytes')}
        audit.require(value['seed100_admission']==expected_admission,'Seed100 resource admission identity')
        expected_names = {'manifest.json','first-step-tensors.pt','first-step-summary.json'} | {f'{POLICY}-{part}-{step:06d}.{extension}' for step in STEPS for part,extension in (('step','pt'),('through','json'))}
        artifacts = value['artifact_receipts']
        audit.require(len(artifacts)==9 and {Path(x['path']).name for x in artifacts}==expected_names, 'Nine-artifact roster')
        audit.require({p.name for p in directory.iterdir()}==expected_names|{'complete.json'},'No extra seed artifacts')
        by_name = {Path(rec['path']).name:rec for rec in artifacts}
        for name, artifact in by_name.items():
            audit.receipt(artifact, directory/name)
        seed_manifest = audit.read(directory/'manifest.json')
        audit.require(all(value[k]==v for k,v in seed_manifest.items()), 'Seed manifest identity')
        previous = audit.read(audit.receipt(old_rec, OLD/f'seed{seed}/complete.json'))
        audit.require(all(batch['source_sha256'].get(name)==sha for name,sha in previous['source_sha256'].items()),'Original scientific source pins retained')
        audit.require(previous['seed']==seed and previous['parent_checkpoint']==value['parent_checkpoint'], 'Archived parent receipt identity')
        audit.require(previous['parent_metrics_sha256']==value['parent_metrics_sha256'],'Archived parent metrics identity')
        audit.require(previous['fork_scientific_state_sha256']==value['fork_scientific_state_sha256'], 'Archived scientific parent hash')
        audit.require(value['archived_reference']['batch_completion_sha256']==OLD_SHA and value['archived_reference']['seed_completion']==old_rec, 'Archived reference binding')
        audit.require(value['checkpoints']==[by_name[f'{POLICY}-step-{t:06d}.pt'] for t in STEPS], 'Three checkpoint receipts')
        completed.append((value, by_name))
    return batch, completed
