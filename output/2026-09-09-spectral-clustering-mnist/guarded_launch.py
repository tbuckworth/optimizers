"""Fixed one-shot launch; verifies limits and committed code before MNIST work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

MAIN = Path(__file__).resolve().parents[2]
WORKTREE = Path('/tmp/spectral-experiment-artifacts/spectral-clustering-pilot-20260909.j2jkfp/worktree')
PARENT = Path('/tmp/spectral-experiment-artifacts/spectral-clustering-mnist-20260909.51ggnu')
OUTPUT = PARENT/'acquisition-001'
UNIT = 'spectral-clustering-mnist-001.service'


def properties(unit, keys):
    output = subprocess.check_output(['systemctl', '--user', 'show', unit,
        *['--property='+key for key in keys]], text=True)
    return dict(line.split('=', 1) for line in output.splitlines())


def validate_bounds(effective, service):
    quota, period = effective['cpu.max'].split()
    if (effective['memory.max'] != str(16*1024**3)
            or effective['memory.swap.max'] != '0' or quota == 'max'
            or int(quota) != int(period) or int(period) <= 0
            or service != {'Type': 'exec', 'RuntimeMaxUSec': '30min',
                           'Restart': 'no', 'KillMode': 'control-group'}):
        raise RuntimeError('Unexpected effective experiment bounds')


def main():
    if len(sys.argv) != 1:
        raise ValueError('Fixed guard has no CLI overrides')
    if (not PARENT.is_dir() or PARENT.is_symlink() or list(PARENT.iterdir())
            or os.stat(PARENT).st_dev == os.stat('/').st_dev
            or OUTPUT.exists() or OUTPUT.is_symlink()):
        raise RuntimeError('Need unused exclusive large-volume output parent')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError('Expected single-threaded math environment')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Missing deterministic CUDA workspace setting')
    cgroup = next(x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines()
                  if x.startswith('0::'))
    if Path(cgroup).name != UNIT:
        raise RuntimeError('Unexpected cgroup')
    root = Path('/sys/fs/cgroup')/cgroup.lstrip('/')
    effective = {key: (root/key).read_text().strip()
                 for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
    service = properties(UNIT, ('Type', 'RuntimeMaxUSec', 'Restart', 'KillMode'))
    validate_bounds(effective, service)
    for prior in ('spectral-base-grokking-function-response-001.service',
                  'spectral-base-grokking-function-response-audit-001.service'):
        if properties(prior, ('MainPID', 'ActiveState', 'Result')) != {
                'MainPID': '0', 'ActiveState': 'inactive', 'Result': 'success'}:
            raise RuntimeError('Prior grokking stage is not successfully terminal')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True).strip()
    for row in gpu.splitlines():
        pid, name, _ = [x.strip() for x in row.split(',', 2)]
        if (pid, name) not in {('2101', '/usr/libexec/gnome-remote-desktop-daemon'), ('8861', 'stremio')}:
            raise RuntimeError('Unrecognized GPU occupant; not launching')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORKTREE, text=True).strip()
    sys.path.insert(0, str(WORKTREE))
    spec = importlib.util.spec_from_file_location('mnist_runner', WORKTREE/'experiments/anchor_graph_mnist.py')
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    pins = runner.source_pins()
    for name, expected in pins.items():
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=WORKTREE)
        if (hashlib.sha256(committed).hexdigest() != expected
                or hashlib.sha256((WORKTREE/name).read_bytes()).hexdigest() != expected):
            raise RuntimeError('Uncommitted or changed experiment source: '+name)
    own_name = str(Path(__file__).resolve().relative_to(MAIN))
    committed_guard = subprocess.check_output(['git', 'show', 'HEAD:'+own_name], cwd=MAIN)
    if committed_guard != Path(__file__).read_bytes():
        raise RuntimeError('Main launch guard not committed')
    print(json.dumps({'resource_guard': 'PASS', 'unit': UNIT, 'pid': os.getpid(),
        'source_commit': commit, 'source_sha256': pins, 'cgroup': cgroup,
        'effective': effective, 'service': service, 'gpu_occupants': gpu,
        'output': str(OUTPUT), 'invocation_id': os.environ.get('INVOCATION_ID')}), flush=True)
    os.execv(sys.executable, [sys.executable, str(WORKTREE/'experiments/anchor_graph_mnist.py'),
                             '--output-dir', str(OUTPUT), '--device', 'cuda'])


if __name__ == '__main__':
    main()
