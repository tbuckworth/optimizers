#!/usr/bin/env python3
"""Independent, CPU-only saved-tensor corroboration; never executes a model."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '1', key
import torch

REPO = Path(__file__).resolve().parents[2]
BATCH = Path('/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD')
OUT = BATCH / 'tensor-audit-001.json'
POLICIES = ('native', 'orthogonal', 'norm_matched')
CHECKS = 0
ERRORS = []


def check(ok, message):
    global CHECKS
    CHECKS += 1
    if not ok:
        ERRORS.append(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def receipt(rec):
    path = Path(rec['path'])
    check(sha(path) == rec['sha256'], f'hash {path}')
    if 'size_bytes' in rec:
        check(path.stat().st_size == rec['size_bytes'], f'size {path}')
    return path


def close(a, b, name, rtol=1e-9, atol=1e-12):
    check(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol),
          f'{name}: observed={a}, expected={b}')


def compare(a, b, name, tolerance=1e-9):
    a, b = a.double(), b.double()
    relative = float((a-b).norm() / max(float(b.norm()), 1e-30))
    check(relative < tolerance, f'{name}: relative norm error={relative}, bound={tolerance}')
    return relative


def cosine(a, b):
    a, b = a.double(), b.double()
    return float(a.dot(b) / (a.norm()*b.norm()))


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            digest.update(b'tensor\0' + str(data.dtype).encode() + b'\0')
            digest.update(json.dumps(list(data.shape)).encode() + b'\0')
            digest.update(data.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict\0')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
            digest.update(b'enddict\0')
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b'\0')
            for child in item:
                visit(child)
            digest.update(b'endsequence\0')
        else:
            digest.update(type(item).__name__.encode() + b'\0')
            digest.update(json.dumps(item, allow_nan=False).encode() + b'\0')
    visit(value)
    return digest.hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def audit_seed(seed, completion_receipt):
    completion = json.loads(receipt(completion_receipt).read_text())
    check(completion['status'] == 'complete' and completion['seed'] == seed, 'seed completion')
    for name, value in completion['source_sha256'].items():
        check(sha(REPO/name) == value, 'scientific source '+name)
    expected = [[p, s] for p in POLICIES for s in ((1501,) if p == 'native' else (1501, 2000, 2500))]
    check(completion['accepted_roster'] == expected, 'fixed seven-state roster')
    for rec in completion['checkpoints']:
        receipt(rec)
    parent = load(receipt(completion['parent_checkpoint']))
    common = load(receipt(completion['shared_first_action']))
    adam = load(receipt(completion['first_step_adam_diagnostics']))
    summary = json.loads((BATCH/f'seed{seed}'/'first-step-summary.json').read_text())
    native = load(BATCH/f'seed{seed}'/'native-step-001501.pt')['native_compatible_state']
    check(tree_hash(native['filter_state']) == common['post_estimator_sha256'], 'post-estimator identity')
    state_keys = ('model_state', 'optimizer_state', 'filter_state', 'filter_configuration',
                  'torch_cpu_rng_state', 'torch_cuda_rng_states', 'device_type', 'cuda_device_count',
                  'config', 'config_sha256', 'split_identity', 'parameter_identity')
    parent_hash = tree_hash({key: parent[key] for key in state_keys})
    check(parent_hash == common['restored_scientific_state_sha256'] == completion['fork_scientific_state_sha256'],
          'full restored-state identity')
    g = common['raw_gradient'].double()
    result = common['actions']
    Q = result['Q']
    V = native['filter_state']['V'].double()
    s = result['singular_values']
    tol = max(V.shape)*torch.finfo(torch.float64).eps*float(s[0])
    close(result['tolerance'], tol, 'rank tolerance')
    rank = int((s > tol).sum())
    check(rank == result['numerical_rank'] == Q.shape[1], 'rank count')
    check(result['full_column_rank'] == (rank == V.shape[1]), 'full rank flag')
    gram_error = float((Q.T@Q - torch.eye(rank, dtype=torch.float64)).abs().max())
    check(gram_error < 1e-10, 'Q orthogonality')
    compressed = Q.T@V
    inferred_s = torch.linalg.svdvals(compressed)
    singular_error = compare(inferred_s, s[:rank], 'singular values from Q.T V', 1e-9)
    span_error = compare(Q@compressed, V, 'V reconstruction', 1e-9)
    del compressed
    actions = result['actions']
    native_error = compare(actions['native'], V@(V.T@g), 'native versus fp64 V(V.T g)', 2e-5)
    q = Q@(Q.T@g)
    orth_error = compare(actions['orthogonal'], q.float(), 'delivered orthogonal action', 1e-8)
    scale = float(actions['native'].double().norm()/actions['orthogonal'].double().norm())
    compare(actions['norm_matched'], (actions['orthogonal'].double()*scale).float(), 'norm-match action', 1e-8)
    close(result['diagnostics']['norm_match_scale'], scale, 'norm-match scale')
    action_norms = {p: float(actions[p].double().norm()) for p in POLICIES}
    action_cosines = {}
    for i, p in enumerate(POLICIES):
        close(result['diagnostics']['action_norms'][p], action_norms[p], 'action norm '+p)
        compare(result['coefficients'][p], Q.T@actions[p].double(), 'Q action coefficients '+p, 1e-9)
        for other in POLICIES[i+1:]:
            key = p+'_'+other
            action_cosines[key] = cosine(actions[p], actions[other])
            close(result['diagnostics']['pairwise_cosines'][key], action_cosines[key], 'action cosine '+key)
    close(action_norms['native'], action_norms['norm_matched'], 'post-cast norm matching', 10*torch.finfo(torch.float32).eps)
    check(result['diagnostics']['norm_match_exact'] and not result['diagnostics']['norm_match_degenerate'], 'nondegenerate match')
    names = [row['name'] for row in parent['parameter_identity']]
    old_parameters = torch.cat([parent['model_state'][name].flatten().double() for name in names])
    groups = parent['optimizer_state']['param_groups']
    check(len(groups) == 1, 'single Adam group')
    group = groups[0]
    ids = group['params']
    check(len(ids) == len(names), 'optimizer/model order length')
    old_m = torch.cat([parent['optimizer_state']['state'][i]['exp_avg'].flatten().double() for i in ids])
    old_v = torch.cat([parent['optimizer_state']['state'][i]['exp_avg_sq'].flatten().double() for i in ids])
    check(all(int(parent['optimizer_state']['state'][i]['step']) == 1500 for i in ids), 'parent Adam steps')
    beta1, beta2 = group['betas']; lr = group['lr']; decay = group['weight_decay']; eps = group['eps']
    adam_checks = {}
    del V, Q, native
    for p in POLICIES:
        checkpoint = load(BATCH/f'seed{seed}'/f'{p}-step-001501.pt')['native_compatible_state']
        check(tree_hash(checkpoint['filter_state']) == common['post_estimator_sha256'], 'common estimator '+p)
        check(adam[p]['restored_scientific_state_sha256'] == parent_hash, 'common restore '+p)
        observed = adam[p]
        delivered = actions[p].double()
        m_error = compare(observed['m'], beta1*old_m+(1-beta1)*delivered, 'm recurrence '+p, 5e-7)
        v_error = compare(observed['v'], beta2*old_v+(1-beta2)*delivered.square(), 'v recurrence '+p, 5e-7)
        states = checkpoint['optimizer_state']['state']
        check(all(int(states[i]['step']) == 1501 for i in ids), 'new Adam steps '+p)
        for key, stored_key in (('m','exp_avg'),('v','exp_avg_sq')):
            flat = torch.cat([states[i][stored_key].flatten().double() for i in ids])
            check(torch.equal(observed[key], flat), 'checkpoint moment identity '+p+key)
        direction = (observed['m']/(1-beta1**1501))/((observed['v']/(1-beta2**1501)).sqrt()+eps)
        compare(observed['adaptive_direction'], direction, 'bias-corrected Adam direction '+p, 1e-12)
        compare(observed['adaptive_movement'], -lr*direction, 'adaptive movement '+p, 1e-12)
        compare(observed['decay_movement'], -lr*decay*old_parameters, 'decay movement '+p, 1e-12)
        new_parameters = torch.cat([checkpoint['model_state'][name].flatten().double() for name in names])
        check(torch.equal(observed['actual_displacement'], new_parameters-old_parameters), 'actual displacement '+p)
        residual = observed['actual_displacement']-observed['adaptive_movement']-observed['decay_movement']
        for name in ('m','v','adaptive_direction','adaptive_movement','decay_movement','actual_displacement'):
            close(observed['summary'][name+'_norm'], observed[name].norm(), 'stored norm '+p+name)
        close(observed['summary']['decomposition_residual_norm'], residual.norm(), 'residual norm '+p)
        close(observed['summary']['decomposition_residual_max_abs'], residual.abs().max(), 'residual max '+p)
        check(summary['policies'][p] == observed['summary'], 'JSON/PT summary identity '+p)
        for other in POLICIES:
            for name in ('adaptive_direction','actual_displacement'):
                close(observed['summary']['pairwise'][other][name+'_cosine'], cosine(observed[name], adam[other][name]), 'Adam cosine '+p+other+name)
        adam_checks[p] = {'m_recurrence_relative_error':m_error, 'v_recurrence_relative_error':v_error, **observed['summary']}
        del checkpoint
    return {'seed':seed, 'completion_receipt':completion_receipt,
            'shared_receipt':completion['shared_first_action'], 'adam_receipt':completion['first_step_adam_diagnostics'],
            'parent_receipt':completion['parent_checkpoint'], 'source_sha256':completion['source_sha256'],
            'P':result['P'], 'k':result['k'], 'rank':rank, 'singular_min':float(s[-1]), 'singular_max':float(s[0]),
            'rank_tolerance':tol, 'Q_orthogonality_max_abs':gram_error,
            'singular_relative_error':singular_error, 'span_reconstruction_relative_error':span_error,
            'native_fp64_relative_error':native_error, 'orthogonal_relative_error':orth_error,
            'action_norms':action_norms, 'action_cosines':action_cosines, 'norm_match_scale':scale,
            'adam':adam_checks}


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'exclusive audit output already exists'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cgroup = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    assert int((cgroup/'memory.max').read_text()) == 4*1024**3
    assert int((cgroup/'memory.swap.max').read_text()) == 0
    quota, period = map(int, (cgroup/'cpu.max').read_text().split()); assert quota == period
    batch = json.loads((BATCH/'batch-complete.json').read_text())
    assert batch['status'] == 'complete' and batch['seeds'] == list(range(100,105))
    rows = []
    for seed, rec in zip(range(100,105),batch['accepted'], strict=True):
        assert rec['seed'] == seed
        rows.append(audit_seed(seed, rec)); gc.collect()
        print(json.dumps({'audited_seed':seed,'checks':CHECKS,'errors':ERRORS}), flush=True)
        assert time.monotonic()-started < 580
    result = {'status':'PASS' if not ERRORS else 'FAIL', 'scope':'independent saved-tensor arithmetic; no training or inference replication',
              'script_sha256':sha(__file__), 'batch_completion_sha256':sha(BATCH/'batch-complete.json'),
              'checks':CHECKS, 'errors':ERRORS, 'rows':rows, 'elapsed_seconds':time.monotonic()-started,
              'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    with OUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)
    assert not ERRORS


if __name__ == '__main__':
    main()
