#!/usr/bin/env python3
"""One frozen CPU example; output JSON, never mutate production source."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from spectral_filter import SpectralGradientFilter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


class Quadratic(torch.nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.theta = torch.nn.Parameter(initial.clone())

    def forward(self):
        return self.theta.square().sum() / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Pass --run to execute the previously frozen protocol.')
    if (HERE / 'result.json').exists():
        raise SystemExit('Refusing to overwrite an existing result.')

    started = now()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dtype = torch.float64
    gradient = torch.tensor([-2., 1.1], dtype=dtype, device='cpu')
    u = torch.tensor([1., 2.], dtype=dtype) / (5. ** .5)
    projector = torch.outer(u, u)
    expected_h = torch.tensor([.04, .08], dtype=dtype)
    lr, epsilon, tolerance = .1, 1e-8, 1e-12
    checks = {}

    def close(name, actual, expected):
        actual = torch.as_tensor(actual, dtype=dtype)
        expected = torch.as_tensor(expected, dtype=dtype)
        error = float((actual - expected).abs().max())
        checks[name] = {'max_abs_error': error, 'threshold': tolerance, 'passed': error <= tolerance}

    close('projector_symmetry', projector, projector.T)
    close('projector_idempotence', projector @ projector, projector)
    close('hand_projection', projector @ gradient, expected_h)
    modes = ['unfiltered_adam', 'projected_gradient_sgd',
             'projected_gradient_adam', 'projected_gradient_adam_then_project_displacement']
    arms = {}
    for mode in modes:
        model = Quadratic(gradient)
        optimizer = (torch.optim.SGD(model.parameters(), lr=lr) if mode == 'projected_gradient_sgd'
                     else torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999),
                                           eps=epsilon, weight_decay=0, foreach=False, fused=False))
        filt = SpectralGradientFilter(model, optimizer, rank=1, warmup=0,
                                     normalize='none', weighting='hard', filter_strength=1.)
        filt.V = u[:, None].clone()
        filt.S = torch.ones(1, dtype=dtype)
        filt.proj_k = None
        before = model.theta.detach().clone()
        loss_before = float(model().detach())
        model().backward()
        raw_gradient = model.theta.grad.detach().clone()
        h = filt._project_gradient(raw_gradient).detach().clone()
        close(mode + ':current_filter_projection', h, expected_h)
        close(mode + ':projection_descent_identity', torch.dot(raw_gradient, h), h.square().sum())
        supplied = raw_gradient if mode == 'unfiltered_adam' else h
        model.theta.grad.copy_(supplied)
        optimizer.step()
        before_reprojection = model.theta.detach().clone() - before
        if mode == 'projected_gradient_adam_then_project_displacement':
            with torch.no_grad():
                model.theta.copy_(before + projector @ before_reprojection)
        displacement = model.theta.detach().clone() - before
        expected = -lr * supplied if mode == 'projected_gradient_sgd' else -lr * supplied / (supplied.abs() + epsilon)
        if mode == 'projected_gradient_adam_then_project_displacement':
            expected = projector @ expected
        close(mode + ':analytic_displacement', displacement, expected)
        directional_change = float(torch.dot(raw_gradient, displacement))
        loss_after = float(model().detach())
        loss_change = loss_after - loss_before
        close(mode + ':quadratic_identity', loss_change,
              directional_change + float(displacement.square().sum()) / 2)
        arms[mode] = {
            'raw_gradient': raw_gradient.tolist(), 'filtered_gradient': h.tolist(),
            'optimizer_input': supplied.tolist(), 'parameters_before': before.tolist(),
            'parameters_after': model.theta.detach().tolist(), 'displacement': displacement.tolist(),
            'optimizer_displacement_before_optional_reprojection': before_reprojection.tolist(),
            'analytic_displacement': expected.tolist(), 'raw_gradient_dot_displacement': directional_change,
            'displacement_norm': float(displacement.norm()),
            'outside_subspace_displacement_norm': float((displacement - projector @ displacement).norm()),
            'loss_before': loss_before, 'loss_after': loss_after, 'loss_change': loss_change,
        }

    sgd, adam = arms['projected_gradient_sgd'], arms['projected_gradient_adam']
    observed_candidate_success = (sgd['raw_gradient_dot_displacement'] < 0 and sgd['loss_change'] < 0
                                  and adam['raw_gradient_dot_displacement'] > 0 and adam['loss_change'] > 0)
    all_valid = all(check['passed'] for check in checks.values())
    filter_path = Path(inspect.getfile(SpectralGradientFilter)).resolve()
    result = {
        'question': 'Can fixed hard projection before first Adam step turn a descent projected gradient into ascent on original quadratic?',
        'status': 'valid' if all_valid else 'invalid_arithmetic',
        'candidate_success': observed_candidate_success,
        'execution': {'started_utc': started, 'ended_utc': now(), 'device': 'cpu', 'dtype': str(dtype),
                      'python': platform.python_version(), 'torch': torch.__version__, 'git_head': git('rev-parse', 'HEAD'),
                      'git_status': git('status', '--short'), 'torch_threads': torch.get_num_threads()},
        'hashes': {name: sha(HERE / name) for name in ['protocol.md', 'theory.md', 'run_diagnostic.py']},
        'filter_source': {'path': str(filter_path), 'sha256': sha(filter_path)},
        'frozen_inputs': {'gradient': gradient.tolist(), 'u': u.tolist(), 'P': projector.tolist(),
                          'learning_rate': lr, 'adam_betas': [.9, .999], 'adam_epsilon': epsilon,
                          'weight_decay': 0, 'covariance_updates': 0, 'optimizer_steps_per_arm': 1},
        'arms': arms, 'validation': checks,
        'scope': 'Fixed initialized basis; no learned temporal trajectory, neural-model frequency, or general convergence claim.'
    }
    print(json.dumps(result, indent=2))
    return 0 if all_valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
