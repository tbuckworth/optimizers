#!/usr/bin/env python3
"""Reproduce the investigation's bounded mathematical counterexamples on CPU.

From the repository root:
  python3 output/2026-09-06-spectral-optimizer-investigation/analysis/analytical_checks.py

No datasets, training runs, network calls or GPU operations are used. The script
prints JSON; it does not overwrite the recorded evidence. It uses the current
canonical implementation, so code changes may intentionally invalidate results.
"""

import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from spectral_filter import SpectralGradientFilter  # noqa: E402
from experiments.spectral_optimizer import SpectralConsensusFilter  # noqa: E402


def make(p=2, **kwargs):
    model = torch.nn.Linear(p, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return SpectralGradientFilter(model, optimizer, **kwargs)


def run_checks():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    out = {}

    filt = make(rank=1, decay=.99, warmup=0)
    for xi in [1., -1., 2., -2.]:
        gradient = torch.tensor([1., xi])
        filt.step_count += 1
        filt._update_svd(gradient)
    filtered = filt._project_gradient(gradient)
    assert torch.allclose(filtered, torch.tensor([0., -2.]))
    out['stationary_mean_counterexample'] = {
        'g': gradient.tolist(), 'filtered': filtered.tolist(), 'basis': filt.V.tolist()
    }

    filt = make(p=3, rank=2, weighting='soft', alpha=100, soft_residual=True)
    filt.V = torch.eye(3)[:, :2]
    filt.S = torch.tensor([2., 1.])
    gradient = torch.ones(3)
    filtered = filt._project_gradient(gradient)
    assert filtered[2] > 1.2
    out['soft_residual_alpha100'] = {
        'g': gradient.tolist(), 'filtered': filtered.tolist(),
        'complement_nonzero': bool(filtered[2] != 0)
    }

    direction = torch.tensor([1., 2.])
    direction /= direction.norm()
    projector = torch.outer(direction, direction)
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=.1)
    parameter.grad = direction.clone()
    optimizer.step()
    update = parameter.detach().clone()
    leakage = (update - projector @ update).norm() / update.norm()
    assert leakage > .31
    out['adam_projection_leakage'] = {
        'gradient_residual': float((direction - projector @ direction).norm()),
        'update': update.tolist(), 'update_residual_fraction': float(leakage)
    }

    filt = make(rank=1, decay=.99, stabilize_every=None)
    filt._update_svd(torch.zeros(2))
    gradient = torch.tensor([1., 0.])
    filt._update_svd(gradient)
    centered = gradient - filt.grad_mean
    regular = .01 * centered.square().sum()
    ratio = filt.S[0] ** 2 / regular
    assert abs(float(ratio) - 100.) < 1e-10
    out['startup_covariance_inflation'] = {
        'represented_eigenvalue': float(filt.S[0] ** 2),
        'regular_ema_eigenvalue': float(regular), 'ratio': float(ratio)
    }

    filt = make(rank=1, decay=.5, stabilize_every=None)
    filt.grad_mean = torch.zeros(2)
    filt.V = torch.tensor([[1.], [0.]])
    filt.S = torch.ones(1)
    gradient = torch.tensor([0., 10.])
    previous = filt._project_gradient(gradient)
    filt._update_svd(gradient)
    filtered = filt._project_gradient(gradient)
    assert previous.norm() == 0 and torch.allclose(filtered, gradient)
    out['self_inclusion_outlier'] = {
        'lagged_projection': previous.tolist(),
        'current_projection': filtered.tolist(), 'new_basis': filt.V.tolist()
    }

    filt = make(rank=1, decay=.99, stabilize_every=None)
    filt.grad_mean = torch.zeros(2)
    filt.V = torch.tensor([[1.], [0.]])
    filt.S = torch.ones(1)
    covariance = torch.diag(torch.tensor([1., 0.]))
    centered = torch.tensor([0., 1.])
    for _ in range(100):
        gradient = filt.grad_mean + centered / .99
        filt._update_svd(gradient)
        covariance = .99 * covariance + .01 * torch.outer(centered, centered)
    assert covariance[1, 1] > covariance[0, 0] and abs(filt.V[0, 0]) > .999
    out['repeated_truncation_hides_emerging_direction'] = {
        'steps': 100, 'full_covariance_diagonal': covariance.diag().tolist(),
        'sketch_basis': filt.V.tolist(), 'sketch_eigenvalue': float(filt.S[0] ** 2)
    }

    filt = make(rank=2, decay=.9, stabilize_every=None, relative_eig_tol=0)
    gradients = [torch.tensor(v) for v in [[0., 0.], [1., 0.], [0., 2.], [2., -1.]]]
    mean = gradients[0].clone()
    central_covariance = torch.zeros(2, 2)
    first_centered = None
    for step, gradient in enumerate(gradients):
        if step:
            delta = gradient - mean
            mean = .9 * mean + .1 * gradient
            central_covariance = .9 * central_covariance + .09 * torch.outer(delta, delta)
        filt._update_svd(gradient)
        if step == 1:
            first_centered = gradient - filt.grad_mean
    represented = filt.V @ torch.diag(filt.S.square()) @ filt.V.T
    expected = .9 * central_covariance + .9 ** 3 * torch.outer(first_centered, first_centered)
    error = (represented - expected).norm()
    assert error < 1e-12
    out['weighted_central_covariance_identity'] = {'frobenius_error': float(error)}

    # The historical hard B x B mode forces k>=1. For a single example it is
    # exactly the ordinary example gradient even if the threshold exceeds 1.
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    old = SpectralConsensusFilter(model, optimizer, mp_factor=2., soft=False)
    raw = torch.tensor([[1., -2., 3.]])
    filtered, diagnostics = old._spectral_filter(raw, 1)
    assert torch.allclose(filtered, raw[0])
    out['legacy_batch_one_hard_identity'] = {
        'input': raw[0].tolist(), 'output': filtered.tolist(), 'kept_rank': diagnostics['k']
    }
    return out


if __name__ == '__main__':
    print(json.dumps({
        'date': '2026-09-06', 'device': 'CPU', 'torch': torch.__version__,
        'dtype': 'float64',
        'repository_commit': '95c48ed208a144c5cf9b857dd6b6cd5caa96208e',
        'tests': run_checks(),
    }, indent=2))
