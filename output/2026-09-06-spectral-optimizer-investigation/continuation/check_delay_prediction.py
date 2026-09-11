#!/usr/bin/env python3
"""Verify a prospectively derived rank-truncation delay against current code."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from spectral_filter import SpectralGradientFilter


def make(rank, beta, initial):
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    base = torch.optim.SGD(model.parameters(), lr=0)
    filt = SpectralGradientFilter(
        model, base, rank=rank, decay=beta, warmup=0,
        stable_update=True, stabilize_every=None, relative_eig_tol=0)
    filt.grad_mean = torch.zeros(2, dtype=torch.float64)
    filt.V = torch.tensor([[1.], [0.]], dtype=torch.float64)
    filt.S = torch.tensor([math.sqrt(initial)], dtype=torch.float64)
    return filt


def first_strict(beta, threshold):
    return max(1, math.floor(math.log(threshold) / math.log(beta)) + 1)


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    started = time.monotonic()
    rows = []
    for beta in (.9, .99, .999):
        for initial in (1., 1. / (1. - beta)):
            exact_prediction = first_strict(beta, 1. / (initial + 1.))
            narrow_prediction = first_strict(beta, (1. - beta) / initial)
            observed = {}
            for rank, expected in ((1, narrow_prediction), (2, exact_prediction)):
                filt = make(rank, beta, initial)
                for step in range(1, expected + 3):
                    innovation = torch.tensor([0., 1.])
                    gradient = filt.grad_mean + innovation / beta
                    filt._update_svd(gradient)
                    overlap = float(filt.V[1, 0].square())
                    if overlap > .999:
                        observed[rank] = step
                        break
                assert observed.get(rank) == expected, (beta, initial, rank, expected, observed)
            rows.append({
                'decay': beta, 'initial_eigenvalue': initial,
                'exact_predicted': exact_prediction, 'wide_observed': observed[2],
                'rank_one_predicted': narrow_prediction, 'rank_one_observed': observed[1],
                'extra_observations': observed[1] - observed[2],
            })
    result = {
        'protocol_sha256': hashlib.sha256((HERE / 'truncation-delay-theory.md').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'filter_sha256': hashlib.sha256((ROOT / 'spectral_filter.py').read_bytes()).hexdigest(),
        'torch': torch.__version__, 'dtype': 'float64', 'device': 'cpu',
        'results': rows, 'elapsed_seconds': time.monotonic() - started,
        'scope': 'Controlled centered innovations; no model training or generalization claim.',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
