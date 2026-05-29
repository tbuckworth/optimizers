#!/usr/bin/env python3
"""Random fixed-subspace gradient filter (Version A control).

This is the apples-to-apples control for WeightCovarianceFilterV2: it uses the
*exact same* projection (g -> P P^T g over the flattened gradient), but P is a
fixed random orthonormal basis chosen once at init, rather than the learned,
time-varying eigenbasis of the gradient covariance.

If our learned filter beats this at matched rank k, the adaptivity is doing the
work. If they tie, the streaming SVD is replaceable by torch.randn(p, k).

This is the Li et al. 2018 ("Measuring the Intrinsic Dimension of Objective
Landscapes") setup expressed as a gradient projection: the parameters stay at
theta0 and only ever move within theta0 + span(P).
"""

import torch
import torch.nn.functional as F


class RandomSubspaceFilter:
    def __init__(self, model, base_optimizer, rank=200, seed=0):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rank = rank
        self.seed = seed
        self.param_list = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.param_list)
        self.P = None  # (p, k) orthonormal, lazily built on the compute device
        self.step_count = 0

    def _init_P(self, device):
        # Draw on CPU for reproducibility, orthonormalize, move to device.
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        k = min(self.rank, self.n_params)
        M = torch.randn(self.n_params, k, generator=g)
        Q, _ = torch.linalg.qr(M)  # (p, k) orthonormal columns
        self.P = Q.to(device)

    def _get_flat_grad(self):
        return torch.cat([p.grad.reshape(-1) for p in self.param_list])

    def _set_flat_grad(self, flat_grad):
        offset = 0
        for p in self.param_list:
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].reshape(p.shape)
            offset += numel

    def step(self, inputs, targets):
        self.step_count += 1
        self.base_optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        loss_val = loss.item()

        g = self._get_flat_grad()
        if self.P is None:
            self._init_P(g.device)
        g_proj = self.P @ (self.P.T @ g)
        self._set_flat_grad(g_proj)

        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        return loss_val, {"step": self.step_count}

    def zero_grad(self):
        self.base_optimizer.zero_grad()
