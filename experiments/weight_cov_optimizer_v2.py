#!/usr/bin/env python3
"""Weight-covariance spectral optimizer (v2 — batch-mean streaming).

Tracks a running low-rank estimate of the p×p weight-gradient covariance
using streaming rank-1 SVD updates from batch-mean gradients. Every step:
1. Normal forward/backward → batch-mean gradient g ∈ R^p
2. Rank-1 SVD update with g (trivially cheap)
3. Project g onto top-k eigenspace
4. Pass projected gradient to base optimizer

No per-sample gradients needed. Barely slower than standard Adam.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightCovarianceFilterV2:
    def __init__(self, model, base_optimizer, rank=200, decay=0.99,
                 warmup=100, filter_strength=1.0):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rank = rank
        self.decay = decay
        self.warmup = warmup
        self.filter_strength = filter_strength

        self.param_list = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.param_list)

        self.V = None  # (p, k) top eigenvectors
        self.S = None  # (k,) singular values
        self.step_count = 0

        # Running mean for centering
        self.grad_mean = None

    def _get_flat_grad(self):
        return torch.cat([p.grad.reshape(-1) for p in self.param_list])

    def _set_flat_grad(self, flat_grad):
        offset = 0
        for p in self.param_list:
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].reshape(p.shape)
            offset += numel

    def _update_svd(self, g):
        """Rank-1 update to the streaming covariance SVD.

        g: (p,) batch-mean gradient vector on the compute device.
        SVD math done on CPU to avoid GPU memory pressure for large p.
        """
        device = g.device
        g_cpu = g.detach().cpu().unsqueeze(0)  # (1, p)

        # Update running mean with same decay
        if self.grad_mean is None:
            self.grad_mean = g_cpu.squeeze(0)
        else:
            self.grad_mean = self.decay * self.grad_mean + (1 - self.decay) * g_cpu.squeeze(0)

        g_centered = g_cpu - self.grad_mean.unsqueeze(0)  # (1, p)

        if self.V is None:
            # First update — just store the direction
            norm = g_centered.norm()
            if norm > 1e-12:
                self.V = (g_centered / norm).T.contiguous().to(device)  # (p, 1)
                self.S = norm.unsqueeze(0)  # (1,)
            return

        # Combine old sketch (decayed) with new observation
        # old_rows: (k, p), new_row: (1, p)
        old_rows = (self.S * math.sqrt(self.decay)).unsqueeze(1) * self.V.T.cpu()
        new_scale = math.sqrt(1 - self.decay)
        new_row = g_centered * new_scale

        combined = torch.cat([old_rows, new_row], dim=0)  # (k+1, p)
        del old_rows, new_row, g_centered

        # SVD of (k+1, p) — since k+1 << p, this is cheap: O(k^2 * p)
        _, s, Vt = torch.linalg.svd(combined, full_matrices=False)
        k = min(self.rank, len(s))
        self.V = Vt[:k].T.contiguous().to(device)  # (p, k)
        self.S = s[:k].clone()

    def _project_gradient(self, g):
        if self.V is None:
            return g
        g_projected = self.V @ (self.V.T @ g)
        if self.filter_strength < 1.0:
            return (1 - self.filter_strength) * g + self.filter_strength * g_projected
        return g_projected

    def step(self, inputs, targets):
        """One optimization step. Returns (loss_value, diagnostics_dict)."""
        self.step_count += 1

        # Standard forward/backward
        self.base_optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        loss_val = loss.item()

        flat_grad = self._get_flat_grad()

        # Update covariance estimate every step
        self._update_svd(flat_grad)

        # Apply filter after warmup
        if self.step_count > self.warmup:
            filtered_grad = self._project_gradient(flat_grad)
            self._set_flat_grad(filtered_grad)

        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

        diagnostics = {
            "loss": loss_val,
            "step": self.step_count,
            "filtering_active": self.step_count > self.warmup,
        }
        if self.S is not None:
            diagnostics["top_singular_values"] = self.S[:5].tolist()
            total = (self.S ** 2).sum().item()
            if total > 1e-12:
                diagnostics["variance_in_top5"] = (self.S[:5] ** 2).sum().item() / total
            diagnostics["effective_rank"] = self._effective_rank()

        return loss_val, diagnostics

    def _effective_rank(self):
        if self.S is None:
            return 0.0
        s2 = self.S ** 2
        s2 = s2[s2 > 1e-12]
        if len(s2) == 0:
            return 0.0
        p = s2 / s2.sum()
        entropy = -(p * p.log()).sum().item()
        return math.exp(entropy)

    def reset(self):
        self.V = None
        self.S = None
        self.step_count = 0
        self.grad_mean = None

    def zero_grad(self):
        self.base_optimizer.zero_grad()
