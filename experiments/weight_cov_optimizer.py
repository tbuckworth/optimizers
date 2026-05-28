#!/usr/bin/env python3
"""Weight-covariance spectral optimizer.

Tracks a running low-rank estimate of the p×p weight-gradient covariance
matrix using incremental SVD. Projects gradients onto the top-k eigenspace
(directions where weights historically co-vary across samples), suppressing
isolated/memorization-specific weight directions.

The key insight: feature-learning creates large co-varying weight clusters
(dominant eigenvalues), while memorization creates many small independent
groups (flat eigenvalues). Projecting onto the top eigenspace keeps
feature-learning signals and filters memorization signals.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightCovarianceFilter:
    """Wraps a base optimizer with weight-covariance spectral filtering.

    Per step:
    1. Compute per-sample gradients every `update_every` steps
    2. Update a running rank-k SVD of the weight-gradient covariance
    3. Project the mean gradient onto the top-k eigenspace
    4. Pass projected gradient to the base optimizer

    On non-update steps, uses cached projection from the last SVD.
    Uses a simple per-sample loop (no vmap) to avoid JIT compilation overhead.
    """

    def __init__(self, model, base_optimizer, loss_fn=None,
                 rank=20, decay=0.95, warmup=10, update_every=1,
                 filter_strength=1.0):
        self.model = model
        self.base_optimizer = base_optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(reduction='none')
        self.rank = rank
        self.decay = decay
        self.warmup = warmup
        self.update_every = update_every
        self.filter_strength = filter_strength

        self.param_list = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.param_list)

        self.V = None
        self.S = None
        self.step_count = 0
        self.update_count = 0

    def _compute_per_sample_grads(self, inputs, targets):
        """Compute per-sample gradients via a loop. Returns (B, p) tensor."""
        B = inputs.size(0)
        grads = []

        for i in range(B):
            self.model.zero_grad()
            out = self.model(inputs[i:i+1])
            loss = F.cross_entropy(out, targets[i:i+1])
            loss.backward()
            flat = torch.cat([p.grad.reshape(-1) for p in self.param_list])
            grads.append(flat.detach())

        return torch.stack(grads), F.cross_entropy(
            self.model(inputs), targets).item()

    def _update_svd(self, G):
        """Update running low-rank SVD with new per-sample gradients.

        G: (B, p) per-sample gradient matrix.

        Combines old sketch (decayed) with new data, recomputes top-k SVD.
        SVD is done on CPU to avoid GPU memory pressure.
        """
        B, p = G.shape
        G_cpu = G.detach().cpu()
        G_centered = G_cpu - G_cpu.mean(dim=0, keepdim=True)
        del G_cpu

        device = G.device

        if self.V is None:
            _, s, Vt = torch.linalg.svd(G_centered, full_matrices=False)
            k = min(self.rank, len(s))
            self.V = Vt[:k].T.contiguous().to(device)
            self.S = s[:k].clone()
            return

        old_rows = (self.S * math.sqrt(self.decay)).unsqueeze(1) * self.V.T.cpu()
        new_scale = math.sqrt((1 - self.decay) / B)
        new_rows = G_centered * new_scale

        combined = torch.cat([old_rows, new_rows], dim=0)
        del old_rows, new_rows, G_centered

        _, s, Vt = torch.linalg.svd(combined, full_matrices=False)
        k = min(self.rank, len(s))
        self.V = Vt[:k].T.contiguous().to(device)
        self.S = s[:k].clone()

    def _project_gradient(self, g):
        """Project gradient onto top-k eigenspace of the running covariance."""
        if self.V is None:
            return g

        g_projected = self.V @ (self.V.T @ g)

        if self.filter_strength < 1.0:
            return (1 - self.filter_strength) * g + self.filter_strength * g_projected
        return g_projected

    def _set_grads(self, flat_grad):
        offset = 0
        for p in self.param_list:
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].reshape(p.shape)
            offset += numel

    def step(self, inputs, targets):
        """One optimization step. Returns (loss, diagnostics_dict)."""
        self.step_count += 1
        do_update = (self.step_count % self.update_every == 0)

        if do_update:
            G, mean_loss = self._compute_per_sample_grads(inputs, targets)
            mean_grad = G.mean(dim=0)

            self._update_svd(G)
            del G
            self.update_count += 1

            if self.update_count > self.warmup:
                filtered_grad = self._project_gradient(mean_grad)
            else:
                filtered_grad = mean_grad

            self._set_grads(filtered_grad)
        else:
            self.base_optimizer.zero_grad()
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            mean_loss = loss.item()

            if self.update_count > self.warmup and self.V is not None:
                flat_grad = torch.cat([p.grad.reshape(-1) for p in self.param_list])
                filtered_grad = self._project_gradient(flat_grad)
                self._set_grads(filtered_grad)

        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

        diagnostics = {
            "loss": mean_loss,
            "step": self.step_count,
            "update_count": self.update_count,
            "filtering_active": self.update_count > self.warmup,
        }
        if self.S is not None:
            diagnostics["top_singular_values"] = self.S[:5].tolist()
            total = (self.S ** 2).sum().item()
            if total > 1e-12:
                diagnostics["variance_in_top5"] = (self.S[:5] ** 2).sum().item() / total
            diagnostics["effective_rank"] = self._effective_rank()

        return mean_loss, diagnostics

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
        """Reset running state (for reuse across HP sweep configs)."""
        self.V = None
        self.S = None
        self.step_count = 0
        self.update_count = 0

    def zero_grad(self):
        self.base_optimizer.zero_grad()
