import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]
                update = (beta1 * m + (1 - beta1) * grad).sign_()
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                m.lerp_(grad, 1 - beta2)

        return loss


class Muon(Optimizer):
    """Vector-mode Muon: normalizes momentum for the update direction."""
    def __init__(self, params, lr=0.02, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)
                norm = buf.norm()
                if norm > 1e-12:
                    p.add_(buf / norm, alpha=-lr)

        return loss


def _zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G (2D) via the quintic Newton-Schulz iteration (Keller Jordan).

    Approximates the closest semi-orthogonal matrix U V^T from G = U S V^T, i.e.
    replaces the singular values with 1 — Muon's defining operation.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonNS(Optimizer):
    """True Muon: Newton-Schulz orthogonalization of the per-matrix momentum.

    For 2D weights the update is the orthogonalized (U V^T) momentum, scaled by
    sqrt(max(1, rows/cols)) as in Keller Jordan's reference. 1D params (biases)
    have no matrix structure to orthogonalize, so they fall back to SGD+momentum.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                if p.ndim == 2:
                    upd = _zeropower_via_newtonschulz5(buf, steps)
                    scale = (max(1.0, p.shape[0] / p.shape[1])) ** 0.5
                    p.add_(upd, alpha=-lr * scale)
                else:
                    # bias / 1D: no orthogonalization possible -> SGD+momentum
                    p.add_(buf, alpha=-lr)
        return loss
