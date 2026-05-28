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
