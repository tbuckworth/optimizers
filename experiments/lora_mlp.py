#!/usr/bin/env python3
"""Random-init MLP + LoRA adapters (Version B baseline).

The base MLP weights are randomly initialised and FROZEN; only the low-rank
adapters (A, B per layer) are trained. This is "train a LoRA adapter on a
randomly initialised network" — the baseline a reviewer suggested as a possible
equivalent to our method.

Architecture matches FlexMNISTNet: input -> 256 -> 128 -> 10, ReLU.
Each Linear is replaced by  y = (W0 x + b0) + (alpha/r) * B (A x),
with W0, b0 frozen, A ~ N(0, 1/in), B = 0 (so the adapter starts as a no-op
and the initial function is the frozen random net).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=32, alpha=32.0, bias=True):
        super().__init__()
        W = torch.empty(out_f, in_f)
        nn.init.kaiming_uniform_(W, a=5 ** 0.5)
        self.weight = nn.Parameter(W, requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_f), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        r_eff = min(r, in_f, out_f)
        self.A = nn.Parameter(torch.randn(r_eff, in_f) / (in_f ** 0.5))
        self.B = nn.Parameter(torch.zeros(out_f, r_eff))
        self.scale = alpha / r_eff

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.A), self.B) * self.scale
        return base + lora


class LoRAMLP(nn.Module):
    def __init__(self, input_dim=784, r=32, alpha=32.0):
        super().__init__()
        self.fc1 = LoRALinear(input_dim, 256, r, alpha)
        self.fc2 = LoRALinear(256, 128, r, alpha)
        self.fc3 = LoRALinear(128, 10, r, alpha)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
