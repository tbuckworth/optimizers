import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from matrix_spectral_filter import PerMatrixSpectralGradientFilter
from spectral_filter import SpectralGradientFilter


class BlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(5, 4)
        self.norm = nn.LayerNorm(4)
        self.output = nn.Linear(4, 2, bias=False)

    def forward(self, inputs):
        return self.output(self.norm(torch.tanh(self.input(inputs))))


class AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 5), requires_grad=False)
        self.A = nn.Parameter(torch.randn(2, 5) * 0.1)
        self.B = nn.Parameter(torch.zeros(6, 2))

    def forward(self, inputs):
        return F.linear(inputs, self.weight) + F.linear(F.linear(inputs, self.A), self.B)


def train_step(model, optimizer, filt, inputs, targets):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    diagnostics = filt.filter_grad()
    optimizer.step()
    return diagnostics


class PerMatrixSpectralFilterTest(unittest.TestCase):
    def test_joint_bias_blocks_and_unfiltered_vectors(self):
        model = BlockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        filt = PerMatrixSpectralGradientFilter(
            model, optimizer, rank=3, bias_mode="joint"
        )
        self.assertEqual(tuple(filt.blocks), ("input.weight", "output.weight"))
        self.assertEqual(
            filt.block_parameter_names["input.weight"],
            ("input.weight", "input.bias"),
        )
        self.assertEqual(
            filt.unfiltered_parameter_names,
            ("norm.weight", "norm.bias"),
        )
        self.assertEqual(filt.n_filtered_params, 32)
        self.assertEqual(filt.max_basis_numel, 96)

    def test_separate_bias_mode_filters_all_vectors(self):
        model = BlockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        filt = PerMatrixSpectralGradientFilter(
            model, optimizer, rank=2, bias_mode="separate"
        )
        self.assertEqual(
            tuple(filt.blocks),
            (
                "input.weight",
                "output.weight",
                "input.bias",
                "norm.weight",
                "norm.bias",
            ),
        )
        self.assertEqual(filt.unfiltered_parameter_names, ())

    def test_frozen_base_and_optimizer_subset_support_lora(self):
        torch.manual_seed(1)
        model = AdapterModel()
        optimizer = torch.optim.Adam([model.A, model.B], lr=1e-3)
        matrix_filter = PerMatrixSpectralGradientFilter(
            model, optimizer, rank=2, warmup=0, report_per_block=True
        )
        self.assertEqual(tuple(matrix_filter.blocks), ("A", "B"))
        self.assertEqual(matrix_filter.n_filtered_params, model.A.numel() + model.B.numel())

        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 6, (8,))
        diagnostics = train_step(
            model, optimizer, matrix_filter, inputs, targets
        )
        self.assertEqual(diagnostics["active_block_count"], 2)
        self.assertEqual(diagnostics["block_count"], 2)

        # The global implementation also filters the optimizer subset rather
        # than trying to include frozen base parameters with missing gradients.
        global_model = AdapterModel()
        global_optimizer = torch.optim.Adam(
            [global_model.A, global_model.B], lr=1e-3
        )
        global_filter = SpectralGradientFilter(
            global_model, global_optimizer, rank=2, warmup=0
        )
        self.assertEqual(
            global_filter.n_params,
            global_model.A.numel() + global_model.B.numel(),
        )
        train_step(
            global_model, global_optimizer, global_filter, inputs, targets
        )

    def test_blocks_are_independent_stable_projections(self):
        torch.manual_seed(2)
        model = BlockModel().double()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        filt = PerMatrixSpectralGradientFilter(
            model,
            optimizer,
            rank=3,
            decay=0.9,
            warmup=0,
            bias_mode="exclude",
        )
        for _ in range(30):
            model.input.weight.grad = torch.randn_like(model.input.weight)
            model.output.weight.grad = torch.randn_like(model.output.weight)
            filt.filter_grad()
        for block in filt.blocks.values():
            self.assertLess(block.orthogonality_error(), 1e-9)
            gradient = torch.randn(block.n_params, dtype=torch.float64)
            projected = block._project_gradient(gradient)
            self.assertLessEqual(
                projected.norm(), gradient.norm() * (1 + 1e-10)
            )
        self.assertIsNot(
            filt.blocks["input.weight"].V,
            filt.blocks["output.weight"].V,
        )

    def test_inactive_matrix_block_is_skipped_without_creating_gradients(self):
        model = BlockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        filt = PerMatrixSpectralGradientFilter(
            model, optimizer, rank=2, warmup=0, report_per_block=True
        )
        optimizer.zero_grad()
        model.input(torch.randn(3, 5)).sum().backward()
        diagnostics = filt.filter_grad()
        self.assertEqual(diagnostics["active_block_count"], 1)
        self.assertEqual(diagnostics["skipped_blocks"], ["output.weight"])
        self.assertIsNone(model.output.weight.grad)

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(3)
        model = BlockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        filt = PerMatrixSpectralGradientFilter(
            model, optimizer, rank=2, warmup=0, bias_mode="joint"
        )
        inputs = torch.randn(7, 5)
        targets = torch.randint(0, 2, (7,))
        for _ in range(5):
            train_step(model, optimizer, filt, inputs, targets)
        state = copy.deepcopy(filt.state_dict())

        restored_model = BlockModel()
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
        restored = PerMatrixSpectralGradientFilter(
            restored_model,
            restored_optimizer,
            rank=2,
            warmup=0,
            bias_mode="joint",
        )
        restored.load_state_dict(state)
        self.assertEqual(restored.step_count, filt.step_count)
        for name in filt.blocks:
            original = filt.blocks[name]
            loaded = restored.blocks[name]
            torch.testing.assert_close(loaded.S, original.S)
            torch.testing.assert_close(loaded.V, original.V)
            torch.testing.assert_close(loaded.grad_mean, original.grad_mean)

    def test_memory_estimate_uses_per_block_rank(self):
        model = BlockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        filt = PerMatrixSpectralGradientFilter(
            model, optimizer, rank=2, bias_mode="joint"
        )
        estimate = filt.memory_estimate()
        self.assertEqual(estimate["filtered_parameters"], 32)
        self.assertEqual(estimate["blocks"], 2)
        self.assertEqual(estimate["max_basis_elements"], 64)
        self.assertEqual(estimate["max_basis_bytes"], 256)


if __name__ == "__main__":
    unittest.main()
