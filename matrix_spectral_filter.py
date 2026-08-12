#!/usr/bin/env python3
"""Block-diagonal spectral filtering, one covariance basis per weight matrix.

This is the scalable counterpart to the global p x p SpectralGradientFilter.
It never concatenates gradients across the model. Each trainable matrix (or
higher-order weight tensor) receives an independent small temporal covariance
basis. Matching affine biases can share their matrix's block; unmatched vectors
such as LayerNorm parameters remain unfiltered by default.

For LoRA, A and B are naturally separate blocks. With k=4 and P trainable
adapter parameters, the basis cap is approximately 4P values rather than the
200P values used by the original global rank-200 experiments.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from spectral_filter import SpectralGradientFilter


class PerMatrixSpectralGradientFilter:
    """Apply an independent streaming spectral filter to each weight tensor.

    Args mirror :class:`SpectralGradientFilter`, except that ``rank`` is the
    cap *per block*. ``bias_mode`` controls optimized 1D parameters:

    - ``"joint"`` (default): concatenate a module's matching affine bias with
      its ``weight`` block. Other 1D parameters are left unchanged.
    - ``"separate"``: give every optimized 1D parameter its own block.
    - ``"exclude"``: leave all 1D parameters unchanged.

    Only parameters present in ``base_optimizer`` are considered. This makes
    the wrapper directly usable for LoRA/PEFT models with a frozen base.
    """

    def __init__(
        self,
        model,
        base_optimizer,
        rank=8,
        decay=0.99,
        warmup=100,
        filter_strength=1.0,
        energy_threshold=None,
        adaptive="none",
        normalize="none",
        weighting="hard",
        alpha=1.0,
        soft_residual=True,
        bias_mode="joint",
        stable_update=True,
        relative_eig_tol=1e-8,
        absolute_eig_floor=0.0,
        stabilize_every=100,
        report_per_block=False,
    ):
        if rank < 1:
            raise ValueError("rank must be positive")
        if bias_mode not in {"joint", "separate", "exclude"}:
            raise ValueError("bias_mode must be 'joint', 'separate', or 'exclude'")

        self.model = model
        self.base_optimizer = base_optimizer
        self.rank = int(rank)
        self.bias_mode = bias_mode
        self.report_per_block = bool(report_per_block)
        self.step_count = 0

        named_parameters = OrderedDict(model.named_parameters())
        optimizer_ids = {
            id(parameter)
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        }
        model_ids = {id(parameter) for parameter in named_parameters.values()}
        unknown = optimizer_ids - model_ids
        if unknown:
            raise ValueError(
                "base_optimizer contains a parameter that is not in model"
            )
        optimized = OrderedDict(
            (name, parameter)
            for name, parameter in named_parameters.items()
            if id(parameter) in optimizer_ids and parameter.requires_grad
        )
        if not optimized:
            raise ValueError("base_optimizer has no trainable model parameters")

        common = dict(
            decay=decay,
            warmup=warmup,
            filter_strength=filter_strength,
            energy_threshold=energy_threshold,
            adaptive=adaptive,
            normalize=normalize,
            weighting=weighting,
            alpha=alpha,
            soft_residual=soft_residual,
            stable_update=stable_update,
            relative_eig_tol=relative_eig_tol,
            absolute_eig_floor=absolute_eig_floor,
            stabilize_every=stabilize_every,
        )

        self.blocks = OrderedDict()
        self.block_parameter_names = OrderedDict()
        assigned = set()
        for name, parameter in optimized.items():
            if parameter.ndim < 2:
                continue
            block_names = [name]
            block_parameters = [parameter]
            if bias_mode == "joint" and (
                name == "weight" or name.endswith(".weight")
            ):
                bias_name = (
                    "bias"
                    if name == "weight"
                    else f"{name[:-len('.weight')]}.bias"
                )
                bias = optimized.get(bias_name)
                if (
                    bias is not None
                    and bias.ndim == 1
                    and parameter.shape[0] == bias.numel()
                ):
                    block_names.append(bias_name)
                    block_parameters.append(bias)
                    assigned.add(bias_name)
            self._add_block(
                name,
                block_names,
                block_parameters,
                min(self.rank, sum(item.numel() for item in block_parameters)),
                common,
            )
            assigned.update(block_names)

        if bias_mode == "separate":
            for name, parameter in optimized.items():
                if name in assigned or parameter.ndim != 1:
                    continue
                self._add_block(
                    name,
                    [name],
                    [parameter],
                    min(self.rank, parameter.numel()),
                    common,
                )
                assigned.add(name)

        self.unfiltered_parameter_names = tuple(
            name for name in optimized if name not in assigned
        )
        if not self.blocks:
            raise ValueError("no matrix parameters were selected for filtering")

        self.n_filtered_params = sum(
            parameter.numel()
            for block in self.blocks.values()
            for parameter in block.param_list
        )
        self.max_basis_numel = sum(
            block.n_params * block.rank for block in self.blocks.values()
        )

    def _add_block(self, name, parameter_names, parameters, rank, common):
        self.blocks[name] = SpectralGradientFilter(
            self.model,
            self.base_optimizer,
            rank=rank,
            parameters=parameters,
            **common,
        )
        self.block_parameter_names[name] = tuple(parameter_names)

    @staticmethod
    def _block_has_gradient(block):
        gradients = [parameter.grad for parameter in block.param_list]
        if all(gradient is None for gradient in gradients):
            return False
        if any(gradient is None for gradient in gradients):
            raise RuntimeError(
                "a joint matrix/bias block has only a partial gradient"
            )
        if any(gradient.is_sparse for gradient in gradients):
            raise RuntimeError("sparse gradients are not supported")
        return True

    def filter_grad(self):
        """Update and apply every active matrix block's filter in place."""
        self.step_count += 1
        active = []
        skipped = []
        block_diagnostics = OrderedDict()
        for name, block in self.blocks.items():
            if not self._block_has_gradient(block):
                skipped.append(name)
                continue
            diagnostics = block.filter_grad()
            active.append(name)
            if self.report_per_block:
                block_diagnostics[name] = {
                    "parameters": self.block_parameter_names[name],
                    "basis_rank": diagnostics.get("basis_rank", 0),
                    "kept_rank": diagnostics.get("kept_rank", 0),
                    "effective_rank": diagnostics.get("effective_rank", 0.0),
                    "max_orthogonality_error": diagnostics[
                        "max_orthogonality_error"
                    ],
                }

        basis_ranks = [
            block.V.shape[1] for block in self.blocks.values() if block.V is not None
        ]
        effective_ranks = [
            block._effective_rank()
            for block in self.blocks.values()
            if block.S is not None
        ]
        result = {
            "step": self.step_count,
            "filtering_active": any(
                block.step_count > block.warmup for block in self.blocks.values()
            ),
            "block_count": len(self.blocks),
            "active_block_count": len(active),
            "skipped_block_count": len(skipped),
            "total_basis_rank": sum(basis_ranks),
            "mean_effective_rank": (
                sum(effective_ranks) / len(effective_ranks)
                if effective_ranks
                else 0.0
            ),
            "max_orthogonality_error": max(
                (block.max_orthogonality_error for block in self.blocks.values()),
                default=0.0,
            ),
        }
        if self.report_per_block:
            result["blocks"] = block_diagnostics
            result["skipped_blocks"] = skipped
        return result

    def step(self, inputs, targets):
        """Convenience cross-entropy training step, matching the global API."""
        self.base_optimizer.zero_grad()
        loss = F.cross_entropy(self.model(inputs), targets)
        loss.backward()
        loss_value = loss.item()
        diagnostics = self.filter_grad()
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        diagnostics["loss"] = loss_value
        return loss_value, diagnostics

    def reset(self):
        self.step_count = 0
        for block in self.blocks.values():
            block.reset()

    def zero_grad(self, *args, **kwargs):
        self.base_optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        """Return optimizer plus block-filter state for checkpointing."""
        block_state = OrderedDict()
        for name, block in self.blocks.items():
            block_state[name] = {
                "V": block.V,
                "S": block.S,
                "proj_k": block.proj_k,
                "step_count": block.step_count,
                "grad_mean": block.grad_mean,
                "stabilization_count": block.stabilization_count,
                "max_orthogonality_error": block.max_orthogonality_error,
            }
        return {
            "version": 1,
            "rank": self.rank,
            "bias_mode": self.bias_mode,
            "step_count": self.step_count,
            "block_parameter_names": dict(self.block_parameter_names),
            "base_optimizer": self.base_optimizer.state_dict(),
            "blocks": block_state,
        }

    def load_state_dict(self, state):
        """Restore a state produced by :meth:`state_dict`."""
        if state.get("version") != 1:
            raise ValueError("unsupported per-matrix filter state version")
        if state.get("rank") != self.rank or state.get("bias_mode") != self.bias_mode:
            raise ValueError("checkpoint configuration does not match filter")
        if state.get("block_parameter_names") != dict(self.block_parameter_names):
            raise ValueError("checkpoint parameter blocks do not match model")
        if set(state.get("blocks", {})) != set(self.blocks):
            raise ValueError("checkpoint block names do not match model")

        self.base_optimizer.load_state_dict(state["base_optimizer"])
        self.step_count = int(state["step_count"])
        for name, saved in state["blocks"].items():
            block = self.blocks[name]
            parameter = block.param_list[0]
            block.V = (
                None
                if saved["V"] is None
                else saved["V"].to(device=parameter.device, dtype=parameter.dtype)
            )
            block.S = None if saved["S"] is None else saved["S"].cpu()
            if block.S is not None and block.stable_update:
                block.S = block.S.double()
            block.proj_k = saved["proj_k"]
            block.step_count = int(saved["step_count"])
            block.grad_mean = (
                None
                if saved["grad_mean"] is None
                else saved["grad_mean"].to(
                    device=parameter.device, dtype=parameter.dtype
                )
            )
            block.stabilization_count = int(saved["stabilization_count"])
            block.max_orthogonality_error = float(
                saved["max_orthogonality_error"]
            )

    def memory_estimate(self):
        """Return the maximum basis storage implied by configured block ranks."""
        basis_bytes = sum(
            block.n_params * block.rank * block.param_list[0].element_size()
            for block in self.blocks.values()
        )
        return {
            "filtered_parameters": self.n_filtered_params,
            "blocks": len(self.blocks),
            "max_basis_elements": self.max_basis_numel,
            "max_basis_bytes": basis_bytes,
            "max_basis_mib": basis_bytes / (1024 ** 2),
        }


MatrixSpectralGradientFilter = PerMatrixSpectralGradientFilter
