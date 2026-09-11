#!/usr/bin/env python3
"""Raw-direction norm control, retaining the unchanged legacy shadow estimator."""
from __future__ import annotations

import math
import time
from typing import Any

import torch

from experiments.grokking_action_policy import (
    LegacyActionPolicyFilter, NORM_MATCH_DENOMINATOR_FLOOR,
    NORM_MATCH_EPS_MULTIPLIER, NORM_MATCH_MIN_RELATIVE_TOLERANCE,
    _cosine, _norm, compute_actions,
)

POLICY = "raw_norm_matched"


@torch.no_grad()
def match_raw_norm(g: torch.Tensor, native: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Match native's incoming norm along raw g, preserving the frozen floor.

    The domain flag and the post-cast tolerance are separate. Returning a zero
    tensor never licenses skipping the optimizer step or setting grad=None.
    """
    if not isinstance(g, torch.Tensor) or not isinstance(native, torch.Tensor):
        raise TypeError("Raw and native gradients must be tensors")
    if (g.ndim != 1 or native.shape != g.shape or g.numel() == 0
            or g.dtype != native.dtype or g.device != native.device
            or not g.dtype.is_floating_point):
        raise ValueError("Require matching nonempty floating gradient vectors")
    if not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(native).all()):
        raise ValueError("Raw and native gradients must be finite")
    raw_norm, target_norm = _norm(g), _norm(native)
    if not math.isfinite(raw_norm) or not math.isfinite(target_norm):
        raise FloatingPointError("Gradient norm overflow")
    if raw_norm == 0.0 and target_norm > 0.0:
        raise FloatingPointError("Cannot match positive norm with zero raw direction")
    floor = NORM_MATCH_DENOMINATOR_FLOOR
    scale = target_norm / max(raw_norm, floor)
    delivered = (g.detach().double() * scale).to(dtype=g.dtype)
    if not math.isfinite(scale) or not bool(torch.isfinite(delivered).all()):
        raise FloatingPointError("Raw norm-matched action is nonfinite")
    actual_norm = _norm(delivered)
    mismatch = abs(actual_norm - target_norm)
    relative = mismatch / max(target_norm, floor)
    tolerance = max(NORM_MATCH_MIN_RELATIVE_TOLERANCE,
                    NORM_MATCH_EPS_MULTIPLIER * torch.finfo(g.dtype).eps)
    exact_domain = raw_norm >= floor or (raw_norm == 0 and target_norm == 0)
    if raw_norm >= floor and relative > tolerance:
        raise FloatingPointError("Raw norm matching exceeds post-cast tolerance")
    return delivered, {
        "raw_norm_match_scale": scale,
        "raw_norm_match_target_norm": target_norm,
        "raw_norm_match_degenerate": raw_norm <= floor,
        "raw_norm_match_denominator_clamped": raw_norm < floor,
        "raw_norm_match_exact": exact_domain,
        "raw_norm_match_relative_tolerance": tolerance,
        "raw_norm_match_absolute_mismatch": mismatch,
        "raw_norm_match_relative_mismatch": relative,
    }


@torch.no_grad()
def compute_raw_direction_actions(
    V: torch.Tensor, g: torch.Tensor, *, retain_basis: bool = False,
) -> dict[str, Any]:
    """Reuse the old three-action diagnostic; append the one new raw action.

    V and g are not mutated. The hypothetical projected actions are diagnostics
    only: exactly one raw_norm_matched action is supplied to the optimizer.
    The original projection-domain admission rules are preserved as well.
    """
    started = time.perf_counter()
    result = compute_actions(V, g, retain_basis=retain_basis)
    actions = result["actions"]
    delivered, raw_match = match_raw_norm(g, actions["native"])
    actions[POLICY] = delivered
    diag = result["diagnostics"]
    diag.update(raw_match)
    diag["action_norms"][POLICY] = _norm(delivered)
    for name, action in actions.items():
        diag["pairwise_cosines"]["raw_" + name] = _cosine(g, action)
        if name != POLICY:
            diag["pairwise_cosines"][name + "_" + POLICY] = _cosine(action, delivered)
    diag["projected_over_raw_norm"] = (
        diag["action_norms"]["orthogonal"] / diag["input_norm"]
        if diag["input_norm"] > 0 else None)
    # Existing coefficients describe the three old diagnostic actions only.
    # The new delivered vector need not lie in Q's span; no missing components
    # are implicitly imputed from those coefficients.
    if retain_basis:
        result["coefficients"][POLICY] = (
            result["Q"].T @ delivered.double()).cpu()
    result["coefficient_scope"] = (
        "raw_norm_matched coordinates included; out-of-span part not represented"
        if retain_basis else "native/orthogonal/norm_matched diagnostics only")
    result["raw_gradient"] = g.detach().clone()
    if g.device.type == "cuda":
        torch.cuda.synchronize(g.device)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


class RawDirectionPolicyFilter(LegacyActionPolicyFilter):
    """Identical legacy observer; native action's norm along unfiltered g."""

    def __init__(self, model, base_optimizer, *, retain_action_basis=False, **kwargs):
        super().__init__(model, base_optimizer, action_policy="native",
                         retain_action_basis=retain_action_basis, **kwargs)
        self.action_policy = POLICY
        self.last_action_result = None
        self.last_raw_gradient = None

    def _project_gradient(self, g):
        self.last_raw_gradient = g.detach().clone()
        if self.V is None:
            # Preserve the historical no-basis identity, not an invented zero
            # projector. All projected-angle/rank interpretations are absent.
            value = super()._project_gradient(g)
            self.last_action_result = None
            return value
        result = compute_raw_direction_actions(
            self.V, g, retain_basis=self.retain_action_basis)
        self.last_action_result = result
        self.last_action_basis = result.get("Q")
        self.last_action_metrics = {
            "step": self.step_count,
            **{name: result[name] for name in (
                "tolerance", "numerical_rank", "P", "k", "sigma_max",
                "basis_column_count", "full_column_rank",
                "svd_truncation_relative_frobenius_residual", "elapsed_seconds",
                "diagnostics", "coefficient_scope")},
            "singular_values": result["singular_values"].tolist(),
            "coefficients": {name: value.tolist() for name, value in result["coefficients"].items()},
        }
        if "geometry_diagnostics" in result:
            self.last_action_metrics["geometry_diagnostics"] = result["geometry_diagnostics"]
        return result["actions"][POLICY]
