#!/usr/bin/env python3
"""Numerical action substitutions for a legacy spectral-filter state.

This module changes only the action delivered to the optimizer.  The wrapper
inherits the historical estimator update verbatim from ``SpectralGradientFilter``.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from spectral_filter import SpectralGradientFilter


ACTION_POLICIES = ("native", "orthogonal", "norm_matched")
NORM_MATCH_DENOMINATOR_FLOOR = 1e-30
NORM_MATCH_MIN_RELATIVE_TOLERANCE = 1e-12
NORM_MATCH_EPS_MULTIPLIER = 10.0


def _validate_inputs(V: torch.Tensor, g: torch.Tensor) -> None:
    if not isinstance(V, torch.Tensor) or not isinstance(g, torch.Tensor):
        raise TypeError("V and g must be torch tensors")
    if V.ndim != 2 or g.ndim != 1 or V.shape[0] != g.shape[0]:
        raise ValueError("expected V with shape (P, k) and g with shape (P,)")
    if V.shape[0] == 0:
        raise ValueError("P must be positive")
    if not V.dtype.is_floating_point or not g.dtype.is_floating_point:
        raise TypeError("V and g must have floating-point dtypes")
    if V.dtype != g.dtype or V.device != g.device:
        raise ValueError("V and g must have the same dtype and device")
    if not bool(torch.isfinite(V).all()) or not bool(torch.isfinite(g).all()):
        raise ValueError("V and g must be finite")


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().double()).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = _norm(left) * _norm(right)
    if denominator <= NORM_MATCH_DENOMINATOR_FLOOR:
        return None
    numerator = torch.dot(left.detach().double(), right.detach().double())
    return float((numerator / denominator).item())


@torch.no_grad()
def compute_actions(
    V: torch.Tensor,
    g: torch.Tensor,
    *,
    selected: str | None = None,
    retain_basis: bool = False,
) -> dict[str, Any]:
    """Compute native and numerical same-span actions without changing inputs.

    ``selected=None`` returns all three actions for the common-state diagnostic.
    A selected policy returns only that delivered action.  The norm-matched
    calculation still constructs the native action because its norm defines the
    scale.  Action tensors have ``g``'s original dtype and device; decomposition
    metadata and common-basis coefficients are float64 CPU tensors.  The
    norm-match denominator is the norm of the dtype-cast action actually
    delivered by the orthogonal arm.  Positive norms below the frozen floor are
    clamped and explicitly marked as approximate; ordinary norms must match
    after the final cast within the frozen dtype-aware rounding tolerance.
    """
    _validate_inputs(V, g)
    if selected is not None and selected not in ACTION_POLICIES:
        raise ValueError(f"selected must be one of {ACTION_POLICIES} or None")
    if not isinstance(retain_basis, bool):
        raise TypeError("retain_basis must be bool")

    p, k = V.shape
    if V.device.type == "cuda":
        torch.cuda.synchronize(V.device)
    computation_started = time.perf_counter()
    # The native legacy action deliberately uses the input dtype and exactly the
    # historical parenthesization.  Do not reuse the float64 decomposition here.
    need_native = selected in (None, "native", "norm_matched")
    need_orthogonal = selected in (None, "orthogonal", "norm_matched")
    native = V @ (V.T @ g) if need_native else None
    if native is not None and not bool(torch.isfinite(native).all()):
        raise FloatingPointError("native action is non-finite")

    V64 = V.detach().to(dtype=torch.float64)
    qr_basis, small_r = torch.linalg.qr(V64, mode="reduced")
    small_u, singular_values, _ = torch.linalg.svd(
        small_r.detach().cpu(), full_matrices=False
    )
    if not bool(torch.isfinite(singular_values).all()):
        raise FloatingPointError("basis singular values are non-finite")
    sigma_max = float(singular_values[0].item()) if singular_values.numel() else 0.0
    tolerance = max(p, k) * torch.finfo(torch.float64).eps * sigma_max
    kept = singular_values > tolerance
    numerical_rank = int(kept.sum().item())
    total_singular_energy = float(singular_values.square().sum().item())
    discarded_singular_energy = float(
        singular_values[~kept].square().sum().item()
    )
    svd_truncation_relative_residual = (
        (discarded_singular_energy / total_singular_energy) ** 0.5
        if total_singular_energy > 0.0
        else 0.0
    )
    if numerical_rank:
        Q = qr_basis @ small_u[:, kept].to(device=V.device)
    else:
        Q = torch.empty((p, 0), dtype=torch.float64, device=V.device)
    if not bool(torch.isfinite(Q).all()):
        raise FloatingPointError("reconstructed numerical basis is non-finite")

    orthogonal64 = None
    orthogonal = None
    if need_orthogonal:
        g64 = g.detach().to(dtype=torch.float64)
        orthogonal64 = Q @ (Q.T @ g64)
        orthogonal = orthogonal64.to(dtype=g.dtype)
        if not bool(torch.isfinite(orthogonal).all()):
            raise FloatingPointError("orthogonal action is non-finite")

    norm_matched = None
    norm_match_scale = None
    degenerate = None
    if selected in (None, "norm_matched"):
        native_norm_for_scale = _norm(native)
        orthogonal_norm_for_scale = _norm(orthogonal)
        degenerate = orthogonal_norm_for_scale <= NORM_MATCH_DENOMINATOR_FLOOR
        denominator_clamped = (
            orthogonal_norm_for_scale < NORM_MATCH_DENOMINATOR_FLOOR
        )
        if orthogonal_norm_for_scale == 0.0 and native_norm_for_scale > 0.0:
            raise FloatingPointError(
                "cannot norm-match a numerically zero delivered orthogonal action"
            )
        norm_match_scale = native_norm_for_scale / max(
            orthogonal_norm_for_scale, NORM_MATCH_DENOMINATOR_FLOOR
        )
        norm_matched = (
            orthogonal.detach().double() * norm_match_scale
        ).to(dtype=g.dtype)
        if not bool(torch.isfinite(norm_matched).all()):
            raise FloatingPointError("norm-matched action is non-finite")

    computed = {
        name: value
        for name, value in (
            ("native", native),
            ("orthogonal", orthogonal),
            ("norm_matched", norm_matched),
        )
        if value is not None
    }
    actions = computed if selected is None else {selected: computed[selected]}
    coefficients = {
        name: (Q.T @ value.detach().double()).cpu()
        for name, value in computed.items()
    }
    if any(not bool(torch.isfinite(value).all()) for value in coefficients.values()):
        raise FloatingPointError("common-basis coefficients are non-finite")
    norms = {name: _norm(value) for name, value in computed.items()}
    cosines = {
        f"{left}_{right}": _cosine(computed[left], computed[right])
        for index, left in enumerate(computed)
        for right in tuple(computed)[index + 1 :]
    }
    diagnostics: dict[str, Any] = {
        "action_norms": norms,
        "pairwise_cosines": cosines,
        "input_norm": _norm(g),
    }
    if norm_matched is not None:
        absolute_mismatch = abs(norms["norm_matched"] - norms["native"])
        relative_mismatch = absolute_mismatch / max(
            norms["native"], NORM_MATCH_DENOMINATOR_FLOOR
        )
        relative_tolerance = max(
            NORM_MATCH_MIN_RELATIVE_TOLERANCE,
            NORM_MATCH_EPS_MULTIPLIER * torch.finfo(g.dtype).eps,
        )
        if orthogonal_norm_for_scale >= NORM_MATCH_DENOMINATOR_FLOOR:
            if relative_mismatch > relative_tolerance:
                raise FloatingPointError(
                    "delivered norm-matched action exceeds rounding tolerance"
                )
            norm_match_exact = True
        else:
            # Both exact zeros satisfy the control.  A positive sub-floor
            # denominator deliberately uses the frozen clamp and is only an
            # explicitly reported approximation to equal input norm.
            norm_match_exact = (
                orthogonal_norm_for_scale == 0.0
                and native_norm_for_scale == 0.0
            )
        diagnostics.update(
            {
                "norm_match_scale": float(norm_match_scale),
                "norm_match_degenerate": bool(degenerate),
                "norm_match_denominator_clamped": bool(denominator_clamped),
                "norm_match_exact": bool(norm_match_exact),
                "norm_match_relative_tolerance": float(relative_tolerance),
                "norm_match_absolute_mismatch": absolute_mismatch,
                "norm_match_relative_mismatch": relative_mismatch,
            }
        )

    result: dict[str, Any] = {
        "actions": actions,
        "singular_values": singular_values,
        "tolerance": float(tolerance),
        "numerical_rank": numerical_rank,
        "P": p,
        "k": k,
        "sigma_max": sigma_max,
        "basis_column_count": k,
        "full_column_rank": numerical_rank == k,
        "svd_truncation_relative_frobenius_residual": (
            svd_truncation_relative_residual
        ),
        "coefficients": coefficients,
        "diagnostics": diagnostics,
    }
    if retain_basis:
        result["Q"] = Q
        if numerical_rank:
            gram_error = Q.T @ Q - torch.eye(
                numerical_rank, dtype=torch.float64, device=V.device
            )
            q_orthogonality_max_abs = float(gram_error.abs().max().item())
        else:
            q_orthogonality_max_abs = 0.0
        reconstructed_V = Q @ (Q.T @ V64)
        V_norm = float(torch.linalg.matrix_norm(V64, ord="fro").item())
        V_residual = float(
            torch.linalg.matrix_norm(V64 - reconstructed_V, ord="fro").item()
        )
        result["geometry_diagnostics"] = {
            "q_orthogonality_max_abs": q_orthogonality_max_abs,
            "v_reconstruction_relative_frobenius_residual": V_residual
            / max(V_norm, NORM_MATCH_DENOMINATOR_FLOOR),
        }
    if V.device.type == "cuda":
        torch.cuda.synchronize(V.device)
    result["elapsed_seconds"] = time.perf_counter() - computation_started
    return result


class LegacyActionPolicyFilter(SpectralGradientFilter):
    """Legacy estimator with only its delivered gradient action substituted."""

    _REQUIRED = {
        "stable_update": False,
        "weighting": "hard",
        "normalize": "none",
        "filter_strength": 1.0,
        "adaptive": "none",
    }

    def __init__(
        self,
        model,
        base_optimizer,
        *,
        action_policy: str,
        retain_action_basis: bool = False,
        **kwargs,
    ):
        if action_policy not in ACTION_POLICIES:
            raise ValueError(f"action_policy must be one of {ACTION_POLICIES}")
        if not isinstance(retain_action_basis, bool):
            raise TypeError("retain_action_basis must be bool")
        for name, required in self._REQUIRED.items():
            if name in kwargs and kwargs[name] != required:
                raise ValueError(
                    f"legacy action substitution requires {name}={required!r}"
                )
            kwargs[name] = required
        super().__init__(model, base_optimizer, **kwargs)
        self.action_policy = action_policy
        self.retain_action_basis = retain_action_basis
        self.last_action_metrics: dict[str, Any] | None = None
        self.last_action_basis: torch.Tensor | None = None

    def _project_gradient(self, g):
        if self.V is None:
            self.last_action_metrics = {
                "step": self.step_count,
                "singular_values": [],
                "tolerance": 0.0,
                "numerical_rank": 0,
                "P": g.numel(),
                "k": 0,
                "sigma_max": 0.0,
                "basis_column_count": 0,
                "full_column_rank": True,
                "svd_truncation_relative_frobenius_residual": 0.0,
                "elapsed_seconds": 0.0,
                "coefficients": {},
                "diagnostics": {"no_basis_identity": True, "input_norm": _norm(g)},
            }
            self.last_action_basis = None
            return g

        result = compute_actions(
            self.V, g, selected=self.action_policy,
            retain_basis=self.retain_action_basis
        )
        self.last_action_metrics = {
            "step": self.step_count,
            "singular_values": result["singular_values"].tolist(),
            "tolerance": result["tolerance"],
            "numerical_rank": result["numerical_rank"],
            "P": result["P"],
            "k": result["k"],
            "sigma_max": result["sigma_max"],
            "basis_column_count": result["basis_column_count"],
            "full_column_rank": result["full_column_rank"],
            "svd_truncation_relative_frobenius_residual": result[
                "svd_truncation_relative_frobenius_residual"
            ],
            "elapsed_seconds": result["elapsed_seconds"],
            "coefficients": {
                name: values.tolist()
                for name, values in result["coefficients"].items()
            },
            "diagnostics": result["diagnostics"],
        }
        if "geometry_diagnostics" in result:
            self.last_action_metrics["geometry_diagnostics"] = result[
                "geometry_diagnostics"
            ]
        self.last_action_basis = result.get("Q")
        return result["actions"][self.action_policy]
