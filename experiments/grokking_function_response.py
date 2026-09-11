"""Finite function-response diagnostics; no model, optimizer or artifact I/O.

The linear utility is a derivative of CE in logit space along a FINITE response,
not a parameter-space Jacobian-vector product or an exact finite loss change.
"""
from __future__ import annotations

import math
import numpy as np

ACTION_NAMES = ("before", "raw", "trunc", "projected", "zero")
CONTRASTS = (("raw", "trunc"), ("trunc", "projected"),
             ("raw", "projected"), ("zero", "raw"),
             ("zero", "trunc"), ("zero", "projected"))


def centered(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Require finite row-by-class matrix")
    return values - values.mean(axis=1, keepdims=True)


def sum_projection(values, sums):
    """Orthogonal projection under the uniform COMPLETE p-by-p input grid."""
    values = np.asarray(values, dtype=np.float64)
    sums = np.asarray(sums)
    if values.ndim != 2 or sums.shape != (len(values),) or sums.dtype.kind not in "iu":
        raise ValueError("Invalid full-grid matrix or sum labels")
    p = values.shape[1]
    if (p < 2 or len(values) != p * p or np.any(sums < 0)
            or np.any(sums >= p) or not np.isfinite(values).all()
            or not np.array_equal(np.bincount(sums, minlength=p), np.full(p, p))):
        raise ValueError("Projection requires p observations per sum on the full grid")
    means = np.zeros((p, p), dtype=np.float64)
    np.add.at(means, sums, values)
    means /= p
    return means[sums]


def _indices(values, n):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values >= n)
            or len(np.unique(values)) != len(values)):
        raise ValueError("Invalid split indices")
    return values


def behavior(logits, sums, indices):
    selected = logits[indices]
    labels = sums[indices]
    row = np.arange(len(indices))
    maxima = selected.max(axis=1)
    lse = maxima + np.log(np.exp(selected - maxima[:, None]).sum(axis=1))
    correct = selected[row, labels]
    competitors = selected.copy()
    competitors[row, labels] = -np.inf
    return {"ce": float(np.mean(lse - correct)),
            "margin": float(np.mean(correct - competitors.max(axis=1))),
            "accuracy": float(np.mean(selected.argmax(axis=1) == labels))}


def response(change, sums, negative_ce_gradient, splits):
    """Global energies; train/test utilities use their own mean normalization."""
    change = centered(change)
    rule = sum_projection(change, sums)
    residual = change - rule
    components = {"total": change, "sum_consistent": rule, "within_sum": residual}
    energies = {name: float(np.mean(np.sum(value * value, axis=1)))
                for name, value in components.items()}
    cross = float(np.mean(np.sum(rule * residual, axis=1)))
    utility = {split: {name: float(np.mean(np.sum(
        negative_ce_gradient[indices] * value[indices], axis=1)))
        for name, value in components.items()} for split, indices in splits.items()}
    result = {"energy": energies, "cross_inner_product": cross,
            "energy_additivity_residual": energies["total"] -
                energies["sum_consistent"] - energies["within_sum"],
            "sum_projection_idempotence_max_abs": float(np.max(np.abs(
                sum_projection(rule, sums) - rule))),
            "within_sum_mean_max_abs": float(np.max(np.abs(
                sum_projection(residual, sums)))),
            "class_mean_max_abs": float(np.max(np.abs(change.mean(axis=1)))),
            "linear_ce_utility": utility,
            "utility_additivity_residual": {split: values["total"] -
                values["sum_consistent"] - values["within_sum"]
                for split, values in utility.items()}}
    energy_tolerance = 1e-10 * (1 + sum(abs(value) for value in energies.values()))
    point_tolerance = 1e-10 * (1 + float(np.max(np.abs(change))))
    if (abs(result["energy_additivity_residual"]) > energy_tolerance
            or abs(cross) > energy_tolerance
            or any(result[key] > point_tolerance for key in (
                "sum_projection_idempotence_max_abs", "within_sum_mean_max_abs",
                "class_mean_max_abs"))
            or any(abs(result["utility_additivity_residual"][split]) >
                   1e-10 * (1 + sum(abs(v) for v in utility[split].values()))
                   for split in splits)):
        raise ArithmeticError("Function projection or utility identity failed")
    return result


def analyze_seed(logits_by_action, sums, train_indices, test_indices):
    if set(logits_by_action) != set(ACTION_NAMES):
        raise ValueError("Require before/raw/trunc/projected/zero logits")
    logits = {name: centered(logits_by_action[name]) for name in ACTION_NAMES}
    shape = logits["before"].shape
    if any(value.shape != shape for value in logits.values()):
        raise ValueError("Counterfactual grids differ")
    sums = np.asarray(sums)
    sum_projection(logits["before"], sums)  # admit the complete-grid contract
    n = shape[0]
    train, test = _indices(train_indices, n), _indices(test_indices, n)
    if not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(n)):
        raise ValueError("Train/test must partition the full grid")
    splits = {"train": train, "test": test}
    base = logits["before"]
    exp = np.exp(base - base.max(axis=1, keepdims=True))
    negative_gradient = -exp / exp.sum(axis=1, keepdims=True)
    negative_gradient[np.arange(n), sums] += 1
    metrics = {name: {split: behavior(value, sums, ids)
                     for split, ids in splits.items()} for name, value in logits.items()}
    arms = {}
    for name in ACTION_NAMES[1:]:
        arm = response(logits[name] - base, sums, negative_gradient, splits)
        arm["finite_improvement"] = {
            split: {"ce": metrics["before"][split]["ce"] - metrics[name][split]["ce"],
                    "margin": metrics[name][split]["margin"] - metrics["before"][split]["margin"],
                    "accuracy": metrics[name][split]["accuracy"] - metrics["before"][split]["accuracy"]}
            for split in splits}
        # Convex logit-space CE makes this remainder nonnegative in exact arithmetic.
        arm["ce_convex_remainder"] = {split: arm["linear_ce_utility"][split]["total"] -
            arm["finite_improvement"][split]["ce"] for split in splits}
        if any(value < -1e-10 for value in arm["ce_convex_remainder"].values()):
            raise ArithmeticError("Logit-space CE convexity check failed")
        arms[name] = arm
    contrasts = {}
    for left, right in CONTRASTS:
        entry = response(logits[right] - logits[left], sums, negative_gradient, splits)
        entry["finite_difference_right_minus_left"] = {
            split: {metric: metrics[right][split][metric] - metrics[left][split][metric]
                    for metric in ("ce", "margin", "accuracy")} for split in splits}
        contrasts[left + "_to_" + right] = entry
    checks = {split: {metric:
        contrasts["raw_to_trunc"]["finite_difference_right_minus_left"][split][metric] +
        contrasts["trunc_to_projected"]["finite_difference_right_minus_left"][split][metric] -
        contrasts["raw_to_projected"]["finite_difference_right_minus_left"][split][metric]
        for metric in ("ce", "margin", "accuracy")} for split in splits}
    if any(abs(value) > 1e-12 for values in checks.values() for value in values.values()):
        raise ArithmeticError("Finite scalar telescoping failed")
    utility_checks = {}
    for split in splits:
        utility_checks[split] = {}
        for component in ("total", "sum_consistent", "within_sum"):
            pieces = [contrasts[key]["linear_ce_utility"][split][component] for key in
                      ("raw_to_trunc", "trunc_to_projected", "raw_to_projected")]
            residual = pieces[0] + pieces[1] - pieces[2]
            if abs(residual) > 1e-10 * (1 + sum(abs(value) for value in pieces)):
                raise ArithmeticError("Linear utility telescoping failed")
            utility_checks[split][component] = residual
    return {"behavior": metrics, "responses_from_before": arms, "contrasts": contrasts,
            "finite_scalar_telescoping_residual": checks,
            "linear_utility_telescoping_residual": utility_checks,
            "semantics": {"positive_linear_utility": "predicted CE reduction at shared before logits",
                "finite_contrast_signs": "right minus left: negative CE favorable; positive margin/accuracy favorable",
                "energy_measure": "mean over full uniform grid of squared class-vector L2 norm",
                "within_sum": "variation within true-sum class, not identified memorization",
                "linear_utility": "CE derivative along finite logit difference, NOT parameter JVP"}}


def paired_summary(seed_results):
    """Aggregate all scalar leaves, retaining individual values and sample SE."""
    if len(seed_results) != 5:
        raise ValueError("Exactly five paired seeds are required")
    def recurse(values):
        first = values[0]
        if isinstance(first, dict):
            if any(set(value) != set(first) for value in values):
                raise ValueError("Seed result schema differs")
            return {key: recurse([value[key] for value in values]) for key in first}
        if isinstance(first, str):
            if any(value != first for value in values):
                raise ValueError("Seed semantics differ")
            return first
        data = np.asarray(values, dtype=np.float64)
        if data.shape != (5,) or not np.isfinite(data).all():
            raise ValueError("Invalid paired scalar")
        return {"values": data.tolist(), "mean": float(data.mean()),
                "sample_se": float(data.std(ddof=1) / math.sqrt(5)),
                "positive_count": int(np.sum(data > 0)),
                "negative_count": int(np.sum(data < 0)), "zero_count": int(np.sum(data == 0))}
    return recurse(seed_results)
