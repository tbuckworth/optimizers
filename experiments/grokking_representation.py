"""Pure representation measurements for the modular-addition checkpoints.

No function in this module trains, calls backward, or changes model parameters.
The acquisition runner owns checkpoint loading, artifact I/O, and CPU-thread
pinning; this module returns CPU tensors or JSON-safe scalar structures.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import torch


def _index_tensor(values: Sequence[int] | torch.Tensor, n: int, name: str) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.int64, device="cpu").flatten()
    if result.numel() == 0 or bool((result < 0).any()) or bool((result >= n).any()):
        raise ValueError(f"{name} contains no rows or an out-of-range row")
    if result.unique().numel() != result.numel():
        raise ValueError(f"{name} contains duplicate rows")
    return result


def _int64_hash(*named_values: tuple[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in named_values:
        tensor = value.detach().cpu().to(torch.int64).contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _float64_hash(*named_values: tuple[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in named_values:
        tensor = value.detach().cpu().to(torch.float64).contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def extract_activations(model, pairs: torch.Tensor, batch_size: int = 1024) -> dict[str, torch.Tensor]:
    """Return logits and two hook readouts, all detached on CPU.

    ``final_hidden`` is the output of ``ln_final`` immediately before
    unembedding. ``pre_attention`` concatenates both token residuals entering
    ``ln1`` in token order. Existing parameters, buffers, gradients, hooks, and
    per-module train/eval flags are left unchanged.
    """
    if not isinstance(pairs, torch.Tensor) or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be an N-by-2 tensor")
    if pairs.shape[0] == 0 or batch_size < 1:
        raise ValueError("pairs must be nonempty and batch_size positive")
    if not hasattr(model, "ln1") or not hasattr(model, "ln_final"):
        raise ValueError("model lacks the registered grokking hook modules")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters") from error

    pre_attention, final_hidden, hooked_logits = [], [], []

    def capture_pre_attention(_module, inputs):
        residual = inputs[0]
        if residual.ndim != 3 or residual.shape[1] != 2:
            raise RuntimeError("ln1 input is not a two-token residual stream")
        pre_attention.append(residual.detach().reshape(residual.shape[0], -1).cpu().clone())

    def capture_final(_module, _inputs, output):
        if output.ndim != 2:
            raise RuntimeError("ln_final output is not an N-by-d hidden state")
        final_hidden.append(output.detach().cpu().clone())

    def capture_logits(_module, _inputs, output):
        hooked_logits.append(output.detach().cpu().clone())

    module_modes = [(module, module.training) for module in model.modules()]
    pre_handle = model.ln1.register_forward_pre_hook(capture_pre_attention)
    final_handle = model.ln_final.register_forward_hook(capture_final)
    logits_handle = model.unembed.register_forward_hook(capture_logits)
    logits = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                before_pre, before_final, before_logits = (
                    len(pre_attention), len(final_hidden), len(hooked_logits))
                output = model(pairs[start:start + batch_size].to(device))
                if (len(pre_attention) != before_pre + 1
                        or len(final_hidden) != before_final + 1
                        or len(hooked_logits) != before_logits + 1
                        or not torch.equal(hooked_logits[-1], output.detach().cpu())):
                    raise RuntimeError("registered hooks did not each fire exactly once")
                logits.append(hooked_logits[-1])
    finally:
        pre_handle.remove()
        final_handle.remove()
        logits_handle.remove()
        for module, training in module_modes:
            module.training = training
    return {"logits": torch.cat(logits), "final_hidden": torch.cat(final_hidden),
            "pre_attention": torch.cat(pre_attention)}


def make_probe_split(sums: torch.Tensor, seed: int) -> dict[str, Any]:
    """Deterministically split every sum stratum into floor-half fit and remainder eval."""
    labels = torch.as_tensor(sums, dtype=torch.int64, device="cpu").flatten()
    if labels.numel() < 2:
        raise ValueError("at least two rows are required")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    fit_parts, eval_parts, counts = [], [], []
    for label in labels.unique(sorted=True).tolist():
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        if rows.numel() < 2:
            raise ValueError(f"sum {label} has fewer than two rows")
        shuffled = rows[torch.randperm(rows.numel(), generator=generator)]
        cut = rows.numel() // 2
        fit_parts.append(shuffled[:cut])
        eval_parts.append(shuffled[cut:])
        counts.append({"sum": label, "total": rows.numel(), "fit": cut,
                       "eval": rows.numel() - cut})
    fit = torch.cat(fit_parts).sort().values
    evaluation = torch.cat(eval_parts).sort().values
    if torch.cat((fit, evaluation)).sort().values.tolist() != list(range(len(labels))):
        raise RuntimeError("probe split is not a disjoint partition")
    return {"fit_indices": fit.tolist(), "eval_indices": evaluation.tolist(),
            "strata": counts, "seed": int(seed),
            "sha256": _int64_hash(("fit", fit), ("eval", evaluation))}


def make_row_permutations(n_rows: int, seed: int, n_nulls: int = 20) -> list[dict[str, Any]]:
    """Create fixed row permutations; these shuffle observations, not class codes."""
    if n_rows < 2 or n_nulls < 1:
        raise ValueError("need at least two rows and one null")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    identity = torch.arange(n_rows, dtype=torch.int64)
    result = []
    for index in range(n_nulls):
        permutation = torch.randperm(n_rows, generator=generator)
        if torch.equal(permutation, identity):
            permutation = permutation.roll(1)
        result.append({"null_index": index, "permutation": permutation.tolist(),
                       "sha256": _int64_hash(("row_permutation", permutation))})
    return result


def _fourier_targets(sums: torch.Tensor, p: int) -> torch.Tensor:
    frequencies = torch.arange(1, (p - 1) // 2 + 1, dtype=torch.float64)
    angles = (2.0 * math.pi / p) * sums.to(torch.float64).unsqueeze(1) * frequencies
    return torch.stack((angles.cos(), angles.sin()), dim=2)


def _select(scores: torch.Tensor, top_k: int) -> list[int]:
    # Python's stable ordering plus the second key fixes exact-score ties by k.
    order = sorted(range(scores.numel()), key=lambda i: (-float(scores[i]), i + 1))
    return [index + 1 for index in order[:top_k]]


def _score_targets(x_fit: torch.Tensor, x_eval: torch.Tensor,
                   targets: torch.Tensor, fit: torch.Tensor, evaluation: torch.Tensor,
                   chol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_fit, y_eval = targets[fit], targets[evaluation]
    target_mean = y_fit.mean(dim=0, keepdim=True)
    centered_fit = y_fit - target_mean
    rhs = x_fit.T @ centered_fit.reshape(len(fit), -1) / len(fit)
    coefficients = torch.cholesky_solve(rhs, chol)
    predictions_fit = (x_fit @ coefficients).reshape_as(y_fit) + target_mean
    predictions_eval = (x_eval @ coefficients).reshape_as(y_eval) + target_mean
    fit_sse = ((y_fit - predictions_fit) ** 2).sum(dim=(0, 2))
    eval_sse = ((y_eval - predictions_eval) ** 2).sum(dim=(0, 2))
    fit_reference = ((y_fit - target_mean) ** 2).sum(dim=(0, 2))
    eval_reference = ((y_eval - target_mean) ** 2).sum(dim=(0, 2))
    if bool((fit_reference <= 0).any()) or bool((eval_reference <= 0).any()):
        raise ValueError("a Fourier target has zero reference variation")
    return (1.0 - fit_sse / fit_reference, 1.0 - eval_sse / eval_reference,
            target_mean.squeeze(0))


def _score_rows(fit_scores: torch.Tensor, eval_scores: torch.Tensor,
                target_means: torch.Tensor) -> list[dict[str, float]]:
    return [{"frequency": index + 1, "fit_r2": float(fit_scores[index]),
             "eval_r2": float(eval_scores[index]),
             "fit_target_mean_cos": float(target_means[index, 0]),
             "fit_target_mean_sin": float(target_means[index, 1])}
            for index in range(fit_scores.numel())]


def evaluate_fourier_probes(
    features: torch.Tensor,
    sums: torch.Tensor,
    fit_indices: Sequence[int] | torch.Tensor,
    eval_indices: Sequence[int] | torch.Tensor,
    *,
    p: int = 113,
    ridge: float = 1e-3,
    top_k: int = 5,
    null_seed: int = 20261909,
    n_nulls: int = 20,
    null_permutations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit/select/evaluate paired Fourier ridge probes without evaluation leakage."""
    values = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    labels = torch.as_tensor(sums, dtype=torch.int64, device="cpu").flatten()
    if values.ndim != 2 or values.shape[0] != labels.numel() or values.shape[1] == 0:
        raise ValueError("features must be finite N-by-d and align with sums")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must be finite")
    if p < 3 or p % 2 == 0 or bool((labels < 0).any()) or bool((labels >= p).any()):
        raise ValueError("p must be odd and sums must be in [0,p)")
    frequency_count = (p - 1) // 2
    if ridge <= 0 or top_k < 1 or top_k > frequency_count:
        raise ValueError("ridge/top_k outside the fixed positive range")
    fit = _index_tensor(fit_indices, len(labels), "fit_indices")
    evaluation = _index_tensor(eval_indices, len(labels), "eval_indices")
    if (fit.numel() + evaluation.numel() != len(labels)
            or torch.cat((fit, evaluation)).unique().numel() != len(labels)):
        raise ValueError("fit/eval must be disjoint and cover every row")

    feature_mean = values[fit].mean(dim=0)
    centered_fit = values[fit] - feature_mean
    feature_rms = float(centered_fit.square().mean().sqrt())
    if not math.isfinite(feature_rms):
        raise ValueError("feature RMS is non-finite")
    degenerate = feature_rms == 0.0
    scale = 1.0 if degenerate else feature_rms
    # Centered features/targets give the unpenalized intercept exactly as the
    # restored fit-target mean, while the solve remains d-dimensional.
    x_fit = centered_fit / scale
    x_eval = (values[evaluation] - feature_mean) / scale
    gram = x_fit.T @ x_fit / len(fit)
    gram.diagonal().add_(ridge)
    chol, info = torch.linalg.cholesky_ex(gram)
    if int(info) != 0:
        raise RuntimeError("ridge Gram matrix was not positive definite")

    true_targets = _fourier_targets(labels, p)
    true_fit, true_eval, true_means = _score_targets(
        x_fit, x_eval, true_targets, fit, evaluation, chol)
    selected = _select(true_fit, top_k)
    selected_eval = [float(true_eval[index - 1]) for index in selected]

    permutations = null_permutations
    if permutations is None:
        permutations = make_row_permutations(len(labels), null_seed, n_nulls)
    if len(permutations) != n_nulls:
        raise ValueError("null permutation count mismatch")
    null_rows, null_selected_means = [], []
    for expected_index, record in enumerate(permutations):
        if set(record) != {"null_index", "permutation", "sha256"}:
            raise ValueError("null permutation schema mismatch")
        permutation = _index_tensor(record["permutation"], len(labels), "permutation")
        if permutation.numel() != len(labels):
            raise ValueError("null permutation does not contain every row")
        expected_hash = _int64_hash(("row_permutation", permutation))
        if record["null_index"] != expected_index or record["sha256"] != expected_hash:
            raise ValueError("null permutation identity mismatch")
        null_targets = _fourier_targets(labels[permutation], p)
        null_fit, null_eval, null_means = _score_targets(
            x_fit, x_eval, null_targets, fit, evaluation, chol)
        null_selected = _select(null_fit, top_k)
        selected_mean = sum(float(null_eval[index - 1]) for index in null_selected) / top_k
        null_selected_means.append(selected_mean)
        null_rows.append({"null_index": expected_index, "permutation_sha256": expected_hash,
                          "selected_frequencies": null_selected,
                          "selected_eval_mean_r2": selected_mean,
                          "per_frequency": _score_rows(null_fit, null_eval, null_means)})

    null_values = torch.tensor(null_selected_means, dtype=torch.float64)
    return {
        "schema": "grokking_fourier_probe_v1", "n_rows": len(labels),
        "feature_dimension": values.shape[1], "p": p,
        "frequency_count": frequency_count, "ridge_mean_loss": ridge,
        "top_k": top_k,
        "split": {"fit_count": len(fit), "eval_count": len(evaluation),
                  "sha256": _int64_hash(("fit", fit.sort().values),
                                        ("eval", evaluation.sort().values))},
        "fit_transform": {"feature_mean": feature_mean.tolist(),
                          "feature_rms_scalar": scale,
                          "degenerate_feature_scale": degenerate,
                          "sha256": _float64_hash(("feature_mean", feature_mean),
                                                  ("feature_rms_scalar",
                                                   torch.tensor([scale], dtype=torch.float64)))},
        "observed": {"selected_frequencies": selected,
                     "selected_eval_r2": selected_eval,
                     "selected_eval_mean_r2": sum(selected_eval) / top_k,
                     "per_frequency": _score_rows(true_fit, true_eval, true_means)},
        "null": {"seed": int(null_seed), "n_nulls": n_nulls,
                 "selected_eval_mean_r2_mean": float(null_values.mean()),
                 "selected_eval_mean_r2_std_population": float(null_values.std(unbiased=False)),
                 "selected_eval_mean_r2_max": float(null_values.max()),
                 "permutations_sha256": canonical_permutations_hash(permutations),
                 "runs": null_rows},
    }


def canonical_permutations_hash(permutations: list[dict[str, Any]]) -> str:
    """Hash the ordered null identities without embedding full permutations in results."""
    digest = hashlib.sha256()
    for record in permutations:
        digest.update(str(record["null_index"]).encode("ascii") + b"\0")
        digest.update(record["sha256"].encode("ascii") + b"\0")
    return digest.hexdigest()
