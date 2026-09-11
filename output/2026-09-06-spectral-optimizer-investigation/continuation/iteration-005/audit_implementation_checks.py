#!/usr/bin/env python3
"""Independent small CPU identities only; never import the experiment launcher.

No MNIST, GPU, training, worst-size arrays, or artifact-store writes occur.
Comparison expectations are derived by explicit dense matrices. Output is JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
import reference_math as reviewed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    count = 0
    max_error = {}

    def compare(label, observed, expected, atol=1e-10, rtol=1e-8):
        nonlocal count
        a, b = float(observed), float(expected)
        error = abs(a - b)
        assert error <= atol + rtol * abs(b), (label, a, b, error)
        max_error[label] = max(max_error.get(label, 0.), error)
        count += 1

    for seed in range(8):
        generator = torch.Generator().manual_seed(80500 + seed)
        dimension, observations, rank = 9, 18, 3
        first = 1 + seed % 4
        beta = [.9, .99, .999][seed % 3]
        innovations = torch.randn(observations, dimension, generator=generator)
        innovations[:first - 1] = 0
        # Explicit chronological recurrence, independent of the weight helper.
        covariance = torch.outer(innovations[first - 1].double(), innovations[first - 1].double())
        for value in innovations[first:]:
            covariance = beta * covariance + (1 - beta) * torch.outer(value.double(), value.double())
        x, weights = reviewed.weighted_columns(innovations, first, beta)
        compare("weighted_dense_reconstruction", torch.linalg.vector_norm(x @ x.T - covariance), 0., atol=1e-11)
        compare("weight_sum", weights.sum(), 1.)
        values, vectors = torch.linalg.eigh(covariance)
        optimum = values[-rank:].sum()
        optimal_projector = vectors[:, -rank:] @ vectors[:, -rank:].T
        reference, tensors = reviewed.reference(x, first, rank=rank)
        compare("reference_optimal_energy", reference["metrics"]["optimal_rank32_energy"], optimum)
        compare("reference_trace", reference["metrics"]["covariance_trace"], torch.trace(covariance))
        compare("reference_squared_frobenius", reference["metrics"]["covariance_squared_frobenius"], covariance.square().sum())
        assert reference["diagnostics"]["reference_projector_valid"]
        count += 1
        clean = torch.randn(dimension, generator=generator)
        noisy = torch.randn(dimension, generator=generator)
        probes = {"clean": clean, "noisy": noisy, "corruption_residual": noisy - clean,
                  "auxiliary_clean": torch.randn(dimension, generator=generator)}
        raw = torch.randn(dimension, generator=generator)
        for width in (3, 6):
            q, _ = torch.linalg.qr(torch.randn(dimension, width, generator=generator).double())
            # Deliberate small nonorthogonality; never treat this as an exact projector.
            basis = q.float()
            basis[:, 0] *= 1.0002
            basis[:, 1] += .0001 * basis[:, 0]
            singular = torch.linspace(.5, 1.3, width, dtype=torch.float64)
            current = basis[:, :rank]
            row = reviewed.observer_metrics(x, reference, tensors, {"V": basis, "S": singular},
                                            None, probes, raw, width, rank=rank)
            full64, current64 = basis.double(), current.double()
            covariance_estimator = full64 @ torch.diag(singular.square()) @ full64.T
            operator = current64 @ current64.T
            native_span, _ = torch.linalg.qr(current64)
            span_projector = native_span @ native_span.T
            compare("covariance_error", row["metrics"]["relative_covariance_error"],
                    torch.linalg.vector_norm(covariance - covariance_estimator) / torch.linalg.vector_norm(covariance))
            compare("estimated_covariance_norm", row["energies"]["estimator_squared_frobenius"], covariance_estimator.square().sum())
            compare("covariance_inner_product", row["energies"]["covariance_inner_product"], (covariance * covariance_estimator).sum())
            compare("represented_output_energy", row["energies"]["represented_operator_output_energy"], torch.trace(operator @ covariance @ operator))
            compare("trace_pc", row["metrics"]["trace_P_C"], torch.trace(operator @ covariance))
            compare("span_fraction", row["metrics"]["span_energy_fraction"], torch.trace(span_projector @ covariance) / optimum)
            compare("projector_distance", row["metrics"]["span_projector_distance"],
                    (span_projector - optimal_projector).square().sum() / (2 * rank))
            for name, value in probes.items():
                transformed = current @ (current.T @ value)
                expected = transformed.double().square().sum() / value.double().square().sum()
                compare("native_probe_retention", row["metrics"][f"native_{name}_retention"], expected)
            actual = current @ (current.T @ raw)
            expected = actual.double().square().sum() / raw.double().square().sum() - 1.
            compare("self_inclusion_previous_identity", row["metrics"]["self_inclusion_retention_increment"], expected)
    # A tie invalidates a unique reference projector, but not matched-span energy.
    x = torch.diag(torch.tensor([3., 2., 2., 1.], dtype=torch.float64))
    ref, tensors = reviewed.reference(x, 1, rank=2)
    assert tensors["projector_basis"] is None
    compare("tie_optimal_energy", ref["metrics"]["optimal_rank32_energy"], 13.)
    count += 1
    # Exact-zero reference with no observer basis keeps native identity semantics.
    x, _ = reviewed.weighted_columns(torch.zeros(4, 4), None)
    ref, tensors = reviewed.reference(x, None, rank=2)
    clean, noisy = torch.arange(1., 5.), torch.arange(4., 0., -1.)
    probes = {"clean": clean, "noisy": noisy, "corruption_residual": noisy - clean,
              "auxiliary_clean": clean.flip(0)}
    row = reviewed.observer_metrics(x, ref, tensors, {"V": None, "S": None}, None, probes, noisy, 2, rank=2)
    compare("absent_basis_identity", row["metrics"]["native_clean_retention"], 1.)
    assert row["metrics"]["relative_covariance_error"] is None
    assert row["metrics"]["span_energy_fraction"] is None
    assert row["metrics"]["span_projector_distance"] is None
    assert set(row["null_reasons"]) == {name for name, value in row["metrics"].items() if value is None}
    count += 4
    result = {"status": "passed", "checks": count, "random_streams": 8,
              "observer_cases": 16, "maximum_dimensions": {"p": 9, "observations": 18},
              "device": "cpu", "threads": 1, "elapsed_seconds": time.time() - started,
              "maximum_absolute_errors": max_error, "audit_source_sha256": sha(__file__),
              "reference_math_sha256": sha(Path(reviewed.__file__)),
              "scope": "Dense algebra and null/action checks only; no MNIST, training, GPU or persistence writes."}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
