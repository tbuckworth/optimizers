#!/usr/bin/env python3
"""Tiny independent CPU checks; prints JSON, never trains or writes run artifacts."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np


def rational(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def keep_diagonal(values, width):
    selected = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:width]
    return [value if i in selected else Fraction(0) for i, value in enumerate(values)]


def diagonal_counterexample(beta):
    exact = narrow = wide = [Fraction(0)] * 3
    rows = []
    for step, axis in enumerate((0, 1, 0, 1, 2, 2), 1):
        contribution = [Fraction(int(i == axis)) for i in range(3)]
        exact = [beta * old + new for old, new in zip(exact, contribution)]
        narrow = keep_diagonal([beta * old + new for old, new in zip(narrow, contribution)], 1)
        wide = keep_diagonal([beta * old + new for old, new in zip(wide, contribution)], 2)
        rows.append({"step": step, "axis": "ABC"[axis],
                     "exact_diagonal": list(map(rational, exact)),
                     "width1_diagonal": list(map(rational, narrow)),
                     "width2_diagonal": list(map(rational, wide))})
    assert exact == [beta**5 + beta**3, beta**4 + beta**2, 1 + beta]
    assert narrow == [0, 0, 1 + beta] and wide == exact[:2] + [0]
    assert exact[2] > exact[1] > exact[0] > 1
    capture_narrow, capture_wide, optimum = exact[2], exact[1], max(exact)
    assert capture_narrow > capture_wide
    narrow_error2 = sum((c - a)**2 for c, a in zip(exact, narrow))
    wide_error2 = sum((c - a)**2 for c, a in zip(exact, wide))
    assert narrow_error2 == beta**4 * (1 + beta**2)**3
    assert wide_error2 == (1 + beta)**2
    return {"beta": rational(beta), "steps": rows, "optimum_rank1_energy": rational(optimum),
            "width1_capture": rational(capture_narrow), "width2_capture": rational(capture_wide),
            "width1_capture_fraction": rational(capture_narrow / optimum),
            "width2_capture_fraction": rational(capture_wide / optimum),
            "width2_minus_width1_capture_fraction": rational((capture_wide - capture_narrow) / optimum),
            "width2_capture_fraction_decimal": float(capture_wide / optimum),
            "width1_squared_covariance_error": rational(narrow_error2),
            "width2_squared_covariance_error": rational(wide_error2),
            "width2_covariance_error_smaller": wide_error2 < narrow_error2}


def top(matrix, width):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values, vectors = values[::-1], vectors[:, ::-1]
    return (vectors[:, :width] * np.maximum(values[:width], 0)) @ vectors[:, :width].T


def dense_checks():
    count, regrets, max_recursion, max_bound_violation, min_psd = 0, 0, 0., 0., 0.
    for beta in (.5, .9, .99):
        for width in (1, 2, 4):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                c, a, error = (np.zeros((6, 6)) for _ in range(3))
                for step in range(1, 13):
                    innovation = rng.normal(size=6)
                    addition = (1. if step == 1 else 1 - beta) * np.outer(innovation, innovation)
                    pre = beta * a + addition
                    c = beta * c + addition
                    a_new = top(pre, width)
                    discarded = pre - a_new
                    new_error = c - a_new
                    scale = max(1., np.linalg.norm(c, ord="fro"))
                    recursion = np.linalg.norm(new_error - beta * error - discarded, ord="fro") / scale
                    max_recursion = max(max_recursion, recursion)
                    psd = min(np.linalg.eigvalsh(new_error).min(), np.linalg.eigvalsh(discarded).min()) / scale
                    min_psd = min(min_psd, float(psd))
                    assert recursion < 1e-11 and psd > -1e-11
                    avals, avecs = np.linalg.eigh(a_new)
                    for rank in sorted({1, min(2, width)}):
                        q = avecs[:, -rank:]
                        optimum = np.linalg.eigvalsh(c)[-rank:].sum()
                        captured = np.trace(q.T @ c @ q)
                        regret = optimum - captured
                        kyfan = np.linalg.eigvalsh(new_error)[-rank:].sum()
                        sharper = kyfan - np.trace(q.T @ new_error @ q)
                        violation = (regret - sharper) / scale
                        max_bound_violation = max(max_bound_violation, float(violation))
                        assert regret >= -1e-11 * scale and regret <= sharper + 1e-11 * scale
                        assert sharper <= kyfan + 1e-11 * scale
                        regrets += 1
                    a, error = a_new, new_error
                    count += 1
    return {"streams": 27, "updates": count, "regret_checks": regrets,
            "max_relative_recursion_residual": max_recursion,
            "minimum_normalized_psd_eigenvalue": min_psd,
            "max_normalized_sharper_bound_violation": max_bound_violation,
            "all_passed": True, "floating_tolerance": 1e-11}


def main():
    alpha, delta = Fraction(1, 100), Fraction(1, 10**13)
    determinant = -(alpha * delta)**2
    assert determinant < 0
    cases = [diagonal_counterexample(Fraction(9, 10)),
             diagonal_counterexample(Fraction(99, 100)),
             diagonal_counterexample(Fraction(17, 20))]
    assert [case["width2_covariance_error_smaller"] for case in cases] == [True, True, False]
    assert cases[2]["width1_squared_covariance_error"] == "27318279949649/10240000000000"
    assert cases[2]["width2_squared_covariance_error"] == "1369/400"
    record = {"scope": "Exact rational counterexamples plus tiny NumPy CPU checks; no production execution, MNIST, GPU or training.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy_version": np.__version__,
              "width_counterexamples": cases,
              "dense_theorem_checks": dense_checks(),
              "residual_rejection_counterexample": {"alpha": rational(alpha), "delta": rational(delta),
                                                     "error_determinant": rational(determinant),
                                                     "strictly_indefinite": True}}
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
