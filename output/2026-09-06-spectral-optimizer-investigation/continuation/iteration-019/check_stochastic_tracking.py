"""Exact rational identities and deterministic numeric bounds; no data/RNG."""
from fractions import Fraction as F
import json
import math


def risk(decay, process, noise):
    return (decay * decay * process + (1 - decay) ** 2 * noise) / (1 - decay ** 2)


def main():
    checks = 0
    # Finite-support tails are legitimate signed-kernel fixtures. Boundary c0=1.
    for q in (F(1, 4), F(1, 2), F(9, 10), F(99, 100)):
        for noise in (F(1), F(5)):
            process = noise * (1 - q) ** 2 / q
            for interior in ([], [F(1, 3)], [F(-1, 2), F(4, 3)],
                             [F(1), F(-2), F(3), F(-1)]):
                c = [F(1)] + interior + [F(0)]
                direct = process * sum(x * x for x in c[1:])
                direct += noise * sum((a - b) ** 2 for a, b in zip(c, c[1:]))
                certificate = noise * (1 - q)
                certificate += noise / q * sum((b - q * a) ** 2 for a, b in zip(c, c[1:]))
                assert direct == certificate and direct >= noise * (1 - q)
                checks += 1
            assert risk(q, process, noise) == noise * (1 - q)
            checks += 1
    beta = F(99, 100)
    measurement_factor = beta ** 2 + (1 - beta) ** 2 * beta ** 2 / (1 - beta ** 2)
    assert measurement_factor == 2 * beta ** 2 / (1 + beta)
    checks += 1
    threshold = 2 * (1 - beta) * (4 - 1)
    for process in (F(1, 100), threshold, F(1, 10)):
        gap = beta ** 2 / (1 - beta ** 2) * process - measurement_factor * 3
        assert (gap > 0) == (process > threshold)
        assert (gap == 0) == (process == threshold)
        checks += 2
    route = risk(F(9, 10), F(1, 10), F(1)) + risk(beta, F(0), F(4))
    assert route == F(18869, 37810)
    # Positive Riccati polynomial root exceeds a positive candidate iff f(candidate)<0.
    assert route > 0 and route ** 2 + F(1, 10) * route - F(1, 2) < 0
    checks += 2
    optimum = (math.sqrt(.1 ** 2 + 4 * .1 * 5) - .1) / 2
    qopt = 1 - optimum / 5
    assert abs(.1 * qopt - 5 * (1 - qopt) ** 2) < 1e-15
    checks += 1
    print(json.dumps({"status": "pass", "checks": checks,
                      "scope": "exact identities/deterministic constants only; no experiment",
                      "route_exact": str(route), "route_risk": float(route),
                      "common_lti_optimum": optimum, "common_optimal_decay": qopt,
                      "population_diagonal_strong": [float(beta ** 2 / (1-beta ** 2) / 10 + measurement_factor),
                                                     float(4 * measurement_factor)],
                      "selection_threshold": float(threshold)}))


if __name__ == "__main__":
    main()
