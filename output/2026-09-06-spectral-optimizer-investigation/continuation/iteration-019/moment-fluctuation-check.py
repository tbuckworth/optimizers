#!/usr/bin/env python3
"""Exact I19 stationary moment-fluctuation identities; no RNG or experiment."""
from fractions import Fraction as F
import json


def gamma(beta, process, measurement, lag):
    if lag == 0:
        return (beta ** 2 * process / (1 - beta ** 2)
                + 2 * beta ** 2 * measurement / (1 + beta))
    return (beta ** (lag + 2) * process / (1 - beta ** 2)
            - (1 - beta) * beta ** (lag + 1) * measurement / (1 + beta))


def variance_diagonal(beta, gamma0, gamma1):
    alpha = 1 - beta
    return alpha / (1 + beta) * (
        2 * gamma0 ** 2 + 4 * gamma1 ** 2 * beta / (1 - beta ** 3))


def variance_off_diagonal(beta, gamma10, gamma20, gamma11, gamma21):
    alpha = 1 - beta
    return alpha / (1 + beta) * (
        gamma10 * gamma20
        + 2 * gamma11 * gamma21 * beta / (1 - beta ** 3))


def main():
    beta = F(99, 100)
    q = (F(1, 10), F(0))
    r = (F(1), F(4))
    g0 = tuple(gamma(beta, qi, ri, 0) for qi, ri in zip(q, r))
    g1 = tuple(gamma(beta, qi, ri, 1) for qi, ri in zip(q, r))
    for coordinate in range(2):
        for lag in range(1, 12):
            assert gamma(beta, q[coordinate], r[coordinate], lag) \
                   == beta ** (lag - 1) * g1[coordinate]
    variances = tuple(variance_diagonal(beta, g0[i], g1[i]) for i in range(2))
    covariance = variance_off_diagonal(beta, g0[0], g0[1], g1[0], g1[1])
    assert all(value >= 0 for value in (*variances, covariance))
    for mean_decay in (F(1, 2), F(9, 10), F(99, 100)):
        pure0 = gamma(mean_decay, F(1), F(0), 0)
        pure1 = gamma(mean_decay, F(1), F(0), 1)
        assert pure1 == mean_decay * pure0
        for moment_decay in (F(1, 2), mean_decay, F(999, 1000)):
            direct = (1-moment_decay)/(1+moment_decay) * (
                2*pure0**2 + 4*moment_decay*pure1**2/(1-moment_decay*mean_decay**2)) / pure0**2
            simplified = 2*(1-moment_decay)/(1+moment_decay) * (
                (1+moment_decay*mean_decay**2)/(1-moment_decay*mean_decay**2))
            assert direct == simplified
        matched = variance_diagonal(mean_decay, pure0, pure1) / pure0**2
        assert matched == 2*(1+mean_decay**3)/((1+mean_decay)*(1+mean_decay+mean_decay**2))
    # Limits evaluated in the nonsingular simplified expressions.
    assert F(2)*(1+F(1)**3)/((1+F(1))*(1+F(1)+F(1)**2)) == F(2,3)
    result = {
        "status": "pass",
        "scope": "stationary Gaussian infinite-past exact arithmetic; no RNG or experiment",
        "beta": str(beta),
        "gamma0": [str(value) for value in g0],
        "gamma1": [str(value) for value in g1],
        "population_gap": str(g0[0] - g0[1]),
        "variance_M11": str(variances[0]),
        "variance_M22": str(variances[1]),
        "variance_M12": str(covariance),
        "matched_pure_process_cv_squared_limit": "2/3",
        "numeric": {
            "gamma0": [float(value) for value in g0],
            "gamma1": [float(value) for value in g1],
            "population_gap": float(g0[0] - g0[1]),
            "sd_M11": float(variances[0] ** F(1, 2)),
            "sd_M22": float(variances[1] ** F(1, 2)),
            "sd_M12": float(covariance ** F(1, 2)),
        },
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
