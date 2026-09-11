#!/usr/bin/env python3
"""Exact rational checks for tracking-risk-steelman.md; no scientific data."""
from fractions import Fraction as F
import json


RHO = F(9, 10)
BETA = F(99, 100)


def lag(q: F) -> F:
    return q / (1 - q)


def noise_factor(q: F) -> F:
    return (1 - q) / (1 + q)


def risk(q: F, drift: F, variance: F) -> F:
    return (lag(q) * drift) ** 2 + noise_factor(q) * variance


def main() -> None:
    # These are the fixed points of mean_error=q*(mean_error-drift) and
    # variance=q^2*variance+(1-q)^2*observation_variance.
    for q in (RHO, BETA):
        drift = F(7, 1000)
        observation_variance = F(13, 10)
        mean_error = -lag(q) * drift
        filtered_variance = noise_factor(q) * observation_variance
        assert mean_error == q * (mean_error - drift)
        assert filtered_variance == (q * q * filtered_variance
                                     + (1 - q) ** 2 * observation_variance)

    assert lag(RHO) == 9 and lag(BETA) == 99
    assert noise_factor(RHO) == F(1, 19)
    assert noise_factor(BETA) == F(1, 199)
    variance_penalty = noise_factor(RHO) - noise_factor(BETA)
    squared_lag_reduction = lag(BETA) ** 2 - lag(RHO) ** 2
    assert variance_penalty == F(180, 3781)
    assert squared_lag_reduction == 9720
    threshold = variance_penalty / squared_lag_reduction
    assert threshold == F(1, 204174)

    drift_1, variance_1 = F(1, 100), F(1)
    drift_2, variance_2 = F(0), F(4)
    routed_1 = risk(RHO, drift_1, variance_1)
    routed_2 = risk(BETA, drift_2, variance_2)
    slow_1 = risk(BETA, drift_1, variance_1)
    fast_2 = risk(RHO, drift_2, variance_2)
    assert routed_1 == F(11539, 190000)
    assert slow_1 == F(1960399, 1990000)
    assert routed_2 == F(4, 199)
    assert fast_2 == F(4, 19)
    routed_total = routed_1 + routed_2
    uniform_slow_total = slow_1 + routed_2
    uniform_fast_total = routed_1 + fast_2
    assert routed_total < uniform_fast_total < uniform_slow_total

    adverse_routed = risk(RHO, F(0), F(1)) + risk(BETA, F(1, 100), F(0))
    adverse_slow = risk(BETA, F(0), F(1)) + risk(BETA, F(1, 100), F(0))
    assert adverse_routed - adverse_slow == F(180, 3781)

    displayed = {
        "schema": "i17_tracking_risk_rational_check_v1",
        "status": "pass",
        "parameters": {"rho": str(RHO), "beta": str(BETA)},
        "fast_better_threshold_a2_over_variance": str(threshold),
        "constructive": {
            "routed": float(routed_total),
            "uniform_slow": float(uniform_slow_total),
            "uniform_fast": float(uniform_fast_total),
        },
        "adverse_excess_risk": float(adverse_routed - adverse_slow),
    }
    expected_decimals = {
        "routed": "0.080832", "uniform_slow": "1.005226",
        "uniform_fast": "0.271258", "adverse": "0.047606",
    }
    assert f"{float(routed_total):.6f}" == expected_decimals["routed"]
    assert f"{float(uniform_slow_total):.6f}" == expected_decimals["uniform_slow"]
    assert f"{float(uniform_fast_total):.6f}" == expected_decimals["uniform_fast"]
    assert f"{float(adverse_routed - adverse_slow):.6f}" == expected_decimals["adverse"]
    print(json.dumps(displayed, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
