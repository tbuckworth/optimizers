#!/usr/bin/env python3
"""Exact prospective algebra checks; no RNG, native observer, or I18 files."""
from fractions import Fraction as F


def ema(values, q):
    state = values[0]
    result = [state]
    for value in values[1:]:
        state = q * state + (1 - q) * value
        result.append(state)
    return result


def trend_corrected(values, q):
    first = ema(values, q)
    second = ema(first, q)
    return [2 * left - right for left, right in zip(first, second)]


def variance_factor(q):
    return (1 - q) * (1 + 4 * q + 5 * q * q) / (1 + q) ** 3


def check():
    identities = 0
    for q in (F(1, 2), F(9, 10), F(99, 100)):
        alpha = 1 - q
        # Exact infinite sums of the signed impulse response.
        weight_sum = alpha * ((1 + q) / (1 - q)
                              - alpha * q / (1 - q) ** 2)
        first_moment = alpha * ((1 + q) * q / (1 - q) ** 2
                                - alpha * q * (1 + q) / (1 - q) ** 3)
        assert weight_sum == 1
        assert first_moment == 0

        r = q * q
        direct_squares = alpha**2 * (
            (1 + q) ** 2 / (1 - r)
            - 2 * (1 + q) * alpha * r / (1 - r) ** 2
            + alpha**2 * r * (1 + r) / (1 - r) ** 3)
        assert direct_squares == variance_factor(q)
        identities += 3

        drift, intercept = F(3, 100), F(-7, 5)
        values = [intercept + drift * n for n in range(31)]
        outputs = trend_corrected(values, q)
        for n, (value, output) in enumerate(zip(values, outputs)):
            assert output - value == -drift * n * q ** (n + 1)
            identities += 1

    q = F(99, 100)
    alpha = 1 - q
    weight = lambda lag: alpha * q**lag * ((1 + q) - alpha * lag)
    assert weight(199) == 0
    assert weight(200) < 0
    identities += 2

    corrected_risk = 5 * variance_factor(q)
    routed_risk = F(3, 100) ** 2 * 9**2 + F(1, 19) + F(4, 199)
    ordinary_zero_drift_risk = 5 * (1 - q) / (1 + q)
    assert corrected_risk == F(493025, 7880599)
    assert routed_risk == F(5506349, 37810000)
    assert corrected_risk < routed_risk
    assert ordinary_zero_drift_risk < corrected_risk
    identities += 4

    print({"status": "pass", "exact_identities": identities,
           "trend_corrected_risk_q0p99": float(corrected_risk),
           "fixed_route_risk": float(routed_risk),
           "zero_drift_ema_q0p99_risk": float(ordinary_zero_drift_risk),
           "scientific_draws": 0, "i18_artifacts_read": 0})


if __name__ == "__main__":
    check()
