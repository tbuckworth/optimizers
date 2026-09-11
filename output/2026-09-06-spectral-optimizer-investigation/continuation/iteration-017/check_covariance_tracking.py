"""Exact theory-only checks, not neural or native-observer acquisition."""
from fractions import Fraction as F
import json


def main():
    beta, rho, drift = F(99, 100), F(9, 10), F(3, 100)
    lag = beta / (1 - beta)
    mean_noise_var = (1 - beta) / (1 + beta)
    residual_factor = 1 + mean_noise_var - 2 * (1 - beta)
    assert residual_factor == 2 * beta**2 / (1 + beta)
    first = lag**2 * drift**2 + residual_factor
    second = 4 * residual_factor
    assert first > second
    routed = (rho / (1 - rho) * drift)**2 + (1 - rho) / (1 + rho) + 4 * mean_noise_var
    lower_bound = min(F(5, 29), drift**2 * F(14)**2)
    assert lower_bound == F(5, 29) and lower_bound > routed
    threshold = 3 * residual_factor / lag**2
    assert threshold == 6 * (1 - beta)**2 / (1 + beta)
    assert F(1, 100)**2 < threshold < drift**2
    print(json.dumps({"status": "pass", "evidence": "own conditional rational theory",
                      "population_moment_diagonal": [float(first), float(second)],
                      "routed_risk": float(routed), "all_uniform_ema_risk_lower_bound": float(lower_bound),
                      "population_selection_drift_squared_threshold": float(threshold)}, indent=2))


if __name__ == "__main__":
    main()
