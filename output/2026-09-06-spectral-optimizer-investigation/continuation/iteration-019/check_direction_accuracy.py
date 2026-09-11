#!/usr/bin/env python3
"""Deterministic fixed-projector risk checks; no RNG or experiment."""
from fractions import Fraction as F
import json
import math


def v(decay):
    return decay * decay / (1 - decay * decay)


def n(decay):
    return (1 - decay) / (1 + decay)


def risk(alignment, fast, slow, process, first_noise, second_noise):
    return (process * (alignment * v(fast) + (1 - alignment) * v(slow))
            + (alignment * first_noise + (1 - alignment) * second_noise) * n(fast)
            + ((1 - alignment) * first_noise + alignment * second_noise) * n(slow))


def main():
    beta = F(99, 100)
    process, first_noise, second_noise = F(1, 10), F(1), F(4)
    checks = 0
    for rho in (F(1, 2), F(9, 10), F(19, 20)):
        useful = risk(F(1), rho, beta, process, first_noise, second_noise)
        nuisance = risk(F(0), rho, beta, process, first_noise, second_noise)
        for alignment in (F(0), F(1, 7), F(1, 2), F(6, 7), F(1)):
            assert risk(alignment, rho, beta, process, first_noise, second_noise) \
                   == alignment * useful + (1 - alignment) * nuisance
            checks += 1
    # If the two response decays coincide, projector orientation is irrelevant.
    equal_decay = risk(F(0), beta, beta, process, first_noise, second_noise)
    for alignment in (F(0), F(1, 3), F(1)):
        assert risk(alignment, beta, beta, process, first_noise, second_noise) == equal_decay
        checks += 1

    common = (math.sqrt(201.0) - 1.0) / 20.0
    rows = []
    for rho in (0.9, (2.1 - math.sqrt(.41)) / 2.0):
        useful = float(risk(1.0, rho, .99, .1, 1.0, 4.0))
        nuisance = float(risk(0.0, rho, .99, .1, 1.0, 4.0))
        threshold = (nuisance - common) / (nuisance - useful)
        assert useful < common < nuisance and 0.0 < threshold < 1.0
        assert math.isclose(risk(threshold, rho, .99, .1, 1.0, 4.0), common,
                            rel_tol=0.0, abs_tol=2e-15)
        rows.append({"rho": rho, "useful_risk": useful,
                     "nuisance_risk": nuisance,
                     "alignment_threshold": threshold,
                     "angle_degrees": math.degrees(math.acos(math.sqrt(threshold)))})
        checks += 2
    print(json.dumps({"status": "pass", "checks": checks,
                      "scope": "fixed data-independent orthoprojector; no RNG or experiment",
                      "common_lti_risk": common, "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
