"""Deterministic algebra check for the prospective normalized-response note.

No dataset, model, random sampling, training, file output or external service.
"""
import json
import math


def check():
    beta, rho = .99, .9
    result = []
    for k in (0., .5, .9, 1.):
        r = rho * k
        a = k - (1-k)*(1-beta)*r/(beta-r)
        c = (1-k)*(1-beta)*beta/(beta-r)
        weights = [(1-r)*(a*r**n + c*beta**n) for n in range(10000)]
        mass = math.fsum(weights)
        lag = math.fsum(n*w for n, w in enumerate(weights))
        variance = (1-r)**2 * (a*a/(1-r*r) + c*c/(1-beta*beta)
                              + 2*a*c/(1-r*beta))
        expected_lag = r/(1-r) + (1-k)*beta/(1-beta)
        for actual, expected in ((mass, 1.), (lag, expected_lag),
                                 (math.fsum(w*w for w in weights), variance)):
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError((k, actual, expected))
        if min(weights) < 0:
            raise AssertionError("Kernel coefficient is negative")
        result.append({"k": k, "dc_gain": mass, "mean_lag": lag,
                       "white_noise_variance_ratio": variance,
                       "first_weight": weights[0]})
    # The explicitly noted counterexample to ordering by mean lag.
    if not (result[1]["mean_lag"] > result[3]["mean_lag"] and
            result[1]["white_noise_variance_ratio"] >
            result[3]["white_noise_variance_ratio"]):
        raise AssertionError("Lag/variance counterexample differs")
    return result


if __name__ == "__main__":
    print(json.dumps({"status": "pass", "rows": check()}, indent=2))
