#!/usr/bin/env python3
"""Exact deterministic fixtures for the I20 response note; no RNG or arrays."""
from fractions import Fraction as F
import json


ZERO = F(0)
ONE = F(1)
I = ((ONE, ZERO), (ZERO, ONE))


def vadd(x, y):
    return tuple(a + b for a, b in zip(x, y))


def vsub(x, y):
    return tuple(a - b for a, b in zip(x, y))


def vscale(a, x):
    return tuple(a * b for b in x)


def mscale(a, matrix):
    return tuple(tuple(a * value for value in row) for row in matrix)


def madd(left, right):
    return tuple(tuple(a + b for a, b in zip(x, y))
                 for x, y in zip(left, right))


def msub(left, right):
    return madd(left, mscale(-ONE, right))


def mv(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), ZERO)
                 for row in matrix)


def norm_squared(vector):
    return sum((value * value for value in vector), ZERO)


def outer(vector):
    return tuple(tuple(a * b for b in vector) for a in vector)


def projector(vector):
    denominator = norm_squared(vector)
    return mscale(ONE / denominator, outer(vector))


def cp_step(previous, mean, gradient, projection, rho):
    complement = msub(I, projection)
    return vadd(
        vadd(vscale(rho, mv(projection, previous)),
             vscale(ONE - rho, mv(projection, gradient))),
        mv(complement, mean),
    )


def smoother_step(previous, gradient, projection, rho, gamma):
    complement = msub(I, projection)
    decay = madd(mscale(rho, projection), mscale(gamma, complement))
    return vadd(mv(decay, previous), mv(msub(I, decay), gradient))


def update_mean(previous, gradient, beta):
    return vadd(vscale(beta, previous), vscale(ONE - beta, gradient))


def response_fixtures():
    rho = F(9, 10)
    beta = F(99, 100)
    fixed = projector((F(3), F(4)))
    moving = (projector((F(1), F(0))), fixed,
              projector((F(4), F(-3))), projector((F(5), F(12))))
    gradients = ((F(2), F(-1)), (F(3), F(5)), (F(-2), F(7)), (F(11), F(1)))

    # Fixed-projector CP and the gamma=beta smoother are exactly identical.
    mean = gradients[0]
    cp = gradients[0]
    smooth = gradients[0]
    for gradient in gradients[1:]:
        mean = update_mean(mean, gradient, beta)
        cp = cp_step(cp, mean, gradient, fixed, rho)
        smooth = smoother_step(smooth, gradient, fixed, rho, beta)
        assert cp == smooth

    # The moving-projector difference identity holds at every step.
    mean = gradients[0]
    cp = gradients[0]
    smooth = gradients[0]
    nonzero_transport_seen = False
    for index, gradient in enumerate(gradients[1:], 1):
        prior_mean, prior_cp = mean, cp
        mean = update_mean(mean, gradient, beta)
        cp = cp_step(cp, mean, gradient, moving[index], rho)
        new_smooth = smoother_step(smooth, gradient, moving[index], rho, beta)
        delta = vsub(smooth, prior_cp)
        complement = msub(I, moving[index])
        decay = madd(mscale(rho, moving[index]), mscale(beta, complement))
        transport = vscale(beta, mv(complement, vsub(prior_cp, prior_mean)))
        predicted = vadd(mv(decay, delta), transport)
        assert vsub(new_smooth, cp) == predicted
        nonzero_transport_seen |= any(value != 0 for value in transport)
        smooth = new_smooth
    assert nonzero_transport_seen
    moving_final = (cp, smooth)

    # Constant preservation is pathwise for both responses.
    constant = (F(7, 3), F(-5, 2))
    mean = cp = smooth = constant
    for projection in moving:
        mean = update_mean(mean, constant, beta)
        cp = cp_step(cp, mean, constant, projection, rho)
        smooth = smoother_step(smooth, constant, projection, rho, beta)
        assert mean == cp == smooth == constant

    # Conditional contraction around a constant, using exact squared norms.
    state = (F(13, 7), F(-9, 5))
    maximum = max(rho, beta)
    for projection in moving:
        next_state = smoother_step(state, constant, projection, rho, beta)
        assert norm_squared(vsub(next_state, constant)) \
               <= maximum ** 2 * norm_squared(vsub(state, constant))
        state = next_state

    # rho=gamma=.9 erases every projector exactly and equals the scalar EMA.
    state = gradients[0]
    scalar = gradients[0]
    for index, gradient in enumerate(gradients[1:], 1):
        state = smoother_step(state, gradient, moving[index], rho, rho)
        scalar = vadd(vscale(rho, scalar), vscale(ONE - rho, gradient))
        assert state == scalar

    return moving_final


def moment_fixtures():
    residuals = ((F(0), F(0)), (F(2), F(-1)),
                 (F(1), F(3)), (F(-4), F(2)))

    def standard(decay):
        moment = ((ZERO, ZERO), (ZERO, ZERO))
        values = []
        for step, residual in enumerate(residuals, 1):
            moment = madd(mscale(decay, moment),
                          mscale(ONE - decay, outer(residual)))
            mass = ONE - decay ** step
            normalized = mscale(ONE / mass, moment)
            closed = ((ZERO, ZERO), (ZERO, ZERO))
            for index, old_residual in enumerate(residuals[:step], 1):
                weight = (ONE - decay) * decay ** (step - index)
                closed = madd(closed, mscale(weight, outer(old_residual)))
            assert moment == closed
            values.append((moment, normalized, mass))
        return values

    standard_99 = standard(F(99, 100))
    standard_999 = standard(F(999, 1000))

    legacy = ((ZERO, ZERO), (ZERO, ZERO))
    initialized = False
    legacy_values = []
    for residual in residuals:
        sample = outer(residual)
        if not initialized and any(residual):
            legacy = sample
            initialized = True
        elif initialized:
            legacy = madd(mscale(F(99, 100), legacy),
                           mscale(F(1, 100), sample))
        legacy_values.append(legacy)

    # At the first nonzero residual, legacy and standard .99 differ by 100x.
    assert legacy_values[1] == outer(residuals[1])
    assert standard_99[1][0] == mscale(F(1, 100), outer(residuals[1]))
    assert legacy_values[1] != standard_99[1][0]

    # Normalizing a diagonal moment scales its eigengap by the same mass.
    diagonal = ((F(3, 10), ZERO), (ZERO, F(1, 10)))
    mass = F(7, 20)
    raw = mscale(mass, diagonal)
    raw_gap = raw[0][0] - raw[1][1]
    normalized_gap = diagonal[0][0] - diagonal[1][1]
    assert raw_gap / mass == normalized_gap
    tolerance = F(1, 10)
    normalized_available = normalized_gap > tolerance * max(ONE, diagonal[0][0])
    rescaled_available = raw_gap > tolerance * max(mass, raw[0][0])
    assert normalized_available == rescaled_available

    # The matched-initialization comparison is standard .999 versus .99.
    assert standard_99[0][0] == standard_999[0][0]
    assert standard_99[1][0] != standard_999[1][0]
    return legacy_values[-1], standard_99[-1][0], standard_999[-1][0]


def risk_fixtures():
    def variation(decay):
        return decay ** 2 / (ONE - decay ** 2)

    def noise(decay):
        return (ONE - decay) / (ONE + decay)

    def risk(alignment, rho, gamma):
        process, r1, r2 = F(1, 10), F(1), F(4)
        return (process * (alignment * variation(rho)
                + (ONE - alignment) * variation(gamma))
                + (alignment * r1 + (ONE - alignment) * r2) * noise(rho)
                + ((ONE - alignment) * r1 + alignment * r2) * noise(gamma))

    rho = F(9, 10)
    assert risk(F(0), rho, rho) == risk(F(1), rho, rho)
    isotropic = F(1, 10) * variation(rho) + F(5) * noise(rho)
    assert risk(F(3, 7), rho, rho) == isotropic == F(131, 190)

    useful_slow = risk(ONE, rho, F(99, 100))
    nuisance_slow = risk(ZERO, rho, F(99, 100))
    assert useful_slow < isotropic < nuisance_slow

    # rho_*=(21-sqrt(41))/20 is represented by its exact rational polynomial.
    # Both roots satisfy it; the stated minus root is the one in (0,1).
    polynomial = lambda x: F(10) * x * x - F(21) * x + F(10)
    lower, upper = F(7298437, 10_000_000), F(729844, 1_000_000)
    assert polynomial(lower) > 0 and polynomial(upper) < 0

    return isotropic, useful_slow, nuisance_slow


def main():
    response = response_fixtures()
    moments = moment_fixtures()
    risks = risk_fixtures()
    result = {
        "status": "pass",
        "scope": "exact Fraction arithmetic on deterministic fixtures; no RNG or scientific arrays",
        "final_response_fixture": [str(value) for value in response[1]],
        "final_moments": {
            "legacy_0.99": [[str(value) for value in row] for row in moments[0]],
            "standard_0.99": [[str(value) for value in row] for row in moments[1]],
            "standard_0.999": [[str(value) for value in row] for row in moments[2]],
        },
        "stationary_risks": {
            "isotropic_0.9": str(risks[0]),
            "useful_0.9_0.99": str(risks[1]),
            "nuisance_0.9_0.99": str(risks[2]),
        },
        "rho_star_polynomial": "10*rho^2-21*rho+10",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
