"""Exact Fraction counterexample; no data, models, tensor state or file output."""
from fractions import Fraction as F


def action(vector, axis):
    return tuple(value if index == axis else F(0) for index, value in enumerate(vector))


def plus(left, right):
    return tuple(x + y for x, y in zip(left, right))


def scale(factor, vector):
    return tuple(factor * value for value in vector)


def check():
    rho, gradient = F(9, 10), (F(1), F(1))
    b1, b2 = (1 + rho, F(1)), (F(1), 1 + rho)
    assert b1 == plus(gradient, scale(rho, action(b2, 0)))
    assert b2 == plus(gradient, scale(rho, action(b1, 1)))
    d1 = plus(b1, scale(-rho, action(b1, 0)))
    d2 = plus(b2, scale(-rho, action(b2, 1)))
    assert d1 == (1 - rho**2, F(1))
    assert d2 == (F(1), 1 - rho**2)
    mean = scale(F(1, 2), plus(d1, d2))
    assert mean == scale(F(119, 200), gradient)
    # Each fixed coordinate projector separately has a constant-state unit response.
    for axis in (0, 1):
        fixed_buffer = tuple(value / (1 - rho) if index == axis else value
                             for index, value in enumerate(gradient))
        assert fixed_buffer == plus(gradient, scale(rho, action(fixed_buffer, axis)))
        assert plus(fixed_buffer, scale(-rho, action(fixed_buffer, axis))) == gradient
        # A distinct, future delivery-state recurrence preserves constants for either action.
        alternative = plus(scale(rho, action(gradient, axis)),
                           plus(gradient, scale(-rho, action(gradient, axis))))
        assert alternative == gradient
    print("PASS: fixed actions have unit constant gain; alternating actions yield exact mean 119/200 = 0.595.")
    print("The distinct delivery-state recurrence preserves the constant state; it is not an I17 arm.")


if __name__ == "__main__":
    check()
