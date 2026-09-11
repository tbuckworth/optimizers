"""Exact rational checks of a prospective control; no neural experiment."""
from fractions import Fraction as F


def add(x, y):
    return tuple(a + b for a, b in zip(x, y))


def mul(a, x):
    return tuple(a * b for b in x)


def sub(x, y):
    return add(x, mul(F(-1), y))


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def project(unit, x):
    assert dot(unit, unit) == 1
    return mul(dot(unit, x), unit)


def check():
    rho = F(9, 10)
    axes = ((F(1), F(0)), (F(3, 5), F(4, 5)), (F(0), F(1)), (F(-4, 5), F(3, 5)))
    gradient, delivered = (F(1), F(1)), (F(-2), F(3))
    initial_error = sub(delivered, gradient)
    for step, axis in enumerate(axes * 5, 1):
        previous_error = sub(delivered, gradient)
        delivered = add(add(mul(rho, project(axis, delivered)),
                            mul(1 - rho, project(axis, gradient))),
                        sub(gradient, project(axis, gradient)))
        error = sub(delivered, gradient)
        assert error == mul(rho, project(axis, previous_error))
        assert dot(error, error) <= rho**2 * dot(previous_error, previous_error)
        assert dot(error, error) <= rho**(2 * step) * dot(initial_error, initial_error)
    previous_gradient, gradient = (F(2), F(-1)), (F(-3), F(4))
    previous_delivery, mean = (F(5), F(-2)), (F(1, 3), F(2, 3))
    old_error = sub(previous_delivery, previous_gradient)
    change = sub(gradient, previous_gradient)
    for axis in axes:
        delivery = add(add(mul(rho, project(axis, previous_delivery)),
                           mul(1 - rho, project(axis, gradient))), sub(mean, project(axis, mean)))
        inside = mul(rho, project(axis, sub(old_error, change)))
        mean_error = sub(mean, gradient)
        outside = sub(mean_error, project(axis, mean_error))
        error = sub(delivery, gradient)
        assert error == add(inside, outside)
        assert dot(inside, outside) == 0
        assert dot(error, error) == dot(inside, inside) + dot(outside, outside)
    print("PASS: 20 exact changing-projector contraction steps and four tracking decompositions.")
    print("Theory-only future control; not an I17 acquisition arm or neural result.")


if __name__ == "__main__":
    check()
