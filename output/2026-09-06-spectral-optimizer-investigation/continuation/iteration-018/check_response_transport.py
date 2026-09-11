#!/usr/bin/env python3
"""Exact arithmetic fixtures only; no native filter, RNG, or scientific files."""
from fractions import Fraction as F


def scale(c, a):
    return [[c * value for value in row] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(left, right)] for left, right in zip(a, b)]


def sub(a, b):
    return add(a, scale(-1, b))


def mul(a, b):
    return [[sum(x * y for x, y in zip(row, column)) for column in zip(*b)] for row in a]


def inv(a):
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    assert determinant != 0
    return scale(1 / determinant, [[a[1][1], -a[0][1]], [-a[1][0], a[0][0]]])


def check():
    eye = [[F(1), F(0)], [F(0), F(1)]]
    rho = F(9, 10)
    e1 = [[F(1), F(0)], [F(0), F(0)]]
    e2 = sub(eye, e1)
    oblique_direction = [[F(9, 25), F(12, 25)], [F(12, 25), F(16, 25)]]
    non_projector = [[F(1, 2), F(1, 4)], [F(1, 4), F(3, 4)]]
    sequences = ((e1, e1, e1, e1), (e1, e2, e1, e2),
                 (e1, oblique_direction, e2, oblique_direction),
                 (eye, non_projector, e1, non_projector))
    identities = 0
    for actions in sequences:
        d = cp = [[F(2)], [F(-3)]]
        prior_q = sub(eye, scale(rho, actions[0]))
        b = mul(inv(prior_q), d)
        for index, action in enumerate(actions[1:], 1):
            prior_action = actions[index - 1]
            q = sub(eye, scale(rho, action))
            h = [[F(index, 3)], [F(2 - index, 5)]]
            transport = scale(-rho**2, mul(mul(action, sub(action, prior_action)), inv(prior_q)))
            coefficient = sub(scale(rho, mul(mul(q, action), inv(prior_q))), scale(rho, action))
            assert coefficient == transport
            next_b = add(scale(rho, mul(action, b)), h)
            next_d = mul(q, next_b)
            next_cp = add(scale(rho, mul(action, cp)), mul(q, h))
            assert next_d == add(add(scale(rho, mul(action, d)), mul(q, h)), mul(transport, d))
            assert sub(next_d, next_cp) == add(scale(rho, mul(action, sub(d, cp))), mul(transport, d))
            constant = [[F(5)], [F(-7)]]
            assert add(scale(rho, mul(action, constant)), mul(q, constant)) == constant
            identities += 4
            b, d, cp, prior_q = next_b, next_d, next_cp, q
    print({"status": "pass", "exact_identities": identities,
           "scientific_draws": 0, "native_updates": 0})


if __name__ == "__main__":
    check()
