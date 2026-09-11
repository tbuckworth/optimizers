#!/usr/bin/env python3
"""Exact rational checks only; no datasets, stochastic runs, Torch or GPU."""
from fractions import Fraction as Q
import argparse
import hashlib
import json
from pathlib import Path


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def add(x, y):
    return [a + b for a, b in zip(x, y)]


def sub(x, y):
    return [a - b for a, b in zip(x, y)]


def mv(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    return [[dot(row, col) for col in transpose(b)] for row in a]


def msub(a, b):
    return [sub(x, y) for x, y in zip(a, b)]


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def outer(x):
    return [[a * b for b in x] for a in x]


def mean(values):
    return sum(values) / Q(len(values))


def norm2(x):
    return dot(x, x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise SystemExit("Refusing to overwrite an exact-check record")
    identity = [[Q(1), Q(0)], [Q(0), Q(1)]]
    projectors = [identity, [[Q(0)] * 2 for _ in range(2)],
                  [[Q(1), Q(0)], [Q(0), Q(0)]],
                  [[Q(1, 2), Q(1, 2)], [Q(1, 2), Q(1, 2)]]]
    native = [[Q(1), Q(1, 5)], [Q(1, 5), Q(1, 2)]]
    cases = [([Q(1), Q(0)], [Q(0), Q(1)], [Q(0), Q(2)]),
             ([Q(1), Q(2)], [Q(-2), Q(1)], [Q(2), Q(3)]),
             ([Q(3), Q(2)], [Q(0), Q(0)], [Q(0), Q(2)])]
    hessian = [[Q(2), Q(1)], [Q(1), Q(3)]]
    eta = Q(1, 10)
    records = []
    for case, (mu, bias, xi) in enumerate(cases):
        m = add(mu, bias)
        gradients = [add(m, xi), sub(m, xi)]
        sigma = outer(xi)
        raw_risk = mean([norm2(sub(g, mu)) for g in gradients])
        assert raw_risk == norm2(bias) + trace(sigma)
        for index, a in enumerate(projectors + [native]):
            projected = [mv(a, g) for g in gradients]
            risk = mean([norm2(sub(v, mu)) for v in projected])
            ata = mm(transpose(a), a)
            delta = (norm2(sub(mv(a, m), mu)) - norm2(bias)
                     + trace(mm(msub(ata, identity), sigma)))
            assert risk - raw_risk == delta
            if index < len(projectors):
                assert mm(a, a) == a and transpose(a) == a
                complement = msub(identity, a)
                simplified = (norm2(mv(complement, mu)) - norm2(mv(complement, bias))
                              - trace(mm(complement, sigma)))
                assert simplified == delta
            aha = mm(mm(transpose(a), hessian), a)
            expected_loss = (-eta * dot(mu, mv(a, m))
                             + eta**2 / 2 * (dot(m, mv(aha, m)) + trace(mm(aha, sigma))))
            losses = []
            for v in projected:
                displacement = [-eta * value for value in v]
                losses.append(dot(mu, displacement) + dot(displacement, mv(hessian, displacement)) / 2)
            assert mean(losses) == expected_loss
            records.append({"case": case, "operator": index, "orthogonal_projector": index < 4,
                            "risk_difference": str(delta), "expected_quadratic_loss_change": str(expected_loss)})
    mu = [Q(3), Q(2)]
    raw = [[Q(3), Q(0)], [Q(3), Q(4)]]
    lagged = [[Q(3), Q(0)], [Q(3), Q(0)]]
    restored = [[Q(3), Q(0)], [Q(5), Q(0)]]
    for g, restored_g in zip(raw, restored):
        assert norm2(g) == norm2(restored_g)
    risks = {name: mean([norm2(sub(g, mu)) for g in values])
             for name, values in (("current", raw), ("lagged", lagged), ("restored", restored))}
    assert risks == {"current": Q(4), "lagged": Q(4), "restored": Q(6)}
    output = {"scope": __doc__, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "general_operator_cases": 15, "orthogonal_projector_cases": 12,
              "exact_quadratic_cases": 15, "records": records,
              "sample_dependent_restoration_risks": {k: str(v) for k, v in risks.items()},
              "all_exact_assertions_passed": True}
    rendered = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as handle:
            handle.write(rendered)
        print(json.dumps({"output": str(args.output), "all_exact_assertions_passed": True,
                          "source_sha256": output["source_sha256"]}))
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
