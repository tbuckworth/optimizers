#!/usr/bin/env python3
"""Small CPU float64 algebra checks; no MNIST data, GPU or training."""
import hashlib
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "reference-identity-checks.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite identity checks")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(2026090605)
    innovations = torch.randn(9, 32, generator=generator, dtype=torch.float64)
    innovations[:, :3] = 0.
    checks = []

    def check(name, actual, expected, tolerance=1e-11):
        error = float(torch.linalg.vector_norm((actual - expected).reshape(-1)))
        scale = max(float(torch.linalg.vector_norm(expected.reshape(-1))), 1.)
        normalized = error / scale
        checks.append({"name": name, "absolute_error": error,
                       "normalized_error": normalized, "tolerance": tolerance,
                       "passed": normalized <= tolerance})
        assert normalized <= tolerance, name

    for beta in (.9, .99):
        covariance = torch.zeros(9, 9, dtype=torch.float64)
        first = None
        for index in range(innovations.shape[1]):
            column = innovations[:, index]
            if first is None:
                if float(torch.linalg.vector_norm(column)) == 0:
                    continue
                first = index
                covariance = torch.outer(column, column)
            else:
                covariance = beta * covariance + (1 - beta) * torch.outer(column, column)
            if index not in (3, 7, 15, 31):
                continue
            weights = torch.tensor([beta ** (index - j) * (1. if j == first else 1 - beta)
                                    for j in range(first, index + 1)], dtype=torch.float64)
            x = innovations[:, first:index + 1] * weights.sqrt()
            gram = x.T @ x
            label = f"beta={beta},step={index + 1}"
            check(label + ":weighted_covariance", x @ x.T, covariance)
            vals, vecs = torch.linalg.eigh((gram + gram.T) / 2)
            positive = vals > 1e-12 * vals[-1]
            vals, vecs = vals[positive].flip(0), vecs[:, positive].flip(1)
            u = x @ vecs / vals.sqrt()
            check(label + ":dual_orthogonality", u.T @ u, torch.eye(len(vals), dtype=torch.float64))
            check(label + ":dual_eigen_residual", covariance @ u, u * vals)
            check(label + ":dual_full_reconstruction", (u * vals) @ u.T, covariance)
            q, _ = torch.linalg.qr(torch.randn(9, 3, generator=generator, dtype=torch.float64))
            mixing = torch.eye(3, dtype=torch.float64)
            mixing[0, 1], mixing[1, 1] = .02, 1.01
            v = q @ mixing  # Deliberately nonorthogonal: sum(lambda²) is wrong.
            lam = torch.tensor([1., .5, .2], dtype=torch.float64)
            represented = (v * lam) @ v.T
            g, a = v.T @ v, v.T @ x
            represented_norm = (lam[:, None] * lam[None, :] * g.square()).sum()
            covariance_norm = gram.square().sum()
            cross = (lam[:, None] * a.square()).sum()
            check(label + ":represented_covariance_norm", represented_norm, represented.square().sum())
            check(label + ":covariance_error", covariance_norm + represented_norm - 2 * cross,
                  (covariance - represented).square().sum())
            p = v @ v.T
            check(label + ":actual_operator_energy", (a * (g @ a)).sum(), (p @ x).square().sum())
            qspan, _ = torch.linalg.qr(v)
            k = min(3, len(vals))
            qk, uk = qspan[:, :k], u[:, :k]
            check(label + ":projector_distance", 1 - (qk.T @ uk).square().sum() / k,
                  ((qk @ qk.T - uk @ uk.T).square().sum()) / (2 * k))
    repeated = torch.diag(torch.tensor([4., 3., 2., 2., 1.], dtype=torch.float64))
    eigenvalues = torch.linalg.eigvalsh(repeated).flip(0)
    check("boundary_tie_is_not_unique", eigenvalues[2] - eigenvalues[3], torch.tensor(0., dtype=torch.float64))
    record = {"scope": "Synthetic CPU float64 identities only; no MNIST, GPU or learning result.",
              "source_sha256": source_sha, "torch": torch.__version__, "checks": checks,
              "all_passed": all(c["passed"] for c in checks), "count": len(checks),
              "max_normalized_error": max(c["normalized_error"] for c in checks)}
    with output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: record[k] for k in ("count", "all_passed", "max_normalized_error")}))


if __name__ == "__main__":
    main()
