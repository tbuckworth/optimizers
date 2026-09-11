"""Focused implementation checks, not the prospective research panel."""
import unittest

import numpy as np
import torch

import run_selective_toy as toy


class ToyChecks(unittest.TestCase):
    def test_fixed_design_gradient_and_hessian(self):
        for angle in toy.ANGLES:
            basis = toy.directions(angle)
            for design, pairs in toy.DESIGNS.items():
                hessian = torch.zeros((2, 2), dtype=torch.float64)
                for pair in pairs:
                    h = torch.tensor(pair, dtype=torch.float64)
                    x = (2 * h).sqrt().unsqueeze(1) * basis.T
                    y = (2 * h).sqrt()
                    theta = torch.tensor([.31, -.14], dtype=torch.float64, requires_grad=True)
                    loss = (x @ theta - y).square().sum() / 4
                    exact, = torch.autograd.grad(loss, theta)
                    torch.testing.assert_close(exact, toy.gradient(theta.detach(), basis, h))
                    hessian += x.T @ x / 8
                torch.testing.assert_close(hessian, torch.eye(2, dtype=torch.float64))

    def test_blocks_and_pairing(self):
        for design in toy.DESIGNS:
            a = toy.batch_order(design, 7, 200)
            np.testing.assert_array_equal(a, toy.batch_order(design, 7, 200))
            np.testing.assert_allclose(a.reshape(-1, 4, 2).mean(1), 1)

    def test_warmup_identity_and_sgd_oracle_freeze(self):
        for optimizer in ["sgd", "adam"]:
            records = [toy.run_one(optimizer, "useful", 30, 999, arm, steps=152)
                       for arm in toy.ARMS]
            for record, curve in records[1:]:
                np.testing.assert_array_equal(curve[:101], records[0][1][:101])
                self.assertEqual(record["batch_sha256"], records[0][0]["batch_sha256"])
            if optimizer == "sgd":
                oracle = records[3][0]
                at100 = next(s for s in oracle["snapshots"] if s["step"] == 100)
                self.assertAlmostEqual(at100["theta_uv"][1], oracle["snapshots"][-1]["theta_uv"][1], places=13)
                self.assertLess(oracle["out_of_applied_subspace_step_energy_fraction"], 1e-20)

    def test_fullbatch_sgd_matches_scalar_formula(self):
        _, curve = toy.run_one("sgd", "none", 30, 999, "raw", steps=152)
        alpha = 1 - (1 - toy.LR) ** np.arange(153)
        np.testing.assert_allclose(curve, .5 * ((1-alpha)**2 + alpha**2), atol=1e-14)

    def test_frozen_basis_stays_fixed(self):
        record, _ = toy.run_one("sgd", "useful", 30, 999, "frozen", steps=152)
        alignments = [s["basis_useful_alignment"] for s in record["snapshots"] if s["step"] >= 100]
        np.testing.assert_array_equal(alignments, np.repeat(alignments[0], len(alignments)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
