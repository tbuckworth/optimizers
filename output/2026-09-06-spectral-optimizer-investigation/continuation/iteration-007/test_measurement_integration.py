"""Parent-authored dataset-free integration of real CPU components.

No scientific seed bundles, datasets, old checkpoints or GPU access. Tiny live
source state, six same-state Adam steps and two independent loss implementations
are tested together; full-shape data below are newly constructed synthetic bytes.
"""
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Explicitly hide CUDA for integration fixtures")

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = load("i7_integrated_state", HERE / "state_core.py")
r = load("i7_integrated_response", HERE / "response_math.py")
l = load("i7_integrated_loss", HERE / "loss_measurements.py")
n = load("i7_integrated_independent", HERE / "independent_numerics.py")
spectral = load("i7_integrated_filter", HERE.parents[3] / "spectral_filter.py")


def make_model():
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                             betas=(.9, .999), eps=1e-8, foreach=False, fused=False)


def make_observer(model, optimizer):
    return spectral.SpectralGradientFilter(model, optimizer, rank=2, decay=.99,
                                           warmup=2, stable_update=True)


def parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def flat(values):
    return torch.cat([x.detach().reshape(-1) for x in values]).clone()


def assign(model, vector):
    offset = 0
    for parameter in model.parameters():
        parameter.grad = vector[offset:offset + parameter.numel()].reshape(parameter.shape).clone()
        offset += parameter.numel()
    assert offset == vector.numel()


def endpoint(model, optimizer):
    return {"parameters": parameters(model), "optimizer": s.clone_tree(optimizer.state_dict())}


class MeasurementIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.x = torch.tensor([[.1, .4, .9], [.7, .2, .5], [.9, .8, .1], [.1, .4, .9]])
        cls.y = torch.tensor([0, 1, 1, 0])

    def compare_loss(self, params, x, y, chunk):
        actual = l.evaluate(params, x, y, chunk_size=chunk)
        reference = n.mlp_ce_gradient([v.numpy() for v in params.values()], x.numpy(), y.numpy(), chunk)
        self.assertLessEqual(abs(actual["mean_cross_entropy"] - reference["mean_ce"]),
                             n.loss_tolerance(actual["mean_cross_entropy"], reference["mean_ce"]))
        q = actual["mean_gradient"].numpy()
        self.assertTrue(np.all(np.abs(q - reference["gradient"]) <= n.gradient_tolerance(q, reference["gradient"])))
        self.assertEqual([row["count"] for row in actual["chunks"]], reference["chunk_counts"])
        for row, mean in zip(actual["chunks"], reference["chunk_means"]):
            self.assertLessEqual(abs(row["mean_cross_entropy"] - mean),
                                 n.loss_tolerance(row["mean_cross_entropy"], mean))
        return actual, reference

    def test_saved_live_source_six_branches_and_independent_measurements(self):
        torch.manual_seed(317)
        model = make_model()
        optimizer = make_optimizer(model)
        observer = make_observer(model, optimizer)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(self.x), self.y).backward()
            observer.filter_grad()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        anchor = s.capture_core(model, optimizer, observer, profile=s.MLP_FIXTURE_PROFILE, completed_updates=4)
        before = parameters(model)
        # The witness comes from the actual live source, BEFORE restoring anything.
        F.cross_entropy(model(self.x), self.y).backward()
        source_raw = flat([p.grad for p in model.parameters()])
        observer.filter_grad()
        source_current = flat([p.grad for p in model.parameters()])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        witness = s.capture_core(model, optimizer, observer, profile=s.MLP_FIXTURE_PROFILE, completed_updates=5)
        live_endpoint = endpoint(model, optimizer)
        self.assertTrue(s._tree_equal(anchor["rng"], witness["rng"]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-anchor.pt"
            s.save_core(path, anchor)
            anchor = s.load_core(path)
        candidate = s.restore_core(anchor, make_model, make_optimizer, make_observer)
        anchor_copy = s.clone_tree(anchor)
        self.assertTrue(s._tree_equal(s._capture_rng(), anchor["rng"]))
        cm, co, cf = candidate["model"], candidate["optimizer"], candidate["observer"]
        previous_basis = cf.V.clone()
        F.cross_entropy(cm(self.x), self.y).backward()
        raw = flat([p.grad for p in cm.parameters()])
        lagged = previous_basis @ (previous_basis.T @ raw)
        cf.filter_grad()  # Exactly one shared candidate observation.
        current = flat([p.grad for p in cm.parameters()])
        self.assertTrue(s._tree_equal(raw, source_raw))
        self.assertTrue(s._tree_equal(current, source_current))
        self.assertTrue(n.audit_projection(raw.numpy(), previous_basis.numpy(), lagged.numpy())["passed"])
        self.assertTrue(n.audit_projection(raw.numpy(), cf.V.numpy(), current.numpy())["passed"])
        self.assertTrue(s._tree_equal(cf.V, witness["observer"]["state"]["V"]["value"]))
        self.assertTrue(s._tree_equal(cf.S, witness["observer"]["state"]["S"]["value"]))
        self.assertEqual(cf.step_count, 5)
        # Observer transition is shared, not repeated once per branch.
        entire_observer = s._capture_observer(cf, cm, co, s.MLP_FIXTURE_PROFILE, 5)
        self.assertTrue(s._tree_equal(entire_observer, witness["observer"]))
        self.assertTrue(s._tree_equal(s._capture_rng(), anchor["rng"]))
        branches = r.construct(raw, current, lagged)["branches"]
        self.assertTrue(s._tree_equal(s._capture_rng(), anchor["rng"]))
        self.assertTrue(all(row["status"] == "defined" for row in branches.values()))

        endpoints = {}
        for order in (r.BRANCHES, tuple(reversed(r.BRANCHES))):
            for name in order:
                restored = s.restore_core(anchor, make_model, make_optimizer, make_observer)
                self.assertTrue(s._tree_equal(s._capture_rng(), anchor["rng"]))
                bm, bo = restored["model"], restored["optimizer"]
                assign(bm, branches[name]["gradient"])
                bo.step()  # Zero is assigned and stepped, never skipped.
                bo.zero_grad(set_to_none=True)
                self.assertTrue(s._tree_equal(s._capture_rng(), anchor["rng"]))
                now = endpoint(bm, bo)
                if name in endpoints:
                    self.assertTrue(s._tree_equal(now, endpoints[name]))
                endpoints[name] = now
                offset = 0
                for index, parameter in enumerate(bm.parameters()):
                    old = anchor["optimizer"]["state"][index]
                    new = bo.state[parameter]
                    gradient = branches[name]["gradient"][offset:offset + parameter.numel()].reshape(parameter.shape)
                    audit = n.audit_adamw(anchor["model"]["parameters"][index]["value"].numpy(),
                        gradient.numpy(), old["exp_avg"]["value"].numpy(), old["exp_avg_sq"]["value"].numpy(), 5,
                        {"theta": parameter.detach().numpy(), "moment": new["exp_avg"].numpy(),
                         "variance": new["exp_avg_sq"].numpy(), "next_step": int(new["step"])},
                        options={key: bo.param_groups[0][key] for key in n.ADAMW_OPTIONS})
                    self.assertTrue(audit["passed"], audit)
                    offset += parameter.numel()
        self.assertTrue(s._tree_equal(endpoints["current"], live_endpoint))
        self.assertTrue(s._tree_equal(anchor, anchor_copy))
        self.assertFalse(s._tree_equal(endpoints["zero"]["parameters"], before))

        probes = {"batch_noisy": (self.x, self.y),
                  "train_probe_noisy": (self.x.clone(), 1 - self.y),
                  "train_probe_clean": (self.x.clone(), self.y.clone()),
                  "auxiliary_clean": (torch.flip(self.x, [0]).clone(), torch.flip(self.y, [0]).clone())}
        for probe, (x, y) in probes.items():
            before_p, before_a = self.compare_loss(before, x, y, 3)
            self.assertEqual(before_p["shape_mode"], "generic_tiny_mlp")
            self.assertTrue(s._tree_equal(s._capture_rng(), anchor["rng"]))
            after_p, after_a = {}, {}
            for name in r.BRANCHES:
                pp, aa = self.compare_loss(endpoints[name]["parameters"], x, y, 3)
                self.assertTrue(s._tree_equal(s._capture_rng(), anchor["rng"]))
                after_p[name], after_a[name] = pp["mean_cross_entropy"], aa["mean_ce"]
                delta = flat(endpoints[name]["parameters"].values()).double() - flat(before.values()).double()
                q, qa = before_p["mean_gradient"], before_a["gradient"]
                observed_dot = float(torch.dot(q, delta))
                independent_dot = math.fsum(float(a) * float(b) for a, b in zip(qa, delta.numpy()))
                ceiling = float((n.gradient_tolerance(q.numpy(), qa) * np.abs(delta.numpy())).sum())
                ceiling += max(n.dot_roundoff(q.numpy(), delta.numpy()), n.dot_roundoff(qa, delta.numpy()))
                self.assertLessEqual(abs(observed_dot - independent_dot), ceiling)
            actual_contrasts = r.contrasts(after_p)
            for key, coefficients in r.COEFFICIENTS.items():
                expected = math.fsum(sign * after_a[name] for name, sign in coefficients.items())
                ceiling = sum(abs(sign) * n.loss_tolerance(after_p[name], after_a[name])
                              for name, sign in coefficients.items())
                ceiling += max(n.sum_roundoff([sign * values[name] for name, sign in coefficients.items()])
                               for values in (after_p, after_a))
                self.assertLessEqual(abs(actual_contrasts[key]["value"] - expected), ceiling, (probe, key))
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_shape_5000_synthetic_examples_ten_chunks(self):
        rng = np.random.default_rng(318)
        params = {key: torch.from_numpy(rng.normal(0, .02, shape).astype(np.float32))
                  for key, shape in zip(l.PARAMETER_ORDER, l.SCIENTIFIC_SHAPES)}
        # Normalize in float32 first, then deliberately duplicate native values.
        base = rng.integers(0, 256, (23, 784), dtype=np.uint8).astype(np.float32) / np.float32(255)
        x = torch.from_numpy(base[rng.integers(0, 23, 5000)].copy())
        y = torch.from_numpy(rng.integers(0, 10, 5000, dtype=np.int64))
        actual, _ = self.compare_loss(params, x, y, 500)
        self.assertEqual(len(actual["chunks"]), 10)
        self.assertEqual(actual["mean_gradient"].shape, (50890,))
        self.assertEqual(actual["shape_mode"], "scientific_784_64_10")
        self.assertFalse(torch.cuda.is_initialized())

    def test_fixture_profile_cannot_be_relabelled_scientific(self):
        torch.manual_seed(319)
        model, optimizer = make_model(), None
        optimizer = make_optimizer(model)
        observer = make_observer(model, optimizer)
        F.cross_entropy(model(self.x), self.y).backward()
        observer.filter_grad()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        core = s.capture_core(model, optimizer, observer, profile=s.MLP_FIXTURE_PROFILE, completed_updates=1)
        for wrong in (s.SCIENTIFIC_PROFILE, s.FIXTURE_PROFILE):
            bad = s.clone_tree(core)
            bad["profile"] = wrong
            with self.assertRaises(ValueError):
                s.validate_core(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
