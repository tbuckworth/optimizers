#!/usr/bin/env python3
"""Dataset-free CPU tests for the strict iteration-007 state core."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def load_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = load_path(HERE / "state_core.py", "iteration007_state_core_tested")
spectral = load_path(REPO / "spectral_filter.py", "iteration007_canonical_spectral")


def make_model(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(3, 2))


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                             betas=(.9, .999), eps=1e-8, foreach=False, fused=False)


def make_observer(model, optimizer):
    return spectral.SpectralGradientFilter(
        model, optimizer, rank=2, decay=.99, warmup=2, stable_update=True,
        stabilize_every=100, relative_eig_tol=1e-8, absolute_eig_floor=0,
        weighting="hard", normalize="none", adaptive="none")


X = torch.tensor([[1., -2., .5], [-.25, .75, 2.], [2., 1., -1.], [.5, .5, .5]])
Y = torch.tensor([0, 1, 0, 1])


def step(model, optimizer, observer):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X), Y)
    loss.backward()
    observer.filter_grad()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def fixture(completed=4):
    model = make_model(23)
    optimizer = make_optimizer(model)
    observer = make_observer(model, optimizer)
    for _ in range(completed):
        step(model, optimizer, observer)
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    core = s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                          completed_updates=completed)
    return model, optimizer, observer, core


def model_factory_with_rng_noise():
    random.seed(9001)
    np.random.seed(9002)
    torch.manual_seed(9003)
    return torch.nn.Sequential(torch.nn.Linear(3, 2))


def tensor_tree(value):
    if type(value) is torch.Tensor:
        return value.detach().clone()
    if type(value) is dict:
        return {k: tensor_tree(v) for k, v in value.items()}
    if type(value) is list:
        return [tensor_tree(v) for v in value]
    if type(value) is tuple:
        return tuple(tensor_tree(v) for v in value)
    return value


def same(left, right):
    if type(left) is torch.Tensor:
        return (type(right) is torch.Tensor and left.dtype == right.dtype and left.shape == right.shape and
                left.detach().cpu().contiguous().numpy().tobytes() ==
                right.detach().cpu().contiguous().numpy().tobytes())
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return tuple(left) == tuple(right) and all(same(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def live_state(model, optimizer, observer):
    return {
        "parameters": [p.detach().clone() for p in model.parameters()],
        "optimizer": tensor_tree(optimizer.state_dict()),
        "observer": {k: tensor_tree(v) for k, v in vars(observer).items()
                     if k not in s.OBSERVER_ALIASES},
    }


class StateCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("state-core tests require CUDA_VISIBLE_DEVICES=''")
        if torch.cuda.is_initialized():
            raise RuntimeError("state-core tests must not initialize CUDA")

    def test_capture_is_owned_and_preserves_all_rng(self):
        model, optimizer, observer, _ = fixture()
        random.seed(7); np.random.seed(8); torch.manual_seed(9)
        py = random.getstate(); nr = np.random.get_state(); tr = torch.get_rng_state().clone()
        core = s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                              completed_updates=4)
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], nr[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), tr))
        before = core["model"]["parameters"][0]["value"].clone()
        with torch.no_grad():
            next(model.parameters()).add_(10)
            optimizer.state[next(model.parameters())]["exp_avg"].add_(10)
            observer.V.add_(10)
        self.assertTrue(torch.equal(core["model"]["parameters"][0]["value"], before))
        self.assertEqual(core["model"]["gradients"], [None, None])

    def test_restricted_serialized_roundtrip(self):
        _, _, _, core = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.pt"
            s.save_core(path, core)
            loaded = s.load_core(path)
            self.assertTrue(same(core, loaded))
            # This direct call documents that the file itself needs no unsafe globals.
            direct = torch.load(path, weights_only=True, map_location="cpu")
            self.assertTrue(same(core, direct))

    def test_atomic_save_never_replaces_existing_or_raced_target(self):
        _, _, _, core = fixture()
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.pt"
            existing.write_bytes(b"existing-owner")
            with self.assertRaises(FileExistsError):
                s.save_core(existing, core)
            self.assertEqual(existing.read_bytes(), b"existing-owner")
            raced = Path(directory) / "raced.pt"
            real_link = os.link
            def racing_link(source, target):
                Path(target).write_bytes(b"racing-owner")
                return real_link(source, target)
            with mock.patch.object(s.os, "link", side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    s.save_core(raced, core)
            self.assertEqual(raced.read_bytes(), b"racing-owner")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["existing.pt", "raced.pt"])

    def test_restore_overwrites_constructor_rng_and_preserves_continuation(self):
        _, _, _, core = fixture()
        restored = s.restore_core(core, model_factory_with_rng_noise, make_optimizer, make_observer)
        witness = core["rng"]["continuation_witness"]
        self.assertEqual([random.random() for _ in range(4)], witness["python"])
        self.assertEqual([float(x) for x in np.random.random_sample(4)], witness["numpy"])
        self.assertTrue(torch.equal(torch.rand(4, dtype=torch.float64), witness["torch_cpu"]))
        self.assertEqual(set(restored), {"model", "optimizer", "observer"})
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_next_step_replay_including_moments_and_observer(self):
        model, optimizer, observer, core = fixture()
        source_loss = step(model, optimizer, observer)
        source = live_state(model, optimizer, observer)
        restored = s.restore_core(core, model_factory_with_rng_noise, make_optimizer, make_observer)
        replay_loss = step(**restored)
        replay = live_state(**restored)
        self.assertEqual(source_loss, replay_loss)
        self.assertTrue(same(source, replay))
        for item in restored["optimizer"].state.values():
            self.assertEqual(float(item["step"]), 5.)
        self.assertEqual(restored["observer"].step_count, 5)

    def test_missing_basis_roundtrip_uses_canonical_state(self):
        model, optimizer, observer, core = fixture(completed=1)
        self.assertIsNone(observer.V)
        self.assertIsNone(core["observer"]["state"]["V"])
        self.assertIsNone(core["observer"]["state"]["S"])
        restored = s.restore_core(core, model_factory_with_rng_noise, make_optimizer, make_observer)
        self.assertIsNone(restored["observer"].V)
        step(model, optimizer, observer)
        step(**restored)
        self.assertTrue(same(live_state(model, optimizer, observer), live_state(**restored)))

    def test_mutable_branch_clones_are_isolated(self):
        _, _, _, core = fixture()
        left, right = s.clone_tree(core), s.clone_tree(core)
        left["model"]["parameters"][0]["value"].add_(3)
        left["optimizer"]["state"][0]["exp_avg"]["value"].mul_(0)
        self.assertFalse(torch.equal(left["model"]["parameters"][0]["value"],
                                     right["model"]["parameters"][0]["value"]))
        self.assertTrue(torch.equal(right["model"]["parameters"][0]["value"],
                                    core["model"]["parameters"][0]["value"]))
        a = s.restore_core(left, model_factory_with_rng_noise, make_optimizer, make_observer)
        b = s.restore_core(right, model_factory_with_rng_noise, make_optimizer, make_observer)
        self.assertFalse(torch.equal(next(a["model"].parameters()), next(b["model"].parameters())))

    def test_rejects_reorder_unknown_bad_counter_device_and_types(self):
        _, _, _, core = fixture()
        mutations = []
        bad = s.clone_tree(core); bad["model"]["parameters"].reverse(); mutations.append(bad)
        bad = s.clone_tree(core); bad["optimizer"]["param_groups"][0]["param_indices"] = [1, 0]; mutations.append(bad)
        bad = s.clone_tree(core); bad["unknown"] = 1; mutations.append(bad)
        bad = s.clone_tree(core); bad["optimizer"]["state_completed_updates"] = 3; mutations.append(bad)
        bad = s.clone_tree(core); bad["model"]["parameters"][0]["native_device"] = "cuda:0"; mutations.append(bad)
        bad = s.clone_tree(core); bad["schema_version"] = True; mutations.append(bad)
        bad = s.clone_tree(core); bad["state_completed_updates"] = True; mutations.append(bad)
        bad = s.clone_tree(core); bad["phase"] = "post_step"; mutations.append(bad)
        bad = s.clone_tree(core); bad["model"]["gradient_null_mask"][0] = False; mutations.append(bad)
        bad = s.clone_tree(core); bad["model"]["gradient_null_mask"][0] = 1; mutations.append(bad)
        bad = s.clone_tree(core); bad["model"]["gradients"][0] = torch.zeros(2, 3); mutations.append(bad)
        bad = s.clone_tree(core); bad["optimizer"]["param_groups"][0]["decoupled_weight_decay"] = False; mutations.append(bad)
        bad = s.clone_tree(core); bad["observer"]["state"]["S"] = None; mutations.append(bad)
        for index, mutated in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                s.validate_core(mutated)

    def test_rejects_unknown_nested_metadata_and_storage_alias(self):
        _, _, _, core = fixture()
        bad = s.clone_tree(core)
        bad["optimizer"]["state"][0]["exp_avg"]["extra"] = 1
        with self.assertRaises(ValueError):
            s.validate_core(bad)

    def test_rejects_negative_second_moment_and_singular_values(self):
        _, _, _, core = fixture()
        bad = s.clone_tree(core)
        bad["optimizer"]["state"][0]["exp_avg_sq"]["value"].view(-1)[0] = -1
        with self.assertRaises(ValueError):
            s.validate_core(bad)
        bad = s.clone_tree(core)
        bad["observer"]["state"]["S"]["value"][0] = -1
        with self.assertRaises(ValueError):
            s.validate_core(bad)

    def test_rejects_malformed_rng_domains_and_lengths(self):
        _, _, _, core = fixture()
        malformed = []
        bad = s.clone_tree(core); bad["rng"]["python"]["internal"][0] = -1; malformed.append(bad)
        bad = s.clone_tree(core); bad["rng"]["python"]["internal"][624] = 625; malformed.append(bad)
        bad = s.clone_tree(core); bad["rng"]["python"]["gauss_next"] = float("inf"); malformed.append(bad)
        bad = s.clone_tree(core); bad["rng"]["torch_cpu"] = bad["rng"]["torch_cpu"][:-1].clone(); malformed.append(bad)
        bad = s.clone_tree(core); bad["rng"]["continuation_witness"]["python"][0] = 1.; malformed.append(bad)
        bad = s.clone_tree(core); bad["rng"]["continuation_witness"]["numpy"][0] = -.1; malformed.append(bad)
        bad = s.clone_tree(core); bad["rng"]["continuation_witness"]["torch_cpu"][0] = 1.; malformed.append(bad)
        for index, bad in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                s.validate_core(bad)
        rng = s.clone_tree(core["rng"])
        rng["torch_cuda"] = [{"device_index": 0, "name": "fixture", "uuid": "",
                              "state": torch.ones(1, dtype=torch.uint8)}]
        rng["continuation_witness"]["torch_cuda"] = [torch.zeros(4)]
        with self.assertRaises(ValueError):
            s._validate_rng(rng, s.SCIENTIFIC_PROFILE)
        rng["torch_cuda"][0]["uuid"] = "fixture-uuid"
        rng["torch_cuda"][0]["state"] = torch.empty(0, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            s._validate_rng(rng, s.SCIENTIFIC_PROFILE)

    def test_exact_tree_comparison_detects_signed_zero_corruption(self):
        _, _, _, core = fixture()
        positive = s.clone_tree(core)
        negative = s.clone_tree(core)
        positive["model"]["parameters"][0]["value"].view(-1)[0] = 0.0
        negative["model"]["parameters"][0]["value"].view(-1)[0] = -0.0
        self.assertTrue(torch.equal(positive["model"]["parameters"][0]["value"],
                                    negative["model"]["parameters"][0]["value"]))
        self.assertFalse(s._tree_equal(positive, negative))
        bad = s.clone_tree(core)
        bad["optimizer"]["state"][0]["exp_avg_sq"]["value"] = \
            bad["optimizer"]["state"][0]["exp_avg"]["value"]
        with self.assertRaises(ValueError):
            s.validate_core(bad)

    def test_capture_rejects_non_none_gradients_and_bad_live_order(self):
        model, optimizer, observer, _ = fixture()
        F.cross_entropy(model(X), Y).backward()
        with self.assertRaises(ValueError):
            s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                           completed_updates=4)
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["params"].reverse()
        with self.assertRaises(ValueError):
            s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                           completed_updates=4)

    def test_capture_checks_actual_module_class_and_failed_restore_preserves_rng(self):
        class DerivedLinear(torch.nn.Linear):
            pass
        model = torch.nn.Sequential(DerivedLinear(3, 2))
        optimizer = make_optimizer(model)
        observer = make_observer(model, optimizer)
        with self.assertRaises(ValueError):
            s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                           completed_updates=1)
        _, _, _, core = fixture()
        random.seed(51); np.random.seed(52); torch.manual_seed(53)
        py = random.getstate(); nr = np.random.get_state(); tr = torch.get_rng_state().clone()
        def wrong_factory():
            random.random(); np.random.random(); torch.rand(1)
            return torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
        with self.assertRaises(ValueError):
            s.restore_core(core, wrong_factory, make_optimizer, make_observer)
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], nr[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), tr))

    def test_capture_rejects_observer_live_alias_mismatch(self):
        model, optimizer, observer, _ = fixture()
        observer.model = make_model(31)
        with self.assertRaises(ValueError):
            s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                           completed_updates=4)
        observer.model = model
        observer.base_optimizer = make_optimizer(model)
        with self.assertRaises(ValueError):
            s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                           completed_updates=4)
        observer.base_optimizer = optimizer
        observer.param_list = list(reversed(list(model.parameters())))
        with self.assertRaises(ValueError):
            s.capture_core(model, optimizer, observer, profile=s.FIXTURE_PROFILE,
                           completed_updates=4)


if __name__ == "__main__":
    unittest.main()
