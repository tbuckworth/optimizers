"""Fabricated CPU-only snapshots; no real checkpoint, image or GPU access."""

import copy
import hashlib
import importlib
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import spectral_component_utility_restore as adapter
from experiments import spectral_strong_augmentation_core as strong


def cpu_rng():
    row = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": row[0], "state": torch.tensor(row[1].tolist(), dtype=torch.uint32),
                      "position": int(row[2]), "has_gauss": int(row[3]), "cached_gaussian": float(row[4])},
            "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": []}


def fixture_parent(step=100):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(23)
        model = nn.Sequential(nn.Linear(784, 256, device="cpu", dtype=torch.float32), nn.ReLU(),
                              nn.Linear(256, 128, device="cpu", dtype=torch.float32), nn.ReLU(),
                              nn.Linear(128, 10, device="cpu", dtype=torch.float32))
    model._strong_augmentation_spec = dict(strong.MODEL_SPEC)
    optimizer = strong.make_optimizer(model)
    tracker = strong.make_tracker(model, optimizer)
    if step:
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, .0001 * (index + 1))
            optimizer.state[parameter] = {"step": torch.tensor(float(step), dtype=torch.float32),
                                          "exp_avg": torch.full_like(parameter, .001 * (index + 1)),
                                          "exp_avg_sq": torch.full_like(parameter, .01 * (index + 1))}
        tracker.step_count = step
        tracker.grad_mean = torch.linspace(-.01, .01, strong.PARAMETER_COUNT, dtype=torch.float32)
        if step > 1:
            tracker.V = torch.zeros(strong.PARAMETER_COUNT, 2, dtype=torch.float32)
            tracker.V[0, 0], tracker.V[1, 1] = 1., 1.
            tracker.S = torch.tensor([.3, .2], dtype=torch.float64)
    return model, optimizer, tracker


def fabricated_snapshot(step=100):
    parent = fixture_parent(step)
    with patch.object(strong.neural_core, "_rng_state", return_value=cpu_rng()):
        return parent, strong.snapshot(*parent)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        # CPU fixtures fail if any restoration/fixture code inspects or touches
        # CUDA. Merely importing torch.cuda is not an accelerator operation.
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("is_available", "device_count", "get_rng_state_all", "set_rng_state_all",
                     "manual_seed_all", "_lazy_init"):
            self.stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError("CUDA call: " + name)))

    def test_import_is_inert(self):
        with patch.object(nn, "Linear", side_effect=AssertionError("model on import")), \
                patch.object(torch, "load", side_effect=AssertionError("load on import")), \
                patch.object(torch, "manual_seed", side_effect=AssertionError("seed on import")):
            importlib.reload(adapter)

    def test_roundtrip_exact_initial_warmup_final_and_distinct_modes(self):
        for step in (0, 100, 56304):
            parent, saved = fabricated_snapshot(step)
            saved["model_modes"] = [("", False), ("0", True), ("1", False),
                                    ("2", True), ("3", False), ("4", True)]
            digest = strong.tree_digest(saved)
            restored = adapter.restore(saved, expected_step=step)
            actual = adapter.snapshot_with_rng(*restored, saved["rng"])
            self.assertTrue(strong.neural_core.equal_tree(saved, actual))
            self.assertEqual(strong.tree_digest(saved), digest)
            self.assertEqual(sum(p.numel() for p in restored[0].parameters()), 235146)
            self.assertEqual([m.training for m in restored[0].modules()], [False, True, False, True, False, True])

    def test_global_python_numpy_torch_rng_unchanged_on_success(self):
        _, saved = fabricated_snapshot()
        # Recorded state differs from ambient, so a covert _restore_rng call is
        # detectable rather than accidentally restoring an identical state.
        random.random(); np.random.random(); torch.rand(2)
        before = strong.tree_digest(cpu_rng())
        adapter.restore(saved, expected_step=100)
        self.assertEqual(strong.tree_digest(cpu_rng()), before)

    def test_rng_isolation_on_construction_failure(self):
        _, saved = fabricated_snapshot()
        before = strong.tree_digest(cpu_rng())
        original = nn.Linear
        calls = 0

        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("fabricated constructor failure")
            return original(*args, **kwargs)

        with patch.object(nn, "Linear", side_effect=fail), self.assertRaises(RuntimeError):
            adapter.restore(saved, expected_step=100)
        self.assertEqual(strong.tree_digest(cpu_rng()), before)

    def test_ownership_and_no_aliasing_between_input_or_restorations(self):
        _, saved = fabricated_snapshot()
        first = adapter.restore(saved, expected_step=100)
        second = adapter.restore(saved, expected_step=100)
        model, optimizer, tracker = first
        params = tuple(model.parameters())
        self.assertIs(tracker.model, model)
        self.assertIs(tracker.base_optimizer, optimizer)
        self.assertTrue(all(p is q for p, q in zip(params, tracker.param_list)))
        self.assertTrue(all(p is q for p, q in zip(params, optimizer.param_groups[0]["params"])))
        saved_digest = strong.tree_digest(saved)
        other_digest = strong.tree_digest(adapter.snapshot_with_rng(*second, saved["rng"]))
        with torch.no_grad():
            params[0].fill_(3)
            params[0].grad.fill_(4)
            optimizer.state[params[0]]["exp_avg"].fill_(5)
            optimizer.state[params[0]]["exp_avg_sq"].fill_(6)
            optimizer.state[params[0]]["step"].fill_(101)
            tracker.V.fill_(7)
            tracker.S.fill_(8)
            tracker.grad_mean.fill_(9)
        self.assertEqual(strong.tree_digest(saved), saved_digest)
        self.assertEqual(strong.tree_digest(adapter.snapshot_with_rng(*second, saved["rng"])), other_digest)

    def test_rejects_schema_keys_spec_and_parameter_order(self):
        _, source = fabricated_snapshot()
        mutations = [lambda x: x.update(schema="i9_neural_snapshot_v1"),
                     lambda x: x.update(extra=1),
                     lambda x: x["model_spec"].update(hidden_dims=(64, 128)),
                     lambda x: x["model_spec"].update(hidden_dims=[256, 128]),
                     lambda x: x.update(model_state=dict(reversed(list(x["model_state"].items())))),
                     lambda x: x["optimizer"]["param_groups"][0].update(params=[1, 0, 2, 3, 4, 5]),
                     lambda x: x["optimizer"]["param_groups"][0].update(params=[False, 1, 2, 3, 4, 5]),
                     lambda x: x.update(tracker=None)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = copy.deepcopy(source); mutate(value)
                with self.assertRaises(ValueError):
                    adapter.restore(value, expected_step=100)

    def test_rejects_each_optimizer_and_tracker_config_change(self):
        _, source = fabricated_snapshot()
        for section, config in (("optimizer", adapter.GROUP_CONFIG), ("tracker", adapter.TRACKER_CONFIG)):
            for key, expected in config.items():
                with self.subTest(section=section, key=key):
                    value = copy.deepcopy(source)
                    row = value["optimizer"]["param_groups"][0] if section == "optimizer" else value["tracker"]
                    row[key] = (not expected if type(expected) is bool else
                                expected + 1 if type(expected) in (int, float) else "changed")
                    with self.assertRaises(ValueError):
                        adapter.restore(value, expected_step=100)

    def test_rejects_tensor_shape_dtype_nonfinite_missing_gradient_and_moment_sign(self):
        _, source = fabricated_snapshot()
        mutations = [lambda x: x["model_state"].update({"0.weight": torch.zeros(256, 783)}),
                     lambda x: x["model_state"].update({"0.weight": x["model_state"]["0.weight"].double()}),
                     lambda x: x["model_state"]["0.bias"].fill_(float("nan")),
                     lambda x: x["gradients"].__setitem__(0, None),
                     lambda x: x["gradients"][0].fill_(float("inf")),
                     lambda x: x["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")),
                     lambda x: x["optimizer"]["state"][0]["exp_avg_sq"].fill_(-.01),
                     lambda x: x["tracker"].update(V=torch.zeros(strong.PARAMETER_COUNT, 201)),
                     lambda x: x["tracker"].update(S=x["tracker"]["S"].float()),
                     lambda x: x["tracker"]["S"].__setitem__(0, -.1),
                     lambda x: x["tracker"]["S"].__setitem__(0, .1),
                     lambda x: x["tracker"]["grad_mean"].fill_(float("inf"))]
        for mutate in mutations:
            value = copy.deepcopy(source); mutate(value)
            with self.assertRaises(ValueError):
                adapter.restore(value, expected_step=100)

    def test_rejects_counters_modes_and_unknown_tracker_fields(self):
        _, source = fabricated_snapshot()
        mutations = [lambda x: x["optimizer"]["state"][0]["step"].fill_(99),
                     lambda x: x["tracker"].update(step_count=99),
                     lambda x: x["tracker"].update(stabilization_count=-1),
                     lambda x: x["tracker"].update(max_orthogonality_error=float("nan")),
                     lambda x: x["tracker"].update(proj_k=1),
                     lambda x: x["tracker"].update(unknown=1),
                     lambda x: x["model_modes"].__setitem__(0, ("", 1)),
                     lambda x: x["model_modes"].reverse()]
        for mutate in mutations:
            value = copy.deepcopy(source); mutate(value)
            with self.assertRaises(ValueError):
                adapter.restore(value, expected_step=100)
        for count in (True, -1, 56306, 100.0):
            with self.assertRaises(ValueError):
                adapter.restore(source, expected_step=count)

    def test_rejects_invalid_rng_without_mutating_ambient(self):
        _, source = fabricated_snapshot()
        mutations = [lambda x: x["rng"].update(python=(3, (), None)),
                     lambda x: x["rng"]["numpy"].update(position=625),
                     lambda x: x["rng"]["numpy"].update(state=torch.zeros(624, dtype=torch.int64)),
                     lambda x: x["rng"].update(torch_cpu=torch.zeros(7, dtype=torch.uint8)),
                     lambda x: x["rng"].update(torch_cuda=[torch.zeros(16)]),
                     lambda x: x["rng"].update(torch_cuda=[torch.zeros(65537, dtype=torch.uint8)])]
        before = strong.tree_digest(cpu_rng())
        for mutate in mutations:
            value = copy.deepcopy(source); mutate(value)
            with self.assertRaises(ValueError):
                adapter.restore(value, expected_step=100)
            self.assertEqual(strong.tree_digest(cpu_rng()), before)

    def test_opaque_recorded_cuda_bytes_preserved_without_cuda_calls(self):
        _, saved = fabricated_snapshot()
        saved["rng"]["torch_cuda"] = [torch.arange(16, dtype=torch.uint8)]
        actual = adapter.restore(saved, expected_step=100)
        self.assertTrue(strong.neural_core.equal_tree(saved, adapter.snapshot_with_rng(*actual, saved["rng"])))

    def test_native_and_bypass_actions_match_direct_reference_at_boundary_and_repair(self):
        inputs = torch.linspace(-.2, .4, 2 * 784, dtype=torch.float32).reshape(2, 784)
        labels = torch.tensor([2, 7], dtype=torch.int64)
        for step in (99, 100, 199, 56304):
            original, saved = fabricated_snapshot(step)
            for native in (False, True):
                reference = copy.deepcopy(original)
                restored = adapter.restore(saved, expected_step=step)
                for parent in (reference, restored):
                    model, optimizer, tracker = parent
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(model(inputs), labels).backward()
                    if native:
                        tracker.filter_grad()
                    # Adam's CPU implementation asks availability in its graph
                    # capture guard. Answer locally; do not inspect a real GPU.
                    with patch.object(torch.cuda, "is_available", return_value=False):
                        optimizer.step()
                self.assertTrue(strong.neural_core.equal_tree(dict(reference[0].state_dict()), dict(restored[0].state_dict())))
                self.assertTrue(strong.neural_core.equal_tree(reference[1].state_dict(), restored[1].state_dict()))
                self.assertTrue(torch.equal(strong.flat_grad(reference[0]), strong.flat_grad(restored[0])))
                self.assertTrue(strong.neural_core.equal_tree(strong.neural_core._tracker_state(reference[2]),
                                                            strong.neural_core._tracker_state(restored[2])))
                if native and step in (99, 199):
                    self.assertGreater(restored[2].stabilization_count, saved["tracker"]["stabilization_count"])

    def test_explicit_cpu_target_no_implicit_cuda_or_other_backend(self):
        _, saved = fabricated_snapshot()
        for target in ("cuda", "meta", "cpu:0"):
            with self.assertRaises(ValueError):
                adapter.restore(saved, target, expected_step=100)


# Explicit load tests use only temporary files generated from fabricated state.
class RestrictedLoadTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name in ("is_available", "device_count", "get_rng_state_all", "set_rng_state_all", "manual_seed_all", "_lazy_init"):
            self.stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError("CUDA call")))
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="component-restore-fixture-"))
        _, self.saved = fabricated_snapshot()
        self.path = Path(self.directory) / "fabricated.pt"
        torch.save(self.saved, self.path)
        blob = self.path.read_bytes()
        self.receipt = {"expected_size": len(blob), "expected_sha256": hashlib.sha256(blob).hexdigest(), "expected_step": 100}

    def test_restricted_load_cpu_and_exact_roundtrip(self):
        original = torch.load
        calls = []
        def wrapped(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)
        with patch.object(torch, "load", side_effect=wrapped):
            value = adapter.load_snapshot(self.path, **self.receipt)
        self.assertEqual(calls, [{"map_location": "cpu", "weights_only": True}])
        self.assertTrue(strong.neural_core.equal_tree(self.saved, value))

    def test_bad_hash_size_symlink_and_relative_path_do_not_deserialize(self):
        with patch.object(torch, "load", side_effect=AssertionError("must reject before deserialization")):
            for change in ({"expected_size": self.receipt["expected_size"] + 1},
                           {"expected_sha256": "0" * 64}, {"expected_size": adapter.MAX_CHECKPOINT_BYTES + 1}):
                with self.assertRaises(ValueError):
                    adapter.load_snapshot(self.path, **(self.receipt | change))
            with self.assertRaises(ValueError):
                adapter.load_snapshot(Path("relative.pt"), **self.receipt)
            link = Path(self.directory) / "link.pt"; link.symlink_to(self.path)
            with self.assertRaises(OSError):
                adapter.load_snapshot(link, **self.receipt)

    def test_mutation_during_load_is_rejected(self):
        original = torch.load
        def mutate(handle, **kwargs):
            result = original(handle, **kwargs)
            with self.path.open("ab") as output:
                output.write(b"x")
            return result
        with patch.object(torch, "load", side_effect=mutate), self.assertRaises(ValueError):
            adapter.load_snapshot(self.path, **self.receipt)

    def test_restricted_failure_never_falls_back_to_broad_pickle(self):
        with patch.object(torch, "load", side_effect=RuntimeError("restricted loader rejection")) as loader:
            with self.assertRaisesRegex(RuntimeError, "restricted loader rejection"):
                adapter.load_snapshot(self.path, **self.receipt)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.kwargs, {"map_location": "cpu", "weights_only": True})


if __name__ == "__main__":
    unittest.main()
