"""Bounded synthetic CPU tests for the I14 cross-optimizer runner."""
from __future__ import annotations

import copy
import io
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

import run_cross_optimizer as runner


def _validation(ce: float, accuracy: float) -> dict[str, float]:
    return {
        "clean_ce": float(ce), "soft_ce": float(ce + .1),
        "clean_accuracy": float(accuracy), "mean_max_probability": .6,
        "mean_true_label_probability": .5,
    }


def _calibration_rows() -> list[dict]:
    rows = []
    for seed in runner.plans.CALIBRATION_SEEDS:
        for base in runner.BASES:
            for index, lr in enumerate(runner.RATES[base]):
                # The middle and largest rates tie, so the middle rate must win.
                final_ce = (.4, .3, .3)[index]
                curve = []
                for horizon in runner.plans.HORIZONS:
                    curve.append({"horizon": horizon,
                                  "validation": _validation(
                                      final_ce if horizon == runner.plans.STEPS else .8,
                                      .9)})
                rows.append({
                    "schema": "i14_trajectory_v1", "phase": "calibration",
                    "seed": seed, "base": base, "lr": lr, "policy": "raw",
                    "target": "clean", "status": "complete",
                    "requested_steps": runner.plans.STEPS,
                    "completed_steps": runner.plans.STEPS, "curve": curve,
                })
    return rows


def _tiny_data() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1401)
    x = torch.rand((12, 4), generator=generator)
    clean = torch.randint(2, (12,), generator=generator)
    noisy = 1 - clean
    vx = torch.rand((7, 4), generator=generator)
    vy = torch.randint(2, (7,), generator=generator)
    ax = torch.rand((9, 4), generator=generator)
    ay = torch.randint(2, (9,), generator=generator)
    return {"x": x, "clean": clean, "noisy": noisy,
            "vx": vx, "vy": vy, "ax": ax, "ay": ay}


def _tiny_plan(steps: int, *, seed: int = 21414) -> dict:
    generator = np.random.default_rng(1402)
    return {"phase": "smoke", "seed": seed, "initialization_seed": 1403,
            "training_batches": generator.integers(
                0, 12, (steps, runner.plans.BATCH), dtype=np.int64)}


class _NoWriteRun:
    def __init__(self):
        self.saved = 0
        self.started = 0.0

    def check(self):
        return None

    def save(self, *_args, **_kwargs):
        self.saved += 1
        raise AssertionError("invalid input reached artifact creation")


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise

    def test_choose_rates_requires_both_seeds_and_smaller_exact_tie(self):
        rows = _calibration_rows()
        result = runner.choose_rates(rows)
        self.assertTrue(result["ready_for_confirmation"])
        self.assertEqual(result["selected_rates"], {
            base: runner.RATES[base][1] for base in runner.BASES})
        self.assertFalse(result["confirmation_outcomes_used"])

        weak = copy.deepcopy(rows)
        row = next(item for item in weak if item["seed"] == 190
                   and item["base"] == "sgd"
                   and item["lr"] == runner.RATES["sgd"][1])
        row["curve"][-1]["validation"]["clean_accuracy"] = .849999
        result = runner.choose_rates(weak)
        self.assertEqual(result["selected_rates"]["sgd"], runner.RATES["sgd"][2])

    def test_choose_rates_rejects_missing_nonfinite_and_wrong_identity(self):
        rows = _calibration_rows()
        with self.assertRaisesRegex(ValueError, "membership"):
            runner.choose_rates(rows[:-1])
        duplicate = copy.deepcopy(rows)
        duplicate.append(copy.deepcopy(rows[-1]))
        with self.assertRaisesRegex(ValueError, "membership"):
            runner.choose_rates(duplicate)

        for field, value in (("phase", "confirmation"), ("policy", "current32"),
                             ("target", "fixed")):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "identity"):
                    runner.choose_rates(changed)
        changed = copy.deepcopy(rows)
        changed[0]["curve"][-1]["validation"]["clean_ce"] = float("nan")
        with self.assertRaisesRegex(ValueError, "metric"):
            runner.choose_rates(changed)
        changed = copy.deepcopy(rows)
        changed[0]["curve"].insert(-1, copy.deepcopy(changed[0]["curve"][-1]))
        with self.assertRaisesRegex(ValueError, "horizons"):
            runner.choose_rates(changed)

    def test_no_eligible_base_blocks_confirmation(self):
        rows = _calibration_rows()
        for row in rows:
            if row["base"] == "sgdm":
                row["curve"][-1]["validation"]["clean_accuracy"] = .84
        result = runner.choose_rates(rows)
        self.assertIsNone(result["selected_rates"]["sgdm"])
        self.assertFalse(result["ready_for_confirmation"])

    def test_selection_excludes_h0_and_uses_earliest_independent_ties(self):
        curve = [
            {"horizon": 0, "validation": {"clean_ce": 0., "clean_accuracy": 1.}},
            {"horizon": 100, "validation": {"clean_ce": .4, "clean_accuracy": .8}},
            {"horizon": 250, "validation": {"clean_ce": .4, "clean_accuracy": .8}},
            {"horizon": 500, "validation": {"clean_ce": .2, "clean_accuracy": .7}},
            {"horizon": 1000, "validation": {"clean_ce": .3, "clean_accuracy": .9}},
            {"horizon": 1500, "validation": {"clean_ce": .2, "clean_accuracy": .9}},
        ]
        self.assertEqual(runner.selection(curve), {
            "minimum_validation_ce": 500,
            "maximum_validation_accuracy": 1000,
        })
        with self.assertRaisesRegex(ValueError, "No selectable"):
            runner.selection(curve[:1])

    def test_evaluate_is_state_mode_gradient_and_rng_neutral(self):
        model = runner.core.i9.make_model(144, "cpu", input_dim=4, width=3, classes=2)
        modules = list(model.modules())
        for index, module in enumerate(modules):
            module.training = bool(index % 2)
        for parameter in model.parameters():
            parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
        data = _tiny_data()
        before_state = {key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()}
        before_grads = [parameter.grad.clone() for parameter in model.parameters()]
        before_modes = [module.training for module in modules]
        before_rng = runner.core.i9._rng_state()
        measured = runner.evaluate(model, data)
        self.assertEqual(tuple(measured), ("train", "validation", "auxiliary"))
        after_state = {key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()}
        self.assertTrue(runner.core.i9.equal_tree(before_state, after_state))
        self.assertTrue(all(torch.equal(left, parameter.grad)
                            for left, parameter in zip(before_grads, model.parameters())))
        self.assertEqual(before_modes, [module.training for module in modules])
        self.assertTrue(runner.core.i9.equal_tree(before_rng, runner.core.i9._rng_state()))

        no_aux = dict(data)
        no_aux["ax"], no_aux["ay"] = data["ax"][:0], data["ay"][:0]
        self.assertEqual(tuple(runner.evaluate(model, no_aux, calibration=True)),
                         ("train", "validation"))
        with self.assertRaisesRegex(ValueError, "Invalid evaluation"):
            runner.evaluate(model, no_aux, calibration=False)

    def test_tiny_healthy_trajectory_and_typed_numeric_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = runner.Run(root, "healthy", 30)
            entry, result = runner.run_trajectory(
                run, _tiny_data(), _tiny_plan(3), "sgd", .03, "raw", "clean",
                steps=3, horizons=(0, 1, 3),
                model_spec={"input_dim": 4, "width": 3, "classes": 2})
            self.assertEqual((entry["status"], entry["completed_steps"]), ("complete", 3))
            self.assertEqual([row["horizon"] for row in result["curve"]], [0, 1, 3])
            self.assertEqual(result["selected_horizons"], runner.selection(result["curve"]))

            failed_run = runner.Run(root, "numeric", 30)
            with mock.patch.object(runner.core, "train_step",
                                   side_effect=runner.core.NumericalFailure("synthetic nonfinite")):
                failed_entry, failed = runner.run_trajectory(
                    failed_run, _tiny_data(), _tiny_plan(3, seed=21415),
                    "sgd", .03, "raw", "clean", steps=3, horizons=(0, 1, 3),
                    model_spec={"input_dim": 4, "width": 3, "classes": 2})
            self.assertEqual(failed_entry["status"], "numerical_failure")
            self.assertEqual(failed_entry["completed_steps"], 0)
            self.assertIsNone(failed["selected_horizons"])
            self.assertEqual(failed["failure"]["type"], "NumericalFailure")
            self.assertTrue((failed_run.path /
                failed["failure"]["state_artifact"]["name"]).is_file())

    def test_generic_resource_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            run = runner.Run(Path(directory), "resource", 30)
            with mock.patch.object(runner.core, "train_step",
                                   side_effect=RuntimeError("synthetic resource stop")):
                with self.assertRaisesRegex(RuntimeError, "resource stop"):
                    runner.run_trajectory(
                        run, _tiny_data(), _tiny_plan(3), "sgd", .03, "raw", "clean",
                        steps=3, horizons=(0, 1, 3),
                        model_spec={"input_dim": 4, "width": 3, "classes": 2})

    def test_actual_warmup_raw_current_states_and_evaluations_match(self):
        data, plan = _tiny_data(), _tiny_plan(100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            results = []
            for policy in runner.POLICIES:
                run = runner.Run(root, "warmup-" + policy, 60)
                entry, result = runner.run_trajectory(
                    run, data, plan, "sgdm", .01, policy, "fixed",
                    steps=100, horizons=(0, 100),
                    model_spec={"input_dim": 4, "width": 3, "classes": 2})
                entries.append(entry)
                results.append(result)
            raw, current = entries
            self.assertEqual(raw["initial_full_state_digest"],
                             current["initial_full_state_digest"])
            self.assertEqual(raw["warmup_full_state_digest"],
                             current["warmup_full_state_digest"])
            self.assertEqual(raw["initial_evaluation_digest"],
                             current["initial_evaluation_digest"])
            self.assertEqual(raw["warmup_evaluation_digest"],
                             current["warmup_evaluation_digest"])
            self.assertEqual(results[0]["curve"], results[1]["curve"])
            self.assertTrue(all(not row["gradient_filter_applied"]
                                for result in results for row in result["steps"]))

    def test_malformed_smoke_plan_and_horizons_reject_before_write(self):
        data = _tiny_data()
        cases = [
            (_tiny_plan(3) | {"training_batches": np.zeros((3,), dtype=np.int64)},
             (0, 1, 3)),
            (_tiny_plan(3) | {"training_batches": np.full((3, 64), 12, dtype=np.int64)},
             (0, 1, 3)),
            (_tiny_plan(3), (0, 3, 1, 3)),
            (_tiny_plan(3), (0, 1, 1, 3)),
        ]
        for bad_plan, horizons in cases:
            fake = _NoWriteRun()
            with self.subTest(shape=np.shape(bad_plan["training_batches"]), horizons=horizons):
                with self.assertRaises(ValueError):
                    runner.run_trajectory(
                        fake, data, bad_plan, "sgd", .03, "raw", "clean",
                        steps=3, horizons=horizons,
                        model_spec={"input_dim": 4, "width": 3, "classes": 2})
                self.assertEqual(fake.saved, 0)

        for key, value in (("vx", data["vx"][:0]),
                           ("clean", data["clean"].to(torch.float32)),
                           ("x", torch.zeros((12, 5), dtype=torch.float32))):
            bad_data = dict(data)
            bad_data[key] = value
            fake = _NoWriteRun()
            with self.subTest(data_key=key):
                with self.assertRaises(ValueError):
                    runner.run_trajectory(
                        fake, bad_data, _tiny_plan(3), "sgd", .03, "raw", "clean",
                        steps=3, horizons=(0, 1, 3),
                        model_spec={"input_dim": 4, "width": 3, "classes": 2})
                self.assertEqual(fake.saved, 0)

    def test_calibration_does_not_read_unused_auxiliary_contents(self):
        generator = torch.Generator().manual_seed(1404)
        data = _tiny_data()
        for key in ("x", "vx", "ax"):
            data[key] = torch.rand((len(data[key]), 784), generator=generator)
        data["ax"][0, 0] = float("nan")
        data["ay"][0] = -1
        plan = _tiny_plan(3)
        plan["phase"] = "calibration"
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner.plans, "STEPS", 3), \
                mock.patch.object(runner.plans, "HORIZONS", (0, 1, 3)), \
                mock.patch.object(runner.plans, "validate_plan", return_value=None):
            run = runner.Run(Path(directory), "calibration-aux-neutral", 30)
            entry, result = runner.run_trajectory(
                run, data, plan, "sgd", .03, "raw", "clean",
                steps=3, horizons=(0, 1, 3))
        self.assertEqual((entry["status"], entry["completed_steps"]), ("complete", 3))
        self.assertTrue(all("auxiliary" not in row for row in result["curve"]))

    def test_validate_pairs_exact_membership_and_warmup_availability(self):
        entries = []
        for seed in runner.plans.CONFIRMATION_SEEDS:
            for base in runner.BASES:
                for target in runner.TARGETS:
                    for policy in runner.POLICIES:
                        prefix = f"{seed}-{base}-{target}"
                        entries.append({"seed": seed, "base": base, "target": target,
                            "policy": policy, "status": "complete",
                            "initial_model_digest": f"model-{seed}",
                            "initial_full_state_digest": "initial-" + prefix,
                            "initial_evaluation_digest": "eval0-" + prefix,
                            "warmup_full_state_digest": "warmup-" + prefix,
                            "warmup_evaluation_digest": "eval100-" + prefix})
        self.assertEqual(runner.validate_pairs(entries), {
            "expected_pairs": 18, "available_pairs_checked_exact": 18,
            "pairs_unavailable_due_to_numerical_failure": 0,
            "all_pairs_reached_warmup": True,
        })
        with self.assertRaises((AssertionError, ValueError)):
            runner.validate_pairs(entries[:-1])
        with self.assertRaises((AssertionError, ValueError)):
            runner.validate_pairs(entries + [copy.deepcopy(entries[-1])])
        incomplete = copy.deepcopy(entries)
        pair = [row for row in incomplete if row["seed"] == runner.plans.CONFIRMATION_SEEDS[0]
                and row["base"] == runner.BASES[0] and row["target"] == runner.TARGETS[0]]
        for row in pair:
            row["status"] = "numerical_failure"
            row["warmup_full_state_digest"] = None
            row["warmup_evaluation_digest"] = None
            row["startup_failure_signature"] = "same-startup-failure"
        self.assertEqual(runner.validate_pairs(incomplete), {
            "expected_pairs": 18, "available_pairs_checked_exact": 17,
            "pairs_unavailable_due_to_numerical_failure": 1,
            "all_pairs_reached_warmup": False,
        })

    def test_budget_writer_and_run_are_bounded_and_create_only(self):
        buffer = io.BytesIO()
        bounded = runner.BudgetWriter(buffer, 3)
        self.assertEqual(bounded.write(b"abc"), 3)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            bounded.write(b"d")
        self.assertEqual(buffer.getvalue(), b"abc")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = runner.Run(root, "writer", 30)
            record = run.save("small.json", {"ok": True})
            self.assertGreater(record["bytes"], 0)
            with self.assertRaises(FileExistsError):
                run.save("small.json", {"ok": False})
            with self.assertRaisesRegex(ValueError, "single filename"):
                run.save("../escape.json", {"ok": True})
            used = run.used()
            with mock.patch.object(runner, "ARTIFACT_CAP", used + runner.RESERVE + 3):
                with self.assertRaisesRegex(RuntimeError, "ceiling"):
                    run.save("too-large.json", {"more": "than three bytes"})
                self.assertFalse((run.path / "too-large.json").exists())


if __name__ == "__main__":
    unittest.main()
