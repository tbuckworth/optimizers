"""Deterministic I20 auditor fixtures; no producer import, observer, Torch, or RNG."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_i20_saved_audit", HERE / "audit_response.py")
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)


def fixture(horizon=24, process=0.1, ri=0, *, tied=False):
    """Build the frozen arrays directly from explicit scalar/vector equations."""
    angle = (0.0, math.pi / 4)[ri]
    axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    useful, eye = np.outer(axis, axis), np.eye(2)
    nuisance = eye - useful
    g = np.empty((horizon, 2), dtype=np.float64)
    for t in range(horizon):
        g[t] = (math.sin(0.41 * t) + 0.2 * math.cos(0.13 * t),
                math.cos(0.27 * t) - 0.1 * math.sin(0.71 * t))
    mu = g.copy() if tied else np.empty_like(g)
    if not tied:
        mu[0] = g[0]
        for t in range(1, horizon):
            mu[t] = 0.99 * mu[t - 1] + 0.01 * g[t]
    s = np.stack((0.03 * np.arange(horizon), -0.02 * np.arange(horizon)), axis=1) @ np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    native = np.empty((horizon, 2, 2), dtype=np.float64)
    full = np.empty_like(native)
    native_present = np.ones(horizon, dtype=np.bool_)
    full_present = np.ones(horizon, dtype=np.bool_)
    native_present[0] = False
    full_present[:2] = False
    for t in range(horizon):
        v = np.array([math.cos(0.08 * t + 0.2), math.sin(0.08 * t + 0.2)])
        w = np.array([math.cos(0.05 * t - 0.1), math.sin(0.05 * t - 0.1)])
        native[t], full[t] = np.outer(v, v), np.outer(w, w)
    # Native is copied exactly (the accepted parent stores I when absent); full
    # absent entries are ignored and replaced by the registered fallback.
    native[~native_present] = eye
    full[~full_present] = -3.0
    parent_names = auditor.parent_policy_names(process)
    parent_output = np.zeros((horizon, len(parent_names), 2), dtype=np.float64)
    old = {name: parent_output[:, i] for i, name in enumerate(parent_names)}
    for name in old:
        old[name][0] = g[0]
    old["ema_q0p9"][0] = g[0]
    for t in range(1, horizon):
        old["ema_q0p9"][t] = 0.9 * old["ema_q0p9"][t - 1] + 0.1 * g[t]

    arrays = {name: np.empty(shape, dtype=np.bool_ if name == "direction_present" else np.float64)
              for name, shape in {
                  "actions": (horizon, 6, 2, 2), "direction_present": (horizon, 6),
                  "weighted_moment": (horizon, 2, 2, 2), "weight_mass": (horizon, 2),
                  "normalized_moment": (horizon, 2, 2, 2), "weighted_eigenvalues": (horizon, 2, 2),
                  "weighted_gap": (horizon, 2), "output": (horizon, 36, 2),
                  "useful_squared_alignment": (horizon, 6), "action_change_norm": (horizon, 6)}.items()}
    arrays["actions"][:, 0] = np.where(native_present[:, None, None], native, eye)
    arrays["actions"][:, 1] = np.where(full_present[:, None, None], full, eye)
    arrays["direction_present"][:, 0] = native_present
    arrays["direction_present"][:, 1] = full_present
    moment = np.zeros((2, 2, 2))
    mass = np.zeros(2)
    decays = np.array((0.99, 0.999))
    for t in range(horizon):
        z = g[t] - mu[t]
        moment = decays[:, None, None] * moment + (1 - decays)[:, None, None] * np.outer(z, z)
        mass = decays * mass + 1 - decays
        arrays["weighted_moment"][t] = moment
        arrays["weight_mass"][t] = mass
        arrays["normalized_moment"][t] = moment / mass[:, None, None]
    values, vectors = np.linalg.eigh(arrays["normalized_moment"])
    arrays["weighted_eigenvalues"][:] = values
    arrays["weighted_gap"][:] = values[:, :, 1] - values[:, :, 0]
    available = arrays["weighted_gap"] > 1e-10 * np.maximum(1, np.abs(values[:, :, 1]))
    arrays["direction_present"][:, 2:4] = available
    for offset in range(2):
        projectors = vectors[:, offset, :, 1, None] * vectors[:, offset, None, :, 1]
        arrays["actions"][:, offset + 2] = np.where(available[:, offset, None, None], projectors, eye)
    arrays["actions"][:, 4], arrays["actions"][:, 5] = useful, nuisance
    arrays["direction_present"][:, 4:6] = True
    alignment = np.einsum("i,teij,j->te", axis, arrays["actions"], axis)
    arrays["useful_squared_alignment"][:] = np.where(arrays["direction_present"], alignment, 0.0)
    arrays["action_change_norm"][0] = 0.0
    arrays["action_change_norm"][1:] = np.linalg.norm(arrays["actions"][1:] - arrays["actions"][:-1], axis=(2, 3))
    outputs = arrays["output"].reshape(horizon, 6, 2, 3, 2)
    outputs[0] = g[0]
    for t in range(1, horizon):
        P = arrays["actions"][t]
        h = np.einsum("eij,j->ei", P, g[t]) + mu[t] - np.einsum("eij,j->ei", P, mu[t])
        for rj, rho in enumerate((0.9, auditor.RHO_STAR)):
            before = outputs[t - 1, :, rj]
            outputs[t, :, rj, 0] = rho * np.einsum("eij,ej->ei", P, before[:, 0]) \
                + np.einsum("eij,ej->ei", eye - rho * P, h)
            B = rho * P + 0.99 * (eye - P)
            outputs[t, :, rj, 1] = np.einsum("eij,ej->ei", B, before[:, 1]) \
                + np.einsum("eij,j->ei", eye - B, g[t])
            B = rho * P + 0.9 * (eye - P)
            outputs[t, :, rj, 2] = np.einsum("eij,ej->ei", B, before[:, 2]) \
                + np.einsum("eij,j->ei", eye - B, g[t])
    new = {name: arrays["output"][:, i] for i, name in enumerate(auditor.POLICIES)}
    for current, accepted in (("native/r0p9/cp", "native_cp"), ("native/rstar/cp", "native_cp_star"),
                              ("oracle_useful/r0p9/cp", "oracle_useful"),
                              ("oracle_useful/rstar/cp", "oracle_useful_star"),
                              ("oracle_nuisance/r0p9/cp", "oracle_nuisance")):
        old[accepted][:] = new[current]
    parent = {"g": g, "mu": mu, "A": native, "basis_present": native_present,
              "full_action": full, "full_basis_present": full_present, "s": s,
              "output": parent_output, "process_variance": process}
    return arrays, parent


class ResponseAuditTests(unittest.TestCase):
    def test_independent_fixture_all_rotations_and_processes(self):
        for process in (0.0, 0.01, 0.1):
            for ri in (0, 1):
                with self.subTest(process=process, rotation=ri):
                    arrays, parent = fixture(process=process, ri=ri)
                    result = auditor.validate_arrays(arrays, parent, ri)
                    self.assertEqual(len(result["policy_outputs"]), 36)
                    self.assertGreater(result["numeric_values"], 0)

    def test_tie_fallback_zero_start_and_first_output(self):
        arrays, parent = fixture(tied=True)
        auditor.validate_arrays(arrays, parent, 0)
        self.assertTrue(np.array_equal(arrays["weighted_moment"], np.zeros_like(arrays["weighted_moment"])))
        np.testing.assert_array_equal(arrays["weight_mass"][0], 1 - np.array([0.99, 0.999]))
        self.assertFalse(arrays["direction_present"][:, 2:4].any())
        np.testing.assert_array_equal(arrays["actions"][:, 2:4],
                                      np.broadcast_to(np.eye(2), arrays["actions"][:, 2:4].shape))
        np.testing.assert_array_equal(arrays["output"][0], np.broadcast_to(parent["g"][0], (36, 2)))

    def test_policy_order_is_frozen_estimator_major(self):
        expected = tuple(f"{e}/{r}/{response}" for e in
            ("native", "full_legacy", "ew99", "ew999", "oracle_useful", "oracle_nuisance")
            for r in ("r0p9", "rstar") for response in ("cp", "rec99", "rec9"))
        self.assertEqual(auditor.POLICIES, expected)

    def test_every_array_and_policy_mutation_is_rejected(self):
        original, parent = fixture()
        for name, value in original.items():
            with self.subTest(array=name):
                changed = {key: item.copy() for key, item in original.items()}
                index = (5,) + (0,) * (value.ndim - 1)
                changed[name][index] = not changed[name][index] if value.dtype == np.bool_ else 91.0
                with self.assertRaises(auditor.AuditError):
                    auditor.validate_arrays(changed, parent, 0)
        for column, name in enumerate(auditor.POLICIES):
            with self.subTest(policy=name):
                changed = {key: item.copy() for key, item in original.items()}
                changed["output"][5, column, 0] += 0.25
                with self.assertRaises(auditor.AuditError):
                    auditor.validate_arrays(changed, parent, 0)

    def test_shape_dtype_order_nonfinite_and_parent_closure_rejected(self):
        original, parent = fixture()
        mutations = []
        changed = dict(original); changed["weighted_gap"] = changed["weighted_gap"].astype(np.float32); mutations.append(changed)
        changed = dict(original); changed["output"] = changed["output"][:-1]; mutations.append(changed)
        changed = dict(reversed(list(original.items()))); mutations.append(changed)
        changed = {key: value.copy() for key, value in original.items()}; changed["actions"][3, 0, 0, 0] = np.nan; mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(auditor.AuditError):
                auditor.validate_arrays(changed, parent, 0)
        bad_parent = copy.deepcopy(parent)
        position = auditor.parent_policy_names(0.1).index("native_cp")
        bad_parent["output"][7, position, 0] += 1e-5
        with self.assertRaisesRegex(auditor.AuditError, "parent CP closure"):
            auditor.validate_arrays(original, bad_parent, 0)

    def test_all_rho_point9_rec9_close_to_parent_ema(self):
        arrays, parent = fixture()
        names = auditor.parent_policy_names(0.1)
        ema = parent["output"][:, names.index("ema_q0p9")]
        for estimator in auditor.ESTIMATORS:
            current = arrays["output"][:, auditor.POLICIES.index(f"{estimator}/r0p9/rec9")]
            np.testing.assert_allclose(current, ema, rtol=2e-11, atol=5e-12)

    def test_npz_exact_members_order_storage_dtype_and_pickle(self):
        arrays, _ = fixture(horizon=8)
        expected_shapes = {name: value.shape for name, value in arrays.items()}
        dtypes = {name: str(value.dtype) for name, value in arrays.items()}
        for mutation in ("valid", "reversed", "extra", "compressed", "object"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "x.npz"
                payload = arrays
                if mutation == "reversed":
                    payload = dict(reversed(list(arrays.items())))
                elif mutation == "extra":
                    payload = dict(arrays, surprise=np.zeros(1))
                elif mutation == "object":
                    payload = dict(arrays); payload["actions"] = np.array([object()], dtype=object)
                saver = np.savez_compressed if mutation == "compressed" else np.savez
                saver(path, **payload)
                checks = auditor.Checks()
                if mutation == "valid":
                    loaded = auditor._load_npz(path, auditor.ARRAY_ORDER, expected_shapes, dtypes, checks)
                    self.assertEqual(list(loaded), list(auditor.ARRAY_ORDER))
                else:
                    with self.assertRaises((auditor.AuditError, ValueError)):
                        auditor._load_npz(path, auditor.ARRAY_ORDER, expected_shapes, dtypes, checks)

    def test_parent_npz_full_archive_roster_loads_selected_subset(self):
        arrays, _ = fixture(horizon=8)
        extra = {"prefix_a": np.zeros((8, 3)), **arrays, "suffix_b": np.ones(8)}
        selected = ("actions", "direction_present")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.npz"
            np.savez(path, **extra)
            loaded = auditor._load_npz(path, selected,
                {name: arrays[name].shape for name in selected},
                {name: str(arrays[name].dtype) for name in selected}, auditor.Checks(), archive_names=tuple(extra))
            self.assertEqual(list(loaded), list(selected))
            with self.assertRaisesRegex(auditor.AuditError, "membership/order"):
                auditor._load_npz(path, selected, {name: arrays[name].shape for name in selected},
                    {name: str(arrays[name].dtype) for name in selected}, auditor.Checks(),
                    archive_names=tuple(reversed(extra)))

    def test_registered_aggregation_counts_signs_and_no_survivors(self):
        metrics, diagnostics, controls = [], [], []
        for seed in auditor.SEEDS:
            arrays, parent = fixture(process=0.1)
            for pi in range(3):
                for ri in range(2):
                    for window, _, _ in auditor.WINDOWS:
                        for policy_index, policy in enumerate(auditor.POLICIES):
                            metrics.append({"seed": seed, "process_index": pi, "rotation_index": ri,
                                            "window": window, "policy": policy, "mse": float(policy_index + seed / 1e6)})
                        controls.extend((
                            {"seed": seed, "process_index": pi, "rotation_index": ri, "window": window,
                             "policy": "ema_q0p9", "mse": 100.0},
                            {"seed": seed, "process_index": pi, "rotation_index": ri, "window": window,
                             "policy": "common_kalman", "mse": 80.0}))
                        for estimator_index, estimator in enumerate(auditor.ESTIMATORS):
                            diagnostics.append({"seed": seed, "process_index": pi, "rotation_index": ri,
                                "window": window, "estimator": estimator, "observations": 1,
                                "direction_present_observations": 1, "direction_absent_observations": 0,
                                "mean_useful_squared_alignment_when_present": estimator_index / 10,
                                "mean_action_change_norm": estimator_index / 20})
        means, contrasts, primary, direction, prediction = auditor._aggregate(metrics, diagnostics, controls)
        self.assertEqual((len(metrics), len(means), len(contrasts), len(primary), len(diagnostics), len(direction)),
                         (27648, 864, 2160, 270, 4608, 144))
        first = contrasts[0]["effect"]
        self.assertEqual((first["positive_seeds"], first["negative_seeds"], first["zero_seeds"]), (32, 0, 0))
        incomplete = {str(seed): 1.0 for seed in auditor.SEEDS[:-1]}
        result = auditor._seed_summary(incomplete)
        self.assertFalse(result["available"])
        self.assertIsNone(result["mean"])
        self.assertTrue(result["no_survivor_averaging"])
        self.assertTrue(prediction["ew999_mean_alignment_exceeds_ew99"])

    def test_auditor_has_no_producer_or_forbidden_imports(self):
        source = (HERE / "audit_response.py").read_text()
        self.assertNotIn("import response_core", source)
        self.assertNotIn("from response_core", source)
        self.assertNotIn("import torch", source.lower())
        self.assertNotIn("default_rng", source)


if __name__ == "__main__":
    unittest.main()
