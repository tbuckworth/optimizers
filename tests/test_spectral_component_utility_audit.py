"""Fabricated numeric arrays only; no producer imports or scientific inputs."""
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from experiments import spectral_component_utility_audit as audit


def saved(root, name, values):
    with (root / name).open("xb") as handle:
        np.savez(handle, **values)
    return audit.file_receipt(root / name)


def action(theta, gradient, native=False, stage=100):
    q = np.eye(len(theta), dtype=np.float32)[:, :1] if native else np.empty((len(theta), 0), np.float32)
    delivered = q @ (q.T @ gradient) if native else gradient.copy()
    before_m = np.arange(len(theta), dtype=np.float32) / 128
    before_v = np.arange(1, len(theta) + 1, dtype=np.float32) / 64
    m = .9 * before_m.astype(np.float64) + .1 * delivered.astype(np.float64)
    v = .999 * before_v.astype(np.float64) + .001 * delivered.astype(np.float64)**2
    decay = (theta * np.float32(.99999)).astype(np.float32)
    step = stage + 1
    end = (decay.astype(np.float64) - .001 * m / (1 - .9**step) /
           (np.sqrt(v / (1 - .999**step)) + 1e-8)).astype(np.float32)
    return dict(theta_before=theta.copy(), theta_after=end, decay_endpoint=decay,
                gradient_raw=gradient.copy(), gradient_delivered=delivered,
                m_before=before_m, v_before=before_v, m_after=m.astype(np.float32), v_after=v.astype(np.float32),
                adam_steps_before=np.full(6, stage, np.int64), adam_steps_after=np.full(6, step, np.int64),
                post_basis=q, post_singular_values=np.ones(q.shape[1], np.float64))


def path_arrays(theta, endpoint, decay, fraction):
    # Explicit independent operations, including direct full endpoint.
    point = endpoint.copy() if fraction == 1 else theta + np.float32(fraction) * (endpoint - theta)
    dpoint = decay.copy() if fraction == 1 else theta + np.float32(fraction) * (decay - theta)
    return dict(point=point, decay_point=dpoint,
                total_delta=point.astype(np.float64) - theta.astype(np.float64),
                decay_delta=dpoint.astype(np.float64) - theta.astype(np.float64),
                data_delta=point.astype(np.float64) - dpoint.astype(np.float64))


class ArithmeticFixtures(unittest.TestCase):
    def test_independent_scalar_formula_and_offset_invariance(self):
        grid = np.array([[[2, -1, 0], [0, 1, 3], [-2, 2, 1]],
                         [[0, 0, 2], [3, -2, 1], [1, 2, 0]]], np.float32)
        panel = dict(evaluation_true=np.array([0, 2]), evaluation_assigned=np.array([1, 2]),
                     reporting_true=np.array([1, 0]))
        values = dict(I=grid, R=grid[::-1].copy(), R_original=grid[:, 1].copy())
        result = audit.metrics(values, panel, 1)
        out = result["objectives"]
        expected = {k: [] for k in ("S", "F", "C", "L", "S_true", "S_uniform")}
        for z, true, assigned in zip(grid.astype(np.float64), panel["evaluation_true"], panel["evaluation_assigned"]):
            mean = [math.fsum(z[:, k]) / 3 for k in range(3)]
            normalizer = math.log(math.fsum(math.exp(x) for x in mean))
            q = [.9 / 3 + (.1 if k == true else 0) for k in range(3)]
            soft = math.fsum(q[k] * mean[k] for k in range(3))
            view_norms = [math.log(math.fsum(math.exp(x) for x in row)) for row in z]
            expected["S"].append(normalizer - soft)
            expected["F"].append(soft - mean[assigned])
            expected["C"].append(math.fsum(view_norms) / 3 - normalizer)
            expected["L"].append(math.fsum(a - row[assigned] for a, row in zip(view_norms, z)) / 3)
            expected["S_true"].append(normalizer - mean[true])
            expected["S_uniform"].append(normalizer - math.fsum(mean) / 3)
        for key in expected:
            self.assertAlmostEqual(out[key], math.fsum(expected[key]) / 2, places=13)
        self.assertAlmostEqual(out["L"], out["S"] + out["F"] + out["C"], places=13)
        shifted = {k: v + np.float32(2**22) for k, v in values.items()}
        audit.Checks().tree(audit.metrics(shifted, panel, 1), result)
        self.assertEqual(result["I"]["wrong_per_view"]["assigned"]["count"], 3)
        self.assertEqual(result["I"]["original"]["true"]["count"], 2)
        self.assertEqual(result["R"]["per_view"]["true"]["count"], 6)

    def test_single_view_and_empty_wrong_subset(self):
        grid = np.array([[[1, 0]], [[0, 2]]], np.float32)
        panel = dict(evaluation_true=np.array([0, 1]), evaluation_assigned=np.array([0, 1]),
                     reporting_true=np.array([0, 1]))
        result = audit.metrics(dict(I=grid, R=grid, R_original=grid[:, 0]), panel, 0)
        self.assertEqual(result["objectives"]["C"], 0.)
        wrong = result["I"]["wrong_per_view"]["true"]
        self.assertFalse(wrong["available"])
        self.assertEqual(wrong["count"], 0)
        self.assertEqual(wrong["ce_sum"], 0.)
        self.assertIsNone(wrong["accuracy"])
        self.assertIsNone(wrong["ce"])
        json.dumps(result, allow_nan=False)

    def test_elementwise_not_only_norm_tolerance(self):
        check = audit.Checks()
        with self.assertRaisesRegex(audit.AuditError, "elementwise"):
            check.norm(np.array([100000., .001]), np.array([100000., 0.]), "small coordinate")
        check.norm(np.array([1. + 1e-6]), np.array([1.]), "allowed")
        self.assertEqual(len(check.array_residuals), 2)

    def test_action_adam_projection_and_clock_tampering(self):
        theta = np.arange(1, 7, dtype=np.float32) / 32
        gradient = np.arange(1, 7, dtype=np.float32) / 16
        for native in (False, True):
            record = action(theta, gradient, native)
            metadata = dict(policy="native" if native else "raw", observer_steps_before=100,
                            observer_steps_after=101 if native else 100, basis_rank=int(native))
            audit.check_action(record, metadata, theta, 100, audit.Checks(), (1,) * 6)
            for key in ("m_after", "v_after", "theta_after", "gradient_delivered", "adam_steps_after"):
                bad = copy.deepcopy(record)
                bad[key][0] += 1
                with self.assertRaises(audit.AuditError, msg=key):
                    audit.check_action(bad, metadata, theta, 100, audit.Checks(), (1,) * 6)
            bad = dict(metadata, observer_steps_after=102)
            with self.assertRaisesRegex(audit.AuditError, "observer"):
                audit.check_action(record, bad, theta, 100, audit.Checks(), (1,) * 6)

    def test_native_no_basis_bypass_is_not_zero_projection(self):
        theta = np.ones(6, np.float32)
        record = action(theta, np.ones(6, np.float32), False)
        metadata = dict(policy="native", basis_rank=0, observer_steps_before=100, observer_steps_after=101)
        audit.check_action(record, metadata, theta, 100, audit.Checks(), (1,) * 6)
        record["gradient_delivered"][:] = 0
        with self.assertRaisesRegex(audit.AuditError, "delivery"):
            audit.check_action(record, metadata, theta, 100, audit.Checks(), (1,) * 6)

    def test_materialized_paths_and_compensated_dot(self):
        theta, endpoint = np.array([2**25], np.float32), np.array([1], np.float32)
        self.assertEqual(audit.materialized(theta, endpoint, 1)[0], 1)
        self.assertEqual(float(audit.materialized(theta, endpoint, .1)[0]),
                         float(theta[0] + np.float32(.1) * (endpoint[0] - theta[0])))
        tiny = np.nextafter(np.float32(1), np.float32(2))
        theta = np.array([1], np.float32)
        data = path_arrays(theta, np.array([tiny], np.float32), theta, .1)
        audit.check_path(data, theta, np.array([tiny], np.float32), theta, .1, audit.Checks())
        self.assertEqual(data["total_delta"][0], 0)
        bad = copy.deepcopy(data)
        bad["total_delta"][0] = .1 * (float(tiny) - 1)
        with self.assertRaises(audit.AuditError):
            audit.check_path(bad, theta, np.array([tiny], np.float32), theta, .1, audit.Checks())
        self.assertEqual(audit.dot(np.array([2**60, 1, -(2**60)], np.float32), np.ones(3)), -1.)

    def test_summary_whole_roster_primary_and_signs(self):
        parents = []
        for row in audit.roster():
            endpoints = []
            for b in (0, 1):
                for policy in ("raw", "native", "decay"):
                    for fraction, fid in audit.FRACTIONS:
                        value = (10 if policy == "native" else 2) + b
                        endpoints.append(dict(batch=b, policy=policy, fraction_id=fid,
                                              effects={k: dict(finite=float(value), linear=float(-value))
                                                       for k in audit.OBJECTIVES}))
            parents.append(dict(row, endpoints=endpoints))
        result = audit.summary(parents)
        self.assertEqual(len(result["cells"]), 8)
        self.assertEqual(result["primary"]["augmentation"], "translate")
        self.assertEqual(result["primary"]["step"], 56304)
        self.assertEqual(result["primary"]["fraction_id"], "full")
        self.assertEqual(len(result["primary"]["seed_rows"]), 3)
        primary = result["primary"]["seed_rows"][0]["objectives"]
        self.assertEqual(set(primary), {"H_O", "C"})
        self.assertEqual(primary["C"]["raw_finite"], 2.5)
        self.assertEqual(primary["C"]["native_finite"], 10.5)
        self.assertEqual(primary["C"]["contrast_finite"], 8.)
        self.assertEqual(primary["C"]["contrast_linear"], -8.)
        self.assertEqual(len(audit.expected_names()), 376)


class ReaderFixtures(unittest.TestCase):
    def test_opened_descriptor_is_regular_and_nonblocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = saved(root, "a.npz", {"x": np.ones(1)})
            path = root / "a.npz"
            fields = list(path.stat())
            fields[0] = stat.S_IFIFO | 0o600
            fake = os.stat_result(fields)
            real_open = os.open
            with mock.patch.object(audit.os, "open", wraps=real_open) as opened, \
                 mock.patch.object(audit.os, "fstat", return_value=fake):
                with self.assertRaisesRegex(audit.AuditError, "nonregular"):
                    audit.file_receipt(path)
                self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)
            reader = audit.Reader(root, [receipt], audit.Checks())
            with mock.patch.object(reader, "path", return_value=path), \
                 mock.patch.object(audit.os, "open", wraps=real_open) as opened, \
                 mock.patch.object(audit.os, "fstat", return_value=fake):
                with self.assertRaisesRegex(audit.AuditError, "regular-file"):
                    reader.npz(receipt)
                self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)

    def test_valid_empty_basis_and_hash_size_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = saved(root, "action.npz", {"post_basis": np.empty((6, 0), np.float32),
                                                  "steps": np.array([100], np.int64)})
            reader = audit.Reader(root, [receipt], audit.Checks())
            result = reader.npz(receipt, ("post_basis", "steps"))
            self.assertEqual(result["post_basis"].shape, (6, 0))
            for key, value in (("sha256", "0" * 64), ("size_bytes", receipt["size_bytes"] + 1)):
                bad = dict(receipt, **{key: value})
                with self.assertRaises(audit.AuditError):
                    audit.Reader(root, [bad], audit.Checks()).npz(bad)

    def test_escape_symlink_and_duplicate_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = saved(root, "a.npz", {"x": np.ones(1)})
            for name in ("../a.npz", "/a.npz", "nested/a.npz"):
                with self.assertRaises(audit.AuditError):
                    audit.Reader(root, [dict(receipt, path=name)], audit.Checks())
            (root / "link.npz").symlink_to(root / "a.npz")
            with self.assertRaises(audit.AuditError):
                audit.Reader(root, [dict(receipt, path="link.npz")], audit.Checks())
            with self.assertRaises(audit.AuditError):
                audit.Reader(root, [receipt, receipt], audit.Checks())

    def test_object_structured_duplicate_and_header_bombs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for i, value in enumerate((np.array([{}], object), np.zeros(2, dtype=[("a", "i4")]))):
                receipt = saved(root, f"bad{i}.npz", {"x": value})
                with self.assertRaises(audit.AuditError):
                    audit.Reader(root, [receipt], audit.Checks()).npz(receipt)
            data = io.BytesIO()
            np.lib.format.write_array_header_1_0(data, dict(descr="<f8", fortran_order=False, shape=(10**12,)))
            with zipfile.ZipFile(root / "bomb.npz", "w") as archive:
                archive.writestr("x.npy", data.getvalue())
            receipt = audit.file_receipt(root / "bomb.npz")
            with self.assertRaisesRegex(audit.AuditError, "payload"):
                audit.Reader(root, [receipt], audit.Checks()).npz(receipt)
            with zipfile.ZipFile(root / "duplicate.npz", "w") as archive:
                archive.writestr("x.npy", data.getvalue())
                with self.assertWarns(UserWarning):
                    archive.writestr("x.npy", data.getvalue())
            receipt = audit.file_receipt(root / "duplicate.npz")
            with self.assertRaisesRegex(audit.AuditError, "duplicate"):
                audit.Reader(root, [receipt], audit.Checks()).npz(receipt)

    def test_json_duplicates_nonfinite_and_default_cli_inert(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(audit.AuditError):
                audit.strict_json(text)
        with mock.patch.object(audit, "audit_archive", side_effect=AssertionError("science")), \
             mock.patch.object(Path, "read_bytes", side_effect=AssertionError("input")):
            self.assertEqual(audit.main(["--input-dir", "/nonexistent"]), 0)

    def test_import_does_not_read_or_inspect_environment(self):
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read")), \
             mock.patch.object(audit.subprocess, "run", side_effect=AssertionError("process")), \
             mock.patch.object(np, "load", side_effect=AssertionError("arrays")):
            spec = importlib.util.spec_from_file_location("inert_component_audit", audit.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


class BindingFixtures(unittest.TestCase):
    def test_provenance_guards_inventory_and_inconsistent_gpu_rows(self):
        manifest = dict(commit="a" * 40, source_pins={"fake.py": "b" * 64})
        inventory = dict(data_pins={"fake-data": "c" * 64})
        acquisition = Path("/fabricated/acquisition-001")
        attempt = dict(schema=audit.SCHEMA, commit=manifest["commit"], worktree_head="d" * 40,
                       source_pins=manifest["source_pins"], data_pins=inventory["data_pins"],
                       input_inventory_sha256="e" * 64, manifest_sha256="f" * 64,
                       unit="spectral-component-utility-001.service", output_dir=str(acquisition),
                       started_utc="2026-09-10T00:00:00+00:00", pid=12345, invocation_id="a" * 32,
                       argv=["runner.py", "--execute"])
        group = "/user.slice/" + attempt["unit"]
        clients = [dict(pid=2101, memory_mib=200), dict(pid=12345, memory_mib=500)]
        guards = dict(unit=attempt["unit"], pid=attempt["pid"], invocation_id=attempt["invocation_id"], cgroup=group,
                      effective={"memory.max": "8589934592", "memory.swap.max": "0", "cpu.max": "100000 100000"},
                      service=dict(Type="exec", RuntimeMaxUSec="30min", Restart="no", KillMode="control-group",
                                   MainPID="12345", InvocationID="a" * 32, ActiveState="active", SubState="running",
                                   ControlGroup=group), gpu_clients=clients, preexisting_gpu_clients=clients[:1],
                      device="cuda:0", device_name="NVIDIA GeForce RTX 3090", gpu_free_bytes_at_configure=12 * 1024**3,
                      gpu_total_bytes=24 * 1024**3, gpu_allocator_limit_bytes=4 * 1024**3,
                      gpu_allocator_fraction=1 / 6, torch_version="2.11.0+cu128", numpy_version="1.26.4",
                      deterministic=dict(thread_env={key: "1" for key in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                          cublas_workspace=":4096:8", intraop_threads=1, interop_threads=1,
                          deterministic_algorithms=True, cudnn_benchmark=False, matmul_allow_tf32=False,
                          cudnn_allow_tf32=False))
        parts = dict(zip(("48_action_tensor_records", "24_post_native_bases", "24_post_native_singular_values",
                         "12_baseline_gradient_theta_sets", "12_baseline_check_sets", "144_path_records",
                         "156_logit_sets", "metadata_container_panel_reserve"),
                        (406336896, 4514803200, 38400, 79009056, 528000, 1083552768, 60702720, 268435456)))
        byte_inventory = dict(component_upper_bytes=parts, total_upper_bytes=sum(parts.values()),
                              failure_reserve_bytes=1048576, cap_bytes=8589934592, expected_receipted_artifacts=376)
        provenance = dict(attempt, guards=guards, inventory=byte_inventory, python="fabricated 3.13",
                          numpy="1.26.4", torch="2.11.0+cu128")
        def check(value):
            audit.verify_provenance(value, attempt, manifest, inventory, "f" * 64, "e" * 64,
                                    acquisition, audit.Checks())
        check(provenance)
        for mutate in (lambda p: p["guards"].update(preexisting_gpu_clients=[]),
                       lambda p: p["guards"].update(gpu_free_bytes_at_configure=30 * 1024**3),
                       lambda p: p["inventory"].update(total_upper_bytes=1),
                       lambda p: p.update(input_inventory_sha256="a" * 64)):
            bad = copy.deepcopy(provenance)
            mutate(bad)
            with self.assertRaises(audit.AuditError):
                check(bad)

    def test_original_roles_fixed_assignments_and_new_panel_draws(self):
        seed = 202609171
        rng = lambda stream: np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))
        ids = rng(0).permutation(60000).astype(np.int64)
        plan = {}
        for role, positions in (("train", ids[:50000]), ("validation", ids[50000:55000]),
                                ("reporting", ids[55000:])):
            plan[role + "_ids"] = positions.copy()
            plan[role + "_labels"] = positions % 10
        corrupt = rng(1)
        plan["corruption_mask"] = corrupt.random(50000) < .9
        plan["replacement_labels"] = corrupt.integers(0, 10, size=50000, dtype=np.int64)
        plan["assigned_labels"] = np.where(plan["corruption_mask"], plan["replacement_labels"], plan["train_labels"])
        plan["changed_mask"] = plan["assigned_labels"] != plan["train_labels"]
        result = audit.panels_from_plan(plan, seed, audit.Checks())
        self.assertEqual(set(result), set(audit.PANEL_KEYS))
        self.assertEqual(result["action_shifts"].dtype, np.int8)
        self.assertEqual(result["action_shifts"].shape, (2, 64, 2))
        self.assertTrue(np.array_equal(result["view_shifts"][12], [0, 0]))
        self.assertEqual(len(set(result["action_positions"].ravel()) & set(result["evaluation_positions"])), 0)
        self.assertTrue(np.array_equal(result["action_assigned"], plan["assigned_labels"][result["action_positions"]]))
        bad = copy.deepcopy(plan)
        bad["assigned_labels"][0] = (bad["assigned_labels"][0] + 1) % 10
        with self.assertRaisesRegex(audit.AuditError, "assigned_labels"):
            audit.panels_from_plan(bad, seed, audit.Checks())
        bad = copy.deepcopy(plan)
        bad["reporting_ids"][0] = bad["train_ids"][0]
        with self.assertRaises(audit.AuditError):
            audit.panels_from_plan(bad, seed, audit.Checks())

    def test_committed_and_current_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            stems = ("", "_actions", "_restore", "_panels", "_objectives", "_guard", "_audit")
            names = [f"experiments/spectral_component_utility{s}.py" for s in stems]
            names += [f"tests/test_spectral_component_utility{s}.py" for s in stems]
            names += ["output/2026-09-10-spectral-component-utility/" + name for name in
                      ("protocol.md", "archive-contract.md", "parent-inventory.json", "implementation-check.md")]
            names += ["old.py"]
            pins = {}
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fabricated\n")
                pins[name] = audit.file_receipt(path)["sha256"]
            inventory = dict(source_pins={"old.py": pins["old.py"]})
            manifest = dict(schema="spectral_component_utility_source_manifest_v1", commit="a" * 40,
                            source_pins=pins, input_inventory_sha256=pins[names[-2]])
            with mock.patch.object(audit.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=b"fabricated\n")):
                audit.verify_sources(root, manifest, inventory, audit.Checks())
                bad = copy.deepcopy(manifest)
                bad["source_pins"].pop("experiments/spectral_component_utility.py")
                with self.assertRaisesRegex(audit.AuditError, "source pin set"):
                    audit.verify_sources(root, bad, inventory, audit.Checks())
            with mock.patch.object(audit.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=b"wrong\n")):
                with self.assertRaisesRegex(audit.AuditError, "committed"):
                    audit.verify_sources(root, manifest, inventory, audit.Checks())

    def test_audit_service_resource_and_output_admission(self):
        unit = "spectral-component-utility-audit-001.service"
        cgroup = "/user.slice/" + unit
        values = {"/proc/self/cgroup": "0::" + cgroup + "\n"}
        for key, value in (("memory.max", "2147483648"), ("memory.swap.max", "0"), ("cpu.max", "100000 100000")):
            values["/sys/fs/cgroup" + cgroup + "/" + key] = value
        props = dict(Type="exec", RuntimeMaxUSec="20min", Restart="no", KillMode="control-group",
                     MainPID=str(os.getpid()), InvocationID="a" * 32, ControlGroup=cgroup)
        environment = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        environment.update(CUDA_VISIBLE_DEVICES="", INVOCATION_ID="a" * 32)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, environment), \
             mock.patch.object(Path, "read_text", lambda p, *a, **k: values[str(p)]), \
             mock.patch.object(audit.subprocess, "check_output", return_value="\n".join(k + "=" + v for k, v in props.items())):
            root = Path(directory).resolve()
            acquisition = root / "acquisition-001"
            acquisition.mkdir()
            audit.audit_environment(root / "audit.json", acquisition)
            with self.assertRaisesRegex(audit.AuditError, "outside"):
                audit.audit_environment(acquisition / "audit.json", acquisition)
            with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"):
                with self.assertRaisesRegex(audit.AuditError, "CPU-only"):
                    audit.audit_environment(root / "audit.json", acquisition)
            values["/sys/fs/cgroup" + cgroup + "/memory.max"] = "8589934592"
            with self.assertRaisesRegex(audit.AuditError, "resource"):
                audit.audit_environment(root / "audit.json", acquisition)


class ParentArchiveFixture(unittest.TestCase):
    def make_parent(self, root):
        row = dict(seed=202609171, augmentation="translate", step=100, parent_id="s202609171-translate-h00100")
        pid = row["parent_id"]
        panel = dict(evaluation_true=np.array([0, 2]), evaluation_assigned=np.array([1, 2]),
                     reporting_true=np.array([1, 0]), baseline_ids=np.array([5, 7]))
        theta = np.arange(1, 7, dtype=np.float32) / 32
        vectors = {key: np.full(6, i + 1, np.float32) / 16 for i, key in enumerate(audit.OBJECTIVES)}
        vectors["L"] = vectors["S"] + vectors["F"] + vectors["C"]
        vectors["theta"] = theta
        grid = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 8
        logits = dict(I=grid, R=grid[::-1].copy(), R_original=grid[:, 1].copy())
        source_readout = logits["R_original"].copy()
        baseline = audit.metrics(logits, panel, 1)
        receipts = []
        def write(name, data):
            receipt = saved(root, name, data)
            receipts.append(receipt)
            return receipt
        row.update(parent_digest_before="a" * 64, parent_digest_after="a" * 64,
                   baseline_check_receipt=write(f"baseline-check-{pid}.npz", dict(actual=source_readout,
                                              expected=source_readout, ids=panel["baseline_ids"])),
                   baseline_receipt=write(f"baseline-{pid}.npz", logits),
                   gradient_receipt=write(f"gradients-{pid}.npz", vectors), baseline_metrics=baseline,
                   actions=[], endpoints=[], contrasts=[])
        records = {}
        for batch in (0, 1):
            gradient = np.arange(1, 7, dtype=np.float32) / (batch + 16)
            for policy in ("raw", "native"):
                record = action(theta, gradient, policy == "native")
                records[batch, policy] = record
                row["actions"].append(dict(batch=batch, policy=policy,
                    receipt=write(f"action-{pid}-b{batch}-{policy}.npz", record),
                    observer_steps_before=100, observer_steps_after=100 + (policy == "native"),
                    basis_rank=int(policy == "native")))
        by_key = {}
        for batch in (0, 1):
            for policy in ("raw", "native", "decay"):
                for fraction, fid in audit.FRACTIONS:
                    record = records[batch, policy if policy != "decay" else "raw"]
                    endpoint = record["theta_after"] if policy != "decay" else record["decay_endpoint"]
                    path = path_arrays(theta, endpoint, record["decay_endpoint"], fraction)
                    # Constant saved predictions are legitimate fabricated no-effect readouts.
                    effect = audit.effects(baseline["objectives"], baseline["objectives"], baseline["objectives"], vectors, path)
                    entry = dict(batch=batch, policy=policy, fraction=fraction, fraction_id=fid,
                                 logit_receipt=write(f"logits-{pid}-b{batch}-{policy}-{fid}.npz", logits),
                                 path_receipt=write(f"path-{pid}-b{batch}-{policy}-{fid}.npz", path),
                                 metrics=copy.deepcopy(baseline), effects=effect,
                                 norms={k.replace("delta", "norm"): float(np.linalg.norm(v))
                                        for k, v in path.items() if k.endswith("delta")})
                    by_key[batch, policy, fid] = entry
                    row["endpoints"].append(entry)
        for batch in (0, 1):
            for fraction, fid in audit.FRACTIONS:
                row["contrasts"].append(dict(batch=batch, fraction=fraction, fraction_id=fid,
                    objectives={key: {kind: by_key[batch, "native", fid]["effects"][key][kind] -
                                     by_key[batch, "raw", fid]["effects"][key][kind]
                                     for kind in ("finite", "linear")} for key in audit.OBJECTIVES}))
        return row, panel, source_readout, receipts

    def test_full_tiny_parent_archive_and_scalar_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            row, panel, source, receipts = self.make_parent(root)
            def run(parent):
                return audit.audit_parent(parent, panel, audit.Reader(root, receipts, audit.Checks()), source,
                                          layout=(6, 2, 2, 3, 3, 2, 1), parameter_sizes=(1,) * 6)
            result = run(row)
            self.assertEqual(len(result["endpoints"]), 12)
            self.assertEqual(len(result["contrasts"]), 4)
            mutations = [lambda p: p["endpoints"][0]["effects"]["H_O"].update(linear=1.),
                         lambda p: p["endpoints"][0]["metrics"]["I"]["per_view"]["true"].update(count=1),
                         lambda p: p["actions"][1].update(observer_steps_after=100),
                         lambda p: p["endpoints"].pop(),
                         lambda p: p.update(parent_digest_after="b" * 64),
                         lambda p: p["contrasts"][0]["objectives"]["C"].update(finite=1.)]
            for mutate in mutations:
                bad = copy.deepcopy(row)
                mutate(bad)
                with self.assertRaises(audit.AuditError):
                    run(bad)
            with self.assertRaises(audit.AuditError):
                audit.audit_parent(row, panel, audit.Reader(root, receipts, audit.Checks()), source + 1,
                                   layout=(6, 2, 2, 3, 3, 2, 1), parameter_sizes=(1,) * 6)


if __name__ == "__main__":
    unittest.main()
