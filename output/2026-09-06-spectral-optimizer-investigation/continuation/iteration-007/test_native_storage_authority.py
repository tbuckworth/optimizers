"""Synthetic decoder/writer contracts only; no measurement or native evidence."""
import copy
import hashlib
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_storage_authority as authority
import native_storage_topology_bound as topology
import native_tensor_inventory as inventory
import process_supervision as supervision
import zip_storage_bound as zip_bound

C, S = "a" * 40, "b" * 64


@lru_cache(maxsize=1)
def _synthetic_measurement_base():
    """Invented bytes/counts exercise parsing, NOT actual serializer evidence."""
    bounds, layouts = topology.compute()["components"], inventory.component_layouts()
    rows = []
    for name, bound in bounds.items():
        raw = sum(r["nbytes"] for r in layouts[name])
        rows.append(dict(component=name, encoding=bound["encoding"],
            payload_count=bound["payload_count"], analytic_pickle_bytes_upper=bound.get("pickle_bytes_upper"),
            analytic_body_bytes_upper=bound["body_bytes_upper"], status="measured",
            observation=dict(pickle_bytes=None if bound["encoding"] == "bytes" else 1,
                body_bytes=raw + 1, storage_count=len(layouts[name]), raw_storage_bytes=raw,
                serialized_sha256="c" * 64, restricted_cpu_roundtrip=True, exact_tree_roundtrip=True)))
    zip_record = {
        "schema": zip_bound.RUNTIME_SCHEMA, "torch_version": zip_bound.TORCH_VERSION,
        "torch_git_revision": zip_bound.TORCH_GIT_REVISION, "pickle_protocol": 2,
        "new_zipfile_serialization_required": True, "byteorder_record_required": True,
        "byteorder": "little", "compute_crc32": True, "storage_alignment_bytes": 64,
        "use_pinned_memory_for_d2h": False, "skip_data": False, "archive_prefix": "archive/",
        "package_registry_verified": True, "save_call_defaults_verified": True,
        "installed_source_hashes_verified": True, "upstream_source_hashes_bound": True,
        "require_cuda_uninitialized": True, "cuda_initialized": False,
        "cuda_initialization_unchanged": True, "source_revision_and_configuration_admitted": True,
        "compiled_binary_provenance_attested": False}
    return dict(schema=authority.MEASUREMENT_SCHEMA, status="complete",
        evidence_role=authority.MEASUREMENT_ROLE, attempt_id=supervision.ATTEMPT_ID,
        repository_revision=C, source_set_sha256=S,
        diagnostic_environment=__import__('test_storage_crosscheck_environment').synthetic_metadata()[1],
        measurement_protocol=authority.MEASUREMENT_PROTOCOL,
        component_rows=rows, serializer_runtime=dict(zip_runtime=zip_record,
            pickle_runtime_checked=True, compiled_binary_provenance_attested=False),
        resource_limits=dict(authority.RESOURCE_LIMITS), observed_resources=dict(
            elapsed_wall_ns=1, elapsed_cpu_ns=1, peak_rss_bytes=1,
            peak_buffer_bytes=max(row["observation"]["body_bytes"] for row in rows),
            runner_pid=1, process_scope="runner_self", cuda_visible_devices="",
            cuda_initialized_before=False, cuda_initialized_after=False,
            thread_settings=dict(authority.THREAD_SETTINGS), rng_preserved=True),
        supervision=None, failure=None, execution_authorized=False, scientific_execution_certified=False)


def synthetic_measurement():
    return bind_synthetic_supervision(copy.deepcopy(_synthetic_measurement_base()))


def bind_synthetic_supervision(value):
    """Invented controller observations for parser tests, never execution evidence."""
    import native_storage_measurement_service as service
    candidate = dict(value)
    candidate["supervision"] = None
    raw = authority.encode_bounded(candidate, maximum=service.STDOUT_MAX)
    snap = dict(device=1, inode=2, memory_max_bytes=service.MEMORY_MAX,
        memory_swap_max_bytes=0, tasks_max=128, memory_peak_bytes=4096,
        memory_events={key: 0 for key in service.EVENT_FIELDS},
        member_pids=[2], sampled_rss_bytes=4096, memory_zswap_max_bytes=0,
        memory_oom_group=1, pids_max_events=0, nr_descendants=0)
    value["supervision"] = dict(schema="i7_storage_service_observation_v1",
        mode="systemd_user_service_cgroup_v2", unit=service.UNIT, control_group=service.CGROUP,
        applied_properties=dict(service.PROPERTIES), process_environment=service.process_environment(),
        controller_pid=2,
        invocation_id="d" * 32, service_active_enter_ns=1, worker_pid=1,
        worker_start_ticks=1, worker_process_group=1,
        worker_exit_code=0 if value["status"] == "complete" else 1,
        candidate_size_bytes=len(raw), candidate_sha256=hashlib.sha256(raw).hexdigest(),
        stderr_bytes=0, elapsed_wall_ns=1, max_poll_wait_ns=service.SAMPLE_NS,
        sampled_aggregate_peak_rss_bytes=4096,
        rss_scope="sum_cgroup_member_VmRSS_sampled_not_kernel_charge",
        wall_scope="service_activation_through_validation_before_slot_finalize",
        before=copy.deepcopy(snap), after=copy.deepcopy(snap), worker_group_gone=True)
    return value


def check(value, **kwargs):
    # Existing tests mutate diagnostic content, so regenerate their explicitly
    # invented channel binding. Dedicated supervision tests never use this shim.
    bind_synthetic_supervision(value)
    return authority.validate_measurement(value, expected_commit=C,
        expected_source_set_sha256=S, **kwargs)


class StorageAuthorityTests(unittest.TestCase):
    def test_runtime_candidate_cross_binds_measurement_root_to_sources(self):
        import identity_codec as codec
        import native_write_ledger as ledger
        from test_native_layout_inspection import _success_report
        fixture = _success_report()
        sources, environment = fixture['source_binding'], fixture['native_environment_binding']
        value = synthetic_measurement()
        digest = codec.tree_digest(sources)
        value['source_set_sha256'] = digest
        value['diagnostic_environment']['repository_root_realpath'] = '/different/root'
        bind_synthetic_supervision(value)
        mraw = authority.encode_bounded(value)
        mpin = dict(path=supervision.LAYOUT_EVIDENCE_PATH, size_bytes=len(mraw),
                    sha256=hashlib.sha256(mraw).hexdigest())
        candidate = authority.make_admission(mpin, expected_commit=C, expected_source_set_sha256=digest)
        raw = authority.encode_bounded(candidate)
        pin = dict(path=supervision.STORAGE_ADMISSION_PATH, size_bytes=len(raw),
                   sha256=hashlib.sha256(raw).hexdigest())
        def read(**kwargs):
            return raw if kwargs['path'] == pin['path'] else mraw
        with mock.patch.object(supervision, 'load_pinned_bytes', side_effect=read), \
             mock.patch.object(ledger, 'compute', side_effect=AssertionError('must reject first')):
            with self.assertRaisesRegex(authority.StorageAuthorityError, 'repository roots differ'):
                authority.validate_runtime_admission(pin, sources=sources, environment=environment)

    def test_measurement_v2_binds_complete_diagnostic_environment(self):
        for field, replacement in (("schema", "i7_native_storage_measurement_v1"),
                                  ("measurement_protocol", "i7_complete_structural_recipe_v1"),
                                  ("diagnostic_environment", None)):
            value = synthetic_measurement()
            value[field] = replacement
            with self.assertRaises(authority.StorageAuthorityError):
                check(value)
        for role in ('cpu_audit', 'native_source', 'fixture_cpu'):
            value = synthetic_measurement()
            value['diagnostic_environment']['runtime_role'] = role
            with self.assertRaises(authority.StorageAuthorityError):
                check(value)
        value = synthetic_measurement()
        value['diagnostic_environment']['operating_system']['release'] = 'x' * 8192
        with self.assertRaises(authority.StorageAuthorityError):
            check(value)

    def test_missing_environment_only_before_any_component_work(self):
        value = synthetic_measurement()
        value.update(status='failed', diagnostic_environment=None,
                     failure=dict(stage='source', component=None, reason='contract_failure'))
        original = copy.deepcopy(value['component_rows'][0])
        for row in value['component_rows']:
            row.update(status='not_run', observation=None)
        check(value, require_complete=False)
        value['component_rows'][0] = original
        with self.assertRaises(authority.StorageAuthorityError):
            check(value, require_complete=False)
        value['component_rows'][0].update(status='not_run', observation=None)
        value['failure']['stage'] = 'recipe'
        with self.assertRaises(authority.StorageAuthorityError):
            check(value, require_complete=False)

    def test_default_import_and_structural_calculation_are_torch_free(self):
        code = (f"import sys; sys.path.insert(0,{str(HERE)!r}); "
                "import native_storage_authority as a; "
                "from test_native_storage_authority import synthetic_measurement,check; "
                "check(synthetic_measurement(),recompute=True); "
                "assert 'torch' not in sys.modules; assert a.main([])==0; "
                "assert 'torch' not in sys.modules")
        run = subprocess.run([sys.executable, "-I", "-c", code],
                             capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_exact_complete_record_roundtrip_below_cap(self):
        value = synthetic_measurement()
        check(value, recompute=True)
        raw = authority.encode_bounded(value)
        self.assertLess(len(raw), 64 << 10)
        self.assertEqual(authority.decode_bounded(raw), value)
        self.assertEqual(tuple(value), authority.MEASUREMENT_FIELDS)

    def test_duplicate_noncanonical_nonfinite_huge_and_cyclic_rejected(self):
        for raw in (b'{"a":1,"a":2}\n', b'{ "a":1}\n', b'{"a":NaN}\n', b'[]', b'1e999\n',
                    b'"' + b'x' * (64 << 10) + b'"\n'):
            with self.subTest(raw=raw[:50]), self.assertRaises(authority.StorageAuthorityError):
                authority.decode_bounded(raw)
        cycle = []
        cycle.append(cycle)
        for value in (cycle, "x" * (64 << 10), {"a": 1 << 65}, {"a": float("nan")},
                      {"a": "\ud800"}, [0] * (64 << 10)):
            with self.assertRaises(authority.StorageAuthorityError):
                authority.encode_bounded(value)

    def test_exact_membership_order_and_source_revision(self):
        base = synthetic_measurement()
        variants = []
        for field, item in (("repository_revision", "d" * 40), ("source_set_sha256", "e" * 64),
                            ("execution_authorized", True), ("scientific_execution_certified", 0)):
            value = copy.deepcopy(base)
            value[field] = item
            variants.append(value)
        value = copy.deepcopy(base)
        value["component_rows"].reverse()
        variants.append(value)
        value = copy.deepcopy(base)
        value["component_rows"].pop()
        variants.append(value)
        variants.append(dict(reversed(list(base.items()))))
        for value in variants:
            with self.assertRaises(authority.StorageAuthorityError):
                check(value)

    def test_recomputation_rejects_self_reported_ceiling_and_inventory(self):
        for field in ("analytic_body_bytes_upper", "analytic_pickle_bytes_upper"):
            value = synthetic_measurement()
            value["component_rows"][0][field] -= 1
            check(value)  # Structural mode explicitly is not arithmetic authority.
            with self.assertRaisesRegex(authority.StorageAuthorityError, "current calculation"):
                check(value, recompute=True)
        value = synthetic_measurement()
        value["component_rows"][0]["observation"]["storage_count"] -= 1
        with self.assertRaisesRegex(authority.StorageAuthorityError, "inventory"):
            check(value, recompute=True)

    def test_observation_limits_and_exact_types(self):
        for field, item in (("body_bytes", 64 << 20), ("pickle_bytes", True),
                            ("storage_count", False), ("serialized_sha256", "bad"),
                            ("restricted_cpu_roundtrip", 1)):
            value = synthetic_measurement()
            value["component_rows"][0]["observation"][field] = item
            with self.assertRaises(authority.StorageAuthorityError):
                check(value, recompute=True)

    def test_resource_conditions_and_serializer_claims(self):
        for key, item in (("elapsed_wall_ns", 120_000_000_001), ("peak_rss_bytes", (2 << 30) + 1),
                          ("peak_buffer_bytes", (64 << 20) + 1), ("peak_buffer_bytes", 1),
                          ("cuda_visible_devices", "0"),
                          ("cuda_initialized_after", True), ("rng_preserved", False),
                          ("process_scope", "all_children")):
            value = synthetic_measurement()
            value["observed_resources"][key] = item
            with self.assertRaises(authority.StorageAuthorityError):
                check(value)
        value = synthetic_measurement()
        value["serializer_runtime"]["compiled_binary_provenance_attested"] = True
        with self.assertRaises(authority.StorageAuthorityError):
            check(value)
        value = synthetic_measurement()
        value["serializer_runtime"]["zip_runtime"]["pickle_protocol"] = 3
        with self.assertRaisesRegex(authority.StorageAuthorityError, "ZIP runtime"):
            check(value)

    def test_failed_reports_preserve_all_eleven_rows_but_cannot_admit(self):
        value = synthetic_measurement()
        value["status"] = "failed"
        for row in value["component_rows"]:
            row["status"], row["observation"] = "not_run", None
        value["serializer_runtime"] = dict(zip_runtime=None, pickle_runtime_checked=False,
                                           compiled_binary_provenance_attested=False)
        value["failure"] = dict(stage="source", component=None, reason="contract_failure")
        check(value, require_complete=False, recompute=True)
        with self.assertRaises(authority.StorageAuthorityError):
            check(value)
        value["component_rows"][0]["status"] = "failed"
        value["failure"]["component"] = "anchor_pilot"
        check(value, require_complete=False)
        value["component_rows"][1]["status"] = "failed"
        with self.assertRaises(authority.StorageAuthorityError):
            check(value, require_complete=False)

    def test_admission_exact_pin_canonical_bytes_and_nested_measurement(self):
        mraw = authority.encode_bounded(synthetic_measurement())
        mpin = dict(path=supervision.LAYOUT_EVIDENCE_PATH, size_bytes=len(mraw),
                    sha256=hashlib.sha256(mraw).hexdigest())
        value = authority.make_admission(mpin, expected_commit=C, expected_source_set_sha256=S)
        raw = authority.encode_bounded(value)
        pin = dict(path=supervision.STORAGE_ADMISSION_PATH, size_bytes=len(raw),
                   sha256=hashlib.sha256(raw).hexdigest())
        with mock.patch.object(supervision, "load_pinned_bytes", return_value=mraw) as reader:
            self.assertEqual(authority.validate_structural_admission(raw, pin,
                expected_commit=C, expected_source_set_sha256=S), value)
            reader.assert_called_once_with(**mpin, maximum=authority.EVIDENCE_MAX)
            changed = dict(pin, sha256="d" * 64)
            with self.assertRaises(authority.StorageAuthorityError):
                authority.validate_structural_admission(raw, changed,
                    expected_commit=C, expected_source_set_sha256=S)

    def test_pin_fields_and_fixed_path(self):
        good = dict(path=supervision.STORAGE_ADMISSION_PATH, size_bytes=1, sha256="a" * 64)
        authority.validate_storage_admission_pin(good)
        for changed in (dict(good, path="/tmp/elsewhere"), dict(good, size_bytes=True),
                        dict(good, size_bytes=(64 << 10) + 1), dict(good, sha256="bad")):
            with self.assertRaises(authority.StorageAuthorityError):
                authority.validate_storage_admission_pin(changed)

    def test_direct_final_short_writes_private_mode_no_sidecars_and_no_retry(self):
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            target = str(Path(directory) / "fixture.json")
            real_write = os.write
            def short(fd, data):
                return real_write(fd, data[:2])
            with mock.patch.object(os, "write", side_effect=short):
                pin = authority._write_exclusive(target, b'{"a":1}\n', maximum=8192)
            self.assertEqual(Path(target).read_bytes(), b'{"a":1}\n')
            self.assertEqual(Path(target).stat().st_mode & 0o777, 0o600)
            self.assertEqual(pin["size_bytes"], 8)
            self.assertEqual(os.listdir(directory), ["fixture.json"])
            with self.assertRaises(FileExistsError):
                authority._write_exclusive(target, b'{"b":2}\n', maximum=8192)
            self.assertEqual(Path(target).read_bytes(), b'{"a":1}\n')

    def test_partial_write_failure_preserved_no_retry(self):
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            target = str(Path(directory) / "fixture.json")
            real_write = os.write
            calls = []
            def fail(fd, data):
                calls.append(None)
                if len(calls) > 1:
                    raise OSError("synthetic write failure")
                return real_write(fd, data[:2])
            with mock.patch.object(os, "write", side_effect=fail), self.assertRaises(OSError):
                authority._write_exclusive(target, b'{"a":1}\n', maximum=8192)
            self.assertEqual(Path(target).read_bytes(), b'{"')
            self.assertEqual(os.listdir(directory), ["fixture.json"])
            with self.assertRaises(FileExistsError):
                authority._write_exclusive(target, b'{"a":1}\n', maximum=8192)

    def test_zero_write_fsync_failure_and_symlinks_preserved(self):
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            root = Path(directory)
            with mock.patch.object(os, "write", return_value=0), self.assertRaises(authority.StorageAuthorityError):
                authority._write_exclusive(str(root / "zero.json"), b'{}\n', maximum=8192)
            self.assertEqual((root / "zero.json").stat().st_size, 0)
            with mock.patch.object(os, "fsync", side_effect=OSError("synthetic fsync failure")), self.assertRaises(OSError):
                authority._write_exclusive(str(root / "fsync.json"), b'{}\n', maximum=8192)
            self.assertEqual((root / "fsync.json").read_bytes(), b'{}\n')
            (root / "link").symlink_to(root, target_is_directory=True)
            with self.assertRaises(OSError):
                authority._write_exclusive(str(root / "link" / "new.json"), b'{}\n', maximum=8192)
            self.assertFalse((root / "new.json").exists())

    def test_existing_admission_blocks_measurement_even_zero_or_symlink(self):
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            apath = str(Path(directory) / "fixture-admission.json")
            mpath = str(Path(directory) / "fixture-measurement.json")
            with mock.patch.object(supervision, "STORAGE_ADMISSION_PATH", apath), \
                 mock.patch.object(supervision, "LAYOUT_EVIDENCE_PATH", mpath):
                Path(apath).touch()
                with self.assertRaisesRegex(authority.StorageAuthorityError, "consumed"):
                    authority.write_measurement(synthetic_measurement(), expected_commit=C,
                                                expected_source_set_sha256=S)
                self.assertFalse(Path(mpath).exists())
                Path(apath).unlink()  # Only this test's empty synthetic file.
                Path(apath).symlink_to(Path(directory) / "absent-target")
                with self.assertRaisesRegex(authority.StorageAuthorityError, "consumed"):
                    authority.write_measurement(synthetic_measurement(), expected_commit=C,
                                                expected_source_set_sha256=S)
                self.assertTrue(Path(apath).is_symlink())
                self.assertFalse(Path(mpath).exists())

    def test_measurement_writer_only_uses_its_fixed_slot(self):
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            apath = str(Path(directory) / "fixture-admission.json")
            mpath = str(Path(directory) / "fixture-measurement.json")
            with mock.patch.object(supervision, "STORAGE_ADMISSION_PATH", apath), \
                 mock.patch.object(supervision, "LAYOUT_EVIDENCE_PATH", mpath):
                value = synthetic_measurement()
                pin = authority.write_measurement(value, expected_commit=C, expected_source_set_sha256=S)
                self.assertEqual(pin["path"], mpath)
                self.assertEqual(authority.decode_bounded(Path(mpath).read_bytes()), value)
                self.assertEqual(os.listdir(directory), ["fixture-measurement.json"])
                with self.assertRaises(authority.StorageAuthorityError):
                    authority.write_measurement(value, expected_commit=C, expected_source_set_sha256=S)

    def test_admission_writer_requires_complete_pinned_measurement(self):
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            apath = str(Path(directory) / "native-storage-admission.json")
            mpath = str(Path(directory) / "native-max-layout-measurement.json")
            with mock.patch.object(supervision, "STORAGE_ADMISSION_PATH", apath), \
                 mock.patch.object(supervision, "LAYOUT_EVIDENCE_PATH", mpath):
                mpin = authority.write_measurement(synthetic_measurement(), expected_commit=C,
                                                   expected_source_set_sha256=S)
                candidate = authority.make_admission(mpin, expected_commit=C, expected_source_set_sha256=S)
                pin = authority.write_reviewed_admission(candidate)
                self.assertEqual(authority.decode_bounded(Path(apath).read_bytes()), candidate)
                self.assertEqual(pin["path"], apath)
                self.assertEqual(set(os.listdir(directory)), {"native-storage-admission.json", "native-max-layout-measurement.json"})
                with self.assertRaises(authority.StorageAuthorityError):
                    authority.write_reviewed_admission(candidate)

    def test_later_writer_authenticates_before_and_after_single_slot(self):
        import native_control as control
        import phase_transition as transition
        # Dependency mocks isolate the writer, not evidence of native authority.
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            permission = dict(phase="primary", expected_commit=C, expected_source_set_sha256=S,
                              storage_admission_pin=dict(path="/fixture-only", size_bytes=1, sha256="a" * 64))
            raw = authority.encode_bounded(permission, maximum=8192)
            with mock.patch.object(control, "DEVELOPMENT_ATTEMPT_PATH", directory), \
                 mock.patch.object(transition, "authenticate_permission_for_write", return_value=(raw, permission)) as before, \
                 mock.patch.object(transition, "_authenticate_actual", return_value=(raw, permission)) as after, \
                 mock.patch.object(supervision, "load_pinned_bytes", wraps=supervision.load_pinned_bytes) as reader, \
                 mock.patch.object(authority, "validate_structural_admission") as validate:
                real_read = reader._mock_wraps
                reader.side_effect = lambda **kw: b'x' if kw["path"] == "/fixture-only" else real_read(**kw)
                pin = authority.write_phase_permission({"fixture": "only"}, prior_boundary={"sealed": "fixture"})
                self.assertEqual(os.listdir(directory), ["native-primary-permission.json"])
                before.assert_called_once_with({"fixture": "only"}, prior_boundary={"sealed": "fixture"},
                                                attempt_parent=directory, fixture=False)
                after.assert_called_once_with(pin, "primary", {"sealed": "fixture"}, "native_producer_attestation")
                validate.assert_called_once()

    def test_invalid_schema_or_exceeded_free_floor_creates_no_file(self):
        with tempfile.TemporaryDirectory(prefix="i7-authority-unit-") as directory:
            target = str(Path(directory) / "fixture.json")
            empty_space = type("Space", (), {"f_bavail": 0, "f_frsize": 4096})()
            with mock.patch.object(os, "fstatvfs", return_value=empty_space), self.assertRaises(authority.StorageAuthorityError):
                authority._write_exclusive(target, b'{}\n', maximum=8192)
            self.assertEqual(os.listdir(directory), [])


if __name__ == "__main__":
    unittest.main()
