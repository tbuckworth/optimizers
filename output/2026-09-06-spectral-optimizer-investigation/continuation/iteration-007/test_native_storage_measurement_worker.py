"""Tiny dependency-substitute tests; never materialize the real inventory."""
import copy
import gc
from pathlib import Path
import subprocess
import sys
import unittest
import weakref
from unittest import mock

import native_storage_measurement_worker as subject


C, S = "a" * 40, "b" * 64
ROOT = "/fixture/frozen-root"
CGROUP = "/fixture.slice/i7-native-storage-measurement-002.service"


class _Tree:
    pass


class _Fixture:
    def __init__(self):
        self.pid = 101
        self.parent = 100
        self.wall = 10
        self.cpu = 10
        self.rss = 1024
        self.python_state = ("python",)
        self.rng_state = 7
        self.source_calls = 0
        self.environment_calls = 0
        self.built = []
        self.live = []
        self.validated = []
        self.fail = None
        self.fail_component = None
        self.reject_complete_report = False
        self.reject_all_reports = False
        self.drift_source = False
        self.drift_environment = False
        self.specs = tuple(dict(component=name, encoding=encoding, payload_count=count,
            analytic_pickle_bytes_upper=None if encoding == "bytes" else 128,
            analytic_body_bytes_upper=256, tensor_count=0)
            for name, encoding, count in subject.COMPONENTS)

    def dependencies(self):
        return {
            "fixed_root": ROOT,
            "attempt_id": "attempt",
            "measurement_schema": "measurement-v3",
            "measurement_protocol": "recipe-v2",
            "measurement_role": "cpu-crosscheck",
            "resource_limits": {"wall_ns": 1_000, "rss_bytes": 1 << 20,
                                "buffer_bytes": 1 << 16},
            "thread_settings": {key: "1" for key in subject.THREAD_NAMES},
            "monotonic_ns": lambda: self.wall,
            "process_time_ns": lambda: self.cpu,
            "peak_rss_bytes": lambda: self.rss,
            "current_pid": lambda: self.pid,
            "parent_pid": lambda: self.parent,
            "assert_service_member": self.assert_service_member,
            "configure_runtime": self.configure_runtime,
            "python_rng": lambda: self.python_state,
            "capture_rng": lambda: self.rng_state,
            "rng_equal": lambda left, right: left == right,
            "cuda_initialized": lambda: False,
            "cuda_visible_devices": lambda: "",
            "observed_threads": lambda: {key: "1" for key in subject.THREAD_NAMES},
            "collect_sources": self.collect_sources,
            "source_digest": lambda value: value["digest"],
            "collect_environment": self.collect_environment,
            "validate_metadata": self.validate_metadata,
            "component_specs": lambda: copy.deepcopy(self.specs),
            "build_template": self.build_template,
            "validate_template": self.validate_template,
            "materialize": self.materialize,
            "json_roundtrip": self.json_roundtrip,
            "serialize_roundtrip": self.serialize_roundtrip,
            "collect_garbage": gc.collect,
            "validate_candidate": self.validate_candidate,
        }

    def assert_service_member(self, *, controller_pid, expected_cgroup):
        if self.fail == "service":
            raise RuntimeError
        if controller_pid != self.parent:
            raise RuntimeError
        return expected_cgroup

    def configure_runtime(self):
        if self.fail == "configure":
            raise RuntimeError

    def collect_sources(self, root):
        self.source_calls += 1
        if self.fail == "source" and self.source_calls == 1:
            raise RuntimeError
        digest = "c" * 64 if self.drift_source and self.source_calls == 2 else S
        return {"repository_root_realpath": root, "repository_revision": C,
                "digest": digest}

    def collect_environment(self, root):
        self.environment_calls += 1
        if self.fail == "runtime" and self.environment_calls == 1:
            raise RuntimeError
        suffix = 1 if self.drift_environment and self.environment_calls == 2 else 0
        return {"repository_root_realpath": root, "runtime_role": "storage_crosscheck_cpu",
                "suffix": suffix}

    def validate_metadata(self, sources, environment):
        if sources["repository_root_realpath"] != environment["repository_root_realpath"]:
            raise RuntimeError

    def build_template(self, component, *, sources, environment):
        if any(item() is not None for item in self.live):
            raise AssertionError("previous materialization retained")
        if self.fail == "recipe" and component == self.fail_component:
            raise RuntimeError
        self.built.append(component)
        return {"component": component}

    def validate_template(self, component, template):
        spec = next(row for row in self.specs if row["component"] == component)
        return {"slot_count": spec["tensor_count"],
                "body_bytes_upper": spec["analytic_body_bytes_upper"]}

    def materialize(self, component, template):
        value = _Tree()
        value.component = component
        self.live.append(weakref.ref(value))
        return value

    @staticmethod
    def _observation(encoding):
        return dict(pickle_bytes=None if encoding == "bytes" else 3, body_bytes=10,
            storage_count=0, raw_storage_bytes=0, serialized_sha256="d" * 64,
            restricted_cpu_roundtrip=True, exact_tree_roundtrip=True)

    def _roundtrip(self, tree, *, on_buffer_peak, on_stage, encoding):
        on_stage("serialize")
        on_buffer_peak(10)
        if self.fail == "serialize" and tree.component == self.fail_component:
            raise RuntimeError
        on_stage("roundtrip")
        if self.fail == "roundtrip" and tree.component == self.fail_component:
            raise RuntimeError
        return self._observation(encoding)

    def json_roundtrip(self, tree, *, body_ceiling, on_buffer_peak, on_stage):
        return self._roundtrip(tree, on_buffer_peak=on_buffer_peak,
                               on_stage=on_stage, encoding="bytes")

    def serialize_roundtrip(self, tree, *, pickle_ceiling, body_ceiling,
                            tensor_count, on_buffer_peak, on_stage):
        return (self._roundtrip(tree, on_buffer_peak=on_buffer_peak,
                                on_stage=on_stage, encoding="torch_weights_only"),
                {"runtime": "fixed"})

    def validate_candidate(self, value, *, expected_commit, expected_source_set_sha256):
        self.validated.append(value["status"])
        if self.reject_all_reports or (self.reject_complete_report and value["status"] == "complete"):
            raise ValueError
        if value["supervision"] is not None or value["repository_revision"] != expected_commit \
                or value["source_set_sha256"] != expected_source_set_sha256:
            raise ValueError
        statuses = [row["status"] for row in value["component_rows"]]
        if value["status"] == "complete":
            if value["failure"] is not None or statuses != ["measured"] * 11:
                raise ValueError
        else:
            if value["failure"] is None:
                raise ValueError
            component = value["failure"]["component"]
            failed = [row["component"] for row in value["component_rows"]
                      if row["status"] == "failed"]
            if failed != ([] if component is None else [component]):
                raise ValueError
        return value


def _run(fixture):
    return subject._build_with_dependencies(expected_commit=C,
        expected_source_set_sha256=S, repository_root=ROOT, entry_wall_ns=0,
        entry_cpu_ns=0, runner_pid=fixture.pid, controller_pid=fixture.parent,
        expected_cgroup=CGROUP, python_rng_before=fixture.python_state,
        dependencies=fixture.dependencies())


class NativeStorageMeasurementWorkerTests(unittest.TestCase):
    def test_default_import_and_cli_are_inert_and_torch_free(self):
        here = Path(__file__).resolve().parent
        code = (f"import sys; sys.path.insert(0,{str(here)!r}); "
                "import native_storage_measurement_worker as w; "
                "assert w.main([])==0; assert 'torch' not in sys.modules; "
                "assert 'numpy' not in sys.modules")
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=here,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_fixed_components_match_actual_pure_topology_and_inventory(self):
        here = Path(__file__).resolve().parent
        code = (f"import sys; sys.path.insert(0,{str(here)!r}); "
                "import native_storage_measurement_worker as w; "
                "import native_storage_topology_bound as t; import native_tensor_inventory as i; "
                "b=t.compute()['components']; l=i.component_layouts(); "
                "assert tuple(b)==tuple(l)==tuple(x[0] for x in w.COMPONENTS); "
                "assert tuple((n,b[n]['encoding'],b[n]['payload_count']) for n in b)==w.COMPONENTS; "
                "assert all(len(l[n])<=132 and b[n]['body_bytes_upper']<=64<<20 for n in b); "
                "assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=here,
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_complete_eleven_are_sequential_released_and_unwritten(self):
        fixture = _Fixture()
        value = _run(fixture)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(tuple(row["component"] for row in value["component_rows"]),
                         tuple(name for name, _, _ in subject.COMPONENTS))
        self.assertEqual([row["status"] for row in value["component_rows"]], ["measured"] * 11)
        self.assertEqual(fixture.source_calls, 2)
        self.assertEqual(fixture.environment_calls, 2)
        self.assertTrue(all(item() is None for item in fixture.live))
        self.assertEqual(value["observed_resources"]["peak_buffer_bytes"], 10)
        self.assertIsNone(value["supervision"])
        self.assertEqual(value["serializer_runtime"], {
            "zip_runtime": {"runtime": "fixed"}, "pickle_runtime_checked": True,
            "compiled_binary_provenance_attested": False})

    def test_source_and_runtime_failures_have_no_environment_or_component_claim(self):
        for stage in ("source", "runtime"):
            with self.subTest(stage=stage):
                fixture = _Fixture()
                fixture.fail = stage
                value = _run(fixture)
                self.assertEqual(value["status"], "failed")
                self.assertIsNone(value["diagnostic_environment"])
                self.assertEqual(value["failure"], {
                    "stage": stage, "component": None, "reason": "contract_failure"})
                self.assertEqual([row["status"] for row in value["component_rows"]],
                                 ["not_run"] * 11)

    def test_component_failure_is_exact_measured_failed_not_run_prefix(self):
        for stage in ("recipe", "serialize", "roundtrip"):
            with self.subTest(stage=stage):
                fixture = _Fixture()
                fixture.fail = stage
                fixture.fail_component = subject.COMPONENTS[3][0]
                value = _run(fixture)
                self.assertEqual([row["status"] for row in value["component_rows"]],
                                 ["measured"] * 3 + ["failed"] + ["not_run"] * 7)
                self.assertEqual(value["failure"], {"stage": stage,
                    "component": fixture.fail_component, "reason": "contract_failure"})
                self.assertTrue(all(item() is None for item in fixture.live))

    def test_post_component_source_or_environment_drift_has_no_false_failed_row(self):
        for field, stage in (("drift_source", "source"), ("drift_environment", "runtime")):
            with self.subTest(field=field):
                fixture = _Fixture()
                setattr(fixture, field, True)
                value = _run(fixture)
                self.assertEqual([row["status"] for row in value["component_rows"]],
                                 ["measured"] * 11)
                self.assertEqual(value["failure"], {
                    "stage": stage, "component": None, "reason": "contract_failure"})

    def test_rng_drift_is_truthful_terminal_runtime_failure(self):
        fixture = _Fixture()
        calls = [7, 8, 8]
        dependencies = fixture.dependencies()
        dependencies["capture_rng"] = lambda: calls.pop(0)
        value = subject._build_with_dependencies(expected_commit=C,
            expected_source_set_sha256=S, repository_root=ROOT, entry_wall_ns=0,
            entry_cpu_ns=0, runner_pid=fixture.pid, controller_pid=fixture.parent,
            expected_cgroup=CGROUP, python_rng_before=fixture.python_state,
            dependencies=dependencies)
        self.assertFalse(value["observed_resources"]["rng_preserved"])
        self.assertEqual(value["failure"], {
            "stage": "runtime", "component": "anchor_pilot", "reason": "contract_failure"})
        self.assertEqual([row["status"] for row in value["component_rows"]],
                         ["failed"] + ["not_run"] * 10)

    def test_resource_limit_before_environment_is_valid_runtime_failure(self):
        fixture = _Fixture()
        fixture.rss = (1 << 20) + 1
        value = _run(fixture)
        self.assertIsNone(value["diagnostic_environment"])
        self.assertEqual(value["failure"], {
            "stage": "runtime", "component": None, "reason": "resource_limit"})
        self.assertEqual([row["status"] for row in value["component_rows"]], ["not_run"] * 11)

    def test_complete_report_rejection_falls_back_without_false_component_failure(self):
        fixture = _Fixture()
        fixture.reject_complete_report = True
        value = _run(fixture)
        self.assertEqual(fixture.validated, ["complete", "failed"])
        self.assertEqual(value["failure"], {
            "stage": "report", "component": None, "reason": "contract_failure"})
        self.assertEqual([row["status"] for row in value["component_rows"]], ["measured"] * 11)

    def test_unknown_bootstrap_and_unreportable_fail_closed(self):
        fixture = _Fixture()
        fixture.fail = "configure"
        with self.assertRaisesRegex(subject.MeasurementWorkerBootstrapError, "runtime_unobserved"):
            _run(fixture)
        fixture = _Fixture()
        fixture.reject_all_reports = True
        with self.assertRaisesRegex(subject.MeasurementWorkerBootstrapError, "report_unobserved"):
            _run(fixture)

    def test_preflight_rejects_wrong_process_or_service_before_work(self):
        for mutation in ("pid", "parent", "root", "cgroup"):
            fixture = _Fixture()
            kwargs = dict(expected_commit=C, expected_source_set_sha256=S,
                repository_root=ROOT, entry_wall_ns=0, entry_cpu_ns=0,
                runner_pid=fixture.pid, controller_pid=fixture.parent,
                expected_cgroup=CGROUP, python_rng_before=fixture.python_state,
                dependencies=fixture.dependencies())
            if mutation == "pid":
                kwargs["runner_pid"] += 1
            elif mutation == "parent":
                kwargs["controller_pid"] += 1
            elif mutation == "root":
                kwargs["repository_root"] = "/wrong"
            else:
                fixture.fail = "service"
                kwargs["dependencies"] = fixture.dependencies()
            with self.subTest(mutation=mutation), self.assertRaises(subject.MeasurementWorkerError):
                subject._build_with_dependencies(**kwargs)

    def test_public_bad_fixed_root_fails_before_loading_runtime_dependencies(self):
        with mock.patch.object(subject, "_production_dependencies",
                               side_effect=AssertionError("must remain Torch-free")):
            with self.assertRaises(subject.MeasurementWorkerBootstrapError):
                subject.build_measurement_candidate(expected_commit=C,
                    expected_source_set_sha256=S, repository_root="/wrong",
                    entry_wall_ns=0, entry_cpu_ns=0, runner_pid=1,
                    controller_pid=2, expected_cgroup=CGROUP)

    def test_public_requires_preconfigured_environment_and_service_before_runtime_import(self):
        here = Path(__file__).resolve().parent
        code = f'''import os,sys,time
sys.path.insert(0,{str(here)!r})
from unittest import mock
import native_storage_measurement_worker as w
import native_storage_measurement_service as service
import process_supervision as supervision
required=service.process_environment()
kwargs=dict(expected_commit={C!r},expected_source_set_sha256={S!r},
 repository_root=supervision.REPOSITORY_ROOT,entry_wall_ns=time.monotonic_ns(),
 entry_cpu_ns=time.process_time_ns(),runner_pid=os.getpid(),controller_pid=os.getppid(),
 expected_cgroup=service.CGROUP)
bad=dict(required);bad["OMP_NUM_THREADS"]="2"
with mock.patch.dict(os.environ,bad,clear=True), \
     mock.patch.object(service,"assert_service_member",side_effect=AssertionError("service")), \
     mock.patch.object(w,"_production_dependencies",side_effect=AssertionError("runtime")):
    try:w.build_measurement_candidate(**kwargs)
    except w.MeasurementWorkerBootstrapError as exc:assert exc.code=="preflight_unobserved"
    else:raise AssertionError("mismatch admitted")
order=[]
def membership(**unused):order.append("service");return service.CGROUP
def runtime():order.append("runtime");raise RuntimeError
with mock.patch.dict(os.environ,required,clear=True), \
     mock.patch.object(service,"assert_service_member",side_effect=membership), \
     mock.patch.object(w,"_production_dependencies",side_effect=runtime):
    try:w.build_measurement_candidate(**kwargs)
    except w.MeasurementWorkerBootstrapError as exc:assert exc.code=="runtime_unobserved"
    else:raise AssertionError("runtime unexpectedly returned")
assert order==["service","runtime"]
assert "torch" not in sys.modules and "numpy" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=here,
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
