"""CPU-hidden protocol tests; no native initialization or scientific payloads."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("runtime initialization tests require hidden CUDA")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_control as control


class RuntimeInitializationTests(unittest.TestCase):
    def runtime(self, *, initialized=False, identity="GPU-review-pinned"):
        guard = mock.Mock()
        runtime = SimpleNamespace(
            torch=SimpleNamespace(cuda=SimpleNamespace(
                is_initialized=lambda:initialized)),
            PHASE_CAPS={"primary":{"wall_seconds":600.0}}, RSS_CAP=12 << 30,
            _native_cuda_identity=lambda:{"stable_identity":identity},
            RuntimeGuard=mock.Mock(return_value=guard))
        return runtime, guard

    def invoke(self, runtime, callback, *, wall=None, pid=None, permission=None):
        return control._prepare_native_runtime(runtime,
            {"expected_gpu_uuid":"GPU-review-pinned"} if permission is None else permission,
            "primary", callback,
            entry_wall_origin=time.monotonic() if wall is None else wall,
            entry_cpu_origin=time.process_time(),
            entry_process_id=os.getpid() if pid is None else pid)

    def test_setup_uses_own_origins_and_validates_identity_before_guard(self):
        runtime, guard = self.runtime()
        callback = mock.Mock(return_value=None)
        wall = time.monotonic()
        self.assertIs(self.invoke(runtime, callback, wall=wall), guard)
        callback.assert_called_once()
        self.assertEqual(runtime.RuntimeGuard.call_args.kwargs["entry_wall_origin"], wall)
        self.assertEqual(runtime.RuntimeGuard.call_args.kwargs["entry_process_id"], os.getpid())
        guard.check.assert_called_once_with("runtime.initialized.after.consumption")

    def test_prechecks_reject_before_callback(self):
        for case in ("initialized", "pid", "expired", "rss"):
            with self.subTest(case=case), ExitStack() as stack:
                runtime, _ = self.runtime(initialized=case == "initialized")
                callback = mock.Mock(return_value=None)
                if case == "rss":
                    stack.enter_context(mock.patch.object(control, "_native_rss", return_value=13 << 30))
                with self.assertRaises(control.ControlError):
                    self.invoke(runtime, callback,
                        wall=time.monotonic()-601.0 if case == "expired" else None,
                        pid=os.getpid()+1 if case == "pid" else None)
                callback.assert_not_called()
                runtime.RuntimeGuard.assert_not_called()

    def test_postchecks_reject_payload_and_wrong_identity(self):
        for result, identity in (({}, "GPU-review-pinned"), (None, "GPU-other")):
            runtime, _ = self.runtime(identity=identity)
            with self.assertRaises(control.ControlError):
                self.invoke(runtime, lambda _:result)
            runtime.RuntimeGuard.assert_not_called()

    def test_callback_cannot_rewrite_reviewed_uuid(self):
        permission = {"expected_gpu_uuid":"GPU-review-pinned"}
        runtime, _ = self.runtime(identity="GPU-other")
        def mutate(copy):
            copy["expected_gpu_uuid"] = "GPU-other"
        with self.assertRaisesRegex(control.ControlError, "differs from permission"):
            self.invoke(runtime, mutate, permission=permission)
        self.assertEqual(permission, {"expected_gpu_uuid":"GPU-review-pinned"})

    def test_closure_mutation_of_authority_is_detected(self):
        permission = {"expected_gpu_uuid":"GPU-review-pinned"}
        runtime, _ = self.runtime()
        def mutate(_):
            permission["expected_gpu_uuid"] = "GPU-other"
        with self.assertRaisesRegex(control.ControlError, "changed authoritative"):
            self.invoke(runtime, mutate, permission=permission)
        runtime.RuntimeGuard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
