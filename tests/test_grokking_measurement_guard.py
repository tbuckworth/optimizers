import importlib.util
from pathlib import Path
import unittest

PATH = (Path(__file__).resolve().parents[1] / "output" /
        "2026-09-09-spectral-grokking-action" / "guarded_measurement.py")
SPEC = importlib.util.spec_from_file_location("action_measurement_guard", PATH)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class MeasurementGuardTest(unittest.TestCase):
    def test_exact_effective_bounds(self):
        effective = {"memory.max": str(16 * 1024**3), "memory.swap.max": "0",
                     "cpu.max": "100000 100000"}
        service = {"RuntimeMaxUSec": "20min", "Restart": "no",
                   "KillMode": "control-group", "Type": "exec"}
        cgroup = "/user.slice/app.slice/" + guard.UNIT
        guard.validate_bounds(cgroup, effective, service)
        for key, value in (("memory.max", "max"), ("memory.swap.max", "1"),
                           ("cpu.max", "max 100000"), ("cpu.max", "400000 100000")):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                guard.validate_bounds(cgroup, {**effective, key: value}, service)
        with self.assertRaises(RuntimeError):
            guard.validate_bounds(cgroup + "-wrong", effective, service)
        with self.assertRaises(RuntimeError):
            guard.validate_bounds(cgroup, effective, {**service, "Type": "oneshot"})

    def test_refuses_live_or_failed_training(self):
        terminal = {"MainPID": "0", "ActiveState": "inactive", "Result": "success"}
        guard.validate_training_terminal(terminal)
        for key, value in (("MainPID", "1234"), ("ActiveState", "active"),
                           ("Result", "timeout")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_training_terminal({**terminal, key: value})


if __name__ == "__main__":
    unittest.main()
