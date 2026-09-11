"""Pure guard fixtures: no service launch, tensors or scientific inputs."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "output/2026-09-09-spectral-raw-direction/guarded_measurement.py"
SPEC = importlib.util.spec_from_file_location("raw_measurement_guard", PATH)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class MeasurementGuardTests(unittest.TestCase):
    def test_effective_limits(self):
        group = "/synthetic/" + guard.UNIT
        actual = {"memory.max": str(16*1024**3), "memory.swap.max": "0", "cpu.max": "100000 100000"}
        service = {"RuntimeMaxUSec": "20min", "Restart": "no", "KillMode": "control-group", "Type": "exec"}
        guard.validate_bounds(group, actual, service)
        for key, value in (("memory.max", "max"), ("memory.swap.max", "max"),
                           ("cpu.max", "400000 100000"), ("cpu.max", "max 100000")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_bounds(group, {**actual, key: value}, service)
        with self.assertRaises(RuntimeError):
            guard.validate_bounds("/different.service", actual, service)
        with self.assertRaises(RuntimeError):
            guard.validate_bounds(group, actual, {**service, "Type": "oneshot"})

    def test_requires_successful_terminal_training(self):
        terminal = {"MainPID": "0", "ActiveState": "inactive", "Result": "success"}
        guard.validate_training_terminal(terminal)
        for key, value in (("MainPID", "123"), ("ActiveState", "active"), ("Result", "exit-code")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_training_terminal({**terminal, key: value})


if __name__ == "__main__":
    unittest.main()
