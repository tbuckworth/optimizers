"""Synthetic limits only; never execute the acquisition guard's main."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "output/2026-09-09-spectral-raw-direction/guarded_launch.py"
SPEC = importlib.util.spec_from_file_location("raw_direction_guard", PATH)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class GuardTests(unittest.TestCase):
    def test_effective_limits_and_service_contract(self):
        actual = {"memory.max": str(16 * 1024**3), "memory.swap.max": "0", "cpu.max": "400000 100000"}
        service = {"RuntimeMaxUSec": "12h", "Restart": "no", "KillMode": "control-group", "Type": "exec"}
        guard.check_limits(actual, service)
        guard.check_limits({**actual, "cpu.max": "100000 100000"}, service)
        for key, value in (("memory.max", "max"), ("memory.swap.max", "max"),
                           ("cpu.max", "max 100000"), ("cpu.max", "500000 100000"),
                           ("cpu.max", "0 100000")):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                guard.check_limits({**actual, key: value}, service)
        for key, value in (("Type", "oneshot"), ("Restart", "on-failure"),
                           ("RuntimeMaxUSec", "infinity"), ("KillMode", "process")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.check_limits(actual, {**service, key: value})


if __name__ == "__main__":
    unittest.main()
