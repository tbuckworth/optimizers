import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "output/2026-09-09-spectral-function-response/guarded_launch.py"
SPEC = importlib.util.spec_from_file_location("function_response_guard", PATH)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class GuardTests(unittest.TestCase):
    def test_valid_limits(self):
        guard.validate_bounds("/user.slice/" + guard.UNIT,
            {"memory.max": str(16*1024**3), "memory.swap.max": "0", "cpu.max": "100000 100000"},
            {"RuntimeMaxUSec": "10min", "Restart": "no", "KillMode": "control-group", "Type": "exec"})

    def test_refuses_relaxed_limits(self):
        effective = {"memory.max": str(16*1024**3), "memory.swap.max": "0", "cpu.max": "100000 100000"}
        service = {"RuntimeMaxUSec": "10min", "Restart": "no", "KillMode": "control-group", "Type": "exec"}
        for key, value in (("memory.max", "max"), ("memory.swap.max", "1"),
                           ("cpu.max", "200000 100000"), ("cpu.max", "max 100000")):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                guard.validate_bounds("/" + guard.UNIT, {**effective, key: value}, service)
        for key, value in (("RuntimeMaxUSec", "20min"), ("Restart", "always"),
                           ("KillMode", "process"), ("Type", "simple")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_bounds("/" + guard.UNIT, effective, {**service, key: value})
        with self.assertRaises(RuntimeError):
            guard.validate_bounds("/wrong.service", effective, service)

    def test_terminal_not_live_or_failed(self):
        record = {"MainPID": "0", "ActiveState": "inactive", "Result": "success"}
        guard.validate_terminal(record)
        for key, value in (("MainPID", "112"), ("ActiveState", "active"), ("Result", "exit-code")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_terminal({**record, key: value})


if __name__ == "__main__":
    unittest.main()
