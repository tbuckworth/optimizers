"""Synthetic layout-only tests for the single audit exception disposition."""
from pathlib import Path
import tempfile
import unittest

import audit_runtime_directory as supplement


class RuntimeDirectoryTests(unittest.TestCase):
    def source(self, root):
        for phase in supplement.PHASES:
            (root / phase).mkdir()
            (root / f"attempt-{phase}.json").write_text("{}")

    def test_empty_identity_and_nonempty_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.source(root)
            cache = root / "torchinductor_titus"
            cache.mkdir()
            first = supplement.runtime_identity(root)
            self.assertEqual(supplement.runtime_identity(root), first)
            (cache / ".hidden").write_text("material")
            with self.assertRaisesRegex(ValueError, "not empty"):
                supplement.runtime_identity(root)

    def test_file_symlink_and_unlisted_entries_reject(self):
        for kind in ("file", "symlink", "extra"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.source(root)
                cache = root / "torchinductor_titus"
                if kind == "file":
                    cache.write_text("")
                elif kind == "symlink":
                    cache.symlink_to(root, target_is_directory=True)
                else:
                    cache.mkdir()
                    (root / "extra").write_text("")
                with self.assertRaises((ValueError, OSError)):
                    supplement.runtime_identity(root)


if __name__ == "__main__":
    unittest.main()
