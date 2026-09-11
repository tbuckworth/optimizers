"""Tiny filesystem tests for the one-shot M slot; no native/scientific work."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native_control as control
import native_storage_authority as authority
import native_storage_measurement_slot as slot
from test_native_storage_authority import C, S, synthetic_measurement


def failed_measurement():
    value = synthetic_measurement()
    value["status"] = "failed"
    value["diagnostic_environment"] = None
    for row in value["component_rows"]:
        row["status"], row["observation"] = "not_run", None
    value["serializer_runtime"] = {
        "zip_runtime": None, "pickle_runtime_checked": False,
        "compiled_binary_provenance_attested": False}
    value["failure"] = {"stage": "source", "component": None,
                        "reason": "contract_failure"}
    from test_native_storage_authority import bind_synthetic_supervision
    return bind_synthetic_supervision(value)


class MeasurementSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        control._verify_big_parent(Path(slot.BIG_TMP))

    def setUp(self):
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            self.assertEqual(os.environ.get(key), "1")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="i7-native-measurement-slot-test-", dir=slot.BIG_TMP)
        self.parent = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def reserve(self):
        return slot._reserve_fixture(str(self.parent), C, S)

    def test_import_and_default_cli_are_inert_and_torch_free(self):
        measurement = Path(__import__('process_supervision').LAYOUT_EVIDENCE_PATH)
        admission = Path(__import__('process_supervision').STORAGE_ADMISSION_PATH)
        before = (measurement.exists(), admission.exists())
        code = (f"import sys;sys.path.insert(0,{str(HERE)!r});"
                "import native_storage_measurement_slot as s;"
                "assert 'torch' not in sys.modules;assert s.main([])==0;"
                "assert 'torch' not in sys.modules")
        run = subprocess.run([sys.executable, "-I", "-c", code],
                             capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr.decode())
        self.assertEqual(before, (measurement.exists(), admission.exists()))

    def test_empty_slot_is_private_and_close_permanently_consumes_it(self):
        reservation = self.reserve()
        path = self.parent / slot.MEASUREMENT_BASENAME
        info = path.stat()
        self.assertEqual((info.st_size, info.st_nlink, info.st_mode & 0o777), (0, 1, 0o600))
        reservation.close()
        with self.assertRaisesRegex(slot.MeasurementSlotError, "measurement_present"):
            self.reserve()
        self.assertEqual(path.read_bytes(), b"")
        with self.assertRaisesRegex(slot.MeasurementSlotError, "closed"):
            reservation.close()

    def test_preexisting_admission_or_measurement_refuses_without_mutation(self):
        admission = self.parent / slot.ADMISSION_BASENAME
        admission.write_bytes(b"review")
        with self.assertRaisesRegex(slot.MeasurementSlotError, "admission_present"):
            self.reserve()
        self.assertFalse((self.parent / slot.MEASUREMENT_BASENAME).exists())
        admission.unlink()
        measurement = self.parent / slot.MEASUREMENT_BASENAME
        measurement.write_bytes(b"partial")
        with self.assertRaisesRegex(slot.MeasurementSlotError, "measurement_present"):
            self.reserve()
        self.assertEqual(measurement.read_bytes(), b"partial")

    def test_noncanonical_bindings_reject_before_slot_creation(self):
        for commit, digest in (("A" * 40, S), (C, "B" * 64), (C[:-1], S),
                               (C, S + "0"), (None, S), (C, None)):
            with self.subTest(commit=commit, digest=digest), self.assertRaisesRegex(
                    slot.MeasurementSlotError, "invalid_binding"):
                slot._reserve_fixture(str(self.parent), commit, digest)
            self.assertFalse((self.parent / slot.MEASUREMENT_BASENAME).exists())

    def test_complete_and_structured_failure_reports_use_actual_authority(self):
        for value in (synthetic_measurement(), failed_measurement()):
            with self.subTest(status=value["status"]):
                child = Path(tempfile.mkdtemp(prefix="case-", dir=self.parent)).resolve()
                with slot._reserve_fixture(str(child), C, S) as reservation:
                    pin = reservation.finalize(value)
                    raw = (child / slot.MEASUREMENT_BASENAME).read_bytes()
                    self.assertEqual(raw, authority.encode_bounded(value))
                    self.assertEqual(pin, {"path": str(child / slot.MEASUREMENT_BASENAME),
                        "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
                    with self.assertRaisesRegex(slot.MeasurementSlotError,
                                                "finalize_already_attempted"):
                        reservation.finalize(value)

    def test_invalid_report_consumes_zero_slot_and_cannot_retry(self):
        reservation = self.reserve()
        value = synthetic_measurement()
        value["execution_authorized"] = True
        with self.assertRaisesRegex(slot.MeasurementSlotError, "measurement_invalid"):
            reservation.finalize(value)
        self.assertEqual((self.parent / slot.MEASUREMENT_BASENAME).read_bytes(), b"")
        with self.assertRaisesRegex(slot.MeasurementSlotError, "finalize_already_attempted"):
            reservation.finalize(synthetic_measurement())
        reservation.close()

    def test_partial_write_is_preserved_and_never_retried(self):
        reservation = self.reserve()
        real = os.pwrite
        calls = 0
        def partial(fd, data, offset):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real(fd, bytes(data[:7]), offset)
            raise OSError("fixture interruption")
        with mock.patch.object(slot, "_pwrite", side_effect=partial):
            with self.assertRaisesRegex(slot.MeasurementSlotError, "write_failed"):
                reservation.finalize(failed_measurement())
        path = self.parent / slot.MEASUREMENT_BASENAME
        self.assertEqual(path.stat().st_size, 7)
        with self.assertRaisesRegex(slot.MeasurementSlotError, "finalize_already_attempted"):
            reservation.finalize(failed_measurement())
        reservation.close()
        self.assertEqual(path.stat().st_size, 7)

    def test_path_replacement_is_detected_before_write(self):
        reservation = self.reserve()
        measurement = self.parent / slot.MEASUREMENT_BASENAME
        displaced = self.parent / "displaced-fixture-only"
        measurement.rename(displaced)
        measurement.write_bytes(b"replacement")
        with self.assertRaisesRegex(slot.MeasurementSlotError, "slot_identity_changed"):
            reservation.finalize(failed_measurement())
        self.assertEqual(measurement.read_bytes(), b"replacement")
        self.assertEqual(displaced.read_bytes(), b"")
        reservation.close()

    def test_path_replacement_during_validation_is_detected_before_write(self):
        reservation = self.reserve()
        measurement = self.parent / slot.MEASUREMENT_BASENAME
        displaced = self.parent / "displaced-during-validation"
        real_validate = authority.validate_measurement
        def mutate(value, **kwargs):
            result = real_validate(value, **kwargs)
            measurement.rename(displaced)
            measurement.write_bytes(b"replacement")
            return result
        with mock.patch.object(authority, "validate_measurement", side_effect=mutate):
            with self.assertRaisesRegex(slot.MeasurementSlotError, "slot_identity_changed"):
                reservation.finalize(failed_measurement())
        self.assertEqual(measurement.read_bytes(), b"replacement")
        self.assertEqual(displaced.read_bytes(), b"")
        reservation.close()

    def test_wrong_thread_cannot_finalize_or_close(self):
        reservation = self.reserve()
        errors = []
        def wrong_owner():
            for action in (lambda: reservation.finalize(failed_measurement()), reservation.close):
                try:
                    action()
                except Exception as exc:
                    errors.append(getattr(exc, "code", None))
        thread = threading.Thread(target=wrong_owner)
        thread.start()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["owner_mismatch", "owner_mismatch"])
        self.assertEqual((self.parent / slot.MEASUREMENT_BASENAME).read_bytes(), b"")
        reservation.close()

    def test_admission_appearing_after_reservation_blocks_finalize(self):
        reservation = self.reserve()
        (self.parent / slot.ADMISSION_BASENAME).write_bytes(b"unexpected")
        with self.assertRaisesRegex(slot.MeasurementSlotError, "admission_appeared"):
            reservation.finalize(failed_measurement())
        self.assertEqual((self.parent / slot.MEASUREMENT_BASENAME).read_bytes(), b"")
        reservation.close()

    def test_fixture_rejects_nonbig_and_production_parents(self):
        production = Path(__import__('process_supervision').LAYOUT_EVIDENCE_PATH).parent
        for parent in (str(HERE), str(production), slot.BIG_TMP):
            with self.subTest(parent=parent), self.assertRaisesRegex(
                    slot.MeasurementSlotError, "invalid_parent"):
                slot._reserve_fixture(parent, C, S)


if __name__ == "__main__":
    unittest.main()
