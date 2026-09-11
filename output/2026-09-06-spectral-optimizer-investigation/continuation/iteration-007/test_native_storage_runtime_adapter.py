"""Synthetic admission integration; not collected M/A or a native permission."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import final_storage_accounting as adapter
import native_storage_authority as authority
import native_write_ledger as ledger
import process_supervision as supervision
import identity_codec as codec
import pickle_storage_bound as pickle_bound
import zip_storage_bound as zip_bound
from test_native_storage_authority import synthetic_measurement, bind_synthetic_supervision
from test_storage_crosscheck_environment import synthetic_metadata


def synthetic_chain(*, mutate_measurement=None):
    """Invented parser fixtures only; every reported measurement is fabricated."""
    sources, environment = synthetic_metadata()
    environment["runtime_role"] = "cpu_audit"
    measurement = synthetic_measurement()
    measurement["repository_revision"] = sources["repository_revision"]
    measurement["source_set_sha256"] = codec.tree_digest(sources)
    if mutate_measurement:
        mutate_measurement(measurement)
    bind_synthetic_supervision(measurement)
    mraw = authority.encode_bounded(measurement)
    mpin = dict(path=supervision.LAYOUT_EVIDENCE_PATH, size_bytes=len(mraw),
                sha256=hashlib.sha256(mraw).hexdigest())
    admission = authority.make_admission(mpin, expected_commit=sources["repository_revision"],
        expected_source_set_sha256=codec.tree_digest(sources))
    araw = authority.encode_bounded(admission)
    apin = dict(path=supervision.STORAGE_ADMISSION_PATH, size_bytes=len(araw),
                sha256=hashlib.sha256(araw).hexdigest())
    def read(**kwargs):
        pin = {k: kwargs[k] for k in ("path", "size_bytes", "sha256")}
        if pin == apin:
            return araw
        if pin == mpin:
            return mraw
        raise FileNotFoundError("synthetic pin not registered")
    return apin, sources, environment, read


class RuntimeAdapterTests(unittest.TestCase):
    def test_default_import_is_inert_torch_free(self):
        run = subprocess.run([sys.executable, "-B", "-c",
            "import sys,final_storage_accounting as a;assert 'torch' not in sys.modules;assert a.main([])==0"],
            cwd=Path(__file__).resolve().parent, capture_output=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_delegation_preserves_exact_objects_and_only_none_return(self):
        pin = dict(path=supervision.STORAGE_ADMISSION_PATH, size_bytes=1, sha256="a"*64)
        sources, environment = {}, {}
        with mock.patch.object(authority, "validate_runtime_admission", return_value=None) as call:
            self.assertIsNone(adapter.validate_runtime_admission(pin, sources=sources, environment=environment))
            args, kwargs = call.call_args
            self.assertIs(args[0], pin)
            self.assertIs(kwargs["sources"], sources)
            self.assertIs(kwargs["environment"], environment)
        for result in (True, False, {}, [], "admitted"):
            with self.subTest(result=result), mock.patch.object(authority, "validate_runtime_admission",
                    return_value=result), self.assertRaises(adapter.StorageAdmissionError):
                adapter.validate_runtime_admission(pin, sources=sources, environment=environment)

    def test_errors_reject_without_swallowing_interrupts(self):
        pin = dict(path=supervision.STORAGE_ADMISSION_PATH, size_bytes=1, sha256="a"*64)
        for error in (ValueError, FileNotFoundError, RuntimeError, authority.StorageAuthorityError):
            with self.subTest(error=error), mock.patch.object(authority, "validate_runtime_admission",
                    side_effect=error("test detail must not render")), self.assertRaisesRegex(
                    adapter.StorageAdmissionError, "^reviewed storage admission rejected$"):
                adapter.validate_runtime_admission(pin, sources={}, environment={})
        for error in (KeyboardInterrupt, SystemExit):
            with mock.patch.object(authority, "validate_runtime_admission", side_effect=error), \
                 self.assertRaises(error):
                adapter.validate_runtime_admission(pin, sources={}, environment={})

    def test_malformed_input_rejects_before_authority(self):
        for pin in ({}, dict(path="/fixed", size_bytes=True, sha256="a"*64),
                    dict(path="/fixed", size_bytes=65537, sha256="a"*64),
                    dict(path="/fixed", size_bytes=1, sha256="bad")):
            with mock.patch.object(authority, "validate_runtime_admission") as call:
                with self.assertRaisesRegex(adapter.StorageAdmissionError, "invalid storage admission input"):
                    adapter.validate_runtime_admission(pin, sources={}, environment={})
                call.assert_not_called()

    def test_real_authority_chain_and_cpu_runtime_no_rng_or_cuda_change(self):
        import source_environment as provenance
        import torch
        pin, sources, environment, read = synthetic_chain()
        before = provenance._rng_snapshot(native=False)
        with mock.patch.object(supervision, "load_pinned_bytes", side_effect=read):
            self.assertIsNone(adapter.validate_runtime_admission(pin, sources=sources, environment=environment))
        self.assertTrue(provenance._rng_equal(before, provenance._rng_snapshot(native=False)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_absent_admission_rejects_before_calculation(self):
        pin, sources, environment, _ = synthetic_chain()
        with mock.patch.object(supervision, "load_pinned_bytes", side_effect=FileNotFoundError), \
             mock.patch.object(ledger, "compute") as compute, self.assertRaises(adapter.StorageAdmissionError):
            adapter.validate_runtime_admission(pin, sources=sources, environment=environment)
        compute.assert_not_called()

    def test_partial_or_tampered_measurement_rejects_before_calculation(self):
        pin, sources, environment, read = synthetic_chain()
        for raw in (b"", b"partial", b"{}\n"):
            def bad_read(**kwargs):
                return raw if kwargs["path"] == supervision.LAYOUT_EVIDENCE_PATH else read(**kwargs)
            with mock.patch.object(supervision, "load_pinned_bytes", side_effect=bad_read), \
                 mock.patch.object(ledger, "compute") as compute, self.assertRaises(adapter.StorageAdmissionError):
                adapter.validate_runtime_admission(pin, sources=sources, environment=environment)
            compute.assert_not_called()

    def test_source_revision_and_root_drift_reject(self):
        for change in ("revision", "root", "role"):
            pin, sources, environment, read = synthetic_chain()
            if change == "revision":
                sources["repository_revision"] = "e"*40
            elif change == "root":
                sources["repository_root_realpath"] = "/other/root"
                environment["repository_root_realpath"] = "/other/root"
            else:
                environment["runtime_role"] = "storage_crosscheck_cpu"
            with mock.patch.object(supervision, "load_pinned_bytes", side_effect=read), \
                 mock.patch.object(ledger, "compute") as compute, self.assertRaises(adapter.StorageAdmissionError):
                adapter.validate_runtime_admission(pin, sources=sources, environment=environment)
            compute.assert_not_called()

    def test_measurement_root_and_protocol_are_not_waived(self):
        for mutate in (lambda m: m["diagnostic_environment"].update(repository_root_realpath="/other/root"),
                       lambda m: m.update(measurement_protocol="inventory_only")):
            pin, sources, environment, read = synthetic_chain(mutate_measurement=mutate)
            with mock.patch.object(supervision, "load_pinned_bytes", side_effect=read), \
                 mock.patch.object(ledger, "compute") as compute, self.assertRaises(adapter.StorageAdmissionError):
                adapter.validate_runtime_admission(pin, sources=sources, environment=environment)
            compute.assert_not_called()

    def test_analytic_ledger_failure_still_denies(self):
        pin, sources, environment, read = synthetic_chain()
        with mock.patch.object(supervision, "load_pinned_bytes", side_effect=read), \
             mock.patch.object(ledger, "compute", return_value={"conditional_arithmetic_fits": False}), \
             self.assertRaises(adapter.StorageAdmissionError):
            adapter.validate_runtime_admission(pin, sources=sources, environment=environment)

    def test_pickle_or_zip_runtime_failure_still_denies(self):
        pin, sources, environment, read = synthetic_chain()
        for module, name in ((pickle_bound, "assert_pinned_pickle_runtime"),
                             (zip_bound, "validate_runtime_save_configuration")):
            with mock.patch.object(supervision, "load_pinned_bytes", side_effect=read), \
                 mock.patch.object(module, name, side_effect=RuntimeError("test runtime mismatch")), \
                 self.assertRaises(adapter.StorageAdmissionError):
                adapter.validate_runtime_admission(pin, sources=sources, environment=environment)


if __name__ == "__main__":
    unittest.main()
