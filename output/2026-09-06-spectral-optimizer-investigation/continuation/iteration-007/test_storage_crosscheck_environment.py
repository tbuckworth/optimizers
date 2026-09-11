"""CPU metadata contracts; synthetic fixtures are not collected run evidence."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import source_environment_schema as schema


def synthetic_metadata():
    # Reuse the already pure synthetic schema fixture, never the inspection run.
    from test_native_layout_inspection import _success_report
    fixture = _success_report()
    environment = copy.deepcopy(fixture["native_environment_binding"])
    environment["runtime_role"] = "storage_crosscheck_cpu"
    environment["safe_environment"]["CUDA_VISIBLE_DEVICES"] = ""
    environment["cuda"] = dict(initialized=False, visible_device_count=None,
        current_device=None, driver_version=None, devices=[])
    environment["rng_layout"]["torch_cuda"] = []
    return fixture["source_binding"], environment


class StorageCrosscheckEnvironmentTests(unittest.TestCase):
    def test_diagnostic_schema_hidden_cuda_and_fixed_cpu_layout(self):
        sources, environment = synthetic_metadata()
        schema.validate_sources(sources, profile=schema.SCIENTIFIC)
        schema.validate_environment(environment, profile=schema.SCIENTIFIC)
        for path, value in ((('safe_environment', 'CUDA_VISIBLE_DEVICES'), '0'),
                            (('torch_settings', 'num_interop_threads'), 2),
                            (('cuda', 'initialized'), True),
                            (('rng_layout', 'torch_cpu', 'state_length'), 1)):
            changed = copy.deepcopy(environment)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(schema.SourceEnvironmentError):
                schema.validate_environment(changed, profile=schema.SCIENTIFIC)

    def test_scientific_state_and_payload_reject_diagnostic_role(self):
        import source_environment as provenance
        import native_payload_guard as guard
        sources, environment = synthetic_metadata()
        with self.assertRaisesRegex(schema.SourceEnvironmentError, 'native-source'):
            provenance.validate_state_environment({'profile': schema.SCIENTIFIC},
                environment, profile=schema.SCIENTIFIC)
        with self.assertRaisesRegex(guard.PayloadAdmissionError, 'runtime role'):
            guard.validate_runtime_metadata(sources, environment)
        with self.assertRaisesRegex(guard.PayloadAdmissionError, 'runtime role'):
            guard._source_environment(sources, environment, audit=True)

    def test_scientific_audit_rejects_before_numerical_work(self):
        import audit_envelope as audit
        import native_storage_recipe_core as core
        sources, environment = synthetic_metadata()
        context = {key: None for key in audit.branch_schema.CONTEXT_KEYS}
        context.update(profile=schema.SCIENTIFIC, sources=sources,
                       identity=core._identity('primary', 71001, 101))
        with mock.patch.object(audit, '_checks', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError, 'audit_runtime_role'):
                audit.build_audit(anchor=None, anchor_receipt=None, witness=None,
                    witness_receipt=None, branch=None, branch_receipt=None,
                    plan_reference=None, auditor_environment=environment,
                    created_utc='2000-01-01T00:00:00Z', **context)

    def test_real_collector_is_cpu_rng_neutral_and_does_not_probe_cuda(self):
        code = '''import os,sys,tempfile
from unittest import mock
import torch
import native_layout_inspection as settings
import source_environment as source
settings._configure_and_assert_torch(torch)
before=source._rng_snapshot(native=False)
with tempfile.TemporaryDirectory(prefix='i7-diagnostic-env-unit-') as root:
    with mock.patch.multiple(torch.cuda, **{name:mock.Mock(side_effect=AssertionError('CUDA probe')) for name in ('is_available','device_count','current_device','get_device_properties','get_rng_state_all')}):
        value=source.collect_runtime_environment(root,profile=source.storage.SCIENTIFIC,runtime_role='storage_crosscheck_cpu')
assert value['runtime_role']=='storage_crosscheck_cpu'
assert value['cuda']['initialized'] is False
assert source._rng_equal(before,source._rng_snapshot(native=False))
assert not torch.cuda.is_initialized()
'''
        environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
        run = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parent,
                             env=environment, capture_output=True, timeout=25)
        self.assertEqual(run.returncode, 0, run.stderr.decode())


if __name__ == '__main__':
    unittest.main()
