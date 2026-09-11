"""Symbolic integration and tiny serializer fixtures, never actual measurement."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import native_storage_crosscheck as subject
import native_storage_recipe_common as common
from test_storage_crosscheck_environment import synthetic_metadata


class NativeStorageCrosscheckTests(unittest.TestCase):
    def test_default_import_cli_is_inert_and_torch_free(self):
        code = "import sys,native_storage_crosscheck as s;assert s.main([])==0;assert 'torch' not in sys.modules"
        run = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parent,
                             capture_output=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_metadata_caps_and_cross_root_binding(self):
        sources, environment = synthetic_metadata()
        subject.validate_metadata(sources, environment)
        changed = copy.deepcopy(environment)
        changed['repository_root_realpath'] = '/different/root'
        with self.assertRaisesRegex(common.RecipeError, 'roots differ'):
            subject.validate_metadata(sources, changed)
        changed = copy.deepcopy(environment)
        changed['runtime_role'] = 'cpu_audit'
        with self.assertRaisesRegex(common.RecipeError, 'diagnostic role'):
            subject.validate_metadata(sources, changed)

    def test_capped_buffer_blocks_write_seek_and_truncate_growth(self):
        with subject._CappedBuffer(4) as buffer:
            buffer.write(b'abcd')
            for action in (lambda: buffer.write(b'e'), lambda: buffer.seek(5),
                           lambda: buffer.truncate(5), lambda: buffer.writelines([b'x'])):
                with self.assertRaises(common.RecipeError):
                    action()
            self.assertEqual(buffer.getvalue(), b'abcd')
            self.assertEqual(buffer.peak, 4)

    def test_tiny_mixed_dtype_torch_roundtrip_and_prewrite_ceiling(self):
        import torch
        tree = {'profile': common.PROFILE, 'a': [torch.zeros((2, 3)),
            torch.zeros((2,), dtype=torch.uint32), torch.zeros((), dtype=torch.float64)],
            'b': (1, -0.0, False, None, 'tiny')}
        value, runtime = subject._serialize_roundtrip(tree, pickle_ceiling=1 << 16,
            body_ceiling=1 << 16, tensor_count=3)
        self.assertEqual(value['raw_storage_bytes'], 40)
        self.assertEqual(value['storage_count'], 3)
        self.assertTrue(value['exact_tree_roundtrip'])
        self.assertFalse(runtime['cuda_initialized'])
        with mock.patch.object(subject._CappedBuffer, 'write', side_effect=AssertionError('must not serialize')):
            with self.assertRaisesRegex(common.RecipeError, 'pickle/count ceiling'):
                subject._serialize_roundtrip(tree, pickle_ceiling=1, body_ceiling=1 << 16,
                                             tensor_count=3)
        self.assertFalse(torch.cuda.is_initialized())

    def test_tiny_json_exact_types_and_cap(self):
        tree = {'profile': common.PROFILE, 'values': [None, True, -0.0, '\u20ac', 1]}
        result = subject._json_roundtrip(tree, body_ceiling=1024)
        self.assertEqual(result['storage_count'], 0)
        with self.assertRaises(common.RecipeError):
            subject._json_roundtrip(tree, body_ceiling=4)
        with self.assertRaisesRegex(common.RecipeError, 'type differs'):
            subject._json_roundtrip({'profile': common.PROFILE, 'bad': (1, 2)}, body_ceiling=1024)
        with self.assertRaises(common.RecipeError):
            subject._json_roundtrip({'profile': 'scientific_mnist_current32_v1'}, body_ceiling=1024)

    def test_actual_body_must_obey_tree_specific_zip_theorem(self):
        import torch
        import zip_storage_bound as zip_bound
        original = zip_bound.torch_save_zip_ceiling
        def understate(*args, **kwargs):
            result = original(*args, **kwargs)
            result['archive_bytes_upper'] = 1
            return result
        with mock.patch.object(zip_bound, 'torch_save_zip_ceiling', side_effect=understate):
            with self.assertRaisesRegex(common.RecipeError, 'tree-specific ZIP theorem'):
                subject._serialize_roundtrip({'profile': common.PROFILE, 'a': torch.zeros(2)},
                    pickle_ceiling=65536, body_ceiling=65536, tensor_count=1)

    def test_serializer_rejects_visible_cuda_before_any_serialization(self):
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaisesRegex(common.RecipeError, 'hidden CUDA'):
                subject._serialize_roundtrip({'profile': common.PROFILE},
                    pickle_ceiling=65536, body_ceiling=65536, tensor_count=0)

    def test_exact_roundtrip_rejects_primitive_bits_order_and_aliases(self):
        import torch
        import pickle_storage_bound as pickle_bound
        with self.assertRaisesRegex(common.RecipeError, 'float bits'):
            subject._equal_tree(-0.0, 0.0, torch)
        with self.assertRaisesRegex(common.RecipeError, 'ordered keys'):
            subject._equal_tree({'a': 1, 'b': 2}, {'b': 2, 'a': 1}, torch)
        tensor = torch.zeros(2)
        with self.assertRaisesRegex(pickle_bound.PickleBoundError, 'repeated Tensor'):
            subject._serialize_roundtrip({'profile': common.PROFILE, 'a': tensor, 'b': tensor},
                pickle_ceiling=65536, body_ceiling=65536, tensor_count=2)


class NativeStorageSymbolicIntegrationTests(unittest.TestCase):
    def test_all_eleven_complete_symbolic_templates_with_full_metadata(self):
        # Isolated process proves symbolic full templates never import Torch.
        code = '''import sys
import native_storage_crosscheck as cross
import native_tensor_inventory as inventory
import native_storage_recipe_common as common
from test_storage_crosscheck_environment import synthetic_metadata
sources,environment=synthetic_metadata()
seen=[]
for name,tree,check in cross.symbolic_components(sources=sources,environment=environment):
    seen.append(name)
    assert check['scientific_artifact'] is False
    assert check['slot_count']==len(inventory.component_layouts()[name])
    if name=='capture_comparison_pilot':
        assert cross._json_roundtrip(tree,body_ceiling=check['body_bytes_upper'])['storage_count']==0
assert tuple(seen)==tuple(inventory.COMPONENT_COUNTS)
assert 'torch' not in sys.modules
'''
        run = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parent,
                             capture_output=True, timeout=100)
        self.assertEqual(run.returncode, 0, run.stderr.decode())


if __name__ == '__main__':
    unittest.main()
