"""Dataset-free tests for the protocol-2 pickle storage proof.

Run with CUDA hidden and a single numerical thread.  These fixtures only use
tiny CPU tensors; they are not a native storage specimen or scientific run.
"""
from __future__ import annotations

import io
import collections
import copyreg
import os
import pickle
import pickletools
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import zipfile

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("pickle-bound tests require CUDA_VISIBLE_DEVICES=''")

import torch

import pickle_storage_bound as subject

HERE = Path(__file__).resolve().parent


def _bound(tree, **changes):
    options = {
        "tensor_count_cap": 256,
        "allowed_dtypes": subject.SUPPORTED_DTYPES,
        "max_depth": 32,
        "max_nodes": 10000,
        "max_utf8_bytes": 32768,
        "max_tensor_rank": 16,
    }
    options.update(changes)
    return subject.bound_protocol2_tree(tree, **options)


def _ops(payload):
    return [opcode.name for opcode, _, _ in pickletools.genops(payload)]


def _torch_data_pickle(tree):
    buffer = io.BytesIO()
    torch.save(
        tree,
        buffer,
        pickle_module=pickle,
        pickle_protocol=2,
        _use_new_zipfile_serialization=True,
        _disable_byteorder_record=False,
    )
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise AssertionError(f"expected one data.pkl, found {names!r}")
        return archive.read(names[0])


class RuntimePinTests(unittest.TestCase):
    def test_import_pure_formulas_and_default_cli_are_torch_free(self):
        code = (
            "import sys; import pickle_storage_bound as p; "
            "assert 'torch' not in sys.modules; "
            "assert p.protocol2_int_bytes(2**32-1) == 7; "
            "assert p.protocol2_tuple_of_ints_bytes(()) == 1; "
            "assert p.protocol2_tensor_bytes(dtype_name='torch.uint32', "
            "storage_nbytes=4, shape=(), stride=(), tensor_count_cap=1) == 198"
        )
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        imported = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=HERE,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual((imported.returncode, imported.stdout, imported.stderr), (0, "", ""))
        cli = subprocess.run(
            [sys.executable, "-B", subject.__file__],
            cwd=HERE,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual((cli.returncode, cli.stdout, cli.stderr), (0, "", ""))

    def test_exact_runtime_and_source_manifest(self):
        subject.assert_pinned_pickle_runtime()

    def test_mutable_serializer_settings_fail_closed(self):
        with mock.patch.object(torch.serialization, "DEFAULT_PROTOCOL", 5):
            with self.assertRaisesRegex(subject.PickleBoundError, "default protocol"):
                _bound(None)
        tls = torch.serialization._serialization_tls
        with mock.patch.object(tls, "skip_data", True):
            with self.assertRaisesRegex(subject.PickleBoundError, "skip-data"):
                _bound(None)

    def test_ambient_copyreg_reducers_fail_closed(self):
        reducer = lambda value: (list, ())
        for value_type in (torch.Tensor, collections.OrderedDict, type(torch.uint32)):
            with self.subTest(value_type=value_type):
                with mock.patch.dict(copyreg.dispatch_table, {value_type: reducer}):
                    with self.assertRaisesRegex(subject.PickleBoundError, "copyreg"):
                        _bound(None)


class PrimitiveFormulaTests(unittest.TestCase):
    def test_integer_boundaries_equal_c_pickler_lengths(self):
        values = (
            -0x80000001,
            -0x80000000,
            -129,
            -1,
            0,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFF,
            1 << 2040,
        )
        for value in values:
            with self.subTest(value=value):
                payload = pickle.dumps(value, protocol=2)
                self.assertEqual(len(payload), 3 + subject.protocol2_int_bytes(value))
        self.assertLessEqual(subject.protocol2_int_bytes(-0x80000000), 5)
        self.assertLessEqual(subject.protocol2_int_bytes(0xFFFFFFFF), 7)
        self.assertIn("LONG4", _ops(pickle.dumps(1 << 2040, protocol=2)))

    def test_rank_zero_and_tensor_constant_itemization(self):
        self.assertEqual(subject.protocol2_tuple_of_ints_bytes(()), 1)
        self.assertEqual(
            subject.protocol2_tensor_bytes(
                dtype_name="torch.float32",
                storage_nbytes=4,
                shape=(),
                stride=(),
                tensor_count_cap=1,
            ),
            169,
        )
        self.assertEqual(
            subject.protocol2_tensor_bytes(
                dtype_name="torch.uint32",
                storage_nbytes=4,
                shape=(),
                stride=(),
                tensor_count_cap=1,
            ),
            198,
        )

    def test_dict_keys_are_charged(self):
        result = _bound({"x": None})
        # 3 envelope + 6 dict + 1 SETITEM + 11 first string + 1 NONE.
        self.assertEqual(result.pickle_bytes, 22)
        self.assertGreaterEqual(result.pickle_bytes, len(pickle.dumps({"x": None}, 2)))

    def test_surrogatepass_utf8_count_and_limit(self):
        tree = ["\ud800", "🙂"]
        result = _bound(tree, max_utf8_bytes=4)
        self.assertGreaterEqual(result.pickle_bytes, len(pickle.dumps(tree, 2)))
        with self.assertRaisesRegex(subject.PickleBoundError, "UTF-8"):
            _bound(tree, max_utf8_bytes=3)


class MemoAndBatchTests(unittest.TestCase):
    def test_string_and_container_aliases_use_five_byte_ceiling(self):
        text = ""
        shared = []
        result = _bound([text, text, shared, shared])
        self.assertEqual(result.unique_string_count, 1)
        self.assertEqual(result.memoized_reference_count, 2)
        self.assertGreaterEqual(10, 5)  # first empty str >= any BINGET/LONG_BINGET
        self.assertGreaterEqual(6, 5)   # first empty list >= any alias GET
        self.assertGreaterEqual(
            result.pickle_bytes, len(pickle.dumps([text, text, shared, shared], 2))
        )

    def test_memo_indices_cross_255(self):
        strings = [f"memo-value-{index:03d}" for index in range(260)]
        tree = [*strings, strings[-1]]
        payload = pickle.dumps(tree, protocol=2)
        names = _ops(payload)
        self.assertIn("LONG_BINPUT", names)
        self.assertIn("LONG_BINGET", names)
        result = _bound(tree)
        self.assertEqual(result.memoized_reference_count, 1)
        self.assertGreater(result.memo_slot_upper, 255)
        self.assertGreaterEqual(result.pickle_bytes, len(payload))

    def test_list_batches_cross_1000(self):
        expected = {
            0: (0, 0),
            1: (0, 1),
            999: (1, 0),
            1000: (1, 0),
            1001: (2, 0),
            2000: (2, 0),
        }
        for count, (appends, append) in expected.items():
            with self.subTest(count=count):
                tree = [None] * count
                payload = pickle.dumps(tree, protocol=2)
                names = _ops(payload)
                self.assertEqual(names.count("APPENDS"), appends)
                self.assertEqual(names.count("APPEND"), append)
                self.assertGreaterEqual(_bound(tree).pickle_bytes, len(payload))

    def test_dict_batches_cross_1000(self):
        expected = {1: (0, 1), 999: (1, 0), 1000: (2, 0), 1001: (2, 0), 2000: (3, 0)}
        for count, (setitems, setitem) in expected.items():
            with self.subTest(count=count):
                tree = {index: None for index in range(count)}
                payload = pickle.dumps(tree, protocol=2)
                names = _ops(payload)
                self.assertEqual(names.count("SETITEMS"), setitems)
                self.assertEqual(names.count("SETITEM"), setitem)
                self.assertGreaterEqual(_bound(tree).pickle_bytes, len(payload))

    def test_cycles_reject_but_shared_acyclic_containers_pass(self):
        shared = [1]
        self.assertEqual(_bound([shared, shared]).memoized_reference_count, 1)
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(subject.PickleBoundError, "cyclic"):
            _bound(cyclic)


class TensorReductionTests(unittest.TestCase):
    def test_all_admitted_v2_and_v3_dtypes_dominate_data_pickle(self):
        dtypes = (torch.uint8, torch.uint32, torch.float32, torch.float64, torch.int64)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.zeros((2, 3), dtype=dtype)
                data_pickle = _torch_data_pickle(tensor)
                result = _bound(tensor, tensor_count_cap=1)
                self.assertGreaterEqual(result.pickle_bytes, len(data_pickle))
                self.assertEqual(result.storage_nbytes, (tensor.numel() * tensor.element_size(),))
                names = _ops(data_pickle)
                self.assertIn("BINPERSID", names)
                expected_rebuild = (
                    "torch._utils _rebuild_tensor_v3"
                    if dtype is torch.uint32 else "torch._utils _rebuild_tensor_v2"
                )
                globals_seen = [
                    argument for opcode, argument, _ in pickletools.genops(data_pickle)
                    if opcode.name == "GLOBAL"
                ]
                self.assertIn(expected_rebuild, globals_seen)
                if dtype is torch.uint32:
                    self.assertIn("torch.storage UntypedStorage", globals_seen)
                    self.assertIn("torch uint32", globals_seen)
                else:
                    storage_name = {
                        torch.uint8: "ByteStorage",
                        torch.float32: "FloatStorage",
                        torch.float64: "DoubleStorage",
                        torch.int64: "LongStorage",
                    }[dtype]
                    self.assertIn(f"torch {storage_name}", globals_seen)

    def test_scalar_uses_rank_zero_tuple_bound(self):
        tensor = torch.tensor(1.0, dtype=torch.float32)
        result = _bound(tensor, tensor_count_cap=1)
        self.assertEqual(result.pickle_bytes, 3 + 169)
        self.assertGreaterEqual(result.pickle_bytes, len(_torch_data_pickle(tensor)))

    def test_132_storages_cover_decimal_key_growth_and_inserted_rng_tensor(self):
        # The leading tensor stands in for a newly inserted CPU materialization
        # of a CUDA RNG state; later storage keys still fit digits(132-1)=3.
        tensors = [torch.zeros((16,), dtype=torch.uint8)]
        tensors.extend(torch.zeros((1,), dtype=torch.uint8) for _ in range(131))
        result = _bound(tensors, tensor_count_cap=132, max_nodes=1000)
        self.assertEqual(result.tensor_count, 132)
        self.assertEqual(result.storage_nbytes[0], 16)
        self.assertGreaterEqual(result.pickle_bytes, len(_torch_data_pickle(tensors)))

    def test_zero_repeated_shared_and_stateful_tensors_reject(self):
        zero = torch.empty((0,), dtype=torch.float32)
        with self.assertRaisesRegex(subject.PickleBoundError, "zero-sized"):
            _bound(zero, tensor_count_cap=1)
        tensor = torch.ones((2,), dtype=torch.float32)
        with self.assertRaisesRegex(subject.PickleBoundError, "repeated Tensor"):
            _bound([tensor, tensor], tensor_count_cap=2)
        alias = tensor.view(2)
        with self.assertRaisesRegex(subject.PickleBoundError, "independently owned"):
            _bound(alias, tensor_count_cap=1)
        stateful = torch.ones((1,), dtype=torch.float32)
        stateful.unproved_attribute = "state"
        with self.assertRaisesRegex(subject.PickleBoundError, "Python state"):
            _bound(stateful, tensor_count_cap=1)

    def test_unknown_dtype_and_tensor_count_fail_closed(self):
        with self.assertRaisesRegex(subject.PickleBoundError, "dtype"):
            _bound(torch.ones((1,), dtype=torch.int32), tensor_count_cap=1)
        with self.assertRaisesRegex(subject.PickleBoundError, "Tensor count"):
            _bound(torch.ones((1,), dtype=torch.float32), tensor_count_cap=0)


class TraversalAdmissionTests(unittest.TestCase):
    def test_node_and_depth_limits_apply_during_traversal(self):
        with self.assertRaisesRegex(subject.PickleBoundError, "node count"):
            _bound([1, 2], tensor_count_cap=0, max_nodes=2)
        with self.assertRaisesRegex(subject.PickleBoundError, "depth"):
            _bound([[[1]]], max_depth=2)

    def test_key_and_value_domains_fail_closed(self):
        with self.assertRaisesRegex(subject.PickleBoundError, "mapping keys"):
            _bound({True: None})
        with self.assertRaisesRegex(subject.PickleBoundError, "unsupported"):
            _bound({"bad"})

    def test_structural_caps_are_bounded_before_traversal(self):
        with self.assertRaisesRegex(subject.PickleBoundError, "unsafe traversal"):
            _bound(None, tensor_count_cap=2, max_nodes=1)
        with self.assertRaisesRegex(subject.PickleBoundError, "unsafe traversal"):
            _bound(None, max_utf8_bytes=1 << 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
