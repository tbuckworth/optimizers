"""Small pure-arithmetic/schema-source checks, not producer experiments."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import core_primitive_bound as bound
import primitive_storage_bound as primitive

HERE = Path(__file__).resolve().parent


def static_assignment(filename, name):
    module = ast.parse((HERE / filename).read_text())
    return ast.literal_eval(next(node.value for node in module.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                               for target in node.targets)))


class CorePrimitiveTests(unittest.TestCase):
    def test_identity_membership_and_owned_constants(self):
        ids = bound.fixed_identities()
        self.assertEqual(len(ids), 18)
        self.assertEqual(len(set(ids)), 18)
        self.assertEqual(sum(role == "pilot" for role, _, _ in ids), 2)
        self.assertEqual(sum(role == "primary" for role, _, _ in ids), 12)
        for role, bundle, update in ids:
            value = bound.identity(role, bundle, update)
            self.assertEqual(tuple(value), static_assignment("identity_codec.py", "IDENTITY_KEYS"))
            self.assertEqual(value["stream_roles"], static_assignment("identity_codec.py", "STREAM_ROLES"))
            value["stream_roles"][0] = "mutated"
            self.assertEqual(bound.identity(role, bundle, update)["stream_roles"][0], "permutation")

    def test_static_source_literals(self):
        self.assertEqual(bound.STREAM_ROLES, static_assignment("identity_codec.py", "STREAM_ROLES"))
        self.assertEqual(bound.EVENTS, static_assignment("source_capture.py", "EVENTS"))
        self.assertEqual(bound.INSTRUMENTATION, static_assignment("source_history.py", "INSTRUMENTATION"))
        self.assertEqual(bound.SOURCE_EVIDENCE_SCOPE, static_assignment("source_history.py", "SOURCE_EVIDENCE_SCOPE"))
        self.assertEqual({k: v for k, v in bound.GROUP.items() if k != "param_indices"},
                         static_assignment("state_core.py", "EXPECTED_GROUP"))
        self.assertEqual(tuple(bound.OBSERVER_CONFIG), static_assignment("state_core.py", "OBSERVER_CONFIG_KEYS"))

    def test_core_cost_topology_matches_primary_key_contracts(self):
        shapes = []
        def collect(fields):
            shapes.append(tuple(fields))
            return primitive.pdict(fields)
        with patch.object(bound, "pdict", collect):
            bound.core_cost()
        for constant in ("ROOT_KEYS", "PARAMETER_ENTRY_KEYS", "NATIVE_TENSOR_KEYS", "OPTIMIZER_KEYS",
                         "OPT_STATE_KEYS", "OBSERVER_KEYS", "OBSERVER_STATE_KEYS", "RNG_KEYS",
                         "PYTHON_RNG_KEYS", "NUMPY_RNG_KEYS", "CUDA_RNG_KEYS", "WITNESS_KEYS"):
            self.assertIn(static_assignment("state_core.py", constant), shapes, constant)

    def test_outer_anchor_witness_history_topologies(self):
        shapes = []
        def collect(fields):
            shapes.append(tuple(fields))
            return primitive.pdict(fields)
        with patch.object(bound, "pdict", collect):
            bound.anchor_cost(role="primary", bundle=71001, update=2000)
            bound.witness_cost(role="primary", bundle=71001, update=2000)
            bound.source_completion_cost(role="primary", bundle=71001)
        for filename, names in (
            ("anchor_envelope.py", ("ROOT_KEYS", "ANCHOR_KEYS", "BINDING_KEYS")),
            ("source_capture.py", ("PAYLOAD_KEYS", "REFERENCE_KEYS", "PROOF_KEYS", "BEFORE_BINDING_KEYS",
                                   "LOSS_KEYS", "STATE_HASH_KEYS")),
            ("source_history.py", ("SOURCE_ROOT_KEYS", "FINGERPRINT_KEYS", "ANCHOR_WITNESS_KEYS",
                                   "PROVENANCE_KEYS", "PLAN_REF_KEYS")),
            ("plan_bindings.py", ("BINDING_KEYS",))):
            for name in names:
                self.assertIn(static_assignment(filename, name), shapes, (filename, name))

    def test_component_membership_and_no_tensor_or_proto_cost(self):
        result = bound.core_component_bounds()
        self.assertEqual(tuple(result), ("anchor_pilot", "anchor_long", "source_witness",
            "source_completion_pilot_on", "source_completion_pilot_off", "source_completion_long_on",
            "plan_pilot", "plan_long"))
        self.assertTrue(all(type(value) is int and value > 0 for value in result.values()))
        self.assertEqual(bound.TENSOR, 0)
        self.assertLess(result["plan_long"], 1000)
        self.assertGreater(result["source_completion_pilot_on"], result["source_completion_pilot_off"])
        self.assertGreater(result["source_completion_long_on"], result["source_completion_pilot_on"])
        for role, bundle, update in bound.fixed_identities():
            self.assertLessEqual(bound.anchor_cost(role=role, bundle=bundle, update=update),
                result["anchor_pilot" if role == "pilot" else "anchor_long"])
            self.assertLessEqual(bound.witness_cost(role=role, bundle=bundle, update=update), result["source_witness"])

    def test_exact_reference_encoding_and_audit_schema(self):
        rows = []
        def collect(fields):
            rows.append(fields.copy())
            return primitive.pdict(fields)
        with patch.object(bound, "pdict", collect):
            bound.artifact_reference_cost(role="pilot", bundle=71990, update=101, kind="anchor")
            bound.plan_reference_cost(71990)
            bound.envelope_cost(0, role="primary", bundle=71001, update=101, kind="independent-audit")
        self.assertEqual([row["encoding"] for row in rows if "encoding" in row],
                         [primitive.literal("torch_weights_only")] * 2)
        self.assertEqual(rows[-1]["schema_name"], primitive.literal("i7_anchor_numerical_audit"))
        for filename in ("source_capture.py", "source_history.py"):
            literals = {node.value for node in ast.walk(ast.parse((HERE / filename).read_text()))
                        if isinstance(node, ast.Constant) and type(node.value) is str}
            self.assertIn("torch_weights_only", literals)
        literals = {node.value for node in ast.walk(ast.parse((HERE / "audit_envelope.py").read_text()))
                    if isinstance(node, ast.Constant) and type(node.value) is str}
        self.assertIn("i7_anchor_numerical_audit", literals)

    def test_rng_integer_and_float_alternatives(self):
        self.assertEqual(primitive.integer(0, 0xFFFFFFFF), 7)
        self.assertGreater(primitive.FLOAT, primitive.NULL)
        self.assertEqual(primitive.integer(0, 4000), 3)
        self.assertGreater(bound.rng_cost(), 624 * 7 + 8192)
        # Nullable/zero-rank basis has less primitive framing than any native wrapper.
        for rank in range(1, 33):
            self.assertGreater(bound.native_tensor_metadata((50890, rank)), primitive.NULL)
            self.assertLessEqual(bound.native_tensor_metadata((50890, rank)),
                                 bound.native_tensor_metadata((50890, 32)))

    def test_reject_nonmember_and_wrong_types(self):
        for values in (("pilot", 71990, 500), ("primary", 71001, True),
                       ("primary", 71990, 101), ("sensitivity", 71001, 101)):
            with self.assertRaises(ValueError):
                bound.identity(*values)
        for bundle in (True, 71990., -1):
            with self.assertRaises(ValueError):
                bound.plan_cost(bundle)
        for mode in (None, "capture_off", "unknown"):
            with self.assertRaises(ValueError):
                bound.source_completion_cost(role="primary", bundle=71001, mode=mode)
        with self.assertRaises(ValueError):
            bound.envelope_cost(-1, role="pilot", bundle=71990, update=101, kind="anchor")

    def test_import_and_default_cli_are_inert(self):
        code = "import sys; import core_primitive_bound as b; b.core_component_bounds(); assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=HERE,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", str(HERE / "core_primitive_bound.py")],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no runtime validation or execution admission", result.stdout)


if __name__ == "__main__":
    unittest.main()
