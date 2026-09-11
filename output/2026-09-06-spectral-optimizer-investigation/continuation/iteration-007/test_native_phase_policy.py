"""In-memory synthetic contract transcripts. No native files, plans, data or work."""
import copy
import hashlib
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for native-policy contract fixtures")

import torch
import artifact_store as storage
import identity_codec as codec
import native_phase_policy as policy

UTC = "2026-09-06T22:40:00Z"
KIND = "synthetic_contract_fixture"


def reference(name):
    """Fake value identity is explicit test metadata, never an on-disk receipt."""
    encoding = "bytes" if name.endswith(".json") else "torch_weights_only"
    digest = hashlib.sha256(("SYNTHETIC:" + name).encode()).hexdigest()
    raw = storage._json_bytes(dict(schema="i7_artifact_receipt_v1", name=name, size=10,
        sha256=digest, status="complete", encoding=encoding))
    return dict(name=name, status="complete", encoding=encoding, size_bytes=10,
        sha256=digest, receipt_name="receipt-"+hashlib.sha256(name.encode()).hexdigest()+".json",
        receipt_size_bytes=len(raw), receipt_sha256=hashlib.sha256(raw).hexdigest())


def rechain(records):
    previous = None
    for i, record in enumerate(records):
        record["sequence"] = i
        record["previous_record_sha256"] = previous
        previous = hashlib.sha256(codec.json_bytes(record)).hexdigest()
    return records


def transcript():
    rows, phase_counts = [], {}
    root = dict(path="/tmp/spectral-experiment-artifacts/SYNTHETIC-CONTRACT-NOT-CREATED", device=1,
                inode=1, header_sha256="c"*64)
    for operation in policy.schedule():
        events = ("started", "sealed") if operation["kind"] == "work" else (operation["kind"],)
        for event in events:
            phase = operation["phase"]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            first = operation["operation"] == "development_go"
            resources = None if first else dict(phase_wall_seconds=float(phase_counts[phase]),
                phase_cpu_seconds=float(phase_counts[phase])/2,
                peak_rss_bytes=1000, peak_cuda_allocated_bytes=0, peak_cuda_reserved_bytes=0,
                root_logical_bytes=10000+len(rows)*1000, root_allocated_bytes=10000+len(rows)*1000)
            rows.append(dict(schema="i7_native_phase_record_v1", profile=policy.PROFILE,
                evidence_kind=KIND, execution_enabled=False, sequence=len(rows), event=event,
                operation=operation["operation"], phase=phase, created_utc=UTC,
                root_binding=None if first else copy.deepcopy(root), sources_sha256="a"*64,
                environment_sha256="b"*64, previous_record_sha256=None,
                artifacts=[reference(name) for name in operation["artifacts"]] if event == "sealed" else [],
                resources=resources, error=None))
    return rechain(rows)


class NativePolicyTests(unittest.TestCase):
    def test_exact_schedule_membership_and_complete_synthetic_transcript(self):
        program, records = policy.schedule(), transcript()
        self.assertEqual(len(program), 49)
        self.assertEqual(sum(row["kind"] == "work" for row in program), 41)
        self.assertEqual(len(records), 90)
        names = [name for row in program for name in row["artifacts"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 82)
        self.assertEqual(sum(name.endswith("--anchor.pt") for name in names), 18)
        self.assertEqual(sum(name.endswith("--source-witness.pt") for name in names), 18)
        self.assertEqual(sum(name.endswith("--branch-results.pt") for name in names), 18)
        self.assertEqual(sum(name.endswith("--independent-audit.pt") for name in names), 16)
        self.assertEqual(sum("-source-capture_" in name for name in names), 6)
        self.assertEqual(sum(name.endswith("-plan.pt") for name in names), 5)
        report = policy.validate_transcript(records, evidence_kind=KIND)
        self.assertEqual(report["status"], "declared_complete")
        for key in ("can_execute", "can_resume", "verified_file_bytes", "verified_resource_observations",
                    "verified_execution_history", "scientific_execution_certified"):
            self.assertIs(report[key], False)
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_sources_before_branches_and_all_plans_before_primary_go(self):
        names = [row["operation"] for row in policy.schedule()]
        sources = [names.index(f"primary.source.b{bundle}") for bundle in policy.PRIMARY]
        branches = [i for i, name in enumerate(names) if name.startswith("primary.branch.")]
        self.assertLess(max(sources), min(branches))
        self.assertLess(names.index("scientific_plans"), names.index("primary_go"))
        self.assertLess(max(branches), names.index("sensitivity_go"))
        self.assertLess(names.index("sensitivity_complete"), names.index("audit_go"))

    def test_scientific_kind_cannot_be_inferred_from_synthetic_shape(self):
        rows = transcript()
        with self.assertRaises(policy.PolicyError):
            policy.validate_transcript(rows, evidence_kind="native_producer_attestation")
        rows[0]["execution_enabled"] = True
        with self.assertRaises(policy.PolicyError):
            policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_development_decision_cannot_have_precreated_root(self):
        rows = transcript()
        rows[0]["root_binding"] = copy.deepcopy(rows[1]["root_binding"])
        with self.assertRaises(policy.PolicyError):
            policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_started_prefix_refuses_resume_and_complete_claim(self):
        rows = transcript()[:2]
        report = policy.validate_transcript(rows, evidence_kind=KIND, require_complete=False)
        self.assertTrue(report["interrupted_work"])
        self.assertFalse(report["can_resume"])
        self.assertEqual(report["status"], "declared_incomplete")
        with self.assertRaises(policy.PolicyError):
            policy.validate_transcript(rows, evidence_kind=KIND)

    def test_skips_duplicates_reorders_and_pilot_audits_are_rejected_after_rehash(self):
        variants = []
        rows = transcript()
        variants.append(rows[:2] + rows[3:])
        variants.append(rows[:2] + [copy.deepcopy(rows[1])] + rows[2:])
        rows = transcript()
        first_source = next(i for i,r in enumerate(rows) if r["operation"] == "primary.source.b71001")
        first_branch = next(i for i,r in enumerate(rows) if r["operation"] == "primary.branch.b71001.u101")
        rows[first_source:first_source+2], rows[first_branch:first_branch+2] = rows[first_branch:first_branch+2], rows[first_source:first_source+2]
        variants.append(rows)
        rows = transcript()
        audit = next(row for row in rows if row["event"] == "sealed" and row["phase"] == "audit")
        audit["artifacts"] = [reference(policy.artifact_name("pilot", 71990, 101, "independent-audit"))]
        variants.append(rows)
        for i, rows in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(policy.PolicyError):
                policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_missing_sensitivity_plan_and_missing_primary_anchor_are_rejected(self):
        for operation in ("scientific_plans", "primary.source.b71002"):
            rows = transcript()
            sealed = next(row for row in rows if row["event"] == "sealed" and row["operation"] == operation)
            sealed["artifacts"].pop()
            with self.subTest(operation=operation), self.assertRaises(policy.PolicyError):
                policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_source_environment_and_root_changes_cannot_pass(self):
        for field in ("sources_sha256", "environment_sha256", "root_binding"):
            rows = transcript()
            if field == "root_binding":
                rows[3][field]["inode"] += 1
            else:
                rows[3][field] = "f"*64
            with self.subTest(field=field), self.assertRaises(policy.PolicyError):
                policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_caps_phase_clock_resets_and_deletion_are_rejected(self):
        for field, value in (("phase_wall_seconds", 601.), ("phase_cpu_seconds", 0.),
                             ("peak_rss_bytes", 13 << 30), ("peak_cuda_allocated_bytes", 9 << 30),
                             ("root_logical_bytes", 1), ("root_allocated_bytes", 1)):
            rows = transcript()
            row = next(row for row in rows if row["event"] == "sealed" and row["operation"] == "primary.source.b71001")
            row["resources"][field] = value
            if field == "peak_cuda_allocated_bytes":
                row["resources"]["peak_cuda_reserved_bytes"] = value
            with self.subTest(field=field), self.assertRaises(policy.PolicyError):
                policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_audit_cpu_cap_and_cuda_exclusion(self):
        for values in (dict(phase_cpu_seconds=601.),
                       dict(peak_cuda_allocated_bytes=1, peak_cuda_reserved_bytes=1)):
            rows = transcript()
            next(row for row in rows if row["phase"] == "audit")["resources"].update(values)
            with self.subTest(values=values), self.assertRaises(policy.PolicyError):
                policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_terminal_failure_preserves_actual_overrun_and_blocks_suffix(self):
        rows = transcript()[:3]
        failed = rows[-1]
        failed["event"], failed["artifacts"] = "failed", []
        failed["resources"]["phase_wall_seconds"] = 181.
        failed["error"] = dict(category="resource", retained_failure_ref=None)
        rechain(rows)
        report = policy.validate_transcript(rows, evidence_kind=KIND, require_complete=False)
        self.assertEqual(report["status"], "declared_failed")
        self.assertFalse(report["can_resume"])
        self.assertEqual(report["phase_resources"]["development"]["phase_wall_seconds"], 181.)
        with self.assertRaises(policy.PolicyError):
            policy.validate_transcript(rechain(rows + [transcript()[3]]), evidence_kind=KIND, require_complete=False)

    def test_exact_types_receipt_rehash_and_unknown_fields(self):
        mutations = [lambda r: r.__setitem__("sequence", False),
                     lambda r: r.__setitem__("unknown", None),
                     lambda r: r["resources"].__setitem__("phase_wall_seconds", 3),
                     lambda r: r["root_binding"].__setitem__("inode", True),
                     lambda r: r["artifacts"][0].__setitem__("size_bytes", 11)]
        for index, mutate in enumerate(mutations):
            rows = transcript()
            mutate(rows[2])
            # The explicit sequence corruption is not repaired by rechain.
            if index != 0:
                rechain(rows)
            with self.subTest(index=index), self.assertRaises(policy.PolicyError):
                policy.validate_transcript(rows, evidence_kind=KIND)

    def test_declared_usage_cannot_undercount_its_own_retained_receipts(self):
        rows = transcript()
        for index, row in enumerate(rows):
            if row["resources"] is not None:
                row["resources"]["root_logical_bytes"] = 10000 + index * 100
        with self.assertRaisesRegex(policy.PolicyError, "payload/receipt lower bound"):
            policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_standalone_record_rejects_event_kind_substitution(self):
        row = transcript()[1]
        row["event"] = "decision"
        with self.assertRaisesRegex(policy.PolicyError, "operation kind"):
            policy.validate_record(row, evidence_kind=KIND)

    def test_primary_sensitivity_environment_must_match_but_audit_may_differ(self):
        rows = transcript()
        for row in rows:
            if row["phase"] == "audit":
                row["environment_sha256"] = "d" * 64
        policy.validate_transcript(rechain(rows), evidence_kind=KIND)
        for row in rows:
            if row["phase"] == "sensitivity":
                row["environment_sha256"] = "d" * 64
        with self.assertRaisesRegex(policy.PolicyError, "sensitivity environment"):
            policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_success_cannot_consume_failure_reserve(self):
        rows = transcript()
        for row in rows:
            if row["resources"] is not None:
                row["resources"]["root_logical_bytes"] = storage.DEFAULT_BUDGET
        with self.assertRaisesRegex(policy.PolicyError, "failure reserve"):
            policy.validate_transcript(rechain(rows), evidence_kind=KIND)

    def test_retained_failure_bytes_must_fit_declared_growth(self):
        rows = transcript()[:3]
        row = rows[-1]
        row.update(event="failed", artifacts=[], error=dict(category="io",
            retained_failure_ref=dict(name="failure-000001.json", size_bytes=16000, sha256="e"*64)))
        with self.assertRaisesRegex(policy.PolicyError, "retained failure bytes"):
            policy.validate_transcript(rechain(rows), evidence_kind=KIND, require_complete=False)
        row["resources"]["root_logical_bytes"] += 16000
        row["resources"]["root_allocated_bytes"] += 16000
        self.assertEqual(policy.validate_transcript(rechain(rows), evidence_kind=KIND,
            require_complete=False)["status"], "declared_failed")


if __name__ == "__main__":
    unittest.main()
