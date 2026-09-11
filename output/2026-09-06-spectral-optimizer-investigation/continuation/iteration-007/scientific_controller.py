"""Explicit later-phase integration; import/default do not execute or import Torch.

Only a consumed phase_transition supplies a live writer. Each instance runs one
fixed phase, then closes it. There is no retry, automatic next-phase permission,
native CLI or claim that a CPU audit function proves a separate process.
"""
from __future__ import annotations

import copy
import os

from native_controller import NativeController, ControllerError, _need


PHASE_START = {"primary": 6, "sensitivity": 24, "audit": 31}
PRIOR_COUNTS = {"primary": 10, "sensitivity": 44, "audit": 56}


def prior_boundary(phase, handoff):
    """Normalize a closed handoff; transition still authenticates actual files."""
    _need(phase in ("development", "primary", "sensitivity", "audit"), "unknown prior phase")
    _need(handoff["status"] == "closed_inspected_boundary" and
          handoff["execution_authorized"] is False and
          handoff["automatic_reopen_enabled"] is False and
          handoff["scientific_execution_certified"] is False, "invalid handoff declaration")
    return dict(phase=phase, phase_records=handoff["inspection"]["phase_records"],
        root_binding=copy.deepcopy(handoff["root_binding"]),
        journal_pin=copy.deepcopy(handoff["journal_pin"]),
        boundary_ref=copy.deepcopy(handoff["boundary_ref"]),
        inspection=copy.deepcopy(handoff["inspection"]))


class ScientificController(NativeController):
    def __init__(self, transition, *, entry_process_id, data_directory, expected_files,
                 utc_now, fixture_work=None):
        import artifact_store as storage
        import identity_codec as codec
        import native_control as control
        import native_phase_policy as policy
        import runtime_guard as runtime
        import source_environment_schema as schema
        self.storage, self.codec, self.control, self.policy = storage, codec, control, policy
        self.store = transition["store"]
        self.guard = transition["guard"]
        self._pid = os.getpid()
        self._terminal = False
        self._handoff_started = False
        self.handoff_failure_metadata_status = None
        self._started = False
        self._data = None
        self._fixture_work = fixture_work
        self.utc_now = utc_now
        self.data_directory = data_directory
        self.expected_files = copy.deepcopy(expected_files)
        try:
            import phase_transition
            transition = phase_transition.validate_acquired_phase(transition)
            self.phase = transition["phase"]
            _need(type(self.phase) is str and self.phase in PHASE_START, "unknown scientific phase")
            self.kind = transition["evidence_kind"]
            _need(self.kind in policy.KINDS, "unknown phase evidence kind")
            native = self.kind == "native_producer_attestation"
            _need(type(self.store) is storage.ArtifactStore and
                  self.store.profile == (storage.SCIENTIFIC if native else storage.MLP_FIXTURE),
                  "phase store/profile differs")
            _need(type(self.guard) is runtime.RuntimeGuard and self.guard.phase == self.phase and
                  self.guard.profile == (runtime.SCIENTIFIC if native else runtime.FIXTURE) and
                  self.guard.entry_origins is not None, "phase entry guard differs")
            _need(type(entry_process_id) is int and entry_process_id == self._pid,
                  "phase entry PID differs")
            _need((fixture_work is None if native else callable(fixture_work)) and callable(utc_now),
                  "native callback override or missing fixture/UTC callback")
            _need(transition["execution_authorized"] is False and
                  transition["scientific_execution_certified"] is False, "transition authority differs")
            self.root_binding = copy.deepcopy(transition["root_binding"])
            self.journal_pin = copy.deepcopy(transition["journal_pin"])
            self.sources = copy.deepcopy(transition["sources"])
            self.environment = copy.deepcopy(transition["environment"])
            self.source_environment = copy.deepcopy(transition["source_environment"])
            self._records = copy.deepcopy(transition["records"])
            self._cursor = PHASE_START[self.phase]
            self._check("scientific_controller.entry")
            _need(control._root_binding(self.store) == self.root_binding, "phase root differs")
            _need(len(self._records) == PRIOR_COUNTS[self.phase], "phase prefix count differs")
            validation = policy.validate_transcript(self._records, evidence_kind=self.kind,
                                                     require_complete=False)
            _need(self._records[-1]["event"] == "boundary" and
                  validation["completed_operations"] == self._cursor and
                  validation["pending_operation"] == policy.schedule()[self._cursor]["operation"],
                  "phase prefix is not the expected completed boundary")
            for index, record in enumerate(self._records):
                raw, ref = self._read(control.phase_name(index))
                _need(raw == codec.json_bytes(record), "actual phase prefix differs")
            _need(ref == transition["previous_boundary_ref"], "previous boundary pin differs")
            schema.validate_sources(self.sources, profile=storage.SCIENTIFIC)
            schema.validate_environment(self.environment, profile=storage.SCIENTIFIC)
            schema.validate_environment(self.source_environment, profile=storage.SCIENTIFIC)
            _need(self.source_environment["runtime_role"] == "native_source", "source environment role differs")
            _need(self.environment["runtime_role"] == ("cpu_audit" if self.phase == "audit" else "native_source"),
                  "phase environment role differs")
            _need(self._json(control.SOURCE_NAME) == self.sources and
                  self._records[0]["sources_sha256"] == codec.tree_digest(self.sources),
                  "phase source metadata differs")
            permission_name = f"native-{self.phase}-permission.json"
            environment_name = f"native-{self.phase}-environment.json"
            permission_bytes, _ = self._read(permission_name)
            pin = transition["permission_pin"]
            _need(control._read_pinned(pin["path"], pin["size_bytes"], pin["sha256"],
                                      8 << 10) == permission_bytes == codec.json_bytes(transition["permission"]),
                  "consumed permission copy differs")
            marker = transition["transition_pin"]
            control._read_pinned(marker["path"], marker["size_bytes"], marker["sha256"], 8 << 10)
            _need(self._json(environment_name) == self.environment, "phase environment copy differs")
            if self.phase == "audit":
                _need(self._json("native-primary-environment.json") == self.source_environment,
                      "audit source environment differs from primary")
            else:
                _need(self.source_environment == self.environment, "producer environment differs from phase")
            self._check_context()
            self._check("scientific_controller.ready")
        except BaseException:
            self._fail("scientific_controller_initialization_failed")
            raise

    def _check_context(self):
        self._check("scientific_controller.context.pre")
        if self.kind == "native_producer_attestation":
            import source_environment as provenance
            root = self.sources["repository_root_realpath"]
            _need(provenance.collect_verified_sources(root, profile=self.storage.SCIENTIFIC) == self.sources,
                  "scientific sources changed")
            role = "cpu_audit" if self.phase == "audit" else "native_source"
            _need(provenance.collect_runtime_environment(root, profile=self.storage.SCIENTIFIC,
                                                         runtime_role=role) == self.environment,
                  "scientific phase environment changed")
        self._check("scientific_controller.context.post")

    def _context(self, identity):
        context = super()._context(identity)
        context["environment"] = self.source_environment
        if self.phase == "audit":
            context.pop("native_device")
            context["auditor_environment"] = self.environment
        return context

    def _member(self, operation):
        for role, bundles in (("primary", self.policy.PRIMARY), ("sensitivity", (71901,))):
            for bundle in bundles:
                if operation == f"{role}.source.b{bundle}":
                    return role, bundle, 101, "source"
                for update in self.policy.ANCHORS:
                    if operation == f"{role}.branch.b{bundle}.u{update}":
                        return role, bundle, update, "branch"
                    if operation == f"audit.{role}.b{bundle}.u{update}":
                        return role, bundle, update, "audit"
        raise ControllerError("unknown scientific operation")

    def _produce(self, row):
        if self.kind == "synthetic_contract_fixture" or row["operation"] == "scientific_plans":
            return super()._produce(row)
        role, bundle, update, work = self._member(row["operation"])
        context = self._context(self.policy.identity(role, bundle, update))
        if work == "source":
            import native_source
            return native_source.run_source(capture_mode="capture_on", **context)
        if work == "branch":
            import native_branches
            return dict(native_branches.run_branches(**context), invariant_pass=True)
        import native_audit
        result = native_audit.run_audit(**context)
        # A failed audit may have sealed its report and terminalized already.
        # The inherited post-producer guard then refuses any policy append.
        return dict(result, invariant_pass=result["artifact"]["payload"]["overall_status"] == "pass")

    def _outputs(self, row, produced):
        if self.kind == "synthetic_contract_fixture" or row["operation"] == "scientific_plans":
            return super()._outputs(row, produced)
        import anchor_envelope as envelopes
        import source_capture
        import source_history
        role, bundle, update, work = self._member(row["operation"])
        refs = []
        for name in row["artifacts"]:
            raw, ref = self._read(name)
            refs.append(ref)
            del raw
        if work in ("branch", "audit"):
            value, ref = self._tensor(row["artifacts"][0])
            kind = "branch-results" if work == "branch" else "independent-audit"
            schema = "i7_branch_results" if work == "branch" else "i7_anchor_numerical_audit"
            envelopes.validate_common(value, schema_name=schema, kind=kind,
                identity=self.policy.identity(role, bundle, update), profile=self.storage.SCIENTIFIC)
            _need(envelopes.same_exact(value, produced["artifact"]), "actual scientific output differs from producer")
            self._returned_receipt(ref, produced["receipt"])
            if work == "audit":
                payload = value["payload"]
                _need(payload["overall_status"] == "pass" and
                      payload["exact_validation"]["completion"]["checks_complete"] is True and
                      envelopes.same_exact(payload["exact_validation"]["auditor_environment"], self.environment),
                      "audit is incomplete or has wrong phase environment")
            return refs, True
        completion, ref = self._tensor(self.policy.source_name(role, bundle))
        source_history.validate_source_completion(completion)
        _need(completion["profile"] == self.storage.SCIENTIFIC and
              completion["artifact_name"] == self.policy.source_name(role, bundle) and
              completion["provenance"] == dict(sources_sha256=self.codec.tree_digest(self.sources),
                                                environment_sha256=self.codec.tree_digest(self.source_environment)),
              "scientific completion provenance differs")
        _need(envelopes.same_exact(completion, produced["completion"]), "actual completion differs from producer")
        self._returned_receipt(ref, produced["completion_receipt"])
        _, plan_ref = self._read(self.policy.plan_name(bundle))
        self._bind_completion_refs(completion, refs, plan_ref, role=role, bundle=bundle, anchors=self.policy.ANCHORS)
        del completion
        for update in self.policy.ANCHORS:
            identity = self.policy.identity(role, bundle, update)
            context = self._context(identity)
            context = {key:context[key] for key in ("identity", "profile", "plan", "plan_artifact",
                "images_bytes", "labels_bytes", "expected_files", "sources", "environment")}
            anchor, anchor_ref = self._tensor(self.policy.artifact_name(role, bundle, update, "anchor"))
            witness, _ = self._tensor(self.policy.artifact_name(role, bundle, update, "source-witness"))
            receipt = dict(schema="i7_artifact_receipt_v1", name=anchor_ref["name"], size=anchor_ref["size_bytes"],
                sha256=anchor_ref["sha256"], status="complete", encoding="torch_weights_only",
                receipt_name=anchor_ref["receipt_name"])
            envelopes.validate_anchor(anchor, **context)
            source_capture.validate_source_witness(witness, anchor=anchor, anchor_receipt=receipt, **context)
            del anchor, witness, context
            self._check("scientific_controller.source_transaction_validated")
        return refs, True

    def run_phase(self):
        _need(not self._started and self._cursor == PHASE_START[self.phase], "phase already started; no replay")
        self._started = True
        try:
            while True:
                row = self.policy.schedule()[self._cursor]
                _need(row["phase"] == self.phase, "phase attempted to cross permission boundary")
                if row["kind"] == "work":
                    self._work()
                else:
                    self._check_context()
                    self._append(row["kind"], row)
                    self._cursor += 1
                    if row["kind"] == "boundary":
                        break
            return self.summary()
        except BaseException:
            self._fail("scientific_phase_failed")
            raise

    def close_phase(self):
        return self._close_boundary(self.phase)

    def run_development(self):
        raise ControllerError("later-phase controller cannot run development")

    def prepare_primary(self, **kwargs):
        raise ControllerError("preparation is part of one explicitly permitted primary phase")

    def summary(self):
        try:
            self._check("scientific_controller.summary")
        except BaseException:
            self._fail("scientific_summary_guard_failed")
            raise
        program = self.policy.schedule()
        done = self._cursor == len(program) or program[self._cursor]["phase"] != self.phase
        return dict(status=self.phase + "_complete" if done else "in_progress", phase=self.phase,
            next_operation=None if self._cursor == len(program) else program[self._cursor]["operation"],
            phase_records=len(self._records), resources=self.guard.summary(), evidence_kind=self.kind,
            automatic_next_phase_enabled=False, scientific_execution_certified=False)


def main(argv=None):
    _need(not argv, "no scientific launch CLI is enabled")
    print('{"status":"inert"}')
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
