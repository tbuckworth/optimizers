"""Fixed development producer integration. No native CLI or later-phase GO.

Only an already-authorized, trusted bootstrap may supply a native store.
Synthetic contract callbacks are confined to the explicit fixture evidence kind.
This module is not a permission, recovery framework or complete launcher.
"""
from __future__ import annotations

import copy
import hashlib
import io
import os


class ControllerError(RuntimeError):
    pass


def _need(condition, message):
    if not condition:
        raise ControllerError(message)


def _receipt_name(name):
    return "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"


class NativeController:
    """One live writer, development then primary plan preparation; no resume.

    Successful records are producer attestations. Native producers perform their
    own full envelope validation; the controller additionally binds persisted
    outputs and the complete pilot bundle. Independent numerical audit is later.
    """

    def __init__(self, bootstrap, *, entry_process_id, data_directory, expected_files, utc_now,
                 fixture_work=None, fixture_guard=None):
        # The supplied launcher origins precede these imports. First guard check
        # charges them; this is cooperative detection, not pre-import preemption.
        import artifact_store as storage
        import identity_codec as codec
        import native_control as control
        import native_phase_policy as policy
        import runtime_guard as runtime
        self.storage, self.codec, self.control, self.policy = storage, codec, control, policy
        self.store = bootstrap["store"]
        self._pid = os.getpid()
        self._terminal = False
        self._handoff_started = False
        self.handoff_failure_metadata_status = None
        self._records = []
        self._cursor = 1
        self._data = None
        self._fixture_work = fixture_work
        self.utc_now = utc_now
        self.data_directory = data_directory
        self.expected_files = copy.deepcopy(expected_files)
        self.guard = None
        try:
            _need(type(self.store) is storage.ArtifactStore, "exact ArtifactStore required")
            self.store._ensure_writable()
            go = bootstrap["go_record"]
            self.kind = go["evidence_kind"]
            policy.validate_record(go, evidence_kind=self.kind)
            _need(go["operation"] == "development_go" and go["sequence"] == 0,
                  "controller requires the initial development bootstrap")
            native = self.kind == "native_producer_attestation"
            _need(self.store.profile == (storage.SCIENTIFIC if native else storage.MLP_FIXTURE),
                  "evidence kind/store profile mismatch")
            _need(callable(utc_now), "UTC producer required")
            _need(type(entry_process_id) is int and entry_process_id == self._pid,
                  "bootstrap origins require their entry-captured process identity")
            if native:
                _need(fixture_work is None and fixture_guard is None,
                      "native controller forbids fixture callbacks")
                self.guard = runtime.RuntimeGuard("development",
                    entry_wall_origin=bootstrap["clock_origins"]["wall_origin"],
                    entry_cpu_origin=bootstrap["clock_origins"]["cpu_origin"],
                    entry_process_id=entry_process_id)
            else:
                _need(callable(fixture_work) and type(fixture_guard) is runtime.RuntimeGuard,
                      "explicit synthetic operation and guard required")
                self.guard = fixture_guard
                _need(self.guard.profile == runtime.FIXTURE and self.guard.phase == "development",
                      "synthetic development guard differs")
            _need(self.guard.entry_origins == bootstrap["clock_origins"],
                  "guard does not cover bootstrap entry")
            self._check("controller.bootstrap.entry")
            self.root_binding = copy.deepcopy(bootstrap["root_binding"])
            _need(control._root_binding(self.store) == self.root_binding, "bootstrap root differs")
            expected_names = {"store-header.json", "store.lock"}
            for name in (control.phase_name(0), control.ROOT_JOURNAL_NAME,
                         control.SOURCE_NAME, control.ENVIRONMENT_NAME):
                expected_names.update((name, _receipt_name(name)))
            _need(set(self.store._indexed) == expected_names,
                  "bootstrap already used or unexpected payload present; no resume")
            raw, _ = self._read(control.phase_name(0))
            _need(raw == bootstrap["go_bytes"] == codec.json_bytes(go), "saved GO differs")
            self.sources = self._json(control.SOURCE_NAME)
            self.environment = self._json(control.ENVIRONMENT_NAME)
            import source_environment_schema as schema
            schema.validate_sources(self.sources, profile=storage.SCIENTIFIC)
            schema.validate_environment(self.environment, profile=storage.SCIENTIFIC)
            _need(codec.tree_digest(self.sources) == go["sources_sha256"] and
                  codec.tree_digest(self.environment) == go["environment_sha256"],
                  "saved metadata differs from GO")
            pin = bootstrap["journal_pin"]
            external = control._read_pinned(pin["path"], pin["size_bytes"], pin["sha256"],
                                            control.EXTERNAL_JOURNAL_MAX)
            saved, _ = self._read(control.ROOT_JOURNAL_NAME)
            _need(saved == external, "saved launch journal differs")
            journal = self.codec.json_loads(saved, max_bytes=control.EXTERNAL_JOURNAL_MAX)
            control._validate_journal(journal)
            _need(journal["launcher_entry_wall_origin"] == bootstrap["clock_origins"]["wall_origin"] and
                  journal["launcher_entry_cpu_origin"] == bootstrap["clock_origins"]["cpu_origin"],
                  "entry clocks differ from consumed journal")
            _need(journal["expected_commit"] == self.sources["repository_revision"] and
                  journal["expected_source_set_sha256"] == go["sources_sha256"] and
                  journal["expected_gpu_uuid"] == self.environment["cuda"]["devices"][0]["uuid"],
                  "journal metadata differs")
            if native:
                _need(pin["path"] == control.DEVELOPMENT_ATTEMPT_PATH + "/" + control.EXTERNAL_JOURNAL_NAME,
                      "native journal is not the registered singleton")
            self.journal_pin = copy.deepcopy(pin)
            self._records = [copy.deepcopy(go)]
            self._check_context()
            self._check("controller.bootstrap.exit")
        except BaseException:
            self._fail("controller_bootstrap_failed")
            raise

    def _check(self, stage):
        _need(os.getpid() == self._pid, "controller/CPU origins cannot cross processes")
        _need(not self._terminal, "controller is terminal")
        self.store._ensure_writable()
        return self.guard.check(stage)

    def _fail(self, reason):
        self._terminal = True
        store = self.store
        if (type(store) is self.storage.ArtifactStore and not store._closed and
                not store._terminal):
            store._fail("native_controller_failed", reason)

    def _read(self, name):
        """Authenticate indexed payload AND exact writer receipt on held dirfd."""
        self._check("controller.read.pre")
        self.store._inventory(self.store._indexed)
        meta = self.store._indexed.get(name)
        receipt_name = _receipt_name(name)
        receipt_meta = self.store._indexed.get(receipt_name)
        _need(meta is not None and receipt_meta is not None, "missing indexed output/receipt")
        raw, _ = self.storage._read_regular(self.store._dirfd, name, maximum=meta["size"])
        _need(len(raw) == meta["size"] and hashlib.sha256(raw).hexdigest() == meta["sha256"],
              "saved output differs from indexed bytes")
        receipt_raw, _ = self.storage._read_regular(self.store._dirfd, receipt_name,
                                                    maximum=self.storage._RECEIPT_MAX)
        _need(len(receipt_raw) == receipt_meta["size"] and
              hashlib.sha256(receipt_raw).hexdigest() == receipt_meta["sha256"],
              "saved receipt differs from indexed bytes")
        encoding = "bytes" if name.endswith(".json") else "torch_weights_only"
        expected = dict(schema="i7_artifact_receipt_v1", name=name, size=len(raw),
                        sha256=meta["sha256"], status="complete", encoding=encoding)
        _need(receipt_raw == self.storage._json_bytes(expected), "saved receipt content differs")
        ref = dict(name=name, status="complete", encoding=encoding, size_bytes=len(raw),
            sha256=meta["sha256"], receipt_name=receipt_name,
            receipt_size_bytes=len(receipt_raw), receipt_sha256=receipt_meta["sha256"])
        self.policy.validate_reference(ref, name=name, encoding=encoding)
        self.store._inventory(self.store._indexed)
        self._check("controller.read.post")
        return raw, ref

    def _json(self, name):
        raw, _ = self._read(name)
        value = self.codec.json_loads(raw, max_bytes=1 << 20)
        _need(raw == self.codec.json_bytes(value), "noncanonical JSON output")
        return value

    def _tensor(self, name):
        raw, ref = self._read(name)
        import torch
        import verified_plan_load as verified
        verified._check_safe_globals()
        value = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        self._check("controller.tensor.post")
        return value, ref

    def _check_context(self):
        self._check("controller.context.pre")
        if self.kind == "native_producer_attestation":
            import source_capture as capture
            capture._verify_live_context(self.sources, self.environment,
                                         self.storage.SCIENTIFIC, include_sources=True)
        self._check("controller.context.post")

    def _resources(self):
        self._check("controller.inventory.pre")
        import verified_plan_load as verified
        _, fd, opened, _ = verified._open_root(str(self.store.root), self.store.profile)
        try:
            _need((opened.st_dev, opened.st_ino) == self.store.root_identity,
                  "actual root path differs from held identity")
        finally:
            os.close(fd)
        header, _ = self.storage._read_regular(self.store._dirfd, "store-header.json", maximum=4096)
        _need(self.control._root_binding(self.store) == self.root_binding and
              hashlib.sha256(header).hexdigest() == self.root_binding["header_sha256"],
              "root identity/header differs")
        logical, allocated = self.store._inventory(self.store._indexed)
        self.store._precheck(0)
        sampled = self._check("controller.inventory.post")
        return dict(phase_wall_seconds=float(sampled["wall_seconds"]),
            phase_cpu_seconds=float(sampled["cpu_seconds"]),
            peak_rss_bytes=sampled["peak_rss_bytes"],
            peak_cuda_allocated_bytes=sampled["peak_cuda_allocated_bytes"],
            peak_cuda_reserved_bytes=sampled["peak_cuda_reserved_bytes"],
            root_logical_bytes=logical, root_allocated_bytes=allocated)

    def _append(self, event, row, refs=()):
        record = dict(schema="i7_native_phase_record_v1", profile=self.policy.PROFILE,
            evidence_kind=self.kind, execution_enabled=False, sequence=len(self._records),
            event=event, operation=row["operation"], phase=row["phase"],
            created_utc=self.utc_now(), root_binding=copy.deepcopy(self.root_binding),
            sources_sha256=self.codec.tree_digest(self.sources),
            environment_sha256=self.codec.tree_digest(self.environment),
            previous_record_sha256=hashlib.sha256(self.codec.json_bytes(self._records[-1])).hexdigest(),
            artifacts=list(refs), resources=self._resources(), error=None)
        self.policy.validate_transcript(self._records + [record], evidence_kind=self.kind,
                                        require_complete=False)
        raw = self.codec.json_bytes(record)
        self._check("controller.record.pre_write")
        self.store.write_bytes(self.control.phase_name(len(self._records)), raw)
        # A failed post-write check leaves the actual retained record and terminal
        # store; no fabricated success return or post-terminal policy append.
        self._records.append(record)
        self._check("controller.record.post_write")

    def _context(self, identity):
        import verified_plan_load as verified
        name = self.policy.plan_name(identity["bundle"])
        self._check("controller.plan_load.pre")
        loaded = verified.load_verified_plan_from_store(self.store, name, identity=identity,
            profile=self.storage.SCIENTIFIC, expected_sha256=self.store._indexed[name]["sha256"])
        self._check("controller.plan_load.post")
        if self._data is None:
            import native_inputs as inputs
            self._data = inputs.read_training_idx(directory=self.data_directory,
                expected_files=self.expected_files, profile=self.storage.SCIENTIFIC,
                checkpoint=self.guard.check)
        return dict(store=self.store, identity=identity, profile=self.storage.SCIENTIFIC,
            native_device="cuda:0", plan=loaded["plan"], plan_reference=loaded["artifact"],
            plan_artifact={key: loaded["artifact"][key] for key in ("sha256", "size_bytes")},
            **self._data, sources=self.sources, environment=self.environment,
            created_utc=self.utc_now(), guard=self.guard.check)

    def _produce(self, row):
        """Only fixed development operations and four-plan preparation are wired."""
        operation = row["operation"]
        if self.kind == "synthetic_contract_fixture":
            return self._fixture_work(copy.deepcopy(row), self.store, self.guard.check)
        if operation in ("development.plan", "scientific_plans"):
            import native_inputs as inputs
            members = (("pilot", 71990),) if operation == "development.plan" else (
                *(("primary", bundle) for bundle in self.policy.PRIMARY), ("sensitivity", 71901))
            for role, bundle in members:
                identity = self.policy.identity(role, bundle, 101)
                plan = inputs.generate_plan(identity=identity, profile=self.storage.SCIENTIFIC,
                                             checkpoint=self.guard.check)
                self._check("controller.plan_write.pre")
                self.store.write_tensor_tree(self.policy.plan_name(bundle), plan)
                del plan
                self._check("controller.plan_write.post")
            return {"invariant_pass":True}
        if operation == "development.source_pair":
            import native_source as source
            return source.run_pilot_pair(**self._context(self.policy.identity("pilot", 71990, 101)))
        for update in self.policy.PILOT_ANCHORS:
            if operation == f"development.branch.u{update}":
                import native_branches as branches
                result = branches.run_branches(**self._context(self.policy.identity("pilot", 71990, update)))
                return dict(result, invariant_pass=True)
        raise ControllerError("later scientific operation has no scoped launcher permission integration")

    def _outputs(self, row, produced):
        refs = []
        for name in row["artifacts"]:
            raw, ref = self._read(name)
            refs.append(ref)
            if self.kind == "synthetic_contract_fixture":
                # Real file/receipt binding only. Never label primitive test
                # payloads as validated scientific envelopes.
                continue
            if name.endswith("-plan.pt"):
                import verified_plan_load as verified
                bundle = next(b for b in (*self.policy.PRIMARY, 71901, 71990)
                              if name == self.policy.plan_name(b))
                role = "pilot" if bundle == 71990 else "sensitivity" if bundle == 71901 else "primary"
                verified.load_verified_plan_from_store(self.store, name,
                    identity=self.policy.identity(role, bundle, 101), profile=self.storage.SCIENTIFIC,
                    expected_sha256=ref["sha256"])
            elif name.endswith(".json"):
                comparison = self.codec.json_loads(raw, max_bytes=1 << 20)
                _need(self.codec.json_bytes(comparison) == raw, "comparison JSON not canonical")
                import source_history as history
                history.validate_capture_comparison(comparison)
            else:
                # Full native producer checks precede writes. Recheck identity/
                # source completion semantics on authenticated bytes here.
                value, _ = self._tensor(name)
                if "-source-capture_" in name:
                    import source_history as history
                    history.validate_source_completion(value)
                    _need(value["artifact_name"] == name and value["profile"] == self.storage.SCIENTIFIC,
                          "completion membership differs")
                    _need(value["provenance"] == dict(sources_sha256=self.codec.tree_digest(self.sources),
                        environment_sha256=self.codec.tree_digest(self.environment)), "completion provenance differs")
                else:
                    import anchor_envelope as envelope
                    kinds = {"anchor":"i7_anchor", "source-witness":"i7_source_step_witness",
                             "branch-results":"i7_branch_results"}
                    match = next(( (update, kind) for update in self.policy.PILOT_ANCHORS for kind in kinds
                        if name == self.policy.artifact_name("pilot", 71990, update, kind)), None)
                    _need(match is not None, "unregistered development output")
                    update, kind = match
                    envelope.validate_common(value, schema_name=kinds[kind], kind=kind,
                        identity=self.policy.identity("pilot", 71990, update), profile=self.storage.SCIENTIFIC)
                    if kind == "branch-results":
                        _need(envelope.same_exact(value, produced["artifact"]),
                              "stored branch differs from validated producer value")
                        self._returned_receipt(ref, produced["receipt"])
                del value
            del raw
            self._check("controller.output_validated")
        if row["operation"] == "development.source_pair" and self.kind == "native_producer_attestation":
            import source_history as history
            import native_source as source
            on, _ = self._tensor(self.policy.source_name("pilot", 71990))
            off, _ = self._tensor(self.policy.source_name("pilot", 71990, "capture_off"))
            comparison = self._json("i7-native-pilot-b71990-capture-comparison.json")
            history.validate_capture_bundle(on, off, comparison)
            _, plan_ref = self._read(self.policy.plan_name(71990))
            self._bind_completion_refs(on, refs, plan_ref)
            _need(off["plan_ref"] == plan_ref, "capture-off plan differs from saved plan")
            import anchor_envelope as envelope
            for key, value in (("capture_on", on), ("capture_off", off), ("comparison", comparison)):
                _need(envelope.same_exact(value, produced[key]),
                      "stored pilot bundle differs from validated producer value")
                name = value["artifact_name"]
                self._returned_receipt(next(ref for ref in refs if ref["name"] == name),
                                       produced[key + "_receipt"])
            invariant = source._pilot_invariant(comparison)
            del on, off, comparison
            # Authenticate each saved transaction and rerun full value/binding
            # validators even for an adverse pair that will not reach branches.
            import source_capture as capture
            for update in self.policy.PILOT_ANCHORS:
                identity = self.policy.identity("pilot", 71990, update)
                context = self._context(identity)
                context = {key: context[key] for key in ("identity", "profile", "plan", "plan_artifact",
                    "images_bytes", "labels_bytes", "expected_files", "sources", "environment")}
                anchor, anchor_ref = self._tensor(self.policy.artifact_name("pilot", 71990, update, "anchor"))
                witness, _ = self._tensor(self.policy.artifact_name("pilot", 71990, update, "source-witness"))
                anchor_receipt = dict(schema="i7_artifact_receipt_v1", name=anchor_ref["name"],
                    size=anchor_ref["size_bytes"], sha256=anchor_ref["sha256"], status="complete",
                    encoding="torch_weights_only", receipt_name=anchor_ref["receipt_name"])
                envelope.validate_anchor(anchor, **context)
                capture.validate_source_witness(witness, anchor=anchor,
                                                anchor_receipt=anchor_receipt, **context)
                del anchor, witness, context
                self._check("controller.transaction_validated")
            self._check("controller.bundle_validated")
        else:
            invariant = True
        return refs, invariant

    @staticmethod
    def _returned_receipt(ref, receipt):
        expected = dict(schema="i7_artifact_receipt_v1", name=ref["name"],
            size=ref["size_bytes"], sha256=ref["sha256"], status="complete",
            encoding=ref["encoding"], receipt_name=ref["receipt_name"])
        _need(type(receipt) is dict and tuple(receipt) == tuple(expected) and receipt == expected,
              "producer receipt differs from actual saved receipt")

    def _bind_completion_refs(self, completion, refs, plan_ref, *, role="pilot", bundle=71990,
                              anchors=None):
        """Bind pure completion references to this operation's authenticated files."""
        _need(completion["plan_ref"] == plan_ref, "completion plan differs from saved plan")
        actual = {ref["name"]:ref for ref in refs}
        pairs = completion["anchor_witness_refs"]
        anchors = self.policy.PILOT_ANCHORS if anchors is None else anchors
        _need([pair["anchor_update"] for pair in pairs] == list(anchors),
              "completion anchor order differs")
        for pair in pairs:
            for key, kind in (("anchor_ref", "anchor"), ("witness_ref", "source-witness")):
                name = self.policy.artifact_name(role, bundle, pair["anchor_update"], kind)
                nested = {field:pair[key][field] for field in self.policy.REF_KEYS}
                _need(nested == actual.get(name), "completion ref differs from actual saved output")

    def _work(self):
        row = self.policy.schedule()[self._cursor]
        _need(row["kind"] == "work", "pending operation is not work")
        try:
            self._check_context()
            self._append("started", row)
            before = set(self.store._indexed)
            result = self._produce(row)
            _need(type(result) is dict and type(result.get("invariant_pass")) is bool,
                  "producer invariant result missing")
            self._check("controller.producer.post")
            additions = {part for name in row["artifacts"] for part in (name, _receipt_name(name))}
            _need(set(self.store._indexed) == before | additions,
                  "producer output membership differs")
            refs, invariant = self._outputs(row, result)
            self._check_context()
            self._append("sealed", row, refs)
            self._cursor += 1
            if not result["invariant_pass"] or not invariant:
                self._fail("capture_path_inequality")
                raise ControllerError("pilot invariant failed; subsequent work forbidden")
        except BaseException:
            self._fail("operation_failed")
            raise

    def run_development(self):
        _need(self._cursor == 1, "development already started; no replay")
        try:
            while self.policy.schedule()[self._cursor]["kind"] == "work":
                self._work()
            row = self.policy.schedule()[self._cursor]
            _need(row["operation"] == "development_complete", "unexpected boundary")
            self._check_context()
            self._append("boundary", row)
            self._cursor += 1
            return self.summary()
        except BaseException:
            self._fail("development_failed")
            raise

    def prepare_primary(self, *, fixture_guard=None):
        _need(self.kind == "synthetic_contract_fixture",
              "native preparation disabled pending scoped primary permission/environment integration")
        # Only the synthetic fixed-schedule test is enabled. Native preparation
        # needs fresh primary permission/environment and entry clocks first.
        import runtime_guard as runtime
        _need(self.policy.schedule()[self._cursor]["operation"] == "scientific_plans",
              "plan preparation requires successful development; no replay")
        try:
            _need(type(fixture_guard) is runtime.RuntimeGuard and
                  fixture_guard.profile == runtime.FIXTURE and fixture_guard.phase == "primary" and
                  fixture_guard.entry_origins is not None, "explicit synthetic primary entry guard required")
            self.guard = fixture_guard
            self._work()
            return self.summary()
        except BaseException:
            self._fail("primary_preparation_failed")
            raise

    def close_development(self):
        """Close one completed development writer and return inspected pins.

        No writable reopen or later-phase permission is granted. The returned
        state is an observed snapshot; it does not keep a lock after return.
        """
        return self._close_boundary("development")

    def _close_boundary(self, phase):
        program = self.policy.schedule()
        _need(phase in ("development", "primary", "sensitivity", "audit"), "unknown closing phase")
        boundary_index = next(i for i, row in enumerate(program)
                              if row["phase"] == phase and row["kind"] == "boundary")
        count = sum(2 if row["kind"] == "work" else 1 for row in program[:boundary_index + 1])
        operation = program[boundary_index]["operation"]
        next_operation = program[boundary_index + 1]["operation"] if boundary_index + 1 < len(program) else None
        final = phase == "audit"
        _need(self._cursor == boundary_index + 1 and not self._handoff_started,
              "handoff requires the untouched completed " + phase + " boundary")
        self._handoff_started = True
        try:
            self._check("controller.handoff.entry")
            self._check_context()
            _need(self.guard.phase == phase, "handoff guard phase differs")
            _need(len(self._records) == count and self._records[-1]["operation"] == operation
                  and self._records[-1]["event"] == "boundary", "handoff boundary differs")
            name = self.control.phase_name(count - 1)
            raw, boundary_ref = self._read(name)
            _need(raw == self.codec.json_bytes(self._records[-1]), "actual boundary differs")
            del raw
            accounting_before_close = self._resources()
            root_binding = copy.deepcopy(self.root_binding)
            journal_pin = copy.deepcopy(self.journal_pin)
            self._data = None
            self._check("controller.handoff.pre_close")
            self.store.close()
            # _check requires a writable store; after close use the same guard
            # directly, preserving its phase origins and process binding.
            self.guard.check("controller.handoff.post_close")
            inspected = self.control.inspect_native_attempt(root_binding["path"], journal_pin,
                expected_store_profile=self.store.profile, expected_evidence_kind=self.kind)
            self.guard.check("controller.handoff.post_inspection")
            _need(inspected["schema"] == self.control.INSPECTION_SCHEMA and
                  inspected["status"] == ("complete" if final else "sealed_boundary") and
                  inspected["phase_records"] == count and inspected["last_event"] == "boundary" and
                  inspected["pending_operation"] == next_operation and
                  inspected["root_identity"] == [root_binding["device"], root_binding["inode"]] and
                  inspected["header_sha256"] == root_binding["header_sha256"] and
                  inspected["journal_pin"] == journal_pin and
                  inspected["metadata_complete"] is True and inspected["actual_artifacts_verified"] is True and
                  inspected["store_terminal"] is False and inspected["eligible_for_writable_reopen"] is (not final) and
                  inspected["can_resume_incomplete"] is False and
                  inspected["execution_authorized"] is False and
                  inspected["scientific_execution_certified"] is False,
                  "closed phase boundary did not pass actual inspection")
            self.control._read_pinned(root_binding["path"] + "/" + boundary_ref["name"],
                boundary_ref["size_bytes"], boundary_ref["sha256"], self.policy.MAX_RECORD_BYTES)
            after = os.lstat(root_binding["path"])
            _need((after.st_dev, after.st_ino) == (root_binding["device"], root_binding["inode"]),
                  "root changed after handoff inspection")
            self.guard.check("controller.handoff.final")
            return dict(schema="i7_development_handoff_v1" if phase == "development" else "i7_scientific_handoff_v1",
                status="closed_inspected_boundary",
                evidence_kind=self.kind, root_binding=root_binding, journal_pin=journal_pin,
                boundary_ref=boundary_ref, inspection=inspected,
                root_accounting_before_close=accounting_before_close, resources=self.guard.summary(),
                execution_authorized=False, automatic_reopen_enabled=False,
                process_exit_verified=False, scientific_execution_certified=False)
        except BaseException:
            # A close may already have released some/all descriptors. Do not
            # manufacture a post-terminal record or reopen the root to write.
            self._terminal = True
            if not self.store._closed and not self.store._terminal:
                try:
                    self.store._fail("native_handoff_failed", "phase_handoff_failed")
                except BaseException:
                    pass
            try:
                self.control.record_boundary_failure(self.journal_pin, self.root_binding,
                                                      expected_evidence_kind=self.kind)
                self.handoff_failure_metadata_status = "retained"
            except BaseException:
                self.handoff_failure_metadata_status = "unavailable_or_preexisting"
            if not self.store._closed:
                try:
                    self.store.close()
                except BaseException:
                    pass
            raise

    def summary(self):
        try:
            self._check("controller.summary")
        except BaseException:
            self._fail("summary_guard_failed")
            raise
        status = {6:"development_complete", 7:"primary_go_required"}.get(self._cursor, "in_progress")
        return dict(status=status,
            next_operation=self.policy.schedule()[self._cursor]["operation"],
            phase_records=len(self._records), resources=self.guard.summary(),
            evidence_kind=self.kind, later_phase_execution_enabled=False,
            numerical_audit_complete=False, scientific_execution_certified=False)


def main(argv=None):
    if argv:
        raise ControllerError("no native CLI is enabled")
    print('{"status":"inert"}')
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
