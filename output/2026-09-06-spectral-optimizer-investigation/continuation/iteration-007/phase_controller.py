"""Single-root fixture phase control. Scientific execution is deliberately disabled."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import io
import os

import torch

import anchor_envelope as envelope
import artifact_store as storage
import identity_codec as codec
import runtime_guard as runtime
import source_capture as source
import state_core as core
import verified_plan_load as verified

FIXTURE = storage.MLP_FIXTURE
EVENTS = ("ready", "primary_go", "source_started", "source_complete",
          "branches_started", "primary_complete", "audit_go", "audit_started", "complete")
BOUNDARIES = ("ready", "primary_complete", "complete")
KEYS = ("schema", "profile", "sequence", "event", "created_utc", "root", "previous",
        "bindings", "inventory_before", "outputs", "runtime", "scientific_execution_certified")
EVENT_MAX = 512 << 10
SOURCE_COMPLETE = "fixture-primary-source-complete.pt"


class PhaseError(RuntimeError):
    pass


def _need(condition, message):
    if not condition:
        raise PhaseError(message)


def scientific_schedule():
    """Pure membership/order blueprint, not plans, data, artifacts or a launch GO."""
    return {
        "profile": storage.SCIENTIFIC,
        "execution_enabled": False,
        "development": {"bundles": [71990], "updates": 220, "anchors": [101, 200],
                        "capture_on_off_required": True},
        "primary": {"bundles": [71001, 71002, 71003], "updates": 2000,
                    "anchors": [101, 500, 1000, 2000]},
        "sensitivity": {"bundles": [71901], "updates": 2000,
                        "anchors": [101, 500, 1000, 2000], "pooled": False},
        "order": ["development_go", "development_complete", "all_scientific_plans_frozen",
                  "primary_go", "all_primary_sources_complete", "all_primary_branches_complete",
                  "sensitivity_go", "sensitivity_source_complete", "sensitivity_branches_complete",
                  "audit_go", "both_sets_audited"],
    }


def _ref(name, data):
    return dict(name=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _receipt_name(name):
    return "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"


def _snapshot(store):
    store._ensure_writable()
    report, scanned = storage._inspect_dirfd(store._dirfd)
    verified._strict_metadata(store._dirfd, scanned)
    _need(report["header"]["profile"] == store.profile and not report["terminal"], "store profile/terminal")
    store._inventory(store._indexed)
    rows = []
    for name in sorted(scanned):
        data, info = storage._read_regular(store._dirfd, name, maximum=store.budget,
                                          expected=scanned[name])
        rows.append(dict(**_ref(name, data), allocated_bytes=info.st_blocks * 512))
    final, _, _ = storage._scan_regular(store._dirfd)
    _need(verified._snapshot(final) == verified._snapshot(scanned), "inventory changed during phase snapshot")
    return rows


def _root(store):
    path, fd, opened, _ = verified._open_root(str(store.root), store.profile)
    try:
        held = os.fstat(store._dirfd)
        _need((opened.st_dev, opened.st_ino) == (held.st_dev, held.st_ino), "phase root replaced")
        header, _ = storage._read_regular(store._dirfd, "store-header.json", maximum=4096)
        return dict(path=path, device=held.st_dev, inode=held.st_ino,
                    header_sha256=hashlib.sha256(header).hexdigest())
    finally:
        os.close(fd)


def _runtime_check(summary, phase):
    keys = ("profile", "phase", "limits", "check_count", "last_record", "terminal",
            "cuda_initialized", "runtime", "rss_scope", "cpu_scope", "preemption")
    _need(type(summary) is dict and tuple(summary) == keys, "runtime summary schema")
    _need(summary["profile"] == FIXTURE and summary["phase"] == phase
          and summary["terminal"] is False and summary["cuda_initialized"] is False
          and type(summary["check_count"]) is int and summary["check_count"] > 0,
          "runtime summary not a passing fixture phase")
    _need(envelope.same_exact(summary["limits"], runtime._limits(phase)), "runtime limits differ")
    last = summary["last_record"]
    _need(type(last) is dict and last.get("status") == "pass"
          and last.get("check_index") == summary["check_count"]
          and last.get("violations") == [] and last.get("probe_error") is None,
          "runtime last check did not pass")


def _expected_outputs(event, identity):
    def name(kind):
        return codec.artifact_id(identity, profile=FIXTURE, kind=kind) + ".pt"
    return {"source_complete": [name("anchor"), name("source-witness"), SOURCE_COMPLETE],
            "primary_complete": [name("branch-results")],
            "complete": [name("independent-audit")]}.get(event, [])


def _payload(store, ref):
    _need(type(ref) is dict and tuple(ref) == ("name", "size_bytes", "sha256"), "output reference schema")
    storage._validate_name(ref["name"])
    _need(type(ref["size_bytes"]) is int and 0 < ref["size_bytes"] <= store.budget,
          "output size invalid")
    data, _ = storage._read_regular(store._dirfd, ref["name"], maximum=ref["size_bytes"])
    _need(envelope.same_exact(_ref(ref["name"], data), ref), "pinned output differs")
    return data


def _validate_output(store, ref, identity):
    data = _payload(store, ref)
    verified._check_safe_globals()
    value = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
    if ref["name"] == SOURCE_COMPLETE:
        core.validate_core(value)
        _need(value["profile"] == FIXTURE and value["state_completed_updates"] == 8,
              "source endpoint not complete")
    else:
        kind = next((kind for kind in codec.ARTIFACT_KINDS
                     if ref["name"] == codec.artifact_id(identity, profile=FIXTURE, kind=kind) + ".pt"), None)
        schema = {"anchor": "i7_anchor", "source-witness": "i7_source_step_witness",
                  "branch-results": "i7_branch_results", "independent-audit": "i7_anchor_numerical_audit"}
        _need(kind in schema, "unknown phase output")
        envelope.validate_common(value, schema_name=schema[kind], kind=kind, identity=identity, profile=FIXTURE)
        if kind == "independent-audit":
            _need(value["payload"]["overall_status"] == "pass"
                  and value["payload"]["exact_validation"]["completion"]["checks_complete"] is True,
                  "numerical audit incomplete or failed")
    return value


class PhaseController:
    """Trusted single-process fixture orchestrator; callbacks cannot authorize science."""

    def __init__(self, store, *, context, plan_name, created_utc):
        _need(type(store) is storage.ArtifactStore and store.profile == FIXTURE,
              "only the registered CPU MLP fixture is executable; scientific GO is closed")
        codec.validate_identity(context["identity"], profile=FIXTURE)
        _need(context["profile"] == FIXTURE, "context profile differs")
        self.store, self.identity = store, copy.deepcopy(context["identity"])
        self.context = copy.deepcopy(context)
        self._check_context()
        loaded = verified.load_verified_plan_from_store(store, plan_name, identity=self.identity,
            profile=FIXTURE, expected_sha256=context["plan_artifact"]["sha256"])
        _need(envelope.same_exact(loaded["plan"], context["plan"]), "context plan differs from sealed bytes")
        self.bindings = dict(identity=codec.tree_digest(self.identity),
            plan=copy.deepcopy(loaded["artifact"]),
            sources_sha256=codec.tree_digest(context["sources"]),
            environment_sha256=codec.tree_digest(context["environment"]))
        self.root = _root(store)
        self.last, self.last_ref, self.guard = None, None, None
        self._append("ready", [], created_utc)

    def _check_context(self):
        source._verify_live_context(self.context["sources"], self.context["environment"],
                                    FIXTURE, include_sources=True)

    def _check(self, label):
        if self.guard is not None:
            self.guard.check(label)

    def _append(self, event, outputs, created_utc):
        sequence = 0 if self.last is None else self.last["sequence"] + 1
        _need(sequence < len(EVENTS) and EVENTS[sequence] == event, "out-of-order phase event")
        codec.validate_created_utc(created_utc)
        self._check("phase.pre_snapshot")
        inventory = _snapshot(self.store)
        if self.last is None:
            plan_name = self.bindings["plan"]["name"]
            _need({row["name"] for row in inventory} ==
                  {"store.lock", "store-header.json", plan_name, _receipt_name(plan_name)},
                  "ready store must contain only one frozen plan")
        _need(envelope.same_exact(_root(self.store), self.root), "phase root identity changed")
        self._check("phase.pre_seal")
        row = dict(schema="i7_fixture_phase_event_v1", profile=FIXTURE, sequence=sequence,
            event=event, created_utc=created_utc, root=copy.deepcopy(self.root),
            previous=copy.deepcopy(self.last_ref), bindings=copy.deepcopy(self.bindings),
            inventory_before=inventory, outputs=outputs,
            runtime=None if self.guard is None else self.guard.summary(),
            scientific_execution_certified=False)
        raw = codec.json_bytes(row)
        _need(len(raw) <= EVENT_MAX, "phase event exceeds fixed bound")
        name = f"phase-{sequence:03d}.json"
        self.store.write_bytes(name, raw)
        self.last, self.last_ref = row, _ref(name, raw)
        self._check("phase.post_seal")
        return copy.deepcopy(self.last_ref)

    def boundary_pin(self):
        _need(self.last["event"] in BOUNDARIES, "unfinished phase has no writable recovery pin")
        return dict(root=copy.deepcopy(self.root), event=copy.deepcopy(self.last_ref))

    def go(self, phase, *, decision, created_utc):
        """The explicit decision is fixture-only; no generic approve flag enables science."""
        try:
            self._verify_chain(self.last_ref)
            self._check_context()
            expected = {"ready": "primary", "primary_complete": "audit"}.get(self.last["event"])
            _need(phase == expected and phase in ("primary", "audit"), "GO phase order")
            _need(decision == "dataset_free_fixture_only", "not a fixture decision")
            self.guard = runtime.RuntimeGuard(phase, profile=FIXTURE)
            return self._append(phase + "_go", [], created_utc)
        except BaseException as error:
            self._fatal(error)
            raise

    def execute(self, operation, body, *, created_utc):
        """Mark started before invoking body; exact newly sealed membership afterward."""
        try:
            self._verify_chain(self.last_ref)
            self._check_context()
            rules = {"source": ("primary_go", "source_started", "source_complete"),
                     "branches": ("source_complete", "branches_started", "primary_complete"),
                     "audit": ("audit_go", "audit_started", "complete")}
            _need(operation in rules and callable(body), "invalid fixture operation")
            before, started, completed = rules[operation]
            _need(self.last["event"] == before, "operation phase order")
            self._append(started, [], created_utc)
            before_files = {row["name"]: row for row in _snapshot(self.store)}
            returned = body(self.store, self.guard.check)
            self._check("phase.body_complete")
            self._check_context()
            _need(type(returned) is list and all(type(row) is dict for row in returned), "output list required")
            expected = _expected_outputs(completed, self.identity)
            _need([row.get("name") for row in returned] == expected, "operation output membership/order")
            current = {row["name"]: row for row in _snapshot(self.store)}
            expected_new = {name for payload in expected for name in (payload, _receipt_name(payload))}
            _need(set(current) - set(before_files) == expected_new, "unreported or missing operation files")
            _need(all(envelope.same_exact(current.get(name), row) for name, row in before_files.items()),
                  "operation changed previous files")
            for ref in returned:
                _validate_output(self.store, ref, self.identity)
            completed_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            result = self._append(completed, copy.deepcopy(returned), completed_utc)
            if completed in BOUNDARIES:
                self.guard = None
            return result
        except BaseException as error:
            self._fatal(error)
            raise

    def _fatal(self, error):
        if not self.store._closed and not self.store._terminal:
            self.store._fail("phase_failed", type(error).__name__)

    @classmethod
    def reopen(cls, pin, *, context):
        """Only a fully verified sealed boundary can acquire a new writer handle."""
        _need(type(pin) is dict and tuple(pin) == ("root", "event"), "boundary pin schema")
        root = pin["root"]
        _need(type(root) is dict and tuple(root) == ("path", "device", "inode", "header_sha256"), "root pin schema")
        store = storage.ArtifactStore.reopen(root["path"], expected_profile=FIXTURE,
            expected_header_sha256=root["header_sha256"], expected_root_identity=(root["device"], root["inode"]))
        try:
            self = cls.__new__(cls)
            self.store, self.root, self.guard = store, copy.deepcopy(root), None
            codec.validate_identity(context["identity"], profile=FIXTURE)
            self.identity = copy.deepcopy(context["identity"])
            _need(context["profile"] == FIXTURE, "reopen context profile differs")
            self.context = copy.deepcopy(context)
            self._check_context()
            events = self._verify_chain(pin["event"])
            self.last, self.last_ref = events[-1], copy.deepcopy(pin["event"])
            self.bindings = copy.deepcopy(self.last["bindings"])
            _need(self.last["event"] in BOUNDARIES, "interrupted phase cannot be resumed or restarted")
            loaded = verified.load_verified_plan_from_store(store, self.bindings["plan"]["name"],
                identity=self.identity, profile=FIXTURE, expected_sha256=context["plan_artifact"]["sha256"])
            expected = dict(identity=codec.tree_digest(self.identity),
                plan=copy.deepcopy(loaded["artifact"]),
                sources_sha256=codec.tree_digest(context["sources"]),
                environment_sha256=codec.tree_digest(context["environment"]))
            _need(envelope.same_exact(self.bindings, expected)
                  and envelope.same_exact(loaded["plan"], context["plan"]), "reopen context changed")
            return self
        except BaseException:
            store.close()
            raise

    def _verify_chain(self, pin):
        final = _snapshot(self.store)
        final_map = {row["name"]: row for row in final}
        _need(envelope.same_exact(_root(self.store), self.root), "reopen root changed")
        last_bytes = _payload(self.store, pin)
        last = codec.json_loads(last_bytes, max_bytes=EVENT_MAX)
        _need(type(last) is dict and type(last.get("sequence")) is int
              and 0 <= last["sequence"] < len(EVENTS), "last event sequence invalid")
        _need(pin["name"] == f"phase-{last['sequence']:03d}.json", "last event filename invalid")
        events, previous, previous_inventory, previous_ref = [], None, None, None
        for i in range(last["sequence"] + 1):
            name = f"phase-{i:03d}.json"
            raw, _ = storage._read_regular(self.store._dirfd, name, maximum=EVENT_MAX)
            value = codec.json_loads(raw, max_bytes=EVENT_MAX)
            _need(type(value) is dict and tuple(value) == KEYS, "event ordered schema")
            _need(type(value["sequence"]) is int and value["sequence"] == i
                  and value["event"] == EVENTS[i] and value["profile"] == FIXTURE
                  and value["schema"] == "i7_fixture_phase_event_v1"
                  and value["scientific_execution_certified"] is False, "event values")
            codec.validate_created_utc(value["created_utc"])
            _need(envelope.same_exact(value["root"], self.root)
                  and envelope.same_exact(value["previous"], previous_ref), "event chain/root mismatch")
            inventory = value["inventory_before"]
            _need(type(inventory) is list and all(type(row) is dict and tuple(row) ==
                  ("name", "size_bytes", "sha256", "allocated_bytes") for row in inventory), "inventory schema")
            names = [row["name"] for row in inventory]
            _need(all(type(name) is str for name in names) and names == sorted(set(names))
                  and all(envelope.same_exact(final_map.get(row["name"]), row) for row in inventory),
                  "historical inventory differs from retained files")
            outputs = value["outputs"]
            _need(type(outputs) is list and all(type(row) is dict for row in outputs)
                  and [row.get("name") for row in outputs] == _expected_outputs(EVENTS[i], self.identity),
                  "historical output membership")
            if i == 0:
                _need(type(value["bindings"]) is dict and tuple(value["bindings"]) ==
                      ("identity", "plan", "sources_sha256", "environment_sha256"), "bindings schema")
                plan_name = value["bindings"]["plan"]["name"]
                _need(set(names) == {"store.lock", "store-header.json", plan_name, _receipt_name(plan_name)},
                      "ready inventory has non-plan artifacts")
                _need(value["runtime"] is None, "pre-GO runtime must be null")
            else:
                _need(envelope.same_exact(value["bindings"], previous["bindings"]), "bindings changed in chain")
                extra = {previous_ref["name"], _receipt_name(previous_ref["name"])}
                extra.update(name for ref in outputs for name in (ref["name"], _receipt_name(ref["name"])))
                _need(set(names) == set(previous_inventory) | extra, "event inventory delta mismatch")
                _need(not (set(previous_inventory) & extra), "event rewrites old output")
                phase = "primary" if i <= 5 else "audit"
                _runtime_check(value["runtime"], phase)
                if i not in (1, 6):
                    prior = previous["runtime"]
                    _need(value["runtime"]["check_count"] > prior["check_count"]
                          and value["runtime"]["last_record"]["wall_seconds"] >= prior["last_record"]["wall_seconds"],
                          "runtime reset inside phase")
            for ref in outputs:
                _validate_output(self.store, ref, self.identity)
            previous, previous_inventory, previous_ref = value, names, _ref(name, raw)
            events.append(value)
        _need(envelope.same_exact(previous_ref, pin), "last event pin mismatch")
        _need(set(final_map) == set(previous_inventory) | {pin["name"], _receipt_name(pin["name"])},
              "stale boundary pin or extra files")
        return events

    def summary(self):
        rows = _snapshot(self.store)
        return dict(event=self.last["event"], root=str(self.store.root), file_count=len(rows),
                    logical_bytes=sum(row["size_bytes"] for row in rows),
                    allocated_bytes=sum(row["allocated_bytes"] for row in rows),
                    scientific_execution_certified=False)
