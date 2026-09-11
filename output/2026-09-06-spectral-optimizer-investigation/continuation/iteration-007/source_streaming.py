#!/usr/bin/env python3
"""Bounded one-pass assembly for iteration-007 source-history records.

This module owns no training loop, artifact writer, data loader, plan generator,
or execution entrypoint.  A caller records each actual post-update state core as
soon as it exists.  Historical cores are immediately reduced to primitive
fingerprints; only the final core is cloned and retained.  Anchor and witness
references must be supplied on the exact update at which their files were
sealed.

``SourceCompletionStream`` assembles one capture-on or capture-off completion.
``PilotPairStream`` compares capture-on/off cores directly in one pass and
assembles both completions plus their comparison artifact without ``tee``.
Neither class grants scientific execution authority or writes artifacts.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import artifact_store as storage
import source_history as history


class SourceStreamingError(RuntimeError):
    """The one-pass stream violated its fixed source-history contract."""


PAIR_RESULT_KEYS = ("capture_on", "capture_off", "comparison")
WRITER_RECEIPT_KEYS = (
    "schema", "name", "size", "sha256", "status", "encoding",
    "receipt_name",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceStreamingError(message)


def _prepare(identity: Any, profile: str, plan_ref: Any,
             sources_sha256: str, environment_sha256: str
             ) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    try:
        track = history.trajectory(identity, profile)
        history._validate_plan_ref(plan_ref, profile, track)
        sources = history._sha(sources_sha256, "sources_sha256")
        environment = history._sha(environment_sha256, "environment_sha256")
        return track, history._clone(plan_ref), sources, environment
    except (history.SourceHistoryError, KeyError, TypeError, ValueError) as exc:
        raise SourceStreamingError("invalid fixed streaming context") from exc


def artifact_reference(artifact: Any, receipt: Any, *, identity: Any,
                       profile: str, kind: str) -> dict[str, Any]:
    """Make one declared history reference from an actual writer receipt.

    The caller must supply the artifact value returned to the writer and the
    exact receipt returned by that successful create-only write.  This checks
    their identity and canonical receipt fields, but deliberately does not
    reread external bytes; source-history evidence records that limitation.
    """
    try:
        _require(type(artifact) is dict and type(receipt) is dict and
                 tuple(receipt) == WRITER_RECEIPT_KEYS,
                 "actual artifact and exact writer receipt required")
        expected_id = history.codec.artifact_id(
            identity, profile=profile, kind=kind)
        expected_schema = ("i7_anchor" if kind == "anchor"
                           else "i7_source_step_witness")
        _require(kind in ("anchor", "source-witness") and
                 artifact.get("artifact_id") == expected_id and
                 artifact.get("schema_name") == expected_schema and
                 artifact.get("profile") == profile,
                 "artifact identity differs from reference request")
        name = expected_id + ".pt"
        _require(receipt["schema"] == "i7_artifact_receipt_v1" and
                 receipt["name"] == name and type(receipt["size"]) is int and
                 receipt["size"] > 0 and receipt["status"] == "complete" and
                 receipt["encoding"] == "torch_weights_only",
                 "writer receipt is not a complete tensor-tree receipt")
        base = {key: receipt[key] for key in WRITER_RECEIPT_KEYS[:-1]}
        encoded = storage._json_bytes(base)
        value = {
            "artifact_id": expected_id,
            "schema_name": expected_schema,
            "name": name,
            "size_bytes": receipt["size"],
            "sha256": receipt["sha256"],
            "status": receipt["status"],
            "encoding": receipt["encoding"],
            "receipt_name": receipt["receipt_name"],
            "receipt_size_bytes": len(encoded),
            "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        history._validate_artifact_ref(
            value, identity=identity, profile=profile, kind=kind)
        return history._clone(value)
    except (history.SourceHistoryError, SourceStreamingError, KeyError,
            TypeError, ValueError) as exc:
        if isinstance(exc, SourceStreamingError):
            raise
        raise SourceStreamingError("invalid actual artifact receipt") from exc


def _anchor_pair(*, update: int, track: dict[str, Any], profile: str,
                 capture_mode: str, anchor_ref: Any, witness_ref: Any
                 ) -> dict[str, Any] | None:
    expected = capture_mode == "capture_on" and update in track["anchor_updates"]
    if not expected:
        _require(anchor_ref is None and witness_ref is None,
                 "references supplied outside a capture-on anchor update")
        return None
    _require(anchor_ref is not None and witness_ref is not None,
             "capture-on anchor update requires its persisted reference pair")
    identity = history._identity_for(track, update)
    try:
        history._validate_artifact_ref(
            anchor_ref, identity=identity, profile=profile, kind="anchor")
        history._validate_artifact_ref(
            witness_ref, identity=identity, profile=profile,
            kind="source-witness")
        return {
            "anchor_update": update,
            "anchor_ref": history._clone(anchor_ref),
            "witness_ref": history._clone(witness_ref),
        }
    except (history.SourceHistoryError, KeyError, TypeError, ValueError) as exc:
        raise SourceStreamingError("invalid persisted anchor/witness references") from exc


def _source_value(*, profile: str, track: dict[str, Any], capture_mode: str,
                  plan_ref: dict[str, Any], sources_sha256: str,
                  environment_sha256: str, refs: list[dict[str, Any]],
                  trace: list[dict[str, Any]], final_core: dict[str, Any]
                  ) -> dict[str, Any]:
    stem, name = history._source_names(profile, track, capture_mode)
    value = {
        "schema_name": "i7_source_completion",
        "schema_version": 1,
        "profile": profile,
        "artifact_id": stem,
        "artifact_name": name,
        "trajectory": track,
        "capture_mode": capture_mode,
        "instrumentation": dict(history.INSTRUMENTATION),
        "plan_ref": plan_ref,
        "provenance": {
            "sources_sha256": sources_sha256,
            "environment_sha256": environment_sha256,
        },
        "anchor_witness_refs": refs,
        "trace": trace,
        "final_state_core": final_core,
        "evidence_scope": dict(history.SOURCE_EVIDENCE_SCOPE),
        "scientific_execution_certified": False,
    }
    try:
        return history.validate_source_completion(value)
    except history.SourceHistoryError as exc:
        raise SourceStreamingError("assembled source completion is invalid") from exc


def _comparison_value(*, profile: str, track: dict[str, Any],
                      initial: dict[str, Any], rows: list[dict[str, Any]]
                      ) -> dict[str, Any]:
    stem, name = history._comparison_names(profile, track)
    value = {
        "schema_name": "i7_capture_comparison",
        "schema_version": 1,
        "profile": profile,
        "artifact_id": stem,
        "artifact_name": name,
        "trajectory": track,
        "instrumentation": dict(history.INSTRUMENTATION),
        "initial_comparison": initial,
        "step_trace": rows,
        "summary": history._summary(rows),
        "evidence_scope": dict(history.COMPARISON_EVIDENCE_SCOPE),
        "scientific_execution_certified": False,
    }
    try:
        return history.validate_capture_comparison(value)
    except history.SourceHistoryError as exc:
        raise SourceStreamingError("assembled capture comparison is invalid") from exc


class SourceCompletionStream:
    """Incrementally assemble one exact source completion.

    ``record`` must be called once for every update, in order.  For capture-on,
    pass ``anchor_ref`` and ``witness_ref`` only at registered anchor updates.
    Any invalid call makes the accumulator terminal; a premature ``finish`` is
    likewise terminal.  Step rows returned by ``record`` are independent
    primitive copies and may be inspected or discarded by the caller.
    """

    __slots__ = (
        "_profile", "_track", "_capture_mode", "_plan_ref", "_sources",
        "_environment", "_refs", "_trace", "_final_core", "_status",
    )

    def __init__(self, *, identity: Any, profile: str, capture_mode: str,
                 plan_ref: Any, sources_sha256: str,
                 environment_sha256: str):
        _require(type(capture_mode) is str and capture_mode in history.CAPTURE_MODES,
                 "unknown capture mode")
        track, plan, sources, environment = _prepare(
            identity, profile, plan_ref, sources_sha256, environment_sha256)
        _require(capture_mode == "capture_on" or
                 track["execution_role"] == "pilot" or
                 profile == history.MLP_FIXTURE,
                 "capture-off is reserved for pilot comparison")
        self._profile = profile
        self._track = track
        self._capture_mode = capture_mode
        self._plan_ref = plan
        self._sources = sources
        self._environment = environment
        self._refs: list[dict[str, Any]] | None = []
        self._trace: list[dict[str, Any]] | None = []
        self._final_core: dict[str, Any] | None = None
        self._status = "open"

    @property
    def completed_updates(self) -> int:
        return len(self._trace) if self._trace is not None else self._track["steps_total"]

    @property
    def retained_full_core_count(self) -> int:
        return int(self._final_core is not None)

    def _open(self) -> None:
        _require(self._status == "open", "stream is terminal")

    def record(self, core: Any, *, anchor_ref: Any = None,
               witness_ref: Any = None) -> dict[str, Any]:
        self._open()
        try:
            assert self._trace is not None and self._refs is not None
            update = len(self._trace) + 1
            _require(update <= self._track["steps_total"],
                     "source stream exceeds steps_total")
            _require(type(core) is dict and core.get("profile") == self._profile,
                     "source core profile differs")
            pair = _anchor_pair(
                update=update, track=self._track, profile=self._profile,
                capture_mode=self._capture_mode, anchor_ref=anchor_ref,
                witness_ref=witness_ref)
            row = history.step_fingerprint(core)
            _require(row["completed_updates"] == update,
                     "source core is missing, duplicated, or reordered")
            final = history._clone(core) if update == self._track["steps_total"] else None
            self._trace.append(row)
            if pair is not None:
                self._refs.append(pair)
            if final is not None:
                self._final_core = final
            return copy.deepcopy(row)
        except BaseException as exc:
            self._status = "failed"
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SourceStreamingError):
                raise
            raise SourceStreamingError("invalid source stream update") from exc

    def finish(self) -> dict[str, Any]:
        self._open()
        try:
            assert self._trace is not None and self._refs is not None
            _require(len(self._trace) == self._track["steps_total"],
                     "source stream ended before steps_total")
            _require(self._final_core is not None,
                     "source stream has no retained final core")
            value = _source_value(
                profile=self._profile, track=self._track,
                capture_mode=self._capture_mode, plan_ref=self._plan_ref,
                sources_sha256=self._sources,
                environment_sha256=self._environment, refs=self._refs,
                trace=self._trace, final_core=self._final_core)
        except BaseException as exc:
            self._status = "failed"
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SourceStreamingError):
                raise
            raise SourceStreamingError("source stream finalization failed") from exc
        self._status = "finished"
        self._refs = None
        self._trace = None
        self._final_core = None
        return value


class PilotPairStream:
    """Incrementally assemble the paired pilot completions and comparison.

    Initial equality is checked directly at construction.  Each ``record`` call
    directly compares the supplied typed on/off cores before retaining their
    hashes.  Unequal valid steps are evidence and are retained, not treated as
    stream failures.  Only the capture-on side accepts anchor/witness refs.
    """

    __slots__ = (
        "_profile", "_track", "_plan_ref", "_sources", "_environment",
        "_initial", "_refs", "_on_trace", "_off_trace", "_rows",
        "_on_final", "_off_final", "_status",
    )

    def __init__(self, *, on_initial: Any, off_initial: Any, identity: Any,
                 profile: str, plan_ref: Any, sources_sha256: str,
                 environment_sha256: str):
        track, plan, sources, environment = _prepare(
            identity, profile, plan_ref, sources_sha256, environment_sha256)
        _require(track["execution_role"] == "pilot" or profile == history.MLP_FIXTURE,
                 "capture comparison is pilot-only")
        try:
            _require(type(on_initial) is dict and on_initial.get("profile") == profile,
                     "initial comparison profile differs")
            initial = history.compare_initial(on_initial, off_initial)
            _require(initial["state_direct_typed_equal"] is True,
                     "capture arms must share the exact initial state")
        except (history.SourceHistoryError, SourceStreamingError, KeyError,
                TypeError, ValueError) as exc:
            if isinstance(exc, SourceStreamingError):
                raise
            raise SourceStreamingError("invalid pilot initial states") from exc
        self._profile = profile
        self._track = track
        self._plan_ref = plan
        self._sources = sources
        self._environment = environment
        self._initial = copy.deepcopy(initial)
        self._refs: list[dict[str, Any]] | None = []
        self._on_trace: list[dict[str, Any]] | None = []
        self._off_trace: list[dict[str, Any]] | None = []
        self._rows: list[dict[str, Any]] | None = []
        self._on_final: dict[str, Any] | None = None
        self._off_final: dict[str, Any] | None = None
        self._status = "open"

    @property
    def completed_updates(self) -> int:
        return len(self._rows) if self._rows is not None else self._track["steps_total"]

    @property
    def retained_full_core_count(self) -> int:
        return int(self._on_final is not None) + int(self._off_final is not None)

    def _open(self) -> None:
        _require(self._status == "open", "stream is terminal")

    def record(self, on_core: Any, off_core: Any, *, anchor_ref: Any = None,
               witness_ref: Any = None) -> dict[str, Any]:
        self._open()
        try:
            assert (self._rows is not None and self._refs is not None and
                    self._on_trace is not None and self._off_trace is not None)
            update = len(self._rows) + 1
            _require(update <= self._track["steps_total"],
                     "pilot pair stream exceeds steps_total")
            _require(type(on_core) is dict and on_core.get("profile") == self._profile,
                     "capture-on core profile differs")
            pair = _anchor_pair(
                update=update, track=self._track, profile=self._profile,
                capture_mode="capture_on", anchor_ref=anchor_ref,
                witness_ref=witness_ref)
            row = history.compare_step(on_core, off_core)
            _require(row["completed_updates"] == update,
                     "pilot cores are missing, duplicated, or reordered")
            on_final = (history._clone(on_core)
                        if update == self._track["steps_total"] else None)
            off_final = (history._clone(off_core)
                         if update == self._track["steps_total"] else None)
            self._rows.append(row)
            self._on_trace.append(copy.deepcopy(row["on"]))
            self._off_trace.append(copy.deepcopy(row["off"]))
            if pair is not None:
                self._refs.append(pair)
            if on_final is not None:
                self._on_final = on_final
                self._off_final = off_final
            return copy.deepcopy(row)
        except BaseException as exc:
            self._status = "failed"
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SourceStreamingError):
                raise
            raise SourceStreamingError("invalid pilot pair stream update") from exc

    def finish(self) -> dict[str, Any]:
        self._open()
        try:
            assert (self._rows is not None and self._refs is not None and
                    self._on_trace is not None and self._off_trace is not None)
            _require(len(self._rows) == self._track["steps_total"],
                     "pilot pair stream ended before steps_total")
            _require(self._on_final is not None and self._off_final is not None,
                     "pilot pair stream has no retained final cores")
            on = _source_value(
                profile=self._profile, track=self._track,
                capture_mode="capture_on", plan_ref=copy.deepcopy(self._plan_ref),
                sources_sha256=self._sources,
                environment_sha256=self._environment, refs=self._refs,
                trace=self._on_trace, final_core=self._on_final)
            off = _source_value(
                profile=self._profile, track=copy.deepcopy(self._track),
                capture_mode="capture_off", plan_ref=copy.deepcopy(self._plan_ref),
                sources_sha256=self._sources,
                environment_sha256=self._environment, refs=[],
                trace=self._off_trace, final_core=self._off_final)
            comparison = _comparison_value(
                profile=self._profile, track=copy.deepcopy(self._track),
                initial=self._initial, rows=self._rows)
            history.validate_capture_bundle(on, off, comparison)
            value = {
                "capture_on": on,
                "capture_off": off,
                "comparison": comparison,
            }
        except BaseException as exc:
            self._status = "failed"
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SourceStreamingError):
                raise
            raise SourceStreamingError("pilot pair stream finalization failed") from exc
        self._status = "finished"
        self._refs = None
        self._on_trace = None
        self._off_trace = None
        self._rows = None
        self._on_final = None
        self._off_final = None
        return value
