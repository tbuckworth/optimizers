"""Analytic canonical-JSON ceiling for the one fixed pilot comparison body.

Import/default CLI are inert and Torch-free. The explicit calculation lazily
imports the actual schema constants (which import CPU Torch); it neither builds
a comparison artifact nor generates a plan/state/trace, and does not serialize
an experimental specimen. Boolean maxima need not be simultaneously attainable.
"""
from __future__ import annotations

import json


def _constant_bytes(value) -> int:
    # Exact encoder options in identity_codec.json_bytes, without its final LF.
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii"))


def _object_bytes(field_sizes: dict[str, int]) -> int:
    return 2 + max(0, len(field_sizes) - 1) + sum(
        _constant_bytes(key) + 1 + size for key, size in field_sizes.items())


def pilot_comparison_json_ceiling() -> dict:
    """Bound every schema-valid value, including unequal post-update traces.

    Counters and trace length are fixed by the actual scientific pilot schema.
    SHA256 strings are exactly64 ASCII characters (66 JSON bytes); each bool
    costs at most5 bytes. Each summary counter is in0..220, so at most3 bytes.
    All other fields are schema-fixed constants. JSON memo/alias identity has no
    effect, and the final newline is charged exactly once.
    """
    import identity_codec as codec
    import native_phase_policy as policy
    import source_history as history

    identity = policy.identity("pilot", 71990, 101)
    track = history.trajectory(identity, codec.SCIENTIFIC_PROFILE)
    if track["steps_total"] != 220:
        raise ValueError("fixed pilot length changed")
    if history.FINGERPRINT_KEYS != (
        "completed_updates", "completed_observations", "next_anchor_update",
        "core_sha256", "model_sha256", "optimizer_sha256", "moments_sha256",
        "observer_sha256", "rng_sha256",
    ):
        raise ValueError("fingerprint fields changed")
    if (len(history.INITIAL_FINGERPRINT_KEYS) != 5 or
            history.INITIAL_FINGERPRINT_KEYS != ("state_sha256", "model_sha256",
                "optimizer_sha256", "observer_sha256", "rng_sha256")):
        raise ValueError("initial fingerprint fields changed")
    equal_fields = ("model_direct_typed_equal", "optimizer_direct_typed_equal",
                    "moments_direct_typed_equal", "observer_direct_typed_equal",
                    "rng_direct_typed_equal", "core_direct_typed_equal")
    initial_fields = ("model_direct_typed_equal", "optimizer_direct_typed_equal",
                      "observer_direct_typed_equal", "rng_direct_typed_equal",
                      "state_direct_typed_equal")
    if (history.COMPARISON_KEYS != ("completed_updates", "on", "off", *equal_fields)
            or history.INITIAL_COMPARISON_KEYS != ("on", "off", *initial_fields)):
        raise ValueError("comparison fields changed")

    rows = []  # Integer byte counts only, not trace values.
    for update in range(1, 221):
        fingerprint = _object_bytes(dict(zip(history.FINGERPRINT_KEYS,
            (len(str(update)), len(str(update)), len(str(update + 1)), *([66] * 6)))))
        rows.append(_object_bytes({"completed_updates": len(str(update)),
            "on": fingerprint, "off": fingerprint, **{key: 5 for key in equal_fields}}))
    trace_bytes = 2 + 219 + sum(rows)
    initial_fingerprint = _object_bytes({key: 66 for key in history.INITIAL_FINGERPRINT_KEYS})
    initial_bytes = _object_bytes({"on": initial_fingerprint, "off": initial_fingerprint,
                                   **{key: 5 for key in initial_fields}})
    if history.SUMMARY_KEYS != ("steps_compared", "model_equal_steps", "optimizer_equal_steps",
                                "moments_equal_steps", "observer_equal_steps", "rng_equal_steps",
                                "core_equal_steps"):
        raise ValueError("summary fields changed")
    stem, name = history._comparison_names(codec.SCIENTIFIC_PROFILE, track)
    fields = {
        "schema_name": _constant_bytes("i7_capture_comparison"), "schema_version": 1,
        "profile": _constant_bytes(codec.SCIENTIFIC_PROFILE),
        "artifact_id": _constant_bytes(stem), "artifact_name": _constant_bytes(name),
        "trajectory": _constant_bytes(track),
        "instrumentation": _constant_bytes(history.INSTRUMENTATION),
        "initial_comparison": initial_bytes, "step_trace": trace_bytes,
        "summary": _object_bytes({key: 3 for key in history.SUMMARY_KEYS}),
        "evidence_scope": _constant_bytes(history.COMPARISON_EVIDENCE_SCOPE),
        "scientific_execution_certified": 5,
    }
    if tuple(fields) != history.CAPTURE_ROOT_KEYS:
        raise ValueError("comparison root fields changed")
    return {"schema": "i7_pilot_comparison_json_bound_v1", "payload_count": 1,
            "trace_row_count": 220, "trace_bytes_upper": trace_bytes,
            "initial_comparison_bytes_upper": initial_bytes,
            "json_bytes_upper": _object_bytes(fields) + 1, "trailing_newline_bytes": 1,
            "basis": "fixed_schema_counters_hash_widths_and_independent_boolean_maxima",
            "native_measurement": False, "whole_study_fit_proven": False,
            "execution_authorized": False, "scientific_execution_certified": False}


if __name__ == "__main__":
    print("I7 comparison JSON bound only; no artifact, data, plan or native execution.")
