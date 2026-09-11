"""Launch the approved attempt once, retaining external terminal observations.

This is an observer, not measurement evidence or permission for training.
An acknowledged D-Bus subscription precedes the existing launch API call.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gi.repository import Gio, GLib

ROOT = Path("/private-artifacts/repositories/optimizers-i7-native-frozen-002")
REL = "output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007"
OUTPUT = Path("/private-artifacts/repositories/optimizers-launch-investigation") / REL
UNIT = "i7-native-storage-measurement-002.service"
UNIT_PATH = "/org/freedesktop/systemd1/unit/" + UNIT.replace("-", "_2d").replace(".", "_2e")
DEST = "org.freedesktop.systemd1"
KEYS = (
    "LoadState", "ActiveState", "SubState", "Result", "MainPID", "InvocationID",
    "ExecMainCode", "ExecMainStatus", "ExecMainStartTimestampMonotonic",
    "ExecMainExitTimestampMonotonic", "MemoryPeak", "ControlGroup",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    destination = OUTPUT / "native-storage-measurement-attempt-002-observation.json"
    if os.path.lexists(destination):
        raise RuntimeError("Attempt002 observation already exists; inspect, do not relaunch")
    sys.path.insert(0, str(ROOT / REL))
    import native_storage_measurement_controller as controller
    import native_storage_measurement_service as service
    assert service.UNIT == UNIT
    connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    context = GLib.MainContext.default()
    origin = time.monotonic()
    record = {
        "schema": "i7_external_attempt002_observation_v1",
        "evidence_role": "external_observer_not_measurement_candidate",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen_root": str(ROOT), "expected_commit": args.commit,
        "expected_source_set_sha256": args.source_sha256, "unit": UNIT,
        "launch_calls": 0, "signals": [], "polls": [],
    }

    def emit(kind, data):
        row = {"elapsed_s": time.monotonic() - origin, "properties": data}
        record[kind].append(row)
        print(json.dumps({kind: row}), flush=True)

    def changed(connection, sender, path, interface, member, parameters, user_data):
        _, properties, invalidated = parameters.unpack()
        selected = {k: v for k, v in properties.items() if k in KEYS}
        if selected:
            emit("signals", selected)

    def removed(connection, sender, path, interface, member, parameters, user_data):
        name, object_path = parameters.unpack()
        if name == UNIT:
            emit("signals", {"UnitRemoved": name})

    subscriptions = [
        connection.signal_subscribe(DEST, "org.freedesktop.DBus.Properties",
            "PropertiesChanged", UNIT_PATH, None, Gio.DBusSignalFlags.NONE, changed, None),
        connection.signal_subscribe(DEST, DEST + ".Manager", "UnitRemoved",
            "/org/freedesktop/systemd1", UNIT, Gio.DBusSignalFlags.NONE, removed, None),
    ]
    connection.call_sync(DEST, "/org/freedesktop/systemd1", DEST + ".Manager",
        "Subscribe", None, None, Gio.DBusCallFlags.NONE, 5000, None)
    record["subscription_acknowledged_before_launch"] = True
    print("D-Bus subscription acknowledged; invoking approved launcher once", flush=True)
    try:
        record["launch_calls"] += 1
        record["launcher_return"] = controller.launch_one_shot(
            expected_commit=args.commit, expected_source_set_sha256=args.source_sha256)
        previous = None
        terminal_at = None
        while time.monotonic() - origin < 135:
            while context.pending():
                context.iteration(False)
            result = subprocess.run(["/usr/bin/systemctl", "--user", "show", UNIT,
                *["--property=" + key for key in KEYS]], capture_output=True,
                text=True, timeout=3, env=service.process_environment())
            properties = dict(line.split("=", 1) for line in result.stdout.splitlines()
                              if "=" in line)
            if properties != previous:
                emit("polls", properties)
                previous = properties
            terminal = (properties.get("ActiveState") in ("failed", "inactive")
                        and properties.get("MainPID") == "0")
            if terminal or properties.get("LoadState") == "not-found":
                terminal_at = terminal_at or time.monotonic()
                if time.monotonic() - terminal_at >= .5:
                    record["terminal_observed"] = True
                    break
            time.sleep(.05)
        else:
            record["terminal_observed"] = False
    finally:
        while context.pending():
            context.iteration(False)
        for subscription in subscriptions:
            connection.signal_unsubscribe(subscription)
        record["elapsed_s"] = time.monotonic() - origin
        measurement = OUTPUT / "native-max-layout-measurement-attempt-002.json"
        if measurement.exists():
            payload = measurement.read_bytes()
            info = measurement.stat()
            record["measurement_file"] = {
                "path": str(measurement), "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "inode": info.st_ino, "device": info.st_dev,
            }
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print("Observer record: " + str(destination), flush=True)


if __name__ == "__main__":
    main()
