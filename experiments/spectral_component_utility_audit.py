"""Independent, NumPy-only saved-array checks; import/default CLI are inert.

No producer arithmetic imports, model replay, autograd, checkpoint unpickling,
or observer-stream reconstruction. Scientific execution requires --execute.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import time
import zipfile

ENTRY_MONOTONIC = time.monotonic()
import numpy as np


SCHEMA = "spectral_component_utility_v1"
SEEDS = (202609171, 202609172, 202609173)
STEPS = (100, 56304)
AUGMENTATIONS = ("none", "translate")
FRACTIONS = ((1.0, "full"), (.1, "tenth"))
OBJECTIVES = ("S", "F", "C", "L", "H_O", "H_T")
P = 235146
PARAMETER_SIZES = (200704, 256, 32768, 128, 1280, 10)
PANEL_KEYS = ("action_positions", "action_ids", "action_true", "action_assigned",
              "action_shifts", "evaluation_positions", "evaluation_ids", "evaluation_true",
              "evaluation_assigned", "evaluation_wrong", "reporting_ids", "reporting_true",
              "baseline_ids", "baseline_true", "view_shifts")
ACTION_VECTORS = ("theta_before", "theta_after", "decay_endpoint", "gradient_raw",
                  "gradient_delivered", "m_before", "v_before", "m_after", "v_after")
ACTION_KEYS = (*ACTION_VECTORS, "adam_steps_before", "adam_steps_after", "post_basis",
               "post_singular_values")
MAX_ARTIFACT = 256 * 1024**2
MAX_JSON = 64 * 1024**2
ARCHIVE_CAP = 8 * 1024**3
SCALAR_ATOL = SCALAR_RTOL = 1e-10
ARRAY_ATOL, ARRAY_RTOL = 1e-7, 5e-5
GRAD_ATOL, GRAD_RTOL = 1e-6, 5e-5
COOPERATIVE_SECONDS = 900
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    pass


class Checks:
    def __init__(self, deadline=None):
        self.count = 0
        self.deadline = deadline
        self.array_residuals = []

    def need(self, condition, message):
        self.count += 1
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise AuditError("cooperative audit deadline exceeded")
        if self.deadline is not None and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 2 * 1024**2:
            raise AuditError("audit 2 GiB high-water RSS bound exceeded")
        if not condition:
            raise AuditError(message)

    def exact(self, actual, expected, label):
        self.need(type(actual) is np.ndarray and actual.dtype == expected.dtype
                  and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
                  label + ": exact array mismatch")

    def near(self, actual, expected, label, atol=SCALAR_ATOL, rtol=SCALAR_RTOL):
        self.need(type(actual) in (int, float) and type(actual) is not bool
                  and math.isfinite(actual) and math.isfinite(float(expected))
                  and abs(actual - expected) <= atol + rtol * abs(expected), label + ": scalar mismatch")

    def norm(self, actual, expected, label, atol=ARRAY_ATOL, rtol=ARRAY_RTOL):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                  label + ": numerical shape/finite mismatch")
        residual = np.abs(a - b)
        self.array_residuals.append(dict(label=label, l2=float(np.linalg.norm(residual)),
                                         reference_l2=float(np.linalg.norm(b)),
                                         max_abs=float(residual.max(initial=0))))
        self.need(bool(np.all(residual <= atol + rtol * np.abs(b))),
                  label + ": elementwise residual exceeds frozen bound; max_abs="
                  + repr(float(residual.max(initial=0))) + "; l2=" + repr(float(np.linalg.norm(residual))))

    def tree(self, actual, expected, label="tree"):
        if isinstance(expected, dict):
            self.need(type(actual) is dict and set(actual) == set(expected), label + ": keys")
            for key in expected:
                self.tree(actual[key], expected[key], label + "/" + str(key))
        elif isinstance(expected, list):
            self.need(type(actual) is list and len(actual) == len(expected), label + ": list length")
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.tree(a, b, label + f"/{i}")
        elif type(expected) is float:
            self.near(actual, expected, label)
        else:
            self.need(type(actual) is type(expected) and actual == expected, label + ": literal")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise AuditError("duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(AuditError("nonfinite JSON " + x)))


def _sha(value):
    return type(value) is str and SHA_RE.fullmatch(value) is not None


def direct_path(root, name):
    root = Path(root)
    if type(name) is not str or name in ("", ".", "..") or Path(name).name != name:
        raise AuditError("receipt is not a direct child")
    target = root / name
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise AuditError("root must be canonical regular directory")
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or target.resolve() != target:
        raise AuditError("receipt target is not a direct regular file")
    return target


def file_receipt(path, max_bytes=MAX_ARTIFACT):
    path = Path(path)
    direct_path(path.parent, path.name)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
            raise AuditError("file is nonregular or exceeds admitted bound")
        digest = hashlib.sha256()
        while chunk := handle.read(1024**2):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise AuditError("file changed during hashing")
    return {"path": path.name, "size_bytes": before.st_size, "sha256": digest.hexdigest()}


def pinned_payload(path, receipt, cap):
    """One bounded, nonblocking regular descriptor; parse only its pinned bytes."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != receipt["size_bytes"] or not 0 < before.st_size <= cap:
            raise AuditError("pinned payload regular-file/size admission")
        payload = handle.read(cap + 1)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise AuditError("pinned payload changed during read")
    if len(payload) != receipt["size_bytes"] or hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise AuditError("pinned payload byte hash differs")
    return payload


class Reader:
    def __init__(self, root, receipts, checks):
        self.root = Path(root)
        self.checks = checks
        checks.need(type(receipts) is list, "receipt list required")
        self.receipts = {}
        self.used = set()
        for entry in receipts:
            checks.need(type(entry) is dict and set(entry) == {"path", "size_bytes", "sha256"},
                        "receipt schema")
            checks.need(type(entry["size_bytes"]) is int and 0 < entry["size_bytes"] <= MAX_ARTIFACT
                        and _sha(entry["sha256"]), "receipt size/hash")
            checks.need(entry["path"] not in self.receipts, "duplicate receipt")
            direct_path(self.root, entry["path"])
            self.receipts[entry["path"]] = entry

    def path(self, receipt):
        self.checks.need(type(receipt) is dict and receipt == self.receipts.get(receipt.get("path")),
                         "record does not bind inventory receipt")
        path = direct_path(self.root, receipt["path"])
        self.checks.tree(file_receipt(path), receipt, "artifact receipt")
        self.used.add(receipt["path"])
        return path

    def json(self, receipt):
        path = self.path(receipt)
        self.checks.need(path.stat().st_size <= MAX_JSON, "JSON cap")
        result = strict_json(pinned_payload(path, receipt, MAX_JSON))
        self.checks.tree(file_receipt(path), receipt, "JSON unchanged")
        return result

    def npz(self, receipt, keys=None):
        path = self.path(receipt)
        payload = pinned_payload(path, receipt, MAX_ARTIFACT)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            names = [x.filename for x in members]
            self.checks.need(len(names) == len(set(names)) and len(names) <= 64
                             and all(Path(x).name == x and x.endswith(".npy") for x in names),
                             "NPZ duplicate/unsafe members")
            self.checks.need(sum(x.file_size for x in members) <= MAX_ARTIFACT, "NPZ expanded cap")
            for member in members:
                self.checks.need(not member.flag_bits & 1, "encrypted NPZ")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    self.checks.need(version in ((1, 0), (2, 0)), "unsupported NPY version")
                    header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)(stream, max_header_size=10000)
                    shape, fortran, dtype = header
                    self.checks.need(dtype.kind in "bifu" and dtype.fields is None and not dtype.hasobject
                                     and len(shape) <= 4 and all(type(n) is int and n >= 0 for n in shape),
                                     "non-numeric or malformed NPY")
                    self.checks.need(math.prod(shape) * dtype.itemsize + stream.tell() == member.file_size,
                                     "NPY header/payload size differs")
        with np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=10000) as archive:
            if keys is not None:
                self.checks.need(set(archive.files) == set(keys), "NPZ key set differs")
            arrays = {key: archive[key] for key in archive.files}
        self.checks.tree(file_receipt(path), receipt, "NPZ unchanged")
        return arrays


def array(checks, value, dtype, shape, label):
    checks.need(type(value) is np.ndarray and value.dtype == np.dtype(dtype)
                and value.shape == shape and np.isfinite(value).all(), label + ": array schema")


def _lse(z):
    maximum = np.max(z, axis=-1)
    return maximum + np.log(np.exp(z - maximum[..., None]).sum(axis=-1, dtype=np.float64))


def _ce(z, labels):
    index = labels[:, None, None] if z.ndim == 3 else labels[:, None]
    return _lse(z) - np.take_along_axis(z, index, axis=-1).squeeze(-1)


def _mean(values):
    return math.fsum(np.asarray(values).reshape(-1).tolist()) / np.size(values)


def _gauge(value):
    z = value.astype(np.float64)
    return z - z[..., :1]


def _stats(logits, labels):
    n, views, _ = logits.shape
    count = n * views
    if count == 0:
        return dict(available=False, example_count=0, view_count=views, count=0,
                    correct=0, accuracy=None, ce_sum=0.0, ce=None)
    total = math.fsum(_ce(_gauge(logits), labels).reshape(-1).tolist())
    correct = int(np.sum(logits.argmax(axis=-1) == labels[:, None]))
    return dict(available=True, example_count=n, view_count=views, count=count,
                correct=correct, accuracy=correct / count, ce_sum=total, ce=total / count)


def metrics(logits, panel, original_index=12):
    """Independent FP64 decomposition and explicitly named prediction metrics."""
    original, grid, reporting = logits["R_original"], logits["I"], logits["R"]
    true, assigned = panel["evaluation_true"], panel["evaluation_assigned"]
    z = _gauge(grid)
    mean = z.mean(axis=1, dtype=np.float64)
    normalizer = _lse(mean)
    rows = np.arange(len(true))
    true_logit, assigned_logit = mean[rows, true], mean[rows, assigned]
    uniform = mean.mean(axis=-1, dtype=np.float64)
    s_true, s_uniform = _mean(normalizer - true_logit), _mean(normalizer - uniform)
    objectives = dict(S=.1 * s_true + .9 * s_uniform,
                      F=_mean(.1 * true_logit + .9 * uniform - assigned_logit),
                      C=_mean(_lse(z).mean(axis=1, dtype=np.float64) - normalizer),
                      L=_mean(_ce(z, assigned)), S_true=s_true, S_uniform=s_uniform,
                      H_O=_mean(_ce(_gauge(original), panel["reporting_true"])),
                      H_T=_mean(_ce(_gauge(reporting), panel["reporting_true"])))
    wrong = assigned != true
    raw_original = grid[:, original_index:original_index + 1]
    blocks = {"original": (raw_original, true, assigned), "per_view": (grid, true, assigned),
              "wrong_original": (raw_original[wrong], true[wrong], assigned[wrong]),
              "wrong_per_view": (grid[wrong], true[wrong], assigned[wrong])}
    return dict(objectives=objectives,
                I={name: dict(true=_stats(values, correct), assigned=_stats(values, target))
                   for name, (values, correct, target) in blocks.items()},
                R=dict(original=dict(true=_stats(original[:, None], panel["reporting_true"])),
                       per_view=dict(true=_stats(reporting, panel["reporting_true"]))),
                metadata=dict(grid_layout="image,view,class", original_view_index=original_index,
                              I_examples=grid.shape[0], I_views=grid.shape[1],
                              I_wrong_examples=int(wrong.sum()), classes=grid.shape[2],
                              R_examples=reporting.shape[0], R_views=reporting.shape[1]))


def panels_from_plan(plan, seed, checks):
    """Reconstruct only fixed new roles/streams from accepted numeric plan."""
    for role, count in (("train", 50000), ("validation", 5000), ("reporting", 5000)):
        array(checks, plan[role + "_ids"], np.int64, (count,), role + " IDs")
        array(checks, plan[role + "_labels"], np.int64, (count,), role + " labels")
        checks.need(np.all((plan[role + "_labels"] >= 0) & (plan[role + "_labels"] < 10)), "digit range")
    all_ids = np.concatenate([plan[k + "_ids"] for k in ("train", "validation", "reporting")])
    checks.exact(np.sort(all_ids), np.arange(60000, dtype=np.int64), "disjoint original roles")
    order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0, seed]))).permutation(60000)
    checks.exact(all_ids, order.astype(np.int64), "original role order")
    labels = plan["train_labels"]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1, seed])))
    selected = rng.random(50000) < .9
    replacements = rng.integers(0, 10, size=50000, dtype=np.int64)
    assigned = np.where(selected, replacements, labels)
    for key, expected in (("corruption_mask", selected), ("replacement_labels", replacements),
                          ("assigned_labels", assigned), ("changed_mask", assigned != labels)):
        checks.exact(plan[key], expected, key)
    order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([40, seed]))).permutation(50000)
    action, evaluation = order[:128].reshape(2, 64), order[128:384]
    shifts = np.random.Generator(np.random.PCG64(np.random.SeedSequence([41, seed]))).integers(
        -2, 3, size=(2, 64, 2), dtype=np.int8)
    values = (action, plan["train_ids"][action], labels[action], assigned[action], shifts,
              evaluation, plan["train_ids"][evaluation], labels[evaluation], assigned[evaluation],
              assigned[evaluation] != labels[evaluation], plan["reporting_ids"][:128],
              plan["reporting_labels"][:128], plan["reporting_ids"][:500], plan["reporting_labels"][:500],
              np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], dtype=np.int8))
    return dict(zip(PANEL_KEYS, values))


def materialized(theta, endpoint, fraction):
    if fraction == 1:
        return endpoint.copy()
    return np.add(theta, np.multiply(np.subtract(endpoint, theta, dtype=np.float32),
                                    np.float32(fraction), dtype=np.float32), dtype=np.float32)


def dot(gradient, displacement):
    return -math.fsum((gradient.astype(np.float64) * displacement).tolist())


def check_action(record, metadata, theta, stage, checks, parameter_sizes=PARAMETER_SIZES):
    """Saved-array Adam/projection checks; no optimizer or observer execution."""
    count = sum(parameter_sizes)
    checks.need(set(record) == set(ACTION_KEYS), "action keys")
    for name in ACTION_VECTORS:
        array(checks, record[name], np.float32, (count,), name)
    for suffix, expected in (("before", stage), ("after", stage + 1)):
        checks.exact(record["adam_steps_" + suffix],
                     np.full(len(parameter_sizes), expected, dtype=np.int64), "Adam clocks " + suffix)
    checks.exact(record["theta_before"], theta, "common parent theta")
    checks.need(type(metadata["observer_steps_before"]) is int and type(metadata["observer_steps_after"]) is int
                and metadata["policy"] in ("raw", "native") and metadata["observer_steps_before"] == stage and
                metadata["observer_steps_after"] == stage + (metadata["policy"] == "native"),
                "separate observer clock")
    q, s = record["post_basis"], record["post_singular_values"]
    rank = metadata["basis_rank"]
    checks.need(type(rank) is int and 0 <= rank <= 200, "basis rank")
    array(checks, q, np.float32, (count, rank), "post basis")
    array(checks, s, np.float64, (rank,), "post singular values")
    checks.need(np.all(s > 0) and np.all(s[:-1] >= s[1:]), "positive ordered singular values")
    g = record["gradient_raw"]
    if metadata["policy"] == "raw" or rank == 0:
        checks.need(rank == 0, "raw basis must be absent")
        checks.exact(record["gradient_delivered"], g, "raw/no-basis delivery")
    else:
        # Sequential FP64 matrix-vector products avoid storing a dense P by P matrix.
        q64 = q.astype(np.float64)
        projected = q64 @ (q64.T @ g.astype(np.float64))
        checks.norm(record["gradient_delivered"], projected, "post-basis delivery")
        del q64, projected
    checks.need(np.all(record["v_before"] >= 0) and np.all(record["v_after"] >= 0), "Adam v positivity")
    delivered = record["gradient_delivered"].astype(np.float64)
    m = .9 * record["m_before"].astype(np.float64) + .1 * delivered
    v = .999 * record["v_before"].astype(np.float64) + .001 * delivered**2
    checks.norm(record["m_after"], m, "Adam first moment")
    checks.norm(record["v_after"], v, "Adam second moment")
    decay = np.multiply(theta, np.float32(1 - .001 * .01), dtype=np.float32)
    checks.exact(record["decay_endpoint"], decay, "rounded decay endpoint")
    step = stage + 1
    data_delta = -.001 * (m / (1 - .9**step)) / (np.sqrt(v / (1 - .999**step)) + 1e-8)
    expected = (decay.astype(np.float64) + data_delta).astype(np.float32)
    checks.norm(record["theta_after"], expected, "Adam endpoint")
    # Also inspect actual movement so a large parent cannot hide an incorrect step.
    checks.norm(record["theta_after"].astype(np.float64) - decay.astype(np.float64),
                expected.astype(np.float64) - decay.astype(np.float64), "Adam data movement")


def check_path(path, theta, endpoint, decay, fraction, checks):
    checks.need(set(path) == {"point", "decay_point", "total_delta", "decay_delta", "data_delta"}, "path keys")
    point, decay_point = materialized(theta, endpoint, fraction), materialized(theta, decay, fraction)
    checks.exact(path["point"], point, "materialized point")
    checks.exact(path["decay_point"], decay_point, "materialized decay point")
    deltas = dict(total_delta=point.astype(np.float64) - theta.astype(np.float64),
                  decay_delta=decay_point.astype(np.float64) - theta.astype(np.float64),
                  data_delta=point.astype(np.float64) - decay_point.astype(np.float64))
    for key, value in deltas.items():
        checks.exact(path[key], value, key)
    return {key.replace("delta", "norm"): float(np.linalg.norm(value)) for key, value in deltas.items()}


def effects(baseline, after, decay, gradients, path):
    result = {}
    for name in OBJECTIVES:
        finite = baseline[name] - after[name]
        linear = dot(gradients[name], path["total_delta"])
        result[name] = dict(finite=finite, linear=linear, residual=finite - linear,
                            finite_decay=baseline[name] - decay[name], finite_data=decay[name] - after[name],
                            linear_decay=dot(gradients[name], path["decay_delta"]),
                            linear_data=dot(gradients[name], path["data_delta"]))
    return result


def summary(parents):
    by_key = {(p["seed"], p["augmentation"], p["step"]): p for p in parents}
    cells = []
    for augmentation in AUGMENTATIONS:
        for step in STEPS:
            for fraction, fraction_id in FRACTIONS:
                rows = []
                for seed in SEEDS:
                    endpoints = by_key[seed, augmentation, step]["endpoints"]
                    rows_by_key = {(e["batch"], e["policy"], e["fraction_id"]): e for e in endpoints}
                    values = {}
                    for objective in OBJECTIVES:
                        values[objective] = {}
                        for kind in ("finite", "linear"):
                            for policy in ("raw", "native"):
                                values[objective][policy + "_" + kind] = math.fsum(
                                    rows_by_key[b, policy, fraction_id]["effects"][objective][kind]
                                    for b in (0, 1)) / 2
                            values[objective]["contrast_" + kind] = math.fsum(
                                rows_by_key[b, "native", fraction_id]["effects"][objective][kind] -
                                rows_by_key[b, "raw", fraction_id]["effects"][objective][kind]
                                for b in (0, 1)) / 2
                    rows.append(dict(seed=seed, objectives=values))
                cells.append(dict(augmentation=augmentation, step=step, fraction=fraction,
                                  fraction_id=fraction_id, seed_rows=rows))
    primary = dict(cells[6])
    primary["seed_rows"] = [dict(seed=row["seed"], objectives={key: row["objectives"][key]
                            for key in ("H_O", "C")}) for row in primary["seed_rows"]]
    return dict(primary=primary, cells=cells)


def roster():
    return [dict(seed=seed, augmentation=aug, step=step, parent_id=f"s{seed}-{aug}-h{step:05d}")
            for seed in SEEDS for aug in AUGMENTATIONS for step in STEPS]


def expected_names():
    names = {"provenance.json", *[f"panels-s{seed}.npz" for seed in SEEDS]}
    for row in roster():
        pid = row["parent_id"]
        names.update(f"{kind}-{pid}.npz" for kind in ("baseline-check", "baseline", "gradients"))
        for batch in (0, 1):
            names.update(f"action-{pid}-b{batch}-{policy}.npz" for policy in ("raw", "native"))
            for policy in ("raw", "native", "decay"):
                for _, fid in FRACTIONS:
                    names.update(f"{kind}-{pid}-b{batch}-{policy}-{fid}.npz" for kind in ("logits", "path"))
    return names


def _named(reader, receipt, name, keys=None):
    reader.checks.need(receipt["path"] == name, "artifact role/name differs")
    return reader.npz(receipt, keys)


def _logit_schema(logits, checks, ni, nr, views, classes):
    checks.need(set(logits) == {"I", "R", "R_original"}, "logit keys")
    for key, shape in (("I", (ni, views, classes)), ("R", (nr, views, classes)),
                       ("R_original", (nr, classes))):
        array(checks, logits[key], np.float32, shape, key + " logits")


def audit_parent(parent, panel, reader, source_readout, *,
                 layout=(P, 256, 128, 25, 10, 500, 12), parameter_sizes=PARAMETER_SIZES):
    """One-parent numeric checker; optional small layout is for fabricated fixtures only."""
    checks = reader.checks
    count, ni, nr, views, classes, baseline_count, original_index = layout
    pid, stage = parent["parent_id"], parent["step"]
    checks.need(_sha(parent["parent_digest_before"]) and
                parent["parent_digest_before"] == parent["parent_digest_after"], "parent immutability digest")
    admission = _named(reader, parent["baseline_check_receipt"], f"baseline-check-{pid}.npz",
                       ("actual", "expected", "ids"))
    for key in ("actual", "expected"):
        array(checks, admission[key], np.float32, (baseline_count, classes), "baseline check " + key)
    checks.exact(admission["ids"], panel["baseline_ids"], "baseline source IDs")
    checks.exact(admission["expected"], source_readout, "baseline expected versus archived source")
    checks.exact(admission["actual"], admission["expected"], "restored baseline exact equality")
    baseline_logits = _named(reader, parent["baseline_receipt"], f"baseline-{pid}.npz")
    _logit_schema(baseline_logits, checks, ni, nr, views, classes)
    baseline = metrics(baseline_logits, panel, original_index)
    checks.tree(parent["baseline_metrics"], baseline, "baseline metrics")
    checks.near(baseline["objectives"]["L"], math.fsum(baseline["objectives"][k] for k in ("S", "F", "C")),
                "baseline component identity")
    del admission, baseline_logits
    gradients = _named(reader, parent["gradient_receipt"], f"gradients-{pid}.npz", (*OBJECTIVES, "theta"))
    for key, value in gradients.items():
        array(checks, value, np.float32, (count,), "gradient/parent " + key)
    closure = sum(gradients[k].astype(np.float64) for k in ("S", "F", "C"))
    checks.need(float(np.linalg.norm(closure - gradients["L"])) <=
                GRAD_ATOL + GRAD_RTOL * float(np.linalg.norm(gradients["L"].astype(np.float64))),
                "differentiated component closure")
    theta = gradients["theta"]
    checks.need([(x["batch"], x["policy"]) for x in parent["actions"]] ==
                [(b, p) for b in (0, 1) for p in ("raw", "native")], "action roster/order")
    actions = {}
    reference = None
    for metadata in parent["actions"]:
        checks.need(type(metadata["batch"]) is int, "action batch type")
        batch, policy = metadata["batch"], metadata["policy"]
        record = _named(reader, metadata["receipt"], f"action-{pid}-b{batch}-{policy}.npz")
        check_action(record, metadata, theta, stage, checks, parameter_sizes)
        if reference is None:
            reference = record
        else:
            for name in ("theta_before", "m_before", "v_before", "adam_steps_before", "decay_endpoint"):
                checks.exact(record[name], reference[name], "shared inherited " + name)
        if policy == "native":
            checks.exact(record["gradient_raw"], actions[batch, "raw"]["gradient_raw"], "identical current gradient")
        # The large basis is consumed once, never retained across action records.
        record.pop("post_basis")
        record.pop("post_singular_values")
        actions[batch, policy] = record
    expected_roster = [(b, p, frac, fid) for b in (0, 1) for p in ("raw", "native", "decay")
                       for frac, fid in FRACTIONS]
    checks.need([(e["batch"], e["policy"], e["fraction"], e["fraction_id"]) for e in parent["endpoints"]]
                == expected_roster, "endpoint roster/order")
    computed, saved_paths = {}, {}
    # First consume all small logit readouts, so every arm can use its matching decay loss.
    for endpoint in parent["endpoints"]:
        checks.need(type(endpoint["batch"]) is int and type(endpoint["fraction"]) is float, "endpoint key types")
        b, policy, fraction, fid = (endpoint[k] for k in ("batch", "policy", "fraction", "fraction_id"))
        logits = _named(reader, endpoint["logit_receipt"], f"logits-{pid}-b{b}-{policy}-{fid}.npz")
        _logit_schema(logits, checks, ni, nr, views, classes)
        report = metrics(logits, panel, original_index)
        checks.tree(endpoint["metrics"], report, "endpoint metrics")
        checks.near(report["objectives"]["L"], math.fsum(report["objectives"][k] for k in ("S", "F", "C")),
                    "endpoint component identity")
        if policy == "decay" and b == 0:
            saved_paths["decay_logits", fid] = logits
        elif policy == "decay":
            for key in logits:
                checks.exact(logits[key], saved_paths["decay_logits", fid][key], "shared decay logits")
        computed[b, policy, fid] = dict(endpoint, metrics=report)
    for endpoint in parent["endpoints"]:
        b, policy, fraction, fid = (endpoint[k] for k in ("batch", "policy", "fraction", "fraction_id"))
        action = actions[b, policy if policy != "decay" else "raw"]
        end = action["theta_after"] if policy != "decay" else action["decay_endpoint"]
        path = _named(reader, endpoint["path_receipt"], f"path-{pid}-b{b}-{policy}-{fid}.npz")
        norms = check_path(path, theta, end, action["decay_endpoint"], fraction, checks)
        value = computed[b, policy, fid]
        value["norms"] = norms
        value["effects"] = effects(baseline["objectives"], value["metrics"]["objectives"],
                                    computed[b, "decay", fid]["metrics"]["objectives"], gradients, path)
        checks.tree(endpoint["norms"], norms, "actual displacement norms")
        checks.tree(endpoint["effects"], value["effects"], "objective effects")
        for kind in ("finite", "linear", "finite_decay", "finite_data", "linear_decay", "linear_data"):
            delta_key = {"linear": "total_delta", "linear_decay": "decay_delta", "linear_data": "data_delta"}.get(kind)
            checks.near(value["effects"]["L"][kind],
                        math.fsum(value["effects"][key][kind] for key in ("S", "F", "C")),
                        "effect component closure " + kind,
                        # FP32 differentiated closure does not imply 1e-10 dot closure.
                        atol=(GRAD_ATOL + GRAD_RTOL * float(np.linalg.norm(gradients["L"].astype(np.float64))))
                             * float(np.linalg.norm(path[delta_key])) + SCALAR_ATOL if delta_key else SCALAR_ATOL,
                        rtol=SCALAR_RTOL)
        del path
    contrasts = []
    for batch in (0, 1):
        for fraction, fid in FRACTIONS:
            contrasts.append(dict(batch=batch, fraction=fraction, fraction_id=fid,
                                  objectives={name: {kind: computed[batch, "native", fid]["effects"][name][kind]
                                      - computed[batch, "raw", fid]["effects"][name][kind]
                                      for kind in ("finite", "linear")} for name in OBJECTIVES}))
    checks.tree(parent["contrasts"], contrasts, "native minus raw contrasts")
    return dict(parent, baseline_metrics=baseline, endpoints=list(computed.values()), contrasts=contrasts)


def bound_json(path, expected_sha, checks):
    receipt = file_receipt(path, MAX_JSON)
    checks.need(_sha(expected_sha) and receipt["sha256"] == expected_sha, "bound JSON hash differs")
    value = strict_json(pinned_payload(path, receipt, MAX_JSON))
    checks.tree(file_receipt(path, MAX_JSON), receipt, "bound JSON unchanged")
    return value, receipt


def verify_sources(root, manifest, inventory, checks):
    checks.need(type(manifest) is dict and set(manifest) ==
                {"schema", "commit", "source_pins", "input_inventory_sha256"}, "source manifest keys")
    checks.need(manifest["schema"] == "spectral_component_utility_source_manifest_v1"
                and type(manifest["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", manifest["commit"]),
                "manifest schema/commit")
    pins = manifest["source_pins"]
    checks.need(type(pins) is dict and set(inventory["source_pins"]) <= set(pins), "inherited source pins")
    for path, sha in inventory["source_pins"].items():
        checks.need(pins[path] == sha, "changed canonical dependency")
    stems = ("", "_actions", "_restore", "_panels", "_objectives", "_guard", "_audit")
    required = {f"experiments/spectral_component_utility{name}.py" for name in stems}
    required |= {f"tests/test_spectral_component_utility{name}.py" for name in stems}
    docs = "output/2026-09-10-spectral-component-utility/"
    required |= {docs + name for name in ("protocol.md", "archive-contract.md", "parent-inventory.json", "implementation-check.md")}
    checks.need(required | set(inventory["source_pins"]) == set(pins), "exact source pin set differs")
    checks.need(pins[docs + "parent-inventory.json"] == manifest["input_inventory_sha256"], "inventory source hash")
    for relative, digest in pins.items():
        rel = Path(relative)
        checks.need(type(relative) is str and not rel.is_absolute() and ".." not in rel.parts
                    and _sha(digest), "unsafe source pin")
        path = root / rel
        checks.need(path.resolve() == path, "symlink in source path")
        checks.need(file_receipt(path, MAX_JSON)["sha256"] == digest, "current source hash differs")
        result = subprocess.run(["git", "show", manifest["commit"] + ":" + relative], cwd=root,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, check=False)
        checks.need(result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == digest,
                    "source does not match frozen committed bytes")


def check_guards(guards, attempt, checks):
    keys = {"unit", "pid", "invocation_id", "cgroup", "effective", "service", "gpu_clients",
            "preexisting_gpu_clients", "device", "device_name", "gpu_free_bytes_at_configure",
            "gpu_total_bytes", "gpu_allocator_limit_bytes", "gpu_allocator_fraction", "torch_version",
            "numpy_version", "deterministic"}
    checks.need(type(guards) is dict and set(guards) == keys, "acquisition guard keys")
    for key in ("unit", "pid", "invocation_id"):
        checks.tree(guards[key], attempt[key], "guard identity " + key)
    checks.need(type(guards["cgroup"]) is str and guards["cgroup"].endswith("/" + attempt["unit"]),
                "acquisition cgroup identity")
    checks.tree(guards["effective"], {"memory.max": "8589934592", "memory.swap.max": "0",
                                       "cpu.max": "100000 100000"}, "effective acquisition bounds")
    checks.tree(guards["service"], dict(Type="exec", RuntimeMaxUSec="30min", Restart="no",
                KillMode="control-group", MainPID=str(attempt["pid"]), InvocationID=attempt["invocation_id"],
                ActiveState="active", SubState="running", ControlGroup=guards["cgroup"]), "acquisition service")
    checks.tree(guards["deterministic"], dict(thread_env={key: "1" for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                cublas_workspace=":4096:8", intraop_threads=1, interop_threads=1,
                deterministic_algorithms=True, cudnn_benchmark=False, matmul_allow_tf32=False,
                cudnn_allow_tf32=False), "determinism settings")
    checks.need(guards["device"] == "cuda:0" and guards["device_name"] == "NVIDIA GeForce RTX 3090"
                and guards["torch_version"] == "2.11.0+cu128" and guards["numpy_version"] == "1.26.4",
                "acquisition device/version")
    checks.need(type(guards["gpu_total_bytes"]) is int and guards["gpu_total_bytes"] > 8 * 1024**3
                and type(guards["gpu_free_bytes_at_configure"]) is int
                and guards["gpu_total_bytes"] >= guards["gpu_free_bytes_at_configure"] >= 8 * 1024**3
                and guards["gpu_allocator_limit_bytes"] == 4 * 1024**3, "GPU resource bounds")
    checks.near(guards["gpu_allocator_fraction"], 4 * 1024**3 / guards["gpu_total_bytes"], "GPU allocator fraction")
    for rows, preexisting in ((guards["gpu_clients"], False), (guards["preexisting_gpu_clients"], True)):
        checks.need(type(rows) is list and len(rows) <= 3, "GPU client roster")
        seen = set()
        for row in rows:
            checks.need(type(row) is dict and set(row) == {"pid", "memory_mib"}
                        and type(row["pid"]) is int and row["pid"] not in seen
                        and type(row["memory_mib"]) is int and row["memory_mib"] >= 0, "GPU client schema")
            caps = {2101: 512, 8861: 128}
            if not preexisting:
                caps[attempt["pid"]] = 4096
            checks.need(row["pid"] in caps and row["memory_mib"] <= caps[row["pid"]], "unknown GPU client")
            seen.add(row["pid"])
    checks.tree(guards["preexisting_gpu_clients"],
                [row for row in guards["gpu_clients"] if row["pid"] != attempt["pid"]], "preexisting client subset")


def verify_provenance(provenance, attempt, manifest, inventory, manifest_sha, inventory_sha,
                      acquisition, checks):
    attempt_keys = {"schema", "commit", "worktree_head", "source_pins", "data_pins",
                    "input_inventory_sha256", "manifest_sha256", "unit", "output_dir", "started_utc",
                    "pid", "invocation_id", "argv"}
    checks.need(type(attempt) is dict and set(attempt) == attempt_keys, "attempt keys")
    expected = dict(schema=SCHEMA, commit=manifest["commit"], source_pins=manifest["source_pins"],
                    data_pins=inventory["data_pins"], input_inventory_sha256=inventory_sha,
                    manifest_sha256=manifest_sha, unit="spectral-component-utility-001.service",
                    output_dir=str(acquisition))
    for key, value in expected.items():
        checks.tree(attempt[key], value, "attempt " + key)
    checks.need(type(attempt["worktree_head"]) is str and re.fullmatch(r"[0-9a-f]{40}", attempt["worktree_head"])
                and type(attempt["pid"]) is int and attempt["pid"] > 0
                and type(attempt["invocation_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", attempt["invocation_id"])
                and type(attempt["started_utc"]) is str and attempt["started_utc"].endswith(("Z", "+00:00"))
                and type(attempt["argv"]) is list and all(type(x) is str for x in attempt["argv"]), "attempt types")
    checks.need(type(provenance) is dict and set(provenance) == attempt_keys |
                {"guards", "inventory", "python", "numpy", "torch"}, "provenance keys")
    checks.tree({k: provenance[k] for k in attempt_keys}, attempt, "provenance attempt binding")
    checks.need(type(provenance["python"]) is str and provenance["numpy"] == "1.26.4"
                and provenance["torch"] == "2.11.0+cu128", "provenance versions")
    check_guards(provenance["guards"], attempt, checks)
    component_upper = {"48_action_tensor_records": 406336896, "24_post_native_bases": 4514803200,
                       "24_post_native_singular_values": 38400, "12_baseline_gradient_theta_sets": 79009056,
                       "12_baseline_check_sets": 528000, "144_path_records": 1083552768,
                       "156_logit_sets": 60702720, "metadata_container_panel_reserve": 268435456}
    checks.tree(provenance["inventory"], dict(component_upper_bytes=component_upper,
                total_upper_bytes=6413406496, failure_reserve_bytes=1048576, cap_bytes=ARCHIVE_CAP,
                expected_receipted_artifacts=376), "enforced byte inventory")
    checks.need(sum(component_upper.values()) == 6413406496 and
                6413406496 + 1048576 <= ARCHIVE_CAP, "byte accounting arithmetic")


def verify_input_metadata(old_results, old_audit, inventory, checks):
    """Bind the selected receipts to accepted original branches without reanalysis."""
    checks.need(old_results["schema"] == "spectral_strong_augmentation_v1"
                and old_results["status"] == "complete", "original producer status")
    checks.need(old_audit["schema"] == "spectral_strong_augmentation_audit_v1" and old_audit["status"] == "PASS"
                and old_audit["input_results_sha256"] == inventory["source_results_sha256"]
                and old_audit["input_dir"] == inventory["archive"], "accepted input audit")
    for value in (old_results, old_audit):
        checks.tree(value["data_pins"], inventory["data_pins"], "accepted data pins")
        checks.tree(value["source_pins"], inventory["source_pins"], "accepted source pins")
    receipts = {r["path"]: r for r in old_results["receipts"]}
    checks.need(len(receipts) == len(old_results["receipts"]), "duplicate old receipt")
    branches = {(p["seed"], p["augmentation"]): p for p in old_results["branches"] if p["policy"] == "native200"}
    checks.need(len([p for p in old_results["branches"] if p["policy"] == "native200"]) == 6
                and set(branches) == {(s, a) for s in SEEDS for a in AUGMENTATIONS}, "old native branch roster")
    for row in inventory["parents"]:
        branch = branches[row["seed"], row["augmentation"]]
        for stage, step in (("warmup", 100), ("final", 56304)):
            checks.tree(row[stage], branch[stage + "_receipt"], "old checkpoint branch binding")
            readouts = [r["readout_receipt"] for r in branch["metrics"] if r["step"] == step]
            checks.tree(readouts, [row[stage + "_readout"]], "old readout branch binding")
            for key in (stage, stage + "_readout"):
                checks.tree(receipts.get(row[key]["path"]), row[key], "original receipt binding")
        checks.need(row["plan_npz"] == f"plan-s{row['seed']}.npz" and
                    row["plan_json"] == f"plan-s{row['seed']}.json", "old parent plan name")
    checks.need([r["path"] for r in inventory["plan_receipts"]] ==
                [f"plan-s{s}.{ext}" for s in SEEDS for ext in ("npz", "json")], "old plan roster")
    for receipt in inventory["plan_receipts"]:
        checks.tree(receipts.get(receipt["path"]), receipt, "old plan receipt binding")


def check_plan_header(plan, header, seed, checks):
    checks.need(set(plan) == {"train_ids", "validation_ids", "reporting_ids", "train_labels", "validation_labels",
                "reporting_labels", "corruption_mask", "replacement_labels", "assigned_labels", "changed_mask",
                "occurrences", "shifts", "batch_boundaries"}, "original plan schema")
    checks.need(header["seed"] == seed and header["numpy_version"] == "1.26.4", "plan seed/version")
    hashes = {}
    for key, value in plan.items():
        head = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
        hashes[key] = hashlib.sha256(head + b"\n" + value.tobytes(order="C")).hexdigest()
    checks.tree(header["array_hashes"], hashes, "plan array hashes")
    checks.tree(header["corruption_selected"], int(plan["corruption_mask"].sum()), "selected replacements")
    checks.tree(header["actually_wrong"], int(plan["changed_mask"].sum()), "actually wrong labels")
    checks.tree(header["true_class_counts"], {role: np.bincount(plan[role + "_labels"], minlength=10).tolist()
                for role in ("train", "validation", "reporting")}, "original class counts")


def audit_archive(acquisition, results_sha, manifest_path, manifest_sha, attempt_path, attempt_sha,
                  inventory_path, inventory_sha, root):
    """Fixed production entry point. Only CLI execution admits scientific reads."""
    checks = Checks(ENTRY_MONOTONIC + COOPERATIVE_SECONDS)
    root, acquisition = Path(root), Path(acquisition)
    checks.need(root.resolve() == root and acquisition.resolve() == acquisition
                and acquisition.is_dir() and not acquisition.is_symlink(), "canonical roots")
    result, result_receipt = bound_json(acquisition / "results.json", results_sha, checks)
    manifest, manifest_receipt = bound_json(manifest_path, manifest_sha, checks)
    inventory, inventory_receipt = bound_json(inventory_path, inventory_sha, checks)
    attempt, attempt_receipt = bound_json(attempt_path, attempt_sha, checks)
    checks.need(inventory["schema"] == "spectral_component_utility_parent_inventory_v1", "input inventory schema")
    checks.need(manifest["input_inventory_sha256"] == inventory_sha, "manifest inventory binding")
    verify_sources(root, manifest, inventory, checks)
    checks.need(result["schema"] == SCHEMA and result["status"] == "complete", "completed result required")
    for key, value in (("source_pins", manifest["source_pins"]), ("data_pins", inventory["data_pins"]),
                        ("input_inventory_sha256", inventory_sha), ("roster", roster())):
        checks.tree(result[key], value, "results " + key)
    checks.need(type(result["seconds"]) in (int, float) and 0 < result["seconds"] <= 1200
                and type(result["serialization_seconds"]) in (int, float)
                and 0 <= result["serialization_seconds"] <= result["seconds"]
                and type(result["gpu_max_allocated_bytes"]) is int
                and 0 <= result["gpu_max_allocated_bytes"] <= 4 * 1024**3
                and type(result["host_high_water_rss_kib"]) is int
                and 0 < result["host_high_water_rss_kib"] <= 8 * 1024**2, "acquisition resources")
    reader = Reader(acquisition, result["receipts"], checks)
    checks.need(set(reader.receipts) == expected_names(), "exact artifact inventory")
    checks.need({x.name for x in acquisition.iterdir()} == expected_names() | {"results.json"}, "unreceipted archive file")
    checks.need(sum(x["size_bytes"] for x in result["receipts"]) + result_receipt["size_bytes"] <= ARCHIVE_CAP,
                "archive byte cap")
    provenance = reader.json(reader.receipts["provenance.json"])
    verify_provenance(provenance, attempt, manifest, inventory, manifest_sha, inventory_sha, acquisition, checks)
    oldroot = Path(inventory["archive"])
    old_results, old_results_receipt = bound_json(oldroot / "results.json", inventory["source_results_sha256"], checks)
    old_audit, old_audit_receipt = bound_json(root / "output/2026-09-10-spectral-strong-augmentation/audit.json",
                                           inventory["source_audit_sha256"], checks)
    verify_input_metadata(old_results, old_audit, inventory, checks)
    old_receipts = list(inventory["plan_receipts"])
    for item in inventory["parents"]:
        old_receipts.extend(item[key] for key in ("warmup", "final", "warmup_readout", "final_readout"))
    oldreader = Reader(oldroot, old_receipts, checks)
    checks.need([(x["seed"], x["augmentation"]) for x in inventory["parents"]] ==
                [(s, a) for s in SEEDS for a in AUGMENTATIONS], "input parent inventory roster")
    checks.need([p["seed"] for p in result["panels"]] == list(SEEDS), "panel seed order")
    panels = {}
    for row in result["panels"]:
        seed = row["seed"]
        plan = oldreader.npz(oldreader.receipts[f"plan-s{seed}.npz"])
        # Its old training occurrences are receipt-bound, not numerically replayed.
        expected_panel = panels_from_plan(plan, seed, checks)
        panel = _named(reader, row["receipt"], f"panels-s{seed}.npz", PANEL_KEYS)
        for key in PANEL_KEYS:
            checks.exact(panel[key], expected_panel[key], "fixed panel " + key)
        panels[seed] = panel
        header = oldreader.json(oldreader.receipts[f"plan-s{seed}.json"])
        check_plan_header(plan, header, seed, checks)
        del plan
    checks.need([{k: p[k] for k in ("seed", "augmentation", "step", "parent_id")} for p in result["parents"]]
                == roster(), "parent order/roster")
    accepted = {(p["seed"], p["augmentation"]): p for p in inventory["parents"]}
    parents = []
    for parent in result["parents"]:
        source = accepted[parent["seed"], parent["augmentation"]]
        stage = "warmup" if parent["step"] == 100 else "final"
        checks.tree(parent["source_checkpoint"], source[stage], "checkpoint receipt binding")
        checks.tree(parent["source_readout"], source[stage + "_readout"], "source readout binding")
        oldreader.path(source[stage])  # Hash raw bytes only; NEVER deserialize checkpoint.
        readout = oldreader.npz(source[stage + "_readout"], ("step", "train", "validation", "reporting"))
        checks.exact(readout["step"], np.array([parent["step"]], dtype=np.int64), "source readout stage")
        array(checks, readout["reporting"], np.float32, (5000, 10), "source reporting logits")
        parents.append(audit_parent(parent, panels[parent["seed"]], reader, readout["reporting"][:500]))
        del readout
    independent = summary(parents)
    checks.tree(result["summary"], independent, "all-cell summary and fixed primary")
    checks.need(reader.used == set(reader.receipts) and oldreader.used == set(oldreader.receipts), "all receipts consumed")
    for collection in (reader, oldreader):
        for receipt in collection.receipts.values():
            collection.path(receipt)
    verify_sources(root, manifest, inventory, checks)
    for path, receipt in ((acquisition / "results.json", result_receipt), (Path(manifest_path), manifest_receipt),
                          (Path(attempt_path), attempt_receipt), (Path(inventory_path), inventory_receipt),
                          (oldroot / "results.json", old_results_receipt),
                          (root / "output/2026-09-10-spectral-strong-augmentation/audit.json", old_audit_receipt)):
        checks.tree(file_receipt(path, MAX_JSON), receipt, "final input unchanged")
    return dict(schema="spectral_component_utility_audit_v1", status="PASS", errors=[], checks=checks.count,
                acquisition_dir=str(acquisition), input_results_sha256=results_sha,
                input_inventory_sha256=inventory_sha, source_manifest_sha256=manifest_sha, attempt_sha256=attempt_sha,
                input_receipts=[result_receipt, *result["receipts"]], source_pins=manifest["source_pins"],
                independent_summary=independent, checked_parents=parents, array_residuals=checks.array_residuals,
                counts=dict(parents=12, action_pairs=24, actions=48, endpoints=144, baseline_readouts=12,
                            panels=3, artifacts=376), seconds=time.monotonic() - ENTRY_MONOTONIC,
                boundary="Saved-array arithmetic and provenance, not model/autograd/observer-history replay.")


def audit_environment(output, acquisition):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise AuditError("audit requires one-thread environment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or np.__version__ != "1.26.4":
        raise AuditError("CPU-only pinned NumPy environment required")
    rows = Path("/proc/self/cgroup").read_text().splitlines()
    cgroup = next((x[3:] for x in rows if x.startswith("0::")), None)
    if not cgroup or not cgroup.endswith("/spectral-component-utility-audit-001.service"):
        raise AuditError("fixed once-only audit unit required")
    cg = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    expected = {"memory.max": "2147483648", "memory.swap.max": "0", "cpu.max": "100000 100000"}
    if any((cg / key).read_text().strip() != value for key, value in expected.items()):
        raise AuditError("effective audit resource limits differ")
    unit = "spectral-component-utility-audit-001.service"
    properties = ("Type", "RuntimeMaxUSec", "Restart", "KillMode", "MainPID", "InvocationID", "ControlGroup")
    invocation = os.environ.get("INVOCATION_ID", "")
    if not re.fullmatch(r"[0-9a-f]{32}", invocation):
        raise AuditError("audit invocation identity required")
    response = subprocess.check_output(["systemctl", "--user", "show", unit,
                                       *["--property=" + key for key in properties]], text=True, timeout=10)
    service = {}
    for line in response.splitlines():
        key, sep, value = line.partition("=")
        if sep != "=" or key not in properties or key in service:
            raise AuditError("audit service response schema")
        service[key] = value
    if service != dict(Type="exec", RuntimeMaxUSec="20min", Restart="no", KillMode="control-group",
                       MainPID=str(os.getpid()), InvocationID=invocation, ControlGroup=cgroup):
        raise AuditError("exact once-only audit service properties differ")
    output, acquisition = Path(output), Path(acquisition)
    if output.exists() or output.is_symlink() or output.parent.resolve() != output.parent or not output.parent.is_dir():
        raise AuditError("exclusive output in canonical existing parent required")
    if output == acquisition or acquisition in output.parents:
        raise AuditError("audit output must be outside immutable acquisition")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("input-dir", "output", "expected-results-sha256", "expected-sources",
                 "expected-sources-sha256", "attempt", "expected-attempt-sha256", "inventory",
                 "expected-inventory-sha256"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    if not args.execute:
        return 0
    if any(getattr(args, name.replace("-", "_")) is None for name in
           ("input-dir", "output", "expected-results-sha256", "expected-sources", "expected-sources-sha256",
            "attempt", "expected-attempt-sha256", "inventory", "expected-inventory-sha256")):
        parser.error("--execute requires all bound input/output arguments")
    audit_environment(args.output, args.input_dir)
    root = Path(__file__).resolve().parents[1]
    try:
        result = audit_archive(Path(args.input_dir), args.expected_results_sha256, Path(args.expected_sources),
                               args.expected_sources_sha256, Path(args.attempt), args.expected_attempt_sha256,
                               Path(args.inventory), args.expected_inventory_sha256, root)
    except Exception as exc:
        result = dict(schema="spectral_component_utility_audit_v1", status="FAIL", errors=[str(exc)],
                      seconds=time.monotonic() - ENTRY_MONOTONIC)
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_JSON:
        raise AuditError("audit output exceeds 64 MiB")
    with Path(args.output).open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
