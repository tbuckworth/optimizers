"""Independent raw-value I7 measurement audit; no producer or Torch imports.

Inputs: ordered parameter maps 0.weight/0.bias/2.weight/2.bias; six ordered
endpoint maps or None; six actual delivered float32 candidate vectors or None;
four ordered probes {inputs:float32 matrix,labels:int64 vector}; and the exact
measurement-schema.md mapping {measurement_before,branches,comparisons}.
branches[b] is None or {displacement,probes}. vector64 values are owned,
contiguous NumPy arrays converted by the caller, with their original metadata.
All native arrays are supplied by the caller: this module opens no artifacts.

Recomputed CPU64 CE/q never depend on producer CE/q. Native32 forward values
are NOT independently reevaluated: only finite scalar arithmetic/concordance
is checked. Source/file provenance, Adam/operator checks and live replay remain
separate. Internal hash bindings do not establish those external properties.
Fatal validation, permitted branch-domain nulls, weak factors, numerical-sign
uncertainty and finite native discordance are distinct report fields.
"""

import hashlib
import itertools
import math
import re
import struct

import numpy as np

import independent_numerics as num

BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
NAMES = ("0.weight", "0.bias", "2.weight", "2.bias")
COEFFICIENTS = {
    "ordering": {"lagged": 1, "current": -1},
    "direction_at_current_norm": {"restored": 1, "current": -1},
    "direction_at_lagged_norm": {"lagged": 1, "reciprocal": -1},
    "norm_at_current_direction": {"current": 1, "reciprocal": -1},
    "norm_at_lagged_direction": {"restored": 1, "lagged": -1},
    "interaction": {"restored": 1, "lagged": -1, "current": -1, "reciprocal": 1},
    "current_minus_raw": {"current": 1, "raw": -1},
    "lagged_minus_raw": {"lagged": 1, "raw": -1},
    "restored_minus_raw": {"restored": 1, "raw": -1},
    "reciprocal_minus_raw": {"reciprocal": 1, "raw": -1},
    "raw_minus_zero": {"raw": 1, "zero": -1},
    "current_minus_zero": {"current": 1, "zero": -1},
    "lagged_minus_zero": {"lagged": 1, "zero": -1},
    "restored_minus_zero": {"restored": 1, "zero": -1},
    "reciprocal_minus_zero": {"reciprocal": 1, "zero": -1},
}
PROFILES = {
    "fixture_tiny_mlp_cpu_v1": ((3, 4, 2), (4, 4, 4, 10), 1),
    "scientific_mnist_current32_v1": ((784, 64, 10), (64, 256, 256, 5000), 500),
}
FORMULA = "i7_assembly_roundoff_v1"


def _need(condition, path, reason):
    if not condition:
        raise ValueError(f"{path}: {reason}")


def _map(value, keys, path, ordered=False):
    _need(type(value) is dict and set(value) == set(keys), path, "mapping membership differs")
    if ordered:
        _need(tuple(value) == tuple(keys), path, "mapping order differs")
    return value


def _float(value, path, nonnegative=False):
    _need(type(value) is float and math.isfinite(value), path, "expected finite Python float")
    _need(not nonnegative or value >= 0, path, "negative value")
    return value


def _hash_array(value):
    return hashlib.sha256(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes(order="C")).hexdigest()


def _hash_scalar(value):
    return hashlib.sha256(struct.pack("<d", value)).hexdigest()


def _hash_probe(x, y):
    return hashlib.sha256(x.astype("<f4", copy=False).tobytes(order="C") +
                          y.astype("<i8", copy=False).tobytes(order="C")).hexdigest()


def _native(value, shape, path):
    _need(isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float32) and value.shape == shape,
          path, "native float32 array shape/type differs")
    _need(bool(np.isfinite(value).all()), path, "nonfinite native array")
    return np.array(value, dtype=np.float64, copy=True)


def _norm(x):
    scale = float(np.max(np.abs(x)))
    value = 0. if scale == 0 else scale*math.sqrt(float((x/scale)@(x/scale)))
    _need(math.isfinite(value), "norm", "nonfinite norm intermediate")
    return value


def _vsum(terms):
    return 2*num.gamma(2*len(terms)+4, num.U64)*np.sum(np.abs(terms), axis=0)+4*len(terms)*num.H64


def _combine(vectors, coefficients):
    result = np.zeros_like(next(iter(vectors.values())))
    for branch, coefficient in coefficients.items():
        result += coefficient*vectors[branch]
    return result


def _nround(x):
    norm = _norm(x)
    return 0. if norm == 0 else num.dot_roundoff(x, x)/(2*norm)+8*num.U64*norm


class _GuardFailure(Exception):
    def __init__(self, original):
        self.original = original


class _Audit:
    def __init__(self, profile, guard):
        self.profile, self.guard = profile, guard
        self.report = dict(profile=profile, scope="raw_value_measurements_only", overall_status="pass", audit_complete=False,
                           completion_counts=dict(before_probes=0, branch_probes=0, defined_branches=0, pairs=0,
                                                  contrasts=0, defined_contrasts=0, auxiliary_before_chunks=0,
                                                  auxiliary_branch_chunks=0, auxiliary_contrast_chunks=0),
                           fatal_failures=[], measurement_audits={}, comparison_audits={},
                           domain_undefined_branches=[], weak_factor_contrasts=[], native_discordant_paths=[],
                           unresolved_geometry_paths=[], binding_artifact_id=None,
                           scope_exclusions=["external_provenance", "Adam_operator_audits", "live_source_replay",
                                             "native32_forward_reexecution"])
        self.seen_arrays = set()

    def tick(self, label):
        if self.guard is not None:
            try:
                self.guard(label)
            except Exception as error:
                raise _GuardFailure(error) from error

    def failure(self, path, reason):
        self.report["overall_status"] = "fatal_validation"
        self.report["fatal_failures"].append(dict(path=path, reason=reason))

    def scalar(self, value, reference, ceiling, path, *, comparison=False, sign=False):
        value, reference, ceiling = _float(value, path), float(reference), float(ceiling)
        _need(math.isfinite(reference) and math.isfinite(ceiling) and ceiling >= 0, path, "invalid audit arithmetic")
        error = abs(value-reference)
        _need(math.isfinite(error), path, "nonfinite scalar discrepancy")
        record = dict(producer_value=value, auditor_value=reference, fixed_ceiling=ceiling,
                      achieved_discrepancy=error, formula_id=FORMULA, audit_status="pass" if error <= ceiling else "fatal_validation")
        if sign:
            resolution = num.numerical_sign(value, reference, ceiling)
            reason = resolution["reason"]
            record.update(producer_sign=resolution["producer_sign"], auditor_sign=resolution["auditor_sign"],
                          resolution_status=resolution["status"] if resolution["status"] != "unresolved" else "unresolved_numerical",
                          resolution_reason=None if reason is None else
                          "sign_disagreement" if reason == "sign_disagreement" else "insufficient_fixed_margin")
        self.report["comparison_audits" if comparison else "measurement_audits"][path] = record
        if error > ceiling:
            self.failure(path, "numerical ceiling exceeded")
        return record

    def arithmetic(self, value, expected, path, *, nonnegative=False):
        _float(value, path, nonnegative)
        ceiling = 2*num.gamma(2*self.p+4, num.U64)*max(abs(value), abs(expected))+4*self.p*num.H64
        return self.scalar(value, expected, ceiling, path)

    def vector(self, row, path):
        _map(row, ("value", "shape", "dtype", "device", "sha256"), path)
        value = row["value"]
        _need(isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float64) and value.shape == (self.p,)
              and value.flags.c_contiguous and value.flags.owndata and np.isfinite(value).all(), path, "invalid owned vector64")
        _need(id(value) not in self.seen_arrays, path, "aliased vector64 storage")
        self.seen_arrays.add(id(value))
        _need(type(row["shape"]) is list and len(row["shape"]) == 1 and type(row["shape"][0]) is int
              and row["shape"] == [self.p] and row["dtype"] == "float64" and row["device"] == "cpu", path, "vector metadata differs")
        _need(row["sha256"] == _hash_array(value), path, "vector hash differs")
        return value

    def vector_check(self, producer, auditor, ceiling, path):
        error = np.abs(producer-auditor)
        failed = error > ceiling
        positive = ceiling > 0
        record = dict(max_absolute_error=float(error.max()), error_l2=_norm(error),
                      max_component_ceiling=float(np.max(ceiling)),
                      max_error_ceiling_ratio=None if np.any(failed & ~positive) else
                      float(np.max(error[positive]/ceiling[positive])) if positive.any() else 0.,
                      failing_flat_indices=np.flatnonzero(failed).tolist(),
                      auditor_sha256=_hash_array(auditor), audit_status="pass" if not failed.any() else "fatal_validation")
        self.report["measurement_audits"][path] = record
        if failed.any():
            self.failure(path, "vector component ceiling exceeded")

    def norm_check(self, value, vector, component_error, path):
        _float(value, path, True)
        reference = _norm(vector)
        rounding = 0. if reference == 0 else num.dot_roundoff(vector, vector)/(value+reference)+8*num.U64*max(value, reference)
        return self.scalar(value, reference, _norm(component_error)+rounding, path)

    def concordance(self, row, cpu, native, path, ceiling=None):
        _map(row, ("abs_discrepancy", "descriptive_ceiling", "status"), path)
        _float(cpu, path+".cpu"); _float(native, path+".native")
        ceiling = 5e-6*max(1., abs(cpu)) if ceiling is None else ceiling
        self.arithmetic(row["abs_discrepancy"], abs(cpu-native), path+".abs_discrepancy", nonnegative=True)
        self.arithmetic(row["descriptive_ceiling"], ceiling, path+".descriptive_ceiling", nonnegative=True)
        status = "discordant" if abs(cpu-native) > ceiling else "within_scale"
        _need(row["status"] == status, path, "concordance status differs")
        if status == "discordant":
            self.report["native_discordant_paths"].append(path)

    def geometry(self, row, left, right, mask, path, left_error=None, right_error=None):
        fields = ("defined", "defined_mask", "reason", "distance", "cosine", "cosine_reason", "left_norm", "right_norm")
        _map(row, fields, path)
        self.mask(row["defined_mask"], mask, path+".defined_mask")
        defined = all(mask.values())
        _need(type(row["defined"]) is bool and row["defined"] == defined, path, "geometry defined flag differs")
        if not defined:
            _need(row["reason"] == "domain_undefined_required_branch" and
                  all(row[key] is None for key in ("distance", "cosine", "cosine_reason", "left_norm", "right_norm")),
                  path, "undefined geometry payload differs")
            return
        _need(row["reason"] is None, path, "defined geometry has reason")
        ex = np.zeros(self.p) if left_error is None else left_error
        ey = np.zeros(self.p) if right_error is None else right_error
        nx, ny = _norm(left), _norm(right)
        self.norm_check(row["left_norm"], left, ex, path+".left_norm")
        self.norm_check(row["right_norm"], right, ey, path+".right_norm")
        self.norm_check(row["distance"], left-right, ex+ey+_vsum([left, -right]), path+".distance")
        if nx == 0 or ny == 0:
            _need(row["cosine"] is None and row["cosine_reason"] == "zero_norm", path, "zero-norm cosine differs")
            return
        value = _float(row["cosine"], path+".cosine")
        _need(row["cosine_reason"] is None, path, "nonzero cosine has reason")
        dx, dy = _norm(ex)+_nround(left), _norm(ey)+_nround(right)
        if nx-dx <= 0 or ny-dy <= 0:
            self.report["unresolved_geometry_paths"].append(path+".cosine")
            self.report["comparison_audits"][path+".cosine"] = dict(producer_value=value,
                auditor_value=float(left@right/(nx*ny)), audit_status="unresolved_numerical",
                reason="denominator_interval_contains_zero")
            return
        dot = float(left@right)
        edot = float(np.sum(ex*np.abs(right)+ey*np.abs(left)+ex*ey))+num.dot_roundoff(left, right)
        endpoints = [(dot+sd*edot)/((nx+sx*dx)*(ny+sy*dy))
                     for sd in (-1, 1) for sx in (-1, 1) for sy in (-1, 1)]
        reference = dot/(nx*ny)
        ceiling = max(abs(reference-min(endpoints)), abs(max(endpoints)-reference))+8*num.U64*max(abs(x) for x in endpoints)
        self.scalar(value, reference, ceiling, path+".cosine", comparison=True)

    def mask(self, actual, expected, path):
        _map(actual, expected, path, ordered=True)
        _need(all(type(actual[k]) is bool and actual[k] == v for k, v in expected.items()), path, "defined mask differs")

    def chunk_meta(self, row, index, probe, path):
        start, end = index*self.chunk, (index+1)*self.chunk
        for key, expected in (("chunk_index", index), ("start", start), ("end", end), ("count", self.chunk)):
            _need(type(row[key]) is int and row[key] == expected, path, "chunk schedule differs")
        x, y = self.probes[probe]["inputs"], self.probes[probe]["labels"]
        _need(row["input_sha256"] == _hash_array(x[start:end]) and row["label_sha256"] == _hash_array(y[start:end]),
              path, "chunk input binding differs")

    def inputs(self, before, endpoints, candidates, probes):
        _need(self.profile in PROFILES, "profile", "unsupported profile")
        (d, h, c), counts, self.chunk = PROFILES[self.profile]
        shapes = ((h, d), (h,), (c, h), (c,))
        self.p = sum(math.prod(shape) for shape in shapes)
        def parameters(mapping, path):
            _map(mapping, NAMES, path, ordered=True)
            return np.concatenate([_native(mapping[name], shape, path+"."+name).ravel()
                                   for name, shape in zip(NAMES, shapes)])
        self.before, self.theta = before, parameters(before, "before_parameters")
        _map(endpoints, BRANCHES, "endpoints", ordered=True)
        _map(candidates, BRANCHES, "candidates", ordered=True)
        self.candidates = {b: _native(candidates[b], (self.p,), "candidates."+b) for b in BRANCHES[:3]}
        nc, nl = _norm(self.candidates["current"]), _norm(self.candidates["lagged"])
        self.defined = {b: not (b == "restored" and nc > 0 and nl == 0 or
                               b == "reciprocal" and nl > 0 and nc == 0) for b in BRANCHES}
        for b in BRANCHES[3:]:
            if not self.defined[b]:
                _need(candidates[b] is None and endpoints[b] is None, b, "domain-null membership differs")
                self.candidates[b] = None
                self.report["domain_undefined_branches"].append(b)
                continue
            a = _native(candidates[b], (self.p,), "candidates."+b)
            target = nc if b == "restored" else nl if b == "reciprocal" else 0.
            direction = self.candidates["lagged" if b == "restored" else "current"]
            if target == 0:
                _need(np.all(a == 0), b, "zero-target delivered vector is nonzero")
            else:
                an = _norm(a)
                _need(an > 0 and abs(an-target)/target <= 1e-6 and _norm(a/an-direction/_norm(direction)) <= 1e-6,
                      b, "positive-target norm/direction/representability gate failed")
            self.candidates[b] = a
        self.endpoints, self.delta, self.data = endpoints, {}, {}
        self.z = (.001*.01)*self.theta
        for b in BRANCHES:
            if self.defined[b]:
                self.delta[b] = parameters(endpoints[b], "endpoints."+b)-self.theta
                self.data[b] = self.delta[b]+self.z
        self.edata = {b: 2*num.gamma(4, num.U64)*(np.abs(v)+np.abs(self.z))+8*num.H64 for b, v in self.delta.items()}
        self.factor = dict(direction="unavailable" if nc == 0 or nl == 0 else
                           "qualified" if _norm(self.candidates["current"]/nc-self.candidates["lagged"]/nl) > 2e-6 else "weak",
                           norm="unavailable" if max(nc, nl) == 0 else
                           "qualified" if abs(nc-nl)/max(nc, nl) > 2e-6 else "weak")
        _map(probes, PROBES, "probes", ordered=True)
        for key, count in zip(PROBES, counts):
            probe = _map(probes[key], ("inputs", "labels"), "probes."+key)
            _native(probe["inputs"], (count, d), "probes."+key+".inputs")
            y = probe["labels"]
            _need(isinstance(y, np.ndarray) and y.dtype == np.dtype(np.int64) and y.shape == (count,)
                  and np.all((y >= 0) & (y < c)), "probes."+key, "label shape/type/domain differs")
        _need(probes[PROBES[1]]["inputs"].tobytes() == probes[PROBES[2]]["inputs"].tobytes(),
              "probes", "clean/noisy training probe inputs differ")
        self.probes = probes

    def evaluate(self, parameters, probe, label):
        self.tick(label+".before")
        inputs = self.probes[probe]
        kwargs = {} if self.guard is None else {"guard": lambda phase: self.tick(label+"."+phase)}
        result = num.mlp_ce_gradient([parameters[k] for k in NAMES], inputs["inputs"], inputs["labels"],
                                    self.chunk if probe == "auxiliary_clean" else len(inputs["labels"]), **kwargs)
        self.tick(label+".after")
        return result

    def before_measurements(self, rows):
        _map(rows, ("probe_order", "probes"), "measurement_before")
        _need(type(rows["probe_order"]) is list and rows["probe_order"] == list(PROBES), "measurement_before", "probe order differs")
        _map(rows["probes"], PROBES, "measurement_before.probes", ordered=True)
        self.before_rows, self.before_ref, self.qp, self.qa, self.tq = rows["probes"], {}, {}, {}, {}
        for probe, row in self.before_rows.items():
            path = "before."+probe
            _map(row, ("sample_count", "sample_identity_sha256", "cpu64", "native32", "concordance", "auxiliary_chunks"), path)
            source = self.probes[probe]
            _need(type(row["sample_count"]) is int and row["sample_count"] == len(source["labels"])
                  and row["sample_identity_sha256"] == _hash_probe(source["inputs"], source["labels"]), path, "probe binding differs")
            cpu, native = _map(row["cpu64"], ("before_ce", "q", "q_norm"), path+".cpu64"), _map(row["native32"], ("before_ce",), path+".native32")
            reference = self.evaluate(self.before, probe, path)
            self.before_ref[probe] = reference
            self.qp[probe] = self.vector(cpu["q"], path+".q")
            self.qa[probe] = reference["gradient"]
            self.tq[probe] = num.gradient_tolerance(self.qp[probe], self.qa[probe])
            self.vector_check(self.qp[probe], self.qa[probe], self.tq[probe], path+".q_values")
            self.norm_check(cpu["q_norm"], self.qa[probe], self.tq[probe], path+".q_norm")
            self.scalar(cpu["before_ce"], reference["mean_ce"], num.loss_tolerance(cpu["before_ce"], reference["mean_ce"]), path+".CE")
            self.concordance(row["concordance"], cpu["before_ce"], native["before_ce"], path+".concordance")
            chunks = row["auxiliary_chunks"]
            _need(type(chunks) is list and len(chunks) == (10 if probe == "auxiliary_clean" else 0), path, "auxiliary chunk count differs")
            for j, item in enumerate(chunks):
                cp = path+f".chunk{j}"
                _map(item, ("chunk_index", "start", "end", "count", "input_sha256", "label_sha256", "cpu64_before_ce", "native32_before_ce", "concordance"), cp)
                self.chunk_meta(item, j, probe, cp)
                loss = reference["chunk_means"][j]
                self.scalar(item["cpu64_before_ce"], loss, num.loss_tolerance(item["cpu64_before_ce"], loss), cp+".CE")
                self.concordance(item["concordance"], item["cpu64_before_ce"], item["native32_before_ce"], cp+".concordance")
                self.report["completion_counts"]["auxiliary_before_chunks"] += 1
            self.report["completion_counts"]["before_probes"] += 1

    def branch_measurements(self, branches):
        _map(branches, BRANCHES, "branches", ordered=True)
        self.branch_rows, self.after_ref, self.producer_delta, self.producer_data = branches, {}, {}, {}
        self.outcomes, self.ceilings = {}, {}
        for b in BRANCHES:
            if not self.defined[b]:
                _need(branches[b] is None, "branches."+b, "domain-null branch must be None")
                continue
            row = _map(branches[b], ("displacement", "probes"), "branches."+b)
            dp = "branches."+b+".displacement"
            dis = _map(row["displacement"], ("delta", "delta_data", "delta_norm", "delta_squared_norm", "delta_data_norm", "delta_data_squared_norm", "delta_delta_data"), dp)
            pd, pdd = self.vector(dis["delta"], dp+".delta"), self.vector(dis["delta_data"], dp+".delta_data")
            self.producer_delta[b], self.producer_data[b] = pd, pdd
            self.vector_check(pd, self.delta[b], np.zeros(self.p), dp+".delta_values")
            self.vector_check(pdd, self.data[b], self.edata[b], dp+".delta_data_values")
            for label, vector, error in (("delta", self.delta[b], np.zeros(self.p)), ("delta_data", self.data[b], self.edata[b])):
                self.norm_check(dis[label+"_norm"], vector, error, dp+"."+label+"_norm")
                energy_bound = float(np.sum(error*(2*np.abs(vector)+error)))+num.dot_roundoff(vector, vector)
                _float(dis[label+"_squared_norm"], dp, True)
                self.scalar(dis[label+"_squared_norm"], float(vector@vector), energy_bound, dp+"."+label+"_squared_norm")
            self.geometry(dis["delta_delta_data"], self.delta[b], self.data[b], {b: True}, dp+".delta_delta_data",
                          right_error=self.edata[b])
            _map(row["probes"], PROBES, "branches."+b+".probes", ordered=True)
            self.after_ref[b], self.outcomes[b], self.ceilings[b] = {}, {}, {}
            for probe, measurement in row["probes"].items():
                path = "branches."+b+"."+probe
                _map(measurement, ("before_binding", "cpu64", "native32", "concordance", "auxiliary_chunks"), path)
                binding = _map(measurement["before_binding"], ("measurement_before_artifact_id", "probe_key", "before_ce_cpu64_sha256", "q_sha256"), path+".binding")
                artifact_id = binding["measurement_before_artifact_id"]
                _need(type(artifact_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", artifact_id) is not None,
                      path, "invalid bounded ASCII artifact ID")
                if self.report["binding_artifact_id"] is None:
                    self.report["binding_artifact_id"] = artifact_id
                before = self.before_rows[probe]
                _need(artifact_id == self.report["binding_artifact_id"] and binding["probe_key"] == probe
                      and binding["before_ce_cpu64_sha256"] == _hash_scalar(before["cpu64"]["before_ce"])
                      and binding["q_sha256"] == before["cpu64"]["q"]["sha256"], path, "before binding differs")
                cpu = _map(measurement["cpu64"], ("after_ce", "Y", "D", "Ddata", "R"), path+".cpu64")
                native = _map(measurement["native32"], ("after_ce", "Y"), path+".native32")
                for key, value in cpu.items():
                    _float(value, path+".cpu64."+key)
                for key, value in native.items():
                    _float(value, path+".native32."+key)
                reference = self.evaluate(self.endpoints[b], probe, path)
                self.after_ref[b][probe] = reference
                aa, ba = reference["mean_ce"], self.before_ref[probe]["mean_ce"]
                ap, bp = _float(cpu["after_ce"], path), before["cpu64"]["before_ce"]
                ta, tb = num.loss_tolerance(ap, aa), num.loss_tolerance(bp, ba)
                q, qp, eq = self.qa[probe], self.qp[probe], self.tq[probe]
                ya, da, dda = aa-ba, float(q@self.delta[b]), float(q@self.data[b])
                ty = ta+tb+max(num.sum_roundoff([ap, -bp]), num.sum_roundoff([aa, -ba]))
                td = float(eq@np.abs(self.delta[b]))+max(num.dot_roundoff(qp, pd), num.dot_roundoff(q, self.delta[b]))
                tdd = float(eq@np.abs(self.data[b])+np.maximum(np.abs(q), np.abs(qp))@self.edata[b])+max(num.dot_roundoff(qp, pdd), num.dot_roundoff(q, self.data[b]))
                tr = ty+td+max(num.sum_roundoff([cpu["Y"], -cpu["D"]]), num.sum_roundoff([ya, -da]))
                values, bounds = dict(Y=ya, D=da, Ddata=dda, R=ya-da), dict(Y=ty, D=td, Ddata=tdd, R=tr)
                self.outcomes[b][probe], self.ceilings[b][probe] = values, bounds
                self.scalar(ap, aa, ta, path+".after_ce")
                for key in values:
                    self.scalar(cpu[key], values[key], bounds[key], path+"."+key)
                for key, expected, ceiling in (("Y", ap-bp, num.sum_roundoff([ap, -bp])),
                        ("D", float(qp@pd), num.dot_roundoff(qp, pd)), ("Ddata", float(qp@pdd), num.dot_roundoff(qp, pdd)),
                        ("R", cpu["Y"]-cpu["D"], num.sum_roundoff([cpu["Y"], -cpu["D"]]))):
                    self.scalar(cpu[key], expected, ceiling, path+".stored_identity."+key)
                _map(measurement["concordance"], ("after", "Y"), path+".concordance")
                self.scalar(native["Y"], _float(native["after_ce"], path)-before["native32"]["before_ce"],
                            num.sum_roundoff([native["after_ce"], -before["native32"]["before_ce"]]), path+".native_Y")
                self.concordance(measurement["concordance"]["after"], ap, native["after_ce"], path+".concordance.after")
                self.concordance(measurement["concordance"]["Y"], cpu["Y"], native["Y"], path+".concordance.Y",
                                 5e-6*(max(1., abs(ap))+max(1., abs(bp)))+
                                 max(num.sum_roundoff([ap, -bp]), num.sum_roundoff([native["after_ce"], -before["native32"]["before_ce"]])))
                self.branch_chunks(b, probe, measurement["auxiliary_chunks"])
                self.report["completion_counts"]["branch_probes"] += 1
            self.report["completion_counts"]["defined_branches"] += 1

    def branch_chunks(self, b, probe, chunks):
        path = "branches."+b+"."+probe+".chunks"
        _need(type(chunks) is list and len(chunks) == (10 if probe == "auxiliary_clean" else 0), path, "chunk count differs")
        for j, row in enumerate(chunks):
            cp = path+str(j)
            _map(row, ("chunk_index", "start", "end", "count", "input_sha256", "label_sha256", "cpu64_after_ce", "cpu64_Y", "native32_after_ce", "native32_Y", "concordance"), cp)
            self.chunk_meta(row, j, probe, cp)
            before = self.before_rows[probe]["auxiliary_chunks"][j]
            aa, ba = self.after_ref[b][probe]["chunk_means"][j], self.before_ref[probe]["chunk_means"][j]
            ap, bp = _float(row["cpu64_after_ce"], cp), before["cpu64_before_ce"]
            ceiling = num.loss_tolerance(ap, aa)+num.loss_tolerance(bp, ba)+max(num.sum_roundoff([ap, -bp]), num.sum_roundoff([aa, -ba]))
            self.scalar(ap, aa, num.loss_tolerance(ap, aa), cp+".after")
            self.scalar(row["cpu64_Y"], aa-ba, ceiling, cp+".Y")
            self.scalar(row["cpu64_Y"], ap-bp, num.sum_roundoff([ap, -bp]), cp+".stored_Y")
            an, bn = _float(row["native32_after_ce"], cp), before["native32_before_ce"]
            self.scalar(row["native32_Y"], an-bn, num.sum_roundoff([an, -bn]), cp+".native_Y")
            _map(row["concordance"], ("after", "Y"), cp+".concordance")
            self.concordance(row["concordance"]["after"], ap, an, cp+".concordance.after")
            self.concordance(row["concordance"]["Y"], row["cpu64_Y"], row["native32_Y"], cp+".concordance.Y",
                             5e-6*(max(1., abs(ap))+max(1., abs(bp)))+
                             max(num.sum_roundoff([ap, -bp]), num.sum_roundoff([an, -bn])))
            self.report["completion_counts"]["auxiliary_branch_chunks"] += 1

    def pairs(self, pairs):
        expected = {a+"__"+b: (a, b) for a, b in itertools.combinations(BRANCHES, 2)}
        _map(pairs, expected, "branch_pairs", ordered=True)
        for name, (a, b) in expected.items():
            path = "pairs."+name
            row = _map(pairs[name], ("defined", "defined_mask", "reason", "delivered_gradient", "delta", "delta_data", "full_data_distance_discrepancy", "full_data_rounding_ceiling", "rounding_status"), path)
            mask = {a: self.defined[a], b: self.defined[b]}
            self.mask(row["defined_mask"], mask, path+".mask")
            defined = all(mask.values())
            _need(type(row["defined"]) is bool and row["defined"] == defined, path, "pair defined flag differs")
            for label, vectors in (("delivered_gradient", self.candidates), ("delta", self.delta), ("delta_data", self.data)):
                error_a = self.edata.get(a) if label == "delta_data" else None
                error_b = self.edata.get(b) if label == "delta_data" else None
                self.geometry(row[label], vectors.get(a), vectors.get(b), mask, path+"."+label, error_a, error_b)
            if not defined:
                _need(row["reason"] == "domain_undefined_required_branch" and row["full_data_distance_discrepancy"] is None
                      and row["full_data_rounding_ceiling"] is None and row["rounding_status"] == "domain_undefined", path, "undefined pair payload differs")
                self.report["completion_counts"]["pairs"] += 1
                continue
            _need(row["reason"] is None, path, "defined pair reason differs")
            pd, pdd = self.producer_delta[a]-self.producer_delta[b], self.producer_data[a]-self.producer_data[b]
            ceiling = _norm(self.edata[a]+self.edata[b]+_vsum([self.producer_delta[a], -self.producer_delta[b]])+
                            _vsum([self.producer_data[a], -self.producer_data[b]]))+_nround(pd)+_nround(pdd)
            self.arithmetic(row["full_data_distance_discrepancy"], abs(row["delta"]["distance"]-row["delta_data"]["distance"]),
                            path+".discrepancy_binding", nonnegative=True)
            self.scalar(row["full_data_distance_discrepancy"], abs(_norm(pd)-_norm(pdd)),
                        _nround(pd)+_nround(pdd), path+".discrepancy_independent")
            self.arithmetic(row["full_data_rounding_ceiling"], ceiling, path+".ceiling", nonnegative=True)
            _need(row["rounding_status"] == ("pass" if abs(_norm(pd)-_norm(pdd)) <= ceiling else "fatal_validation"), path, "pair rounding status differs")
            if row["rounding_status"] != "pass":
                self.failure(path, "full/data pair ceiling exceeded")
            self.report["completion_counts"]["pairs"] += 1

    def scalar_record(self, row, components, direct, identity_ceiling, path):
        _map(row, ("component_values", "direct_value", "component_sum_value", "identity_abs_discrepancy", "identity_rounding_ceiling", "identity_status"), path)
        _map(row["component_values"], components, path+".components", ordered=True)
        for key, value in components.items():
            _need(type(row["component_values"][key]) is float and row["component_values"][key] == value, path, "component binding differs")
        component_sum = math.fsum(self.coeff[b]*value for b, value in components.items())
        self.scalar(row["direct_value"], direct, identity_ceiling, path+".direct_route")
        self.scalar(row["component_sum_value"], component_sum, identity_ceiling, path+".component_route")
        discrepancy = abs(row["direct_value"]-row["component_sum_value"])
        self.arithmetic(row["identity_abs_discrepancy"], discrepancy, path+".identity_discrepancy", nonnegative=True)
        self.arithmetic(row["identity_rounding_ceiling"], identity_ceiling, path+".identity_ceiling", nonnegative=True)
        status = "pass" if discrepancy <= identity_ceiling else "fatal_validation"
        _need(row["identity_status"] == status, path, "identity status differs")
        if status != "pass":
            self.failure(path, "direct/component identity ceiling exceeded")

    def y_identity(self, after, before, components):
        coeff = self.coeff
        return (num.sum_roundoff([coeff[b]*after[b] for b in coeff])+
                num.sum_roundoff([coeff[b]*components[b] for b in coeff])+
                sum(abs(coeff[b])*num.sum_roundoff([after[b], -before]) for b in coeff))

    def contrast_probe(self, row, probe, path, chunk_index=None):
        coeff, w, wd, ew, ewd, ec = self.coeff, self.w, self.wd, self.ew, self.ewd, self.ec
        chunk = chunk_index is not None
        allowed = ("Y",) if chunk else ("Y", "D", "Ddata", "R")
        _map(row["cpu64"], allowed, path+".cpu64"); _map(row["native32"], ("Y",), path+".native32")
        _map(row["concordance"], ("Y",), path+".concordance")
        def values(native=False):
            before = self.before_rows[probe]
            if chunk:
                before_value = before["auxiliary_chunks"][chunk_index]["native32_before_ce" if native else "cpu64_before_ce"]
                blocks = {b: self.branch_rows[b]["probes"][probe]["auxiliary_chunks"][chunk_index] for b in coeff}
                after = {b: blocks[b]["native32_after_ce" if native else "cpu64_after_ce"] for b in coeff}
                component = {b: blocks[b]["native32_Y" if native else "cpu64_Y"] for b in coeff}
            else:
                kind = "native32" if native else "cpu64"
                before_value = before[kind]["before_ce"]
                after = {b: self.branch_rows[b]["probes"][probe][kind]["after_ce"] for b in coeff}
                component = {b: self.branch_rows[b]["probes"][probe][kind]["Y"] for b in coeff}
            return after, before_value, component
        after, before, cy = values()
        native_after, native_before, native_y = values(True)
        direct_y = math.fsum(coeff[b]*after[b] for b in coeff)
        direct_native = math.fsum(coeff[b]*native_after[b] for b in coeff)
        ey = self.y_identity(after, before, cy)
        self.scalar_record(row["cpu64"]["Y"], cy, direct_y, ey, path+".Y")
        self.scalar_record(row["native32"]["Y"], native_y, direct_native, self.y_identity(native_after, native_before, native_y), path+".nativeY")
        self.concordance(row["concordance"]["Y"], row["cpu64"]["Y"]["direct_value"], row["native32"]["Y"]["direct_value"],
                         path+".concordance.Y", math.fsum(abs(coeff[b])*5e-6*max(1., abs(after[b])) for b in coeff)+
                         max(num.sum_roundoff([coeff[b]*after[b] for b in coeff]),
                             num.sum_roundoff([coeff[b]*native_after[b] for b in coeff])))
        auditor_after = {b: self.after_ref[b][probe]["chunk_means"][chunk_index] if chunk else self.after_ref[b][probe]["mean_ce"] for b in coeff}
        ya = math.fsum(coeff[b]*auditor_after[b] for b in coeff)
        ty = sum(abs(coeff[b])*num.loss_tolerance(after[b], auditor_after[b]) for b in coeff)+max(
            num.sum_roundoff([coeff[b]*after[b] for b in coeff]), num.sum_roundoff([coeff[b]*auditor_after[b] for b in coeff]))
        self.scalar(row["cpu64"]["Y"]["direct_value"], ya, ty, path+".Y.independent", comparison=True, sign=True)
        if chunk:
            return
        qp, qa = self.qp[probe], self.qa[probe]
        cd = {b: self.branch_rows[b]["probes"][probe]["cpu64"]["D"] for b in coeff}
        cdd = {b: self.branch_rows[b]["probes"][probe]["cpu64"]["Ddata"] for b in coeff}
        cr = {b: self.branch_rows[b]["probes"][probe]["cpu64"]["R"] for b in coeff}
        dp, ddp = float(qp@w), float(qp@wd)
        ed = num.dot_roundoff(qp, w)+sum(abs(coeff[b])*num.dot_roundoff(qp, self.producer_delta[b]) for b in coeff)+num.sum_roundoff([coeff[b]*cd[b] for b in coeff])+float(np.abs(qp)@ew)
        edd = num.dot_roundoff(qp, wd)+sum(abs(coeff[b])*num.dot_roundoff(qp, self.producer_data[b]) for b in coeff)+num.sum_roundoff([coeff[b]*cdd[b] for b in coeff])+float(np.abs(qp)@(ewd+ec))
        er = ey+ed+num.sum_roundoff([direct_y, -dp])+sum(abs(coeff[b])*num.sum_roundoff([cy[b], -cd[b]]) for b in coeff)+num.sum_roundoff([coeff[b]*cr[b] for b in coeff])
        for key, component, direct, ceiling in (("D", cd, dp, ed), ("Ddata", cdd, ddp, edd), ("R", cr, direct_y-dp, er)):
            self.scalar_record(row["cpu64"][key], component, direct, ceiling, path+"."+key)
        wa, wda = _combine(self.delta, coeff), _combine(self.data, coeff)
        ewa, ewda = _vsum([coeff[b]*self.delta[b] for b in coeff]), _vsum([coeff[b]*self.data[b] for b in coeff])
        da, dda = float(qa@wa), float(qa@wda)
        td = float(self.tq[probe]@np.abs(wa)+np.maximum(np.abs(qp), np.abs(qa))@np.maximum(ew, ewa))+max(num.dot_roundoff(qp, w), num.dot_roundoff(qa, wa))
        tdd = float(self.tq[probe]@np.abs(wda)+np.maximum(np.abs(qp), np.abs(qa))@(np.maximum(ewd, ewda)+ec))+max(num.dot_roundoff(qp, wd), num.dot_roundoff(qa, wda))
        tr = ty+td+max(num.sum_roundoff([direct_y, -dp]), num.sum_roundoff([ya, -da]))
        for key, value, ceiling in (("D", da, td), ("Ddata", dda, tdd), ("R", ya-da, tr)):
            self.scalar(row["cpu64"][key]["direct_value"], value, ceiling, path+"."+key+".independent", comparison=True, sign=True)

    def contrasts(self, rows):
        _map(rows, COEFFICIENTS, "contrasts", ordered=True)
        for name, coeff in COEFFICIENTS.items():
            self.tick("contrast."+name)
            path = "contrasts."+name
            row = _map(rows[name], ("coefficients", "branch_defined_mask", "defined", "reason", "factor_requirements", "factor_leverage", "vector", "probes"), path)
            _map(row["coefficients"], coeff, path+".coefficients", ordered=True)
            _need(all(type(row["coefficients"][b]) is int and row["coefficients"][b] == c for b, c in coeff.items()), path, "contrast coefficients differ")
            self.mask(row["branch_defined_mask"], self.defined, path+".mask")
            defined = all(self.defined[b] for b in coeff)
            _need(type(row["defined"]) is bool and row["defined"] == defined, path, "contrast defined flag differs")
            required = ["direction", "norm"] if name == "interaction" else ["direction"] if name.startswith("direction_") else ["norm"] if name.startswith("norm_") else []
            _need(type(row["factor_requirements"]) is list and row["factor_requirements"] == required, path, "factor requirements differ")
            _map(row["factor_leverage"], ("direction", "norm"), path+".leverage")
            expected = {key: self.factor[key] if defined and key in required else
                        "unavailable" if key in required else "not_required" for key in ("direction", "norm")}
            _need(row["factor_leverage"] == expected, path, "factor leverage differs")
            if any(value == "weak" for value in expected.values()):
                self.report["weak_factor_contrasts"].append(name)
            if not defined:
                _need(row["reason"] == "domain_undefined_required_branch" and row["vector"] is None and row["probes"] is None, path, "undefined contrast payload differs")
                self.report["completion_counts"]["contrasts"] += 1
                continue
            _need(row["reason"] is None, path, "defined contrast reason differs")
            self.coeff = coeff
            self.w, self.wd = _combine(self.producer_delta, coeff), _combine(self.producer_data, coeff)
            self.ew, self.ewd = _vsum([coeff[b]*self.producer_delta[b] for b in coeff]), _vsum([coeff[b]*self.producer_data[b] for b in coeff])
            self.ec = sum(abs(coeff[b])*self.edata[b] for b in coeff)
            vectors = _map(row["vector"], ("delta", "delta_data", "full_data_agreement"), path+".vector")
            for key, value in (("delta", self.w), ("delta_data", self.wd)):
                summary = _map(vectors[key], ("representation", "canonical_sha256", "norm", "squared_norm", "component_count"), path+"."+key)
                _need(summary["representation"] == "derived_from_bound_branch_vectors" and summary["canonical_sha256"] == _hash_array(value)
                      and type(summary["component_count"]) is int and summary["component_count"] == self.p, path, "contrast vector binding differs")
                self.norm_check(summary["norm"], value, np.zeros(self.p), path+"."+key+".norm")
                _float(summary["squared_norm"], path, True)
                self.scalar(summary["squared_norm"], float(value@value), num.dot_roundoff(value, value), path+"."+key+".energy")
            agreement = _map(vectors["full_data_agreement"], ("difference_norm", "rounding_ceiling", "status"), path+".agreement")
            difference = self.w-self.wd
            ceiling = _norm(self.ew+self.ewd+self.ec+_vsum([self.w, -self.wd]))+_nround(difference)
            self.arithmetic(agreement["difference_norm"], _norm(difference), path+".agreement.discrepancy", nonnegative=True)
            self.arithmetic(agreement["rounding_ceiling"], ceiling, path+".agreement.ceiling", nonnegative=True)
            _need(agreement["status"] == ("pass" if _norm(difference) <= ceiling else "fatal_validation"), path, "agreement status differs")
            if agreement["status"] != "pass":
                self.failure(path, "full/data vector ceiling exceeded")
            _map(row["probes"], PROBES, path+".probes", ordered=True)
            for probe, item in row["probes"].items():
                pp = path+"."+probe
                _map(item, ("cpu64", "native32", "concordance", "auxiliary_chunks"), pp)
                self.contrast_probe(item, probe, pp)
                chunks = item["auxiliary_chunks"]
                _need(type(chunks) is list and len(chunks) == (10 if probe == "auxiliary_clean" else 0), pp, "contrast chunk count differs")
                for j, chunk in enumerate(chunks):
                    cp = pp+f".chunk{j}"
                    _map(chunk, ("chunk_index", "start", "end", "count", "input_sha256", "label_sha256", "cpu64", "native32", "concordance"), cp)
                    self.chunk_meta(chunk, j, probe, cp)
                    self.contrast_probe(chunk, probe, cp, j)
                    self.report["completion_counts"]["auxiliary_contrast_chunks"] += 1
            self.report["completion_counts"]["contrasts"] += 1
            self.report["completion_counts"]["defined_contrasts"] += 1


def audit(before_parameters, endpoints, candidates, probes, measurement, *,
          profile="fixture_tiny_mlp_cpu_v1", guard=None):
    """Return retained scalar diagnostics; malformed values return fatal_validation.

    Exact six-branch/four-probe order, profile shapes/counts, coefficient order,
    masks, vector/scalar bindings and every nested schema are checked. No silent
    tolerance enlargement, available-case averaging or scientific null fallback.
    guard(label) is optional and may raise to abort resource work; guard exceptions
    are not converted into allowed domain nulls. This is not the full artifact audit.
    """
    _need(guard is None or callable(guard), "guard", "must be callable or None")
    runner = _Audit(profile, guard)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            runner.tick("measurement_audit.begin")
            runner.inputs(before_parameters, endpoints, candidates, probes)
            _map(measurement, ("measurement_before", "branches", "comparisons"), "measurement")
            runner.before_measurements(measurement["measurement_before"])
            runner.branch_measurements(measurement["branches"])
            comparisons = _map(measurement["comparisons"], ("branch_pairs", "contrasts"), "comparisons")
            runner.pairs(comparisons["branch_pairs"])
            runner.contrasts(comparisons["contrasts"])
            runner.tick("measurement_audit.complete")
            runner.report["audit_complete"] = True
    except _GuardFailure as error:
        raise error.original
    except (ValueError, OverflowError, FloatingPointError) as error:
        runner.failure("validation", str(error))
    return runner.report
