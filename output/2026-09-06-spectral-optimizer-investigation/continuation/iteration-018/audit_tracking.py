#!/usr/bin/env python3
"""I18 saved-array audit; NumPy/stdlib only, no native observer or RNG replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SEEDS = tuple(range(18000, 18032))
DRIFTS = (0.0, .01, .03)
ROTATIONS = (0, 1)
N_STEPS = 4000
BETA, RHO = .99, .9
WINDOWS = (("startup", 1, 100), ("transition", 101, 1000), ("late", 1001, 4000))
SHA = re.compile(r"[0-9a-f]{64}")
RTOL, ATOL = 2e-11, 5e-12
ARRAY_CAP, ROOT_CAP = 1024**3, 2 * 1024**3
SOURCE_PATHS = ("spectral_filter.py",) + tuple(str((HERE / name).relative_to(REPO)) for name in
    ("tracking_core.py", "test_tracking_core.py", "run_tracking.py", "test_tracking_runner.py",
     "protocol.md", "best-practices-check.md"))
FILTER_CONFIGURATION = {"rank": 1, "decay": .99, "warmup": 0, "filter_strength": 1.0,
    "adaptive": "none", "normalize": "none", "weighting": "hard", "stable_update": True,
    "relative_eig_tol": 1e-8, "absolute_eig_floor": 0.0, "stabilize_every": 100}


def policy_names(drift):
    names = ["raw", "ema_q0p9", "ema_q0p99", "ema_q0p999"]
    if drift != 0:
        names.append("ema_optimal_oracle")
    return names + ["scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1", "oracle_useful",
                    "oracle_nuisance", "native_i17", "native_cp"]


def array_shapes(horizon, policies):
    n, p = horizon, len(policies)
    return {"noise": (n,2), "epsilon": (n,2), "g": (n,2), "s": (n,2), "mu": (n,2), "A": (n,2,2),
        "native_delivery": (n,2), "native_h": (n,2), "full_moment": (n,2,2),
        "full_action": (n,2,2), "full_gap": (n,), "full_basis_present": (n,),
        "full_basis_V": (n,2), "basis_present": (n,), "basis_V": (n,2), "basis_S": (n,),
        "output": (n,p,2), "native_i17_buffer": (n,2), "scalar_buffer": (n,4,2),
        "native_action_raw_error": (n,2), "native_i17_response_residual": (n,2),
        "native_cp_response_residual": (n,2), "scalar_response_residual": (n,4,2),
        "full_moment_residual": (n,2,2), "native_action_idempotence_error": (n,),
        "native_basis_orthogonality_error": (n,), "native_useful_squared_alignment": (n,),
        "full_useful_squared_alignment": (n,)}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=lambda value:
            (_ for _ in ()).throw(ValueError("nonfinite JSON constant: " + value)))


class Audit:
    def __init__(self):
        self.errors = []
        self.checks = 0
        self.numeric_values = 0
        self.maximum_absolute_error = {}
        self.hash_files = 0
        self.hash_bytes = 0
        self.stream_id = None

    def require(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append((self.stream_id + ": " if self.stream_id else "") + message)
        return bool(condition)

    def close(self, observed, expected, label, *, rtol=RTOL, atol=ATOL):
        observed, expected = np.asarray(observed), np.asarray(expected)
        if not self.require(observed.shape == expected.shape, label + ": shape differs"):
            return False
        self.numeric_values += observed.size
        if not self.require(np.isfinite(observed).all() and np.isfinite(expected).all(), label + ": nonfinite"):
            return False
        difference = np.abs(observed - expected)
        self.maximum_absolute_error[label] = max(self.maximum_absolute_error.get(label, 0.0),
                                                 float(np.max(difference, initial=0)))
        return self.require(np.all(difference <= atol + rtol * np.abs(expected)), label + ": numerical identity differs")


def rotation(index):
    if index == 0:
        return np.eye(2, dtype=np.float64)
    if index != 1:
        raise ValueError("unregistered rotation")
    angle = math.pi / 4
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]], dtype=np.float64)


def optimal_decay(drift):
    if drift == 0:
        return None
    if drift not in DRIFTS:
        raise ValueError("unregistered drift")
    low, high = 0.0, 10000.0
    for _ in range(100):
        lag = (low + high) / 2
        derivative = 2 * drift * drift * lag - 10 / (2 * lag + 1)**2
        if derivative > 0:
            high = lag
        else:
            low = lag
    lag = (low + high) / 2
    return lag / (1 + lag)


def error_metrics(delivery, signal):
    error = np.asarray(delivery, dtype=np.float64) - np.asarray(signal, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 2 or len(error) == 0 or not np.isfinite(error).all():
        raise ValueError("finite nonempty (time,2) error array required")
    mean = error.mean(axis=0)
    mse = float(np.mean(np.sum(error * error, axis=1)))
    squared_bias = float(mean @ mean)
    centered = error - mean
    dispersion = float(np.mean(np.sum(centered * centered, axis=1)))
    return {"observations": len(error), "mse": mse, "mean_error_vector": mean.tolist(),
            "squared_mean_error": squared_bias, "temporal_error_dispersion": dispersion,
            "mse_decomposition_residual": mse - squared_bias - dispersion,
            "dispersion_is_not_independent_sample_uncertainty": True}


def seed_summary(values):
    """Missing seeds never turn into survivor means; time samples are not n."""
    complete = set(values) == {str(seed) for seed in SEEDS} and all(
        type(value) in (int, float) and math.isfinite(value) for value in values.values())
    ordered = [values.get(str(seed)) for seed in SEEDS]
    if not complete:
        return {"per_seed": values, "available": False, "n_independent_seeds": len(SEEDS),
                "mean": None, "standard_error": None, "positive_seeds": None,
                "negative_seeds": None, "zero_seeds": None, "no_survivor_averaging": True}
    array = np.asarray(ordered, dtype=np.float64)
    return {"per_seed": values, "available": True, "n_independent_seeds": len(SEEDS),
            "mean": float(array.mean()), "standard_error": float(array.std(ddof=1) / math.sqrt(len(SEEDS))),
            "positive_seeds": int(np.sum(array > 0)), "negative_seeds": int(np.sum(array < 0)),
            "zero_seeds": int(np.sum(array == 0)), "no_survivor_averaging": True}


def population_predictions():
    rows = []
    for drift in DRIFTS:
        first = (BETA / (1-BETA) * drift)**2 + 2 * BETA**2 / (1+BETA)
        second = 8 * BETA**2 / (1+BETA)
        risks = {"oracle_useful": (RHO / (1-RHO) * drift)**2 + (1-RHO)/(1+RHO)
                     + 4*(1-BETA)/(1+BETA),
                 "oracle_nuisance": (BETA / (1-BETA) * drift)**2 + (1-BETA)/(1+BETA)
                     + 4*(1-RHO)/(1+RHO)}
        for q in (.9, .99, .999):
            risks["uniform_" + str(q)] = (q/(1-q)*drift)**2 + 5*(1-q)/(1+q)
        q = optimal_decay(drift)
        if q is not None:
            risks["oracle_uniform_optimal"] = (q/(1-q)*drift)**2 + 5*(1-q)/(1+q)
        rows.append({"drift": drift, "identity_rotation_population_residual_moment_diagonal": [first, second],
                     "population_prefers_generating_axis": first > second,
                     "asymptotic_fixed_route_risks": risks, "oracle_optimal_decay": q,
                     "scope": "fixed-route population theory, not finite native empirical performance"})
    return rows


def validate_metadata(metadata, drift, ri, horizon, audit):
    policies = policy_names(drift)
    shapes = array_shapes(horizon, policies)
    expected = {"schema": "i18_tracking_stream_metadata_v1", "array_schema": "i18_tracking_arrays_v1",
        "horizon": horizon, "dimension": 2, "drift": drift, "rotation_radians": 0.0 if ri == 0 else math.pi/4,
        "rho": .9, "beta": .99, "policy_names": policies, "policy_count": len(policies),
        "optimal_ema_decay": optimal_decay(drift), "full_gap_relative_tolerance": 1e-10,
        "filter_configuration": FILTER_CONFIGURATION,
        "initialization": {"all_policy_outputs_at_t1_equal_g1": True,
            "native_i17_buffer": "solve(I-rho*A_1,g_1)", "native_cp_delivery": "g_1",
            "scalar_buffer": "g_1/(1-rho*k)", "uniform_ema": "g_1",
            "full_moment": "zero until first nonzero post-ingest residual, then z*z^T unscaled"},
        "native_absent_basis_action": "identity", "full_absent_basis_arrays": "zero-filled with full_basis_present=false",
        "array_order": list(shapes), "array_shapes": {key: list(value) for key,value in shapes.items()},
        "array_dtypes": {key: "bool" if key in ("basis_present", "full_basis_present") else "float64" for key in shapes},
        "noise_generated_by_core": False,
        "noise_array_semantics": "unrotated shared input draw",
        "epsilon_array_semantics": "rotation-applied disturbance used in g=s+epsilon"}
    audit.require(metadata == expected, "core metadata/schema/registered configuration differs")
    return expected


def validate_arrays(arrays, metadata, drift, ri, audit, *, horizon=N_STEPS):
    """Check saved-state equations, not a native observer/RNG reconstruction."""
    expected = validate_metadata(metadata, drift, ri, horizon, audit)
    shapes, policies = array_shapes(horizon, policy_names(drift)), expected["policy_names"]
    if not audit.require(type(arrays) is dict and list(arrays) == list(shapes), "array membership/order differs"):
        return None
    for name, shape in shapes.items():
        value = arrays[name]
        dtype = np.dtype(np.bool_ if name in ("basis_present", "full_basis_present") else np.float64)
        if not audit.require(type(value) is np.ndarray and value.shape == shape and value.dtype == dtype
                and np.isfinite(value).all(), "array shape/dtype/finiteness differs: " + name):
            return None
    g, s, mu, action = (arrays[name] for name in ("g", "s", "mu", "A"))
    q, eye = rotation(ri), np.eye(2)
    signal = np.column_stack((drift * np.arange(horizon, dtype=np.float64), np.zeros(horizon))) @ q.T
    audit.close(s, signal, "signal construction")
    audit.close(arrays["epsilon"], arrays["noise"] @ q.T, "rotation-applied disturbance")
    audit.close(g, s + arrays["epsilon"], "signal plus disturbance")
    audit.close(mu[0], g[0], "first native mean")
    audit.close(mu[1:], BETA * mu[:-1] + (1-BETA)*g[1:], "post-ingest mean recurrence")
    z = g - mu
    outer = np.einsum("ni,nj->nij", z, z)
    present, vectors = arrays["basis_present"], arrays["basis_V"]
    expected_action = np.einsum("ni,nj->nij", vectors, vectors)
    expected_action[~present] = eye
    audit.close(action, expected_action, "native action from saved basis/fallback")
    audit.require(not present[0], "initial native basis must be absent")
    audit.close(vectors[~present], np.zeros_like(vectors[~present]), "absent native basis zero fill")
    audit.close(arrays["basis_S"][~present], np.zeros(np.sum(~present)), "absent native singular value zero fill")
    audit.require(np.all(arrays["basis_S"][present] > 0), "native singular value must be positive when present")
    audit.close(np.sum(vectors[present]**2, axis=1), np.ones(np.sum(present)), "native basis unit length")
    # This checks the one-step represented covariance eigen-relation from saved
    # predecessor V/S, not native's implementation, eigensolver or observer replay.
    covariance = np.zeros((horizon,2,2), dtype=np.float64)
    for t in range(1, horizon):
        if present[t-1]:
            prior = arrays["basis_S"][t-1]**2 * np.outer(vectors[t-1], vectors[t-1])
            covariance[t] = BETA*prior + (1-BETA)*outer[t]
        else:
            covariance[t] = outer[t]
    eigenvalues = np.linalg.eigvalsh(covariance)
    audit.close(arrays["basis_S"][present]**2, eigenvalues[present,-1], "native represented leading eigenvalue")
    audit.close(np.einsum("nij,nj->ni", covariance[present], vectors[present]),
                arrays["basis_S"][present,None]**2*vectors[present], "native represented leading eigenvector")
    audit.require(np.array_equal(present, eigenvalues[:,-1] > 0),
                  "native absence/positive leading eigenvalue classification differs")
    native_expected = np.einsum("nij,nj->ni", action, g)
    audit.close(arrays["native_delivery"], native_expected, "native raw delivery")
    audit.close(arrays["native_action_raw_error"], arrays["native_delivery"]-native_expected, "native delivery residual")
    h = arrays["native_delivery"] + mu - np.einsum("nij,nj->ni", action, mu)
    audit.close(arrays["native_h"], h, "mean-restored native input")
    full = arrays["full_moment"]
    full_expected = np.zeros_like(full)
    initialized = False
    for t in range(horizon):
        if initialized:
            full_expected[t] = BETA*full[t-1] + (1-BETA)*outer[t]
        elif np.any(z[t] != 0):
            full_expected[t] = outer[t]
            initialized = True
    audit.close(full, full_expected, "full residual moment recurrence")
    audit.close(arrays["full_moment_residual"], full-full_expected, "full moment saved residual")
    audit.close(full, np.swapaxes(full,1,2), "full moment symmetry")
    vals, vecs = np.linalg.eigh((full + np.swapaxes(full,1,2))/2)
    gap = vals[:,-1]-vals[:,0]
    available = gap > 1e-10*np.maximum(1.0, np.abs(vals[:,-1]))
    audit.close(arrays["full_gap"], gap, "full moment eigenvalue gap")
    audit.require(np.array_equal(arrays["full_basis_present"], available), "full moment unavailable/tied classification differs")
    full_vectors = arrays["full_basis_V"]
    full_expected_action = np.zeros_like(action)
    full_expected_action[available] = np.einsum("ni,nj->nij", vecs[available,:,-1], vecs[available,:,-1])
    audit.close(arrays["full_action"], full_expected_action, "full moment leading action", rtol=1e-8, atol=1e-10)
    audit.close(arrays["full_action"], np.einsum("ni,nj->nij", full_vectors, full_vectors), "full action from saved eigenvector")
    audit.close(full_vectors[~available], np.zeros_like(full_vectors[~available]), "unavailable full direction zero fill")
    outputs = {policy: arrays["output"][:,column,:] for column,policy in enumerate(policies)}
    audit.require(np.array_equal(arrays["output"][0], np.broadcast_to(g[0], (len(policies),2))),
                  "all estimators initial delivery must be exactly g1")
    audit.close(outputs["raw"], g, "raw control")
    ema_pairs = [("ema_q0p9", .9), ("ema_q0p99", .99), ("ema_q0p999", .999)]
    optimum = optimal_decay(drift)
    if optimum is not None:
        ema_pairs.append(("ema_optimal_oracle", optimum))
    for name, decay in ema_pairs:
        audit.close(outputs[name][1:], decay*outputs[name][:-1]+(1-decay)*g[1:], name + " recurrence")
    ks = np.array([0., .5, .9, 1.])
    factors = 1-RHO*ks
    buffers = arrays["scalar_buffer"]
    audit.close(buffers[0], g[0][None,:]/factors[:,None], "scalar initial buffers")
    scalar_expected = RHO*ks[None,:,None]*buffers[:-1] + ks[None,:,None]*g[1:,None,:] + (1-ks)[None,:,None]*mu[1:,None,:]
    audit.close(buffers[1:], scalar_expected, "scalar buffer recurrence")
    audit.close(arrays["scalar_response_residual"][1:], buffers[1:]-scalar_expected, "scalar saved residual")
    audit.close(arrays["scalar_response_residual"][0], np.zeros((4,2)), "scalar initial residual")
    for position, name in enumerate(("scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1")):
        audit.close(outputs[name], factors[position]*buffers[:,position], name + " normalized delivery")
    audit.close(outputs["scalar_k0"], outputs["ema_q0p99"], "k0 slow EMA duplicate parity")
    audit.close(outputs["scalar_k1"], outputs["ema_q0p9"], "k1 fast EMA duplicate parity")
    projector = np.outer(q[:,0],q[:,0])
    audit.close(outputs["oracle_useful"], outputs["ema_q0p9"]@projector.T+outputs["ema_q0p99"]@(eye-projector).T,
                "useful direction oracle construction")
    audit.close(outputs["oracle_nuisance"], outputs["ema_q0p9"]@(eye-projector).T+outputs["ema_q0p99"]@projector.T,
                "nuisance direction oracle construction")
    inverse_gain = eye[None,:,:] - RHO*action
    b = arrays["native_i17_buffer"]
    audit.close(inverse_gain[0] @ b[0], g[0], "native i17 initial solve")
    expected_b = RHO*np.einsum("nij,nj->ni", action[1:], b[:-1]) + h[1:]
    audit.close(b[1:], expected_b, "native i17 buffer recurrence")
    audit.close(outputs["native_i17"], np.einsum("nij,nj->ni", inverse_gain, b), "native i17 normalized delivery")
    expected_cp = RHO*np.einsum("nij,nj->ni", action[1:], outputs["native_cp"][:-1]) + np.einsum("nij,nj->ni", inverse_gain[1:], h[1:])
    audit.close(outputs["native_cp"][1:], expected_cp, "native CP recurrence")
    delta = outputs["native_i17"] - outputs["native_cp"]
    transport = -RHO**2 * np.einsum("nij,njk,nk->ni", action[1:], action[1:]-action[:-1], b[:-1])
    audit.close(delta[1:], RHO*np.einsum("nij,nj->ni", action[1:], delta[:-1])+transport,
                "native i17 versus CP response transport identity")
    for name, difference in (("native_i17_response_residual", b[1:]-expected_b),
                             ("native_cp_response_residual", outputs["native_cp"][1:]-expected_cp)):
        audit.close(arrays[name][1:], difference, name)
        audit.close(arrays[name][0], np.zeros(2), name + " initialization")
    audit.close(arrays["native_action_idempotence_error"], np.linalg.norm(action@action-action, axis=(1,2)), "native idempotence diagnostic")
    orthogonality = np.abs(np.sum(vectors*vectors,axis=1)-1)
    orthogonality[~present] = 0
    audit.close(arrays["native_basis_orthogonality_error"], orthogonality, "native orthogonality diagnostic")
    alignment = (vectors @ q[:,0])**2
    alignment[~present] = 0.0
    audit.close(arrays["native_useful_squared_alignment"], alignment, "native generating-axis alignment")
    audit.close(arrays["full_useful_squared_alignment"], (full_vectors @ q[:,0])**2, "full generating-axis alignment")
    return outputs


def stream_metrics(arrays, policies, seed, di, ri, *, windows=WINDOWS):
    rows, directions = [], []
    for window, first, last in windows:
        sl = slice(first-1, last)
        if last > len(arrays["g"]):
            raise ValueError("metric window exceeds saved coverage")
        for column, policy in enumerate(policies):
            value = error_metrics(arrays["output"][sl,column], arrays["s"][sl])
            rows.append({"seed": seed, "drift_index": di, "rotation_index": ri, "window": window,
                         "policy": policy, **value})
        present = arrays["basis_present"][sl]
        full_present = arrays["full_basis_present"][sl]
        directions.append({"seed": seed, "drift_index": di, "rotation_index": ri, "window": window,
            "observations": last-first+1, "native_basis_present_observations": int(present.sum()),
            "native_absent_fallback_observations": int((~present).sum()),
            "native_mean_squared_alignment_when_present": float(arrays["native_useful_squared_alignment"][sl][present].mean()) if present.any() else None,
            "full_direction_available_observations": int(full_present.sum()),
            "full_direction_unavailable_observations": int((~full_present).sum()),
            "full_mean_squared_alignment_when_available": float(arrays["full_useful_squared_alignment"][sl][full_present].mean()) if full_present.any() else None,
            "maximum_native_idempotence_error": float(arrays["native_action_idempotence_error"][sl].max()),
            "maximum_native_orthogonality_error": float(arrays["native_basis_orthogonality_error"][sl].max()),
            "alignment_scope": "generating coordinate axis; no changing direction exists in zero-drift cell"})
    return rows, directions


def aggregate_metrics(rows, directions):
    indexed = {(r["seed"],r["drift_index"],r["rotation_index"],r["window"],r["policy"]):r for r in rows}
    means, paired, oracle_paired = [], [], []
    for di, drift in enumerate(DRIFTS):
        for ri in ROTATIONS:
            for window, _, _ in WINDOWS:
                for policy in policy_names(drift):
                    values = {str(seed): indexed.get((seed,di,ri,window,policy),{}).get("mse") for seed in SEEDS}
                    means.append({"drift_index":di,"rotation_index":ri,"window":window,"policy":policy,
                                  "mse":seed_summary(values)})
                    if policy == "native_cp":
                        continue
                    differences = {}
                    for seed in SEEDS:
                        native = indexed.get((seed,di,ri,window,"native_cp"),{}).get("mse")
                        other = values[str(seed)]
                        differences[str(seed)] = None if native is None or other is None else other-native
                    paired.append({"drift_index":di,"rotation_index":ri,"window":window,
                        "comparison":"comparator_mse_minus_native_cp_mse", "comparator":policy,
                        "positive_favors":"native_cp", "effect":seed_summary(differences)})
                    if policy == "raw" or policy.startswith("ema_") or policy.startswith("scalar_"):
                        oracle_effects = {}
                        for seed in SEEDS:
                            oracle = indexed.get((seed,di,ri,window,"oracle_useful"),{}).get("mse")
                            other = values[str(seed)]
                            oracle_effects[str(seed)] = None if oracle is None or other is None else other-oracle
                        oracle_paired.append({"drift_index":di,"rotation_index":ri,"window":window,
                            "comparison":"comparator_mse_minus_useful_oracle_mse","comparator":policy,
                            "positive_favors":"oracle_useful","effect":seed_summary(oracle_effects),
                            "scope":"prespecified directional-oracle headroom; not a native performance claim"})
    direction_index = {(r["seed"],r["drift_index"],r["rotation_index"],r["window"]):r for r in directions}
    alignment = []
    for di in range(3):
        for ri in ROTATIONS:
            for window, _, _ in WINDOWS:
                value = {"drift_index":di,"rotation_index":ri,"window":window}
                for name in ("native_mean_squared_alignment_when_present", "full_mean_squared_alignment_when_available"):
                    value[name] = seed_summary({str(seed):direction_index.get((seed,di,ri,window),{}).get(name) for seed in SEEDS})
                alignment.append(value)
    primary = [r for r in paired if r["rotation_index"]==0 and r["window"]=="late"]
    strong = [r for r in primary if r["drift_index"]==2 and
              (r["comparator"].startswith("ema_") or r["comparator"].startswith("scalar_"))]
    alignment_strong = next(r for r in alignment if r["drift_index"]==2 and r["rotation_index"]==0 and r["window"]=="late")
    aligned = alignment_strong["native_mean_squared_alignment_when_present"]
    available = aligned["available"] and all(r["effect"]["available"] for r in strong)
    prediction = {"scope":"identity rotation, late window, all 32 paired independent seeds",
        "all_conjuncts_available":available, "native_mean_alignment_exceeds_half":
            aligned["mean"]>.5 if aligned["available"] else None,
        "all_prespecified_uniform_and_mixture_mean_mse_effects_positive":
            all(r["effect"]["mean"]>0 for r in strong) if available else None,
        "conjunction_met":(aligned["mean"]>.5 and all(r["effect"]["mean"]>0 for r in strong)) if available else None,
        "not_a_multiplicity_adjusted_significance_test":True}
    return {"per_seed_window_metrics":rows, "per_seed_direction_diagnostics":directions,
            "equal_seed_mse_summaries":means, "paired_cp_contrasts":paired,
            "paired_useful_oracle_contrasts":oracle_paired,
            "primary_identity_late_paired_contrasts":primary, "equal_seed_alignment_summaries":alignment,
            "strong_drift_registered_conjunction":prediction}


def rotation_check(identity, rotated, drift, seed, audit, *, windows=WINDOWS):
    q = rotation(1)
    audit.require(np.array_equal(rotated["noise"], identity["noise"]), "paired raw noise differs")
    audit.close(rotated["epsilon"], identity["epsilon"]@q.T, "paired rotated disturbance construction")
    audit.close(rotated["g"], identity["g"]@q.T, "paired rotated gradient construction")
    audit.close(rotated["s"], identity["s"]@q.T, "paired rotated signal construction")
    # Actions and native deliveries are empirical equivariance diagnostics:
    # do not reject a real numerical/path sensitivity or repair it to pass.
    action_error = rotated["A"]-q[None,:,:]@identity["A"]@q.T[None,:,:]
    full_error = rotated["full_moment"]-q[None,:,:]@identity["full_moment"]@q.T[None,:,:]
    expected_output = identity["output"]@q.T
    output_error = rotated["output"]-expected_output
    diagnostics = []
    for window, first, last in windows:
        sl = slice(first-1,last)
        policies = []
        for column,policy in enumerate(policy_names(drift)):
            original = error_metrics(identity["output"][sl,column],identity["s"][sl])["mse"]
            changed = error_metrics(rotated["output"][sl,column],rotated["s"][sl])["mse"]
            policies.append({"policy":policy,"maximum_rotated_output_difference_norm":
                float(np.linalg.norm(output_error[sl,column],axis=1).max()),
                "identity_mse":original,"rotated_mse":changed,"rotated_minus_identity_mse":changed-original})
        diagnostics.append({"seed":seed,"drift":drift,"window":window,
            "maximum_action_equivariance_difference_frobenius":float(np.linalg.norm(action_error[sl],axis=(1,2)).max()),
            "maximum_full_moment_equivariance_difference_frobenius":float(np.linalg.norm(full_error[sl],axis=(1,2)).max()),
            "policies":policies,"rotation_is_not_an_independent_seed":True})
    return diagnostics


def summarize_rotations(rows):
    """Same-seed sensitivity effects only; never doubles independent n."""
    index = {(row["seed"],row["drift"],row["window"],policy["policy"]):policy
             for row in rows for policy in row["policies"]}
    result = []
    for drift in DRIFTS:
        for window,_,_ in WINDOWS:
            for policy in policy_names(drift):
                result.append({"drift":drift,"window":window,"policy":policy,
                    "rotation_is_not_an_independent_seed":True,
                    "rotated_minus_identity_mse":seed_summary({str(seed):index.get(
                        (seed,drift,window,policy),{}).get("rotated_minus_identity_mse") for seed in SEEDS})})
    return result


def verify_file(root, record, expected_path, audit):
    if not audit.require(type(record) is dict and set(record)=={"path","size","sha256"}
            and record.get("path")==expected_path and type(record.get("size")) is int and record["size"]>=0
            and type(record.get("sha256")) is str and SHA.fullmatch(record["sha256"]), "malformed file record: "+expected_path):
        return None
    path = root / expected_path
    info = path.lstat()
    if not audit.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "not a regular nonsymlink file: "+expected_path):
        return None
    digest = sha256(path)
    audit.hash_files += 1
    audit.hash_bytes += info.st_size
    if not audit.require(info.st_size==record["size"] and digest==record["sha256"], "file size/hash differs: "+expected_path):
        return None
    return path


def load_arrays(path, policies, audit):
    names = list(array_shapes(N_STEPS,policies))
    with zipfile.ZipFile(path) as archive:
        records = archive.infolist()
        if not audit.require([r.filename for r in records]==[name+".npy" for name in names]
                and all(0<=r.file_size<=16*1024**2 for r in records)
                and sum(r.file_size for r in records)<=32*1024**2, "NPZ member roster or expanded size differs"):
            return None
    with np.load(path, allow_pickle=False, max_header_size=10000) as archive:
        return {name:archive[name] for name in names}


def verify_sources(attempt, analysis_commit, audit):
    commit, sources = attempt.get("frozen_commit"), attempt.get("sources")
    audit.require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}",commit) is not None,"invalid acquisition commit")
    audit.require(type(sources) is list and len(sources)==len(SOURCE_PATHS)
        and [r.get("path") for r in sources if type(r) is dict]==list(SOURCE_PATHS),"exact acquisition source closure differs")
    for name,record in zip(SOURCE_PATHS,sources if type(sources) is list else []):
        path=verify_file(REPO,record,name,audit)
        if path is not None and type(commit) is str:
            frozen=subprocess.check_output(["git","show",commit+":"+name],cwd=REPO)
            audit.require(hashlib.sha256(frozen).hexdigest()==record["sha256"],"acquisition frozen source differs: "+name)
    audit.require(type(analysis_commit) is str and re.fullmatch(r"[0-9a-f]{40}",analysis_commit) is not None,"invalid analysis commit")
    analysis_sources=[]
    for path in (HERE/"audit_tracking.py",HERE/"test_audit_tracking.py"):
        name=str(path.relative_to(REPO))
        digest=sha256(path)
        frozen=subprocess.check_output(["git","show",analysis_commit+":"+name],cwd=REPO)
        audit.require(hashlib.sha256(frozen).hexdigest()==digest,"analysis source not frozen: "+name)
        analysis_sources.append({"path":name,"size":path.stat().st_size,"sha256":digest})
    return {"acquisition_commit":commit,"acquisition_sources":sources,
            "analysis_commit":analysis_commit,"analysis_sources":analysis_sources}


def audit_root(root, analysis_commit, audit):
    attempt_path, completion_path = root/"attempt.json", root/"completion.json"
    for path in (attempt_path,completion_path):
        audit.require(path.is_file() and not path.is_symlink(),"missing/nonregular terminal record: "+path.name)
    attempt_sha,completion_sha=sha256(attempt_path),sha256(completion_path)
    attempt,completion=read_json(attempt_path),read_json(completion_path)
    expected_attempt_keys={"schema","created_utc","root","frozen_commit","sources","seeds","drifts","rotations","horizon",
        "cooperative_seconds","array_limit_bytes","root_limit_bytes","pid","service_invocation_id","python","numpy","torch",
        "cpu_threads","platform","cuda_visible_devices","paid_spend_usd","paid_reserved_usd"}
    audit.require(set(attempt)==expected_attempt_keys and attempt.get("schema")=="i18_tracking_attempt_v1"
        and attempt.get("root")==str(root) and attempt.get("seeds")==list(SEEDS) and attempt.get("drifts")==list(DRIFTS)
        and attempt.get("rotations")==[0.0,math.pi/4] and attempt.get("horizon")==N_STEPS
        and attempt.get("cooperative_seconds")==1600 and attempt.get("array_limit_bytes")==ARRAY_CAP
        and attempt.get("root_limit_bytes")==ROOT_CAP and attempt.get("cpu_threads")==1
        and attempt.get("cuda_visible_devices")=="" and attempt.get("paid_spend_usd")==0
        and attempt.get("paid_reserved_usd")==0 and type(attempt.get("pid")) is int and attempt["pid"]>0,
        "attempt schema/identity/resources differ")
    expected_completion_keys={"schema","status","created_utc","frozen_commit","attempt_sha256","expected_streams","completed_streams",
        "completed_observations","elapsed_seconds","max_rss_kib","array_bytes","root_bytes_before_completion","failure","streams"}
    records=completion.get("streams",[])
    expected_ids=[f"{seed}-d{di}-r{ri}" for seed in SEEDS for di in range(3) for ri in ROTATIONS]
    audit.require(set(completion)==expected_completion_keys and completion.get("schema")=="i18_tracking_completion_v1"
        and completion.get("frozen_commit")==attempt.get("frozen_commit")
        and completion.get("attempt_sha256")==attempt_sha and completion.get("expected_streams")==192
        and type(records) is list and completion.get("completed_streams")==len(records)
        and completion.get("completed_observations")==len(records)*N_STEPS,
        "completion schema/attempt/count differs")
    audit.require(type(records) is list and [r.get("id") for r in records if type(r) is dict]==expected_ids[:len(records)],
                  "stream inventory is not the exact registered prefix")
    status=completion.get("status")
    audit.require(status in ("complete","failed") and ((status=="complete" and len(records)==192 and completion.get("failure") is None)
        or (status=="failed" and type(completion.get("failure")) is dict and set(completion["failure"])=={"type","message"})),
        "completion failure/coverage differs")
    audit.require(type(completion.get("elapsed_seconds")) in (float,int) and math.isfinite(completion["elapsed_seconds"])
        and 0<=completion["elapsed_seconds"]<=1805 and type(completion.get("max_rss_kib")) is int
        and 0<=completion["max_rss_kib"]<=4*1024**2,"completion time/memory exceeds hard bounds")
    expected_files={"attempt.json","completion.json"}|{r[role]["path"] for r in records for role in ("array","metadata")}
    actual_files=set()
    total_bytes=0
    for current,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:
            path=Path(current)/name
            audit.require(not path.is_symlink() and path==root/"streams","unexpected/symlink directory in artifact root")
        for name in files:
            path=Path(current)/name
            audit.require(path.is_file() and not path.is_symlink(),"nonregular artifact")
            actual_files.add(str(path.relative_to(root)))
            total_bytes+=path.lstat().st_size
    audit.require({p.name for p in root.iterdir()}=={"attempt.json","completion.json","streams"}
        and (root/"streams").is_dir() and not (root/"streams").is_symlink()
        and actual_files==expected_files,"physical artifact membership differs")
    audit.require(total_bytes<=ROOT_CAP and completion.get("root_bytes_before_completion")==total_bytes-completion_path.stat().st_size
        and completion.get("array_bytes")==sum(r["array"]["size"] for r in records)
        and 0<=completion["array_bytes"]<=ARRAY_CAP,"artifact accounting/cap differs")
    sources=verify_sources(attempt,analysis_commit,audit)
    if audit.errors:
        raise ValueError("root/source admission failed; no stream arrays loaded")
    rows,directions,rotations,stream_checks=[],[],[],[]
    cached={}
    stream_seconds=0.0
    for record in records:
        identity=record["id"]
        audit.stream_id=identity
        errors_before=len(audit.errors)
        match=re.fullmatch(r"(180[0-3][0-9])-d([0-2])-r([01])",identity)
        seed,di,ri=map(int,match.groups())
        drift=DRIFTS[di]
        audit.require(set(record)=={"id","array","metadata"},"stream record topology differs")
        array_path=verify_file(root,record["array"],f"streams/{identity}.npz",audit)
        meta_path=verify_file(root,record["metadata"],f"streams/{identity}.json",audit)
        if array_path is None or meta_path is None:
            raise ValueError("stream hash binding failed")
        metadata=read_json(meta_path)
        audit.require(set(metadata)=={"schema","id","seed","drift_index","rotation_index","drift","rotation","horizon","noise_sha256","frozen_commit","elapsed_seconds","core"}
            and metadata.get("schema")=="i18_tracking_stream_v1" and metadata.get("id")==identity and metadata.get("seed")==seed
            and metadata.get("drift_index")==di and metadata.get("rotation_index")==ri and metadata.get("drift")==drift
            and metadata.get("rotation")==[0.0,math.pi/4][ri] and metadata.get("horizon")==N_STEPS
            and metadata.get("frozen_commit")==attempt["frozen_commit"] and type(metadata.get("noise_sha256")) is str
            and SHA.fullmatch(metadata["noise_sha256"]) is not None and type(metadata.get("elapsed_seconds")) in (int,float)
            and math.isfinite(metadata["elapsed_seconds"]) and 0<=metadata["elapsed_seconds"]<=completion["elapsed_seconds"],
            "stream outer metadata differs")
        if len(audit.errors)>errors_before:
            raise ValueError("stream metadata admission failed: "+identity)
        stream_seconds+=metadata["elapsed_seconds"]
        arrays=load_arrays(array_path,policy_names(drift),audit)
        if arrays is None:
            raise ValueError("NPZ admission failed")
        audit.require(sha256(array_path)==record["array"]["sha256"] and sha256(meta_path)==record["metadata"]["sha256"],
                      "stream artifact changed during read")
        if di==0 and ri==0:
            cached={"seed":seed,"noise":arrays["noise"].copy(),"noise_sha256":metadata["noise_sha256"]}
        audit.require(hashlib.sha256(arrays["noise"].tobytes(order="C")).hexdigest()==metadata["noise_sha256"],
                      "raw shared noise bytes/hash differ")
        audit.require(cached.get("seed")==seed and cached.get("noise_sha256")==metadata["noise_sha256"],
                      "within-seed shared noise binding differs")
        audit.require(np.array_equal(arrays["noise"],cached["noise"]),"shared raw noise across drift/rotation differs")
        outputs=validate_arrays(arrays,metadata.get("core"),drift,ri,audit)
        if outputs is None or len(audit.errors)>errors_before:
            raise ValueError("saved array integrity failed: "+identity)
        metrics,alignment=stream_metrics(arrays,policy_names(drift),seed,di,ri)
        rows.extend(metrics)
        directions.extend(alignment)
        stream_checks.append({"id":identity,"status":"pass","arrays":len(arrays),"observations":N_STEPS})
        if ri==0:
            cached["identity"]=arrays
        else:
            rotations.extend(rotation_check(cached.pop("identity"),arrays,drift,seed,audit))
        if len(audit.errors)>errors_before:
            raise ValueError("paired stream integrity failed: "+identity)
    audit.stream_id=None
    audit.require(stream_seconds<=completion["elapsed_seconds"]+1e-6,"summed stream time exceeds completion time")
    audit.require(sha256(attempt_path)==attempt_sha and sha256(completion_path)==completion_sha,
                  "attempt/completion changed during audit")
    audit.require(verify_sources(attempt,analysis_commit,audit)==sources,"source provenance changed during audit")
    summary=aggregate_metrics(rows,directions)
    summary.update({"schema":"i18_tracking_summary_v1","artifact_root":str(root),"acquisition_status":status,
        "completed_streams":len(records),"expected_streams":192,"missing_streams":expected_ids[len(records):],
        "all_scientific_streams_complete":len(records)==192 and status=="complete",
        "input_provenance":sources,"attempt_sha256":attempt_sha,"completion_sha256":completion_sha,
        "array_audit_streams":stream_checks,"population_predictions_theory":population_predictions(),
        "paired_rotation_diagnostics":rotations,
        "paired_rotation_mse_summaries":summarize_rotations(rotations),
        "scope":["NumPy/stdlib audit of hash-bound saved arrays and one-step algebra; no native observer or RNG replay.",
            "Identity-rotation late-window results use 32 paired independent noise seeds; rotations are not pooled as extra seeds.",
            "Within-window error dispersion is not causal noise variance or uncertainty from independent time samples.",
            "Native absence and full-moment tied directions are reported separately, not counted as learned alignment.",
            "Noise bytes/sharing and acquisition source are pinned; the Gaussian draw or seed-to-draw mapping is not replayed.",
            "All pairwise comparisons retained; no multiplicity-adjusted significance or neural-generalization claim."]})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frozen-commit",required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or any(os.environ.get(name)!="1" for name in
            ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")):
        raise SystemExit("CPU-only one-thread audit environment required")
    root=args.root.resolve(strict=True)
    if args.root.absolute()!=root or root.parent!=Path("/tmp/spectral-experiment-artifacts") or not root.name.startswith("spectral-i18-001."):
        raise SystemExit("direct canonical I18 artifact root required")
    if not os.path.ismount("/private-artifacts/storage") or root.stat().st_dev!=Path("/private-artifacts/storage").stat().st_dev:
        raise SystemExit("I18 artifact root must be on the intended mounted volume")
    if args.output.parent.resolve(strict=True)!=HERE or args.output.name!="analysis-001" or args.output.is_symlink():
        raise SystemExit("only exclusive iteration-018/analysis-001 is admitted")
    args.output.mkdir(exist_ok=False)
    audit=Audit()
    try:
        summary=audit_root(root,args.frozen_commit,audit)
    except Exception as exc:
        audit.errors.append(type(exc).__name__+": "+str(exc))
        summary={"schema":"i18_tracking_summary_v1","artifact_root":str(root),"status":"unavailable_after_integrity_error"}
    status="pass" if not audit.errors else "fail"
    summary["audit_status"]=status
    payload={"schema":"i18_tracking_audit_v1","status":status,"artifact_root":str(root),
        "attempt_sha256":summary.get("attempt_sha256"),"completion_sha256":summary.get("completion_sha256"),
        "input_provenance":summary.get("input_provenance"),
        "checks":audit.checks,"numeric_values_checked":audit.numeric_values,"errors":audit.errors,
        "maximum_absolute_error_by_relation":audit.maximum_absolute_error,
        "hash_files_verified":audit.hash_files,"hash_bytes_streamed":audit.hash_bytes,
        "scope":"saved-array communication-independent audit; no native observer/RNG replay"}
    for name,value in (("summary.json",summary),("audit.json",payload)):
        if name=="audit.json":
            value["summary_sha256"]=sha256(args.output/"summary.json")
        with (args.output/name).open("x",encoding="utf-8") as handle:
            json.dump(value,handle,indent=2,allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status":status,"checks":audit.checks,"errors":len(audit.errors),"numeric_values":audit.numeric_values}))
    return 0 if status=="pass" else 1


if __name__=="__main__":
    raise SystemExit(main())
