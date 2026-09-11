"""Synthetic NumPy-only fixtures for the independent I18 saved-array auditor."""
import importlib.util
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_i18_audit_under_test", HERE / "audit_tracking.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(horizon=24, drift=.03, ri=0, *, zero=False):
    """Hand-constructed deterministic arrays, never native code or random draws.

    A rank-one covariance is diagonalized directly in ambient 2D to provide
    algebra fixtures. This is not an implementation/native-observer test.
    """
    angle = (0., math.pi/4)[ri]
    q = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    names = ["raw", "ema_q0p9", "ema_q0p99", "ema_q0p999"]
    optimum = None
    if drift:
        # Independent convex scalar minimization by derivative bracketing.
        lo, hi = 0., 10000.
        for _ in range(100):
            m = (lo+hi)/2
            if 2*drift**2*m > 10/(2*m+1)**2:
                hi = m
            else:
                lo = m
        lag = (lo+hi)/2
        optimum = lag/(1+lag)
        names.append("ema_optimal_oracle")
    names += ["scalar_k0", "scalar_k0p5", "scalar_k0p9", "scalar_k1", "oracle_useful",
              "oracle_nuisance", "native_i17", "native_cp"]
    shapes = {"noise": (horizon,2), "epsilon": (horizon,2), "g": (horizon,2), "s": (horizon,2),
        "mu": (horizon,2), "A": (horizon,2,2), "native_delivery": (horizon,2),
        "native_h": (horizon,2), "full_moment": (horizon,2,2), "full_action": (horizon,2,2),
        "full_gap": (horizon,), "full_basis_present": (horizon,), "full_basis_V": (horizon,2),
        "basis_present": (horizon,), "basis_V": (horizon,2), "basis_S": (horizon,),
        "output": (horizon,len(names),2), "native_i17_buffer": (horizon,2),
        "scalar_buffer": (horizon,4,2), "native_action_raw_error": (horizon,2),
        "native_i17_response_residual": (horizon,2), "native_cp_response_residual": (horizon,2),
        "scalar_response_residual": (horizon,4,2), "full_moment_residual": (horizon,2,2),
        "native_action_idempotence_error": (horizon,), "native_basis_orthogonality_error": (horizon,),
        "native_useful_squared_alignment": (horizon,), "full_useful_squared_alignment": (horizon,)}
    arrays = {key: np.zeros(shape, dtype=np.bool_ if key.endswith("present") else np.float64)
              for key,shape in shapes.items()}
    t = np.arange(horizon, dtype=np.float64)
    if not zero:
        arrays["noise"][:] = np.column_stack((np.sin(t*.73)+.2, 2*np.cos(t*.39)-.3))
    arrays["epsilon"][:] = arrays["noise"]@q.T
    arrays["s"][:] = np.column_stack((drift*t,np.zeros(horizon)))@q.T
    arrays["g"][:] = arrays["s"]+arrays["epsilon"]
    arrays["A"][:] = np.eye(2)
    initialized = False
    if not zero or drift:
        for i in range(horizon):
            g = arrays["g"][i]
            arrays["mu"][i] = g if i==0 else .99*arrays["mu"][i-1]+.01*g
            z = g-arrays["mu"][i]
            zz = np.outer(z,z)
            candidate = zz if i==0 or not arrays["basis_present"][i-1] else (
                .99*arrays["basis_S"][i-1]**2*np.outer(arrays["basis_V"][i-1],arrays["basis_V"][i-1])+.01*zz)
            vals,vecs = np.linalg.eigh(candidate)
            if vals[-1]>0:
                arrays["basis_present"][i] = True
                arrays["basis_V"][i] = vecs[:,-1]
                arrays["basis_S"][i] = math.sqrt(vals[-1])
                arrays["A"][i] = np.outer(vecs[:,-1],vecs[:,-1])
            full = .99*arrays["full_moment"][i-1]+.01*zz if initialized else zz
            initialized = initialized or np.any(z!=0)
            arrays["full_moment"][i] = full
            vals,vecs = np.linalg.eigh(full)
            arrays["full_gap"][i] = vals[1]-vals[0]
            if vals[1]-vals[0] > 1e-10*max(1.,abs(vals[-1])):
                arrays["full_basis_present"][i] = True
                arrays["full_basis_V"][i] = vecs[:,-1]
                arrays["full_action"][i] = np.outer(vecs[:,-1],vecs[:,-1])
            A,mu = arrays["A"][i],arrays["mu"][i]
            arrays["native_delivery"][i] = A@g
            h = A@g+mu-A@mu
            arrays["native_h"][i] = h
            Q = np.eye(2)-.9*A
            b = np.linalg.solve(Q,g) if i==0 else .9*A@arrays["native_i17_buffer"][i-1]+h
            arrays["native_i17_buffer"][i] = b
            value = {"raw":g,"native_i17":Q@b}
            value["native_cp"] = g if i==0 else .9*A@arrays["output"][i-1,names.index("native_cp")]+Q@h
            for name,decay in [("ema_q0p9",.9),("ema_q0p99",.99),("ema_q0p999",.999)]+(
                    [("ema_optimal_oracle",optimum)] if optimum is not None else []):
                value[name] = g if i==0 else decay*arrays["output"][i-1,names.index(name)]+(1-decay)*g
            for j,(name,k) in enumerate(zip(("scalar_k0","scalar_k0p5","scalar_k0p9","scalar_k1"),(0.,.5,.9,1.))):
                state = g/(1-.9*k) if i==0 else .9*k*arrays["scalar_buffer"][i-1,j]+k*g+(1-k)*mu
                arrays["scalar_buffer"][i,j] = state
                value[name] = (1-.9*k)*state
            P = np.outer(q[:,0],q[:,0])
            value["oracle_useful"] = P@value["ema_q0p9"]+(np.eye(2)-P)@value["ema_q0p99"]
            value["oracle_nuisance"] = P@value["ema_q0p99"]+(np.eye(2)-P)@value["ema_q0p9"]
            arrays["output"][i] = np.stack([g if i==0 else value[name] for name in names])
    arrays["native_action_idempotence_error"][:] = np.linalg.norm(arrays["A"]@arrays["A"]-arrays["A"],axis=(1,2))
    present = arrays["basis_present"]
    arrays["native_basis_orthogonality_error"][present] = abs(np.sum(arrays["basis_V"][present]**2,axis=1)-1)
    arrays["native_useful_squared_alignment"][:] = (arrays["basis_V"]@q[:,0])**2
    arrays["full_useful_squared_alignment"][:] = (arrays["full_basis_V"]@q[:,0])**2
    metadata = {"schema":"i18_tracking_stream_metadata_v1","array_schema":"i18_tracking_arrays_v1",
        "horizon":horizon,"dimension":2,"drift":drift,"rotation_radians":angle,"rho":.9,"beta":.99,
        "policy_names":names,"policy_count":len(names),"optimal_ema_decay":optimum,
        "full_gap_relative_tolerance":1e-10,
        "filter_configuration":{"rank":1,"decay":.99,"warmup":0,"filter_strength":1.,"adaptive":"none",
            "normalize":"none","weighting":"hard","stable_update":True,"relative_eig_tol":1e-8,
            "absolute_eig_floor":0.,"stabilize_every":100},
        "initialization":{"all_policy_outputs_at_t1_equal_g1":True,"native_i17_buffer":"solve(I-rho*A_1,g_1)",
            "native_cp_delivery":"g_1","scalar_buffer":"g_1/(1-rho*k)","uniform_ema":"g_1",
            "full_moment":"zero until first nonzero post-ingest residual, then z*z^T unscaled"},
        "native_absent_basis_action":"identity","full_absent_basis_arrays":"zero-filled with full_basis_present=false",
        "array_order":list(arrays),"array_shapes":{key:list(value.shape) for key,value in arrays.items()},
        "array_dtypes":{key:str(value.dtype) for key,value in arrays.items()},"noise_generated_by_core":False,
        "noise_array_semantics":"unrotated shared input draw",
        "epsilon_array_semantics":"rotation-applied disturbance used in g=s+epsilon"}
    return arrays,metadata


def json_file(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def synthetic_root(root):
    """One all-zero artificial stream; failed registered prefix, not science."""
    (root/"streams").mkdir()
    commit = "a"*40
    attempt = {"schema":"i18_tracking_attempt_v1","created_utc":"2026-09-08T00:00:00+00:00",
        "root":str(root),"frozen_commit":commit,"sources":[],"seeds":list(audit.SEEDS),
        "drifts":list(audit.DRIFTS),"rotations":[0.,math.pi/4],"horizon":4000,
        "cooperative_seconds":1600,"array_limit_bytes":1024**3,"root_limit_bytes":2*1024**3,
        "pid":123,"service_invocation_id":None,"python":"fixture","numpy":"fixture","torch":"fixture",
        "cpu_threads":1,"platform":"fixture","cuda_visible_devices":"","paid_spend_usd":0,"paid_reserved_usd":0}
    json_file(root/"attempt.json",attempt)
    arrays,core = fixture(4000,0.,0,zero=True)
    identity = "18000-d0-r0"
    metadata = {"schema":"i18_tracking_stream_v1","id":identity,"seed":18000,"drift_index":0,
        "rotation_index":0,"drift":0.,"rotation":0.,"horizon":4000,
        "noise_sha256":hashlib.sha256(arrays["noise"].tobytes()).hexdigest(),
        "frozen_commit":commit,"elapsed_seconds":1.,"core":core}
    np.savez(root/"streams"/(identity+".npz"),**arrays)
    json_file(root/"streams"/(identity+".json"),metadata)
    row = {"id":identity}
    for role,suffix in (("array","npz"),("metadata","json")):
        path = root/"streams"/(identity+"."+suffix)
        row[role] = {"path":str(path.relative_to(root)),"size":path.stat().st_size,"sha256":audit.sha256(path)}
    completion = {"schema":"i18_tracking_completion_v1","status":"failed","created_utc":attempt["created_utc"],
        "frozen_commit":commit,"attempt_sha256":audit.sha256(root/"attempt.json"),"expected_streams":192,
        "completed_streams":1,"completed_observations":4000,"elapsed_seconds":1.,"max_rss_kib":1000,
        "array_bytes":row["array"]["size"],"root_bytes_before_completion":(root/"attempt.json").stat().st_size+
            row["array"]["size"]+row["metadata"]["size"],
        "failure":{"type":"SyntheticFixture","message":"Intentional artificial prefix"},"streams":[row]}
    json_file(root/"completion.json",completion)
    return attempt,completion


class TrackingAuditTests(unittest.TestCase):
    def test_error_decomposition_is_temporal_not_seed_variance(self):
        signal = np.zeros((3, 2))
        delivery = np.array([[1., -1.], [2., 1.], [3., 3.]])
        row = audit.error_metrics(delivery, signal)
        self.assertEqual(row["mean_error_vector"], [2., 1.])
        self.assertAlmostEqual(row["mse"], 25/3)
        self.assertAlmostEqual(row["squared_mean_error"], 5.)
        self.assertAlmostEqual(row["temporal_error_dispersion"], 10/3)
        self.assertAlmostEqual(row["mse_decomposition_residual"], 0.)
        self.assertTrue(row["dispersion_is_not_independent_sample_uncertainty"])

    def test_seed_mean_se_and_missing_seed_has_no_survivor_average(self):
        values = {str(seed): float(index) for index, seed in enumerate(audit.SEEDS)}
        result = audit.seed_summary(values)
        self.assertEqual(result["n_independent_seeds"], 32)
        self.assertEqual(result["mean"], 15.5)
        expected_se = math.sqrt(sum((index-15.5)**2 for index in range(32)) / 31 / 32)
        self.assertAlmostEqual(result["standard_error"], expected_se)
        self.assertEqual((result["positive_seeds"], result["zero_seeds"]), (31, 1))
        del values["18000"]
        missing = audit.seed_summary(values)
        self.assertIsNone(missing["mean"])
        self.assertIsNone(missing["standard_error"])
        self.assertFalse(missing["available"])

    def test_optimum_and_population_positive_and_adverse_cells(self):
        self.assertIsNone(audit.optimal_decay(0.0))
        for drift in (.01, .03):
            q = audit.optimal_decay(drift)
            lag = q/(1-q)
            self.assertAlmostEqual(2*drift**2*lag - 10/(2*lag+1)**2, 0., places=12)
        rows = audit.population_predictions()
        self.assertEqual([row["population_prefers_generating_axis"] for row in rows], [False, False, True])
        self.assertAlmostEqual(rows[2]["identity_rotation_population_residual_moment_diagonal"][0], 9.80592512562814)
        self.assertAlmostEqual(rows[2]["asymptotic_fixed_route_risks"]["oracle_useful"], .14563208145993123)

    def test_rotation_and_numerical_identity_checks(self):
        q = audit.rotation(1)
        np.testing.assert_allclose(q.T @ q, np.eye(2), rtol=0, atol=1e-15)
        instance = audit.Audit()
        self.assertTrue(instance.close(q.T @ q, np.eye(2), "orthogonal"))
        self.assertFalse(instance.close(q, np.eye(2), "not identity"))
        self.assertTrue(instance.errors)

    def test_maximum_error_is_global_and_failures_identify_stream(self):
        instance = audit.Audit()
        instance.close(np.array([3.]),np.array([1.]),"global")
        instance.close(np.array([1.01]),np.array([1.]),"global")
        self.assertEqual(instance.maximum_absolute_error["global"],2.)
        instance.stream_id = "fixture-stream"
        instance.require(False,"broken")
        self.assertEqual(instance.errors[-1],"fixture-stream: broken")

    def test_all_saved_equations_on_independent_deterministic_fixtures(self):
        for drift in (0.,.01,.03):
            for ri in (0,1):
                with self.subTest(drift=drift,rotation=ri):
                    arrays,metadata = fixture(drift=drift,ri=ri)
                    instance = audit.Audit()
                    result = audit.validate_arrays(arrays,metadata,drift,ri,instance,horizon=24)
                    self.assertIsNotNone(result)
                    self.assertEqual(instance.errors,[])
                    self.assertGreater(instance.numeric_values,2000)
                    self.assertIn("native i17 versus CP response transport identity",instance.maximum_absolute_error)

    def test_zero_residual_absent_and_tied_directions_not_learned(self):
        arrays,metadata = fixture(drift=0.,zero=True)
        instance = audit.Audit()
        audit.validate_arrays(arrays,metadata,0.,0,instance,horizon=24)
        self.assertEqual(instance.errors,[])
        _,directions = audit.stream_metrics(arrays,metadata["policy_names"],18000,0,0,windows=(("fixture",1,24),))
        self.assertIsNone(directions[0]["native_mean_squared_alignment_when_present"])
        self.assertIsNone(directions[0]["full_mean_squared_alignment_when_available"])
        self.assertEqual(directions[0]["native_absent_fallback_observations"],24)
        arrays["native_useful_squared_alignment"][0]=1.
        mutated = audit.Audit()
        audit.validate_arrays(arrays,metadata,0.,0,mutated,horizon=24)
        self.assertTrue(any("generating-axis alignment" in error for error in mutated.errors))

    def test_mutations_of_every_load_bearing_state_are_rejected(self):
        original,metadata = fixture()
        mutations = {
            "noise":lambda a: a["noise"].__setitem__((5,0),99.),
            "epsilon":lambda a: a["epsilon"].__setitem__((5,0),99.),
            "mu":lambda a: a["mu"].__setitem__((5,0),99.),
            "basis_S":lambda a: a["basis_S"].__setitem__(5,99.),
            "basis_V":lambda a: a["basis_V"].__setitem__((5,0),99.),
            "basis_missing":lambda a: a["basis_present"].__setitem__(5,False),
            "full_moment":lambda a: a["full_moment"].__setitem__((5,0,0),99.),
            "full_action":lambda a: a["full_action"].__setitem__((5,0,0),99.),
            "full_tie":lambda a: a["full_basis_present"].__setitem__(5,False),
            "native_delivery":lambda a: a["native_delivery"].__setitem__((5,0),99.),
            "native_h":lambda a: a["native_h"].__setitem__((5,0),99.),
            "native_buffer":lambda a: a["native_i17_buffer"].__setitem__((5,0),99.),
            "scalar_buffer":lambda a: a["scalar_buffer"].__setitem__((5,2,0),99.),
            "residual":lambda a: a["native_cp_response_residual"].__setitem__((5,0),99.),
            "dtype":lambda a: a.__setitem__("noise",a["noise"].astype(np.float32)),
            "nonfinite":lambda a: a["noise"].__setitem__((5,0),np.nan),
        }
        for policy in metadata["policy_names"]:
            column = metadata["policy_names"].index(policy)
            mutations[policy] = lambda a,column=column:a["output"].__setitem__((5,column,0),99.)
        for name,mutate in mutations.items():
            with self.subTest(mutation=name):
                arrays = {key:value.copy() for key,value in original.items()}
                mutate(arrays)
                instance = audit.Audit()
                audit.validate_arrays(arrays,metadata,.03,0,instance,horizon=24)
                self.assertTrue(instance.errors)

    def test_initialization_is_exact_and_metadata_drift_rejected(self):
        arrays,metadata = fixture()
        arrays["output"][0,0,0] = np.nextafter(arrays["g"][0,0],np.inf)
        instance = audit.Audit()
        audit.validate_arrays(arrays,metadata,.03,0,instance,horizon=24)
        self.assertIn("all estimators initial delivery must be exactly g1",instance.errors)
        changed = copy.deepcopy(metadata)
        changed["filter_configuration"]["stable_update"] = False
        instance = audit.Audit()
        audit.validate_metadata(changed,.03,0,24,instance)
        self.assertTrue(instance.errors)

    def test_rotation_is_paired_and_native_path_difference_not_silently_repaired(self):
        original,_ = fixture(ri=0)
        rotated,_ = fixture(ri=1)
        instance = audit.Audit()
        rows = audit.rotation_check(original,rotated,.03,18000,instance,windows=(("fixture",1,24),))
        self.assertEqual(instance.errors,[])
        self.assertLess(rows[0]["maximum_action_equivariance_difference_frobenius"],1e-12)
        rotated["output"][2,-1,0] += 1.
        rows = audit.rotation_check(original,rotated,.03,18000,instance,windows=(("fixture",1,24),))
        self.assertEqual(instance.errors,[])
        self.assertGreater(rows[0]["policies"][-1]["maximum_rotated_output_difference_norm"],.9)
        self.assertTrue(rows[0]["rotation_is_not_an_independent_seed"])

    def test_npz_members_pickle_and_hash_tampering(self):
        arrays,metadata = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/"fixture.npz"
            np.savez(path,**arrays)
            instance = audit.Audit()
            loaded = audit.load_arrays(path,metadata["policy_names"],instance)
            self.assertEqual(list(loaded),list(arrays))
            record = {"path":"fixture.npz","size":path.stat().st_size,"sha256":audit.sha256(path)}
            self.assertEqual(audit.verify_file(root,record,"fixture.npz",instance),path)
            record["sha256"] = "0"*64
            self.assertIsNone(audit.verify_file(root,record,"fixture.npz",instance))
            with zipfile.ZipFile(path,"a") as archive:
                archive.writestr("extra.npy",b"unexpected")
            self.assertIsNone(audit.load_arrays(path,metadata["policy_names"],instance))
            arrays["noise"] = np.array([[object()]],dtype=object)
            np.savez(path,**arrays)
            with self.assertRaises(ValueError):
                audit.load_arrays(path,metadata["policy_names"],audit.Audit())

    def test_failed_prefix_preserved_without_survivor_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic_root(root)
            instance = audit.Audit()
            with mock.patch.object(audit,"verify_sources",return_value={"synthetic":True}):
                result = audit.audit_root(root,"a"*40,instance)
            self.assertEqual(instance.errors,[])
            self.assertEqual(result["completed_streams"],1)
            self.assertEqual(len(result["missing_streams"]),191)
            self.assertFalse(result["all_scientific_streams_complete"])
            self.assertTrue(result["per_seed_window_metrics"])
            self.assertTrue(all(row["mse"]["mean"] is None for row in result["equal_seed_mse_summaries"]))
            self.assertIsNone(result["strong_drift_registered_conjunction"]["conjunction_met"])

    def test_complete_32_seed_aggregation_sign_se_conjunction_and_missing(self):
        rows,directions = [],[]
        for index,seed in enumerate(audit.SEEDS):
            for di,drift in enumerate(audit.DRIFTS):
                for ri in (0,1):
                    for window,_,_ in audit.WINDOWS:
                        for policy in audit.policy_names(drift):
                            mse = (1+index/64 if policy=="native_cp" else
                                   .5+index/64 if policy=="oracle_useful" else 2+index/32)
                            rows.append({"seed":seed,"drift_index":di,"rotation_index":ri,
                                "window":window,"policy":policy,"mse":mse})
                        directions.append({"seed":seed,"drift_index":di,"rotation_index":ri,
                            "window":window,"native_mean_squared_alignment_when_present":.75,
                            "full_mean_squared_alignment_when_available":.9})
        result = audit.aggregate_metrics(rows,directions)
        target = next(row for row in result["primary_identity_late_paired_contrasts"]
                      if row["drift_index"]==2 and row["comparator"]=="ema_optimal_oracle")
        expected = np.array([1+index/64 for index in range(32)])
        self.assertEqual(target["positive_favors"],"native_cp")
        self.assertEqual(target["effect"]["mean"],expected.mean())
        self.assertAlmostEqual(target["effect"]["standard_error"],expected.std(ddof=1)/math.sqrt(32))
        self.assertEqual(target["effect"]["positive_seeds"],32)
        self.assertEqual(len(result["primary_identity_late_paired_contrasts"]),35)
        self.assertTrue(result["strong_drift_registered_conjunction"]["conjunction_met"])
        oracle = next(row for row in result["paired_useful_oracle_contrasts"]
                      if row["drift_index"]==2 and row["rotation_index"]==0 and row["window"]=="late"
                      and row["comparator"]=="ema_optimal_oracle")
        self.assertEqual(oracle["effect"]["mean"],expected.mean()+.5)
        # Failing one member of the deliberately conjunctive prediction is not
        # concealed by the remaining favorable controls or another rotation.
        for row in rows:
            if row["drift_index"]==2 and row["rotation_index"]==0 and row["window"]=="late" and row["policy"]=="ema_optimal_oracle":
                row["mse"] = 0.
        result = audit.aggregate_metrics(rows,directions)
        self.assertFalse(result["strong_drift_registered_conjunction"]["conjunction_met"])
        removed = next(row for row in rows if row["seed"]==18000 and row["drift_index"]==2
                       and row["rotation_index"]==0 and row["window"]=="late" and row["policy"]=="ema_optimal_oracle")
        rows.remove(removed)
        result = audit.aggregate_metrics(rows,directions)
        self.assertFalse(result["strong_drift_registered_conjunction"]["all_conjuncts_available"])
        self.assertIsNone(result["strong_drift_registered_conjunction"]["conjunction_met"])
        target = next(row for row in result["primary_identity_late_paired_contrasts"]
                      if row["drift_index"]==2 and row["comparator"]=="ema_optimal_oracle")
        self.assertIsNone(target["effect"]["mean"])

    def test_rotation_seed_summary_never_pools_rotation_as_extra_seeds(self):
        rows = [{"seed":seed,"drift":.03,"window":"late","policies":[
            {"policy":"native_cp","rotated_minus_identity_mse":float(index)/32}]}
            for index,seed in enumerate(audit.SEEDS)]
        result = audit.summarize_rotations(rows)
        target = next(row for row in result if row["drift"]==.03 and row["window"]=="late" and row["policy"]=="native_cp")
        self.assertEqual(target["rotated_minus_identity_mse"]["n_independent_seeds"],32)
        self.assertEqual(target["rotated_minus_identity_mse"]["mean"],15.5/32)
        self.assertTrue(target["rotation_is_not_an_independent_seed"])

    def test_noise_hash_rejected_even_after_outer_artifacts_are_rebound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _,completion = synthetic_root(root)
            record = completion["streams"][0]["metadata"]
            path = root/record["path"]
            metadata = audit.read_json(path)
            metadata["noise_sha256"] = "0"*64
            json_file(path,metadata)
            old_size = record["size"]
            record.update(size=path.stat().st_size,sha256=audit.sha256(path))
            completion["root_bytes_before_completion"] += record["size"]-old_size
            json_file(root/"completion.json",completion)
            instance = audit.Audit()
            with mock.patch.object(audit,"verify_sources",return_value={}):
                with self.assertRaisesRegex(ValueError,"saved array integrity failed"):
                    audit.audit_root(root,"a"*40,instance)
            self.assertTrue(any("raw shared noise bytes/hash differ" in error for error in instance.errors))

    def test_root_inventory_or_attempt_tampering_blocks_array_load(self):
        for mutation in ("attempt","complete_short","extra_file","extra_directory","reordered_id","accounting"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                attempt,completion = synthetic_root(root)
                if mutation=="attempt":
                    attempt["horizon"] = 4001
                    json_file(root/"attempt.json",attempt)
                elif mutation=="complete_short":
                    completion["status"],completion["failure"] = "complete",None
                elif mutation=="extra_file":
                    (root/"streams"/"unlisted").write_text("unexpected")
                elif mutation=="extra_directory":
                    (root/"streams"/"empty").mkdir()
                elif mutation=="reordered_id":
                    completion["streams"][0]["id"] = "18001-d0-r0"
                else:
                    completion["array_bytes"] += 1
                json_file(root/"completion.json",completion)
                instance = audit.Audit()
                with mock.patch.object(audit,"verify_sources",return_value={}),mock.patch.object(audit,"load_arrays") as load:
                    with self.assertRaises(ValueError):
                        audit.audit_root(root,"a"*40,instance)
                    load.assert_not_called()
                self.assertTrue(instance.errors)

    def test_source_closure_checks_both_commits_and_current_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            paths = ("native.py","core.py","runner.py")
            frozen = {}
            for name in paths+("audit_tracking.py","test_audit_tracking.py"):
                (repo/name).write_text("# artificial "+name)
                frozen[name] = (repo/name).read_bytes()
            records = [{"path":name,"size":(repo/name).stat().st_size,"sha256":audit.sha256(repo/name)} for name in paths]
            attempt = {"frozen_commit":"a"*40,"sources":records}
            def git_show(command,**kwargs):
                self.assertEqual(command[:2],["git","show"])
                self.assertIn(command[2].split(":",1)[0],("a"*40,"b"*40))
                return frozen[command[2].split(":",1)[1]]
            with mock.patch.multiple(audit,REPO=repo,HERE=repo,SOURCE_PATHS=paths),mock.patch.object(
                    audit.subprocess,"check_output",side_effect=git_show):
                instance = audit.Audit()
                receipt = audit.verify_sources(attempt,"b"*40,instance)
                self.assertEqual(instance.errors,[])
                self.assertEqual(len(receipt["analysis_sources"]),2)
                (repo/"core.py").write_text("# changed")
                instance = audit.Audit()
                audit.verify_sources(attempt,"b"*40,instance)
                self.assertTrue(any("size/hash differs" in error for error in instance.errors))
                frozen["audit_tracking.py"] = b"different frozen bytes"
                instance = audit.Audit()
                audit.verify_sources(attempt,"b"*40,instance)
                self.assertTrue(any("analysis source not frozen" in error for error in instance.errors))

    def test_import_has_no_native_or_torch_or_rng_side_effect(self):
        code = ("import importlib.util,sys,numpy as np; "
            "np.random.seed=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('RNG called')); "
            "s=importlib.util.spec_from_file_location('auditor',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; "
            "assert not any('tracking_core' in key or 'spectral_filter' in key for key in sys.modules)")
        result = subprocess.run([sys.executable,"-c",code,str(HERE/"audit_tracking.py")],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.json"
            for content in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    audit.read_json(path)


if __name__ == "__main__":
    unittest.main()
