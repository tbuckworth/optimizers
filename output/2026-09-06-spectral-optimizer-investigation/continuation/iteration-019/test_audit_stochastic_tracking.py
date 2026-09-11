"""Deterministic NumPy fixtures only: no producer/native imports or RNG draws."""
import copy
import hashlib
import importlib.util
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
spec = importlib.util.spec_from_file_location("_i19_audit_test", HERE/"audit_stochastic_tracking.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(horizon=24, process=.1, ri=0, *, zero=False, first_process=17.):
    """Independent ambient-2D algebra fixture, not a native implementation test.

    This fixture uses explicit deterministic trigonometric inputs and dense 2D
    eigenpairs. It does not call any scientific producer, observer, or RNG.
    """
    angle = (0., math.pi/4)[ri]
    q = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    eye = np.eye(2)
    star = (2.1-math.sqrt(.41))/2
    names = ["raw","ema_q0p9","ema_q0p99","ema_q0p999","dema_q0p9","dema_q0p99","dema_q0p999"]
    optimum = None
    if process:
        optimum = 1-(math.sqrt(process*process+20*process)-process)/10
        names.append("ema_common_steady_oracle")
    names += ["common_kalman","useful_oracle_kalman","scalar_k0","scalar_k0p5","scalar_k0p9","scalar_k1",
              "oracle_useful","oracle_nuisance","native_i17","native_cp","native_cp_star","oracle_useful_star"]
    # Schema duplicated deliberately; no shape/metadata construction helpers
    # are imported from either the producer or the auditor under test.
    shapes = {"canonical":(horizon,3),"noise":(horizon,2),"epsilon":(horizon,2),"g":(horizon,2),"s":(horizon,2),
        "mu":(horizon,2),"A":(horizon,2,2),"native_delivery":(horizon,2),"native_h":(horizon,2),
        "full_moment":(horizon,2,2),"full_action":(horizon,2,2),"full_gap":(horizon,),
        "full_basis_present":(horizon,),"full_basis_V":(horizon,2),"basis_present":(horizon,),
        "basis_V":(horizon,2),"basis_S":(horizon,),"output":(horizon,len(names),2),
        "ema_fixed_state":(horizon,3,2),"dema_first_state":(horizon,3,2),"dema_second_state":(horizon,3,2),
        "ema_common_state":(horizon,2),"common_kalman_state":(horizon,2),"common_kalman_prior_weight":(horizon,),
        "common_kalman_posterior_variance":(horizon,),"useful_oracle_kalman_state_base":(horizon,2),
        "useful_oracle_kalman_prior_weight":(horizon,2),"useful_oracle_kalman_posterior_variance":(horizon,2),
        "oracle_useful_fast_state":(horizon,2),"oracle_useful_slow_state":(horizon,2),"oracle_useful_star_fast_state":(horizon,2),
        "scalar_buffer":(horizon,4,2),"native_i17_buffer":(horizon,2),"native_cp_buffer":(horizon,2),
        "native_cp_star_buffer":(horizon,2),"native_action_raw_error":(horizon,2),
        "ema_fixed_response_residual":(horizon,3,2),"dema_first_response_residual":(horizon,3,2),
        "dema_second_response_residual":(horizon,3,2),"ema_common_response_residual":(horizon,2),
        "common_kalman_response_residual":(horizon,2),"common_kalman_variance_residual":(horizon,),
        "useful_oracle_kalman_response_residual":(horizon,2),"useful_oracle_kalman_variance_residual":(horizon,2),
        "oracle_useful_fast_response_residual":(horizon,2),"oracle_useful_slow_response_residual":(horizon,2),
        "oracle_useful_star_fast_response_residual":(horizon,2),"scalar_response_residual":(horizon,4,2),
        "native_i17_response_residual":(horizon,2),"native_cp_response_residual":(horizon,2),
        "native_cp_star_response_residual":(horizon,2),"full_moment_residual":(horizon,2,2),
        "native_action_idempotence_error":(horizon,),"native_basis_orthogonality_error":(horizon,),
        "native_useful_squared_alignment":(horizon,),"full_useful_squared_alignment":(horizon,)}
    arrays = {name:np.zeros(shape,dtype=np.bool_ if name.endswith("present") else np.float64)
              for name,shape in shapes.items()}
    if not zero:
        for t in range(horizon):
            arrays["canonical"][t] = [math.sin(.73*t)+.2,math.cos(.39*t)-.15,math.sin(.17*t)-.1]
    arrays["canonical"][0,2] = first_process
    arrays["noise"][:] = arrays["canonical"][:,:2]*np.array([1.,2.])
    level = 0.
    for t in range(1,horizon):
        level += arrays["canonical"][t,2]
        arrays["s"][t] = math.sqrt(process)*level*q[:,0]
    arrays["epsilon"][:] = arrays["noise"]@q.T
    arrays["g"][:] = arrays["s"]+arrays["epsilon"]
    P = np.outer(q[:,0],q[:,0])
    full_initialized = False
    for t in range(horizon):
        g = arrays["g"][t]
        mu = g.copy() if t==0 else .99*arrays["mu"][t-1]+.01*g
        arrays["mu"][t] = mu
        z = g-mu
        zz = np.outer(z,z)
        covariance = zz.copy()
        if t and arrays["basis_present"][t-1]:
            covariance = .99*arrays["basis_S"][t-1]**2*np.outer(arrays["basis_V"][t-1],arrays["basis_V"][t-1])+.01*zz
        eigenvalues,eigenvectors = np.linalg.eigh(covariance)
        A = eye.copy()
        if eigenvalues[-1]>0:
            arrays["basis_present"][t] = True
            arrays["basis_V"][t] = eigenvectors[:,-1]
            arrays["basis_S"][t] = math.sqrt(eigenvalues[-1])
            A = np.outer(eigenvectors[:,-1],eigenvectors[:,-1])
        arrays["A"][t] = A
        full = .99*arrays["full_moment"][t-1]+.01*zz if full_initialized else zz
        full_initialized = full_initialized or bool(np.any(z))
        arrays["full_moment"][t] = full
        ev,V = np.linalg.eigh(full)
        arrays["full_gap"][t] = ev[-1]-ev[0]
        if ev[-1]-ev[0]>1e-10*max(1.,abs(ev[-1])):
            arrays["full_basis_present"][t] = True
            arrays["full_basis_V"][t] = V[:,-1]
            arrays["full_action"][t] = np.outer(V[:,-1],V[:,-1])
        arrays["native_delivery"][t] = A@g
        h = A@g+mu-A@mu
        arrays["native_h"][t] = h
        values = {"raw":g}
        for j,(suffix,decay) in enumerate(zip(("q0p9","q0p99","q0p999"),(.9,.99,.999))):
            E = g.copy() if t==0 else decay*arrays["ema_fixed_state"][t-1,j]+(1-decay)*g
            F = g.copy() if t==0 else decay*arrays["dema_second_state"][t-1,j]+(1-decay)*E
            arrays["ema_fixed_state"][t,j] = arrays["dema_first_state"][t,j] = E
            arrays["dema_second_state"][t,j] = F
            values["ema_"+suffix],values["dema_"+suffix] = E,2*E-F
        if optimum is not None:
            E = g.copy() if t==0 else optimum*arrays["ema_common_state"][t-1]+(1-optimum)*g
            arrays["ema_common_state"][t] = E
            values["ema_common_steady_oracle"] = E
        for prefix,R,Q,observation in (("common_kalman",5.,process,g),
                ("useful_oracle_kalman",np.array([1.,4.]),np.array([process,0.]),q.T@g)):
            state_key = prefix+("_state_base" if prefix.startswith("useful") else "_state")
            if t==0:
                arrays[prefix+"_posterior_variance"][t] = R
                estimate = observation.copy()
            else:
                prior_P = arrays[prefix+"_posterior_variance"][t-1]+Q
                weight = R/(prior_P+R)
                arrays[prefix+"_prior_weight"][t] = weight
                arrays[prefix+"_posterior_variance"][t] = R*prior_P/(prior_P+R)
                estimate = weight*arrays[state_key][t-1]+(1-weight)*observation
            arrays[state_key][t] = estimate
            values[prefix] = q@estimate if prefix.startswith("useful") else estimate
        for name,decay in (("fast",.9),("slow",.99),("star_fast",star)):
            key = "oracle_useful_"+name+"_state"
            arrays[key][t] = g if t==0 else decay*arrays[key][t-1]+(1-decay)*g
        fast,slow,star_fast = (arrays["oracle_useful_"+name+"_state"][t] for name in ("fast","slow","star_fast"))
        values["oracle_useful"] = P@fast+(eye-P)@slow
        values["oracle_nuisance"] = P@slow+(eye-P)@fast
        values["oracle_useful_star"] = P@star_fast+(eye-P)@slow
        for j,(name,k) in enumerate(zip(("scalar_k0","scalar_k0p5","scalar_k0p9","scalar_k1"),(0.,.5,.9,1.))):
            b = g/(1-.9*k) if t==0 else .9*k*arrays["scalar_buffer"][t-1,j]+k*g+(1-k)*mu
            arrays["scalar_buffer"][t,j] = b
            values[name] = (1-.9*k)*b
        C = eye-.9*A
        b = np.linalg.solve(C,g) if t==0 else .9*A@arrays["native_i17_buffer"][t-1]+h
        arrays["native_i17_buffer"][t] = b
        values["native_i17"] = C@b
        for name,decay in (("native_cp",.9),("native_cp_star",star)):
            d = g.copy() if t==0 else decay*A@arrays[name+"_buffer"][t-1]+(eye-decay*A)@h
            arrays[name+"_buffer"][t] = d
            values[name] = d
        arrays["output"][t] = np.stack([g if t==0 else values[name] for name in names])
    arrays["native_action_idempotence_error"][:] = np.linalg.norm(arrays["A"]@arrays["A"]-arrays["A"],axis=(1,2))
    present = arrays["basis_present"]
    arrays["native_basis_orthogonality_error"][present] = abs(np.sum(arrays["basis_V"][present]**2,axis=1)-1)
    arrays["native_useful_squared_alignment"][:] = (arrays["basis_V"]@q[:,0])**2
    arrays["full_useful_squared_alignment"][:] = (arrays["full_basis_V"]@q[:,0])**2
    metadata = {"schema":"i19_stochastic_tracking_stream_metadata_v1","array_schema":"i19_stochastic_tracking_arrays_v1",
        "horizon":horizon,"dimension":2,"canonical_dimension":3,"process_variance":process,"rotation_radians":angle,
        "rho":.9,"beta":.99,"rho_star":star,"policy_names":names,"policy_count":len(names),
        "optimal_common_ema_decay":optimum,"full_gap_relative_tolerance":1e-10,
        "filter_configuration":{"rank":1,"decay":.99,"warmup":0,"filter_strength":1.,"adaptive":"none",
            "normalize":"none","weighting":"hard","stable_update":True,"relative_eig_tol":1e-8,"absolute_eig_floor":0.,"stabilize_every":100},
        "initialization":{"all_policy_outputs_at_t1_equal_g1":True,"signal_at_t1":"zero",
            "process_increments":"sqrt(Q1)*canonical[1:,2]; canonical[0,2] unused",
            "native_i17_buffer":"solve(I-rho*A_1,g_1)","native_cp_delivery":"g_1","native_cp_star_delivery":"g_1",
            "scalar_buffer":"g_1/(1-rho*k)","uniform_ema_and_dema_states":"g_1",
            "common_kalman":"d_1=g_1,P_1=5; prior-weight q_1 sentinel is zero",
            "useful_oracle_kalman":"base-coordinate d_1=rotation_matrix^T*g_1,P_1=(1,4); prior-weight q_1 sentinel is zero",
            "full_moment":"zero until first nonzero post-ingest residual, then z*z^T unscaled"},
        "native_absent_basis_action":"identity","full_absent_basis_arrays":"zero-filled with full_basis_present=false",
        "array_order":list(arrays),"array_shapes":{key:list(value.shape) for key,value in arrays.items()},
        "array_dtypes":{key:str(value.dtype) for key,value in arrays.items()},"canonical_generated_by_core":False,
        "canonical_array_semantics":"unmodified supplied [measurement0,measurement1,process] draw",
        "noise_array_semantics":"unrotated canonical[:,:2] scaled by measurement std (1,2)",
        "epsilon_array_semantics":"rotation-applied measurement disturbance used in g=s+epsilon",
        "common_ema_label_semantics":"theoretical steady state; not finite-horizon optimal or learned",
        "kalman_prior_weight_semantics":"q_t multiplies prior state; index zero is a sentinel with no update",
        "kalman_control_semantics":"matched-first-observation model-based controls, not unrestricted finite-horizon optima",
        "oracle_star_semantics":"fixed strong-cell rho_star used in every process-variance cell"}
    return arrays,metadata


def json_file(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))


def synthetic_root(root):
    """One deterministic all-zero stream, failed prefix; not a scientific draw."""
    (root/"streams").mkdir()
    commit = "a"*40
    attempt = {"schema":"i19_tracking_attempt_v1","created_utc":"2026-09-08T00:00:00+00:00",
        "root":str(root),"frozen_commit":commit,"sources":[],"seeds":list(range(19000,19032)),
        "process_variances":[0.,.01,.1],"rotations":[0.,math.pi/4],"horizon":4000,
        "cooperative_seconds":1600,"array_limit_bytes":2*1024**3,"root_limit_bytes":3*1024**3,
        "pid":123,"service_invocation_id":None,"python":"fixture","numpy":"fixture","torch":"fixture",
        "cpu_threads":1,"platform":"fixture","cuda_visible_devices":"","paid_spend_usd":0,"paid_reserved_usd":0}
    json_file(root/"attempt.json",attempt)
    arrays,core = fixture(4000,0.,zero=True)
    identity = "19000-p0-r0"
    metadata = {"schema":"i19_tracking_stream_v1","id":identity,"seed":19000,"process_index":0,
        "rotation_index":0,"process_variance":0.,"rotation":0.,"horizon":4000,
        "canonical_sha256":hashlib.sha256(arrays["canonical"].tobytes()).hexdigest(),
        "frozen_commit":commit,"elapsed_seconds":1.,"core":core}
    np.savez(root/"streams"/(identity+".npz"),**arrays)
    json_file(root/"streams"/(identity+".json"),metadata)
    row = {"id":identity}
    for role,suffix in (("array","npz"),("metadata","json")):
        path = root/"streams"/(identity+"."+suffix)
        row[role] = {"path":str(path.relative_to(root)),"size":path.stat().st_size,"sha256":audit.sha256(path)}
    completion = {"schema":"i19_tracking_completion_v1","status":"failed","created_utc":attempt["created_utc"],
        "frozen_commit":commit,"attempt_sha256":audit.sha256(root/"attempt.json"),"expected_streams":192,
        "completed_streams":1,"completed_observations":4000,"elapsed_seconds":1.,"max_rss_kib":1000,
        "array_bytes":row["array"]["size"],"root_bytes_before_completion":(root/"attempt.json").stat().st_size+
            row["array"]["size"]+row["metadata"]["size"],
        "failure":{"type":"SyntheticFixture","message":"Intentional artificial prefix"},"streams":[row]}
    json_file(root/"completion.json",completion)
    return attempt,completion


class StochasticTrackingAuditTests(unittest.TestCase):
    def test_all_six_deterministic_fixture_cells(self):
        for process in (0.,.01,.1):
            for ri in (0,1):
                with self.subTest(process=process,ri=ri):
                    arrays,metadata = fixture(process=process,ri=ri)
                    instance = audit.Audit()
                    self.assertIsNotNone(audit.validate_arrays(arrays,metadata,process,ri,instance,horizon=24))
                    self.assertEqual(instance.errors,[])
                    self.assertEqual(len(arrays),56)
                    self.assertEqual(len(metadata["policy_names"]),19 if process==0 else 20)
                    self.assertEqual(sum(a.nbytes for a in arrays.values())//24,1442 if process==0 else 1458)
                    self.assertIn("native i17 versus CP response transport identity",instance.maximum_absolute_error)

    def test_unused_first_process_and_zero_initial_signal(self):
        first,meta = fixture(first_process=-1000.)
        second,_ = fixture(first_process=3000.)
        self.assertFalse(np.array_equal(first["canonical"],second["canonical"]))
        for key in first:
            if key!="canonical":
                np.testing.assert_array_equal(first[key],second[key])
        instance = audit.Audit()
        audit.validate_arrays(second,meta,.1,0,instance,horizon=24)
        self.assertEqual(instance.errors,[])
        second["s"][:,0] += 1.
        instance = audit.Audit()
        audit.validate_arrays(second,meta,.1,0,instance,horizon=24)
        self.assertTrue(any("signal construction" in e for e in instance.errors))

    def test_every_saved_array_and_policy_mutation_rejected(self):
        original,meta = fixture()
        for name in original:
            with self.subTest(array=name):
                arrays = {key:value.copy() for key,value in original.items()}
                position = (5,)+(0,)*(arrays[name].ndim-1)
                arrays[name][position] = not arrays[name][position] if arrays[name].dtype==np.bool_ else 99.
                instance = audit.Audit()
                audit.validate_arrays(arrays,meta,.1,0,instance,horizon=24)
                self.assertTrue(instance.errors,name)
        for column,name in enumerate(meta["policy_names"]):
            with self.subTest(policy=name):
                arrays = {key:value.copy() for key,value in original.items()}
                arrays["output"][5,column,0] = 99.
                instance = audit.Audit()
                audit.validate_arrays(arrays,meta,.1,0,instance,horizon=24)
                self.assertTrue(instance.errors)

    def test_initialization_exact_and_bad_schema_dtype_nonfinite(self):
        original,meta = fixture()
        for mutation in ("initial","metadata","order","dtype","nan"):
            arrays,metadata = copy.deepcopy(original),copy.deepcopy(meta)
            if mutation=="initial":
                arrays["output"][0,0,0] = np.nextafter(arrays["g"][0,0],np.inf)
            elif mutation=="metadata":
                metadata["filter_configuration"]["stable_update"] = False
            elif mutation=="order":
                arrays = dict(reversed(list(arrays.items())))
            elif mutation=="dtype":
                arrays["canonical"] = arrays["canonical"].astype(np.float32)
            else:
                arrays["canonical"][4,1] = np.nan
            instance = audit.Audit()
            audit.validate_arrays(arrays,metadata,.1,0,instance,horizon=24)
            self.assertTrue(instance.errors,mutation)

    def test_dema_current_first_stage_and_kalman_prior_weight(self):
        arrays,meta = fixture()
        g = arrays["g"]
        expected_E = .9*g[0]+.1*g[1]
        expected_F = .9*g[0]+.1*expected_E
        np.testing.assert_allclose(arrays["dema_second_state"][1,0],expected_F)
        self.assertGreater(np.linalg.norm(expected_F-g[0]),.001)
        self.assertAlmostEqual(arrays["common_kalman_prior_weight"][1],5/10.1)
        np.testing.assert_allclose(arrays["useful_oracle_kalman_prior_weight"][1],[1/2.1,.5])
        for key in ("common_kalman_prior_weight","useful_oracle_kalman_prior_weight"):
            changed = copy.deepcopy(arrays)
            changed[key][1:] = 1-changed[key][1:]
            instance = audit.Audit()
            audit.validate_arrays(changed,meta,.1,0,instance,horizon=24)
            self.assertTrue(instance.errors)

    def test_zero_residual_alignment_mask_and_common_control(self):
        arrays,metadata = fixture(process=0.,zero=True)
        instance = audit.Audit()
        audit.validate_arrays(arrays,metadata,0.,0,instance,horizon=24)
        self.assertEqual(instance.errors,[])
        np.testing.assert_allclose(arrays["common_kalman_posterior_variance"],5/np.arange(1,25))
        _,rows = audit.stream_metrics(arrays,metadata["policy_names"],19000,0,0,windows=(("fixture",1,24),))
        self.assertIsNone(rows[0]["native_mean_squared_alignment_when_present"])
        self.assertIsNone(rows[0]["full_mean_squared_alignment_when_available"])
        self.assertEqual(rows[0]["native_absent_fallback_observations"],24)
        self.assertIn("reference-only",rows[0]["alignment_scope"])
        arrays["native_useful_squared_alignment"][0] = 1.
        changed = audit.Audit()
        audit.validate_arrays(arrays,metadata,0.,0,changed,horizon=24)
        self.assertTrue(changed.errors)

    def test_theory_stationary_common_bound_and_population_selection(self):
        self.assertIsNone(audit.optimal_decay(0.))
        for process in (.01,.1):
            q = audit.optimal_decay(process)
            self.assertAlmostEqual(process*q,5*(1-q)**2,places=14)
        rows = audit.population_predictions()
        self.assertEqual([r["population_prefers_generating_axis"] for r in rows],[False,False,True])
        self.assertAlmostEqual(rows[2]["asymptotic_fixed_route_risks"]["oracle_useful"],18869/37810)
        self.assertAlmostEqual(rows[2]["asymptotic_fixed_route_risks"]["oracle_uniform_optimal"],(math.sqrt(201)-1)/20)
        self.assertAlmostEqual(rows[2]["oracle_optimal_decay"],(101-math.sqrt(201))/100)

    def test_metric_decomposition_and_seed_se(self):
        row = audit.error_metrics(np.array([[1.,-1.],[2.,1.],[3.,3.]]),np.zeros((3,2)))
        self.assertEqual(row["mean_error_vector"],[2.,1.])
        self.assertAlmostEqual(row["mse"],25/3)
        self.assertAlmostEqual(row["squared_mean_error"],5)
        self.assertAlmostEqual(row["temporal_error_dispersion"],10/3)
        values = {str(seed):float(i) for i,seed in enumerate(audit.SEEDS)}
        result = audit.seed_summary(values)
        self.assertEqual(result["mean"],15.5)
        self.assertAlmostEqual(result["standard_error"],np.arange(32).std(ddof=1)/math.sqrt(32))
        del values["19000"]
        self.assertIsNone(audit.seed_summary(values)["mean"])

    def test_global_max_error_and_stream_identity(self):
        instance = audit.Audit()
        instance.close(np.array([3.]),np.array([1.]),"global")
        instance.close(np.array([1.01]),np.array([1.]),"global")
        self.assertEqual(instance.maximum_absolute_error["global"],2.)
        instance.stream_id = "fixture"
        instance.require(False,"broken")
        self.assertEqual(instance.errors[-1],"fixture: broken")

    def test_rotated_input_integrity_and_native_sensitivity_diagnostic(self):
        identity,_ = fixture(ri=0)
        rotated,_ = fixture(ri=1)
        instance = audit.Audit()
        rows = audit.rotation_check(identity,rotated,.1,19000,instance,windows=(("fixture",1,24),))
        self.assertEqual(instance.errors,[])
        self.assertLess(rows[0]["maximum_action_equivariance_difference_frobenius"],1e-12)
        rotated["output"][2,-1,0] += 1.
        rows = audit.rotation_check(identity,rotated,.1,19000,instance,windows=(("fixture",1,24),))
        self.assertEqual(instance.errors,[])
        self.assertGreater(rows[0]["policies"][-1]["maximum_rotated_output_difference_norm"],.9)
        self.assertTrue(rows[0]["rotation_is_not_an_independent_seed"])
        rotated["canonical"][0,2] += 1
        audit.rotation_check(identity,rotated,.1,19000,instance,windows=(("fixture",1,24),))
        self.assertTrue(instance.errors)

    def test_full_32_seed_aggregate_roster_signs_and_missing(self):
        rows,directions = [],[]
        for index,seed in enumerate(audit.SEEDS):
            for di,process in enumerate((0.,.01,.1)):
                for ri in (0,1):
                    for window,_,_ in audit.WINDOWS:
                        for policy in audit.policy_names(process):
                            mse = (1+index/64 if policy=="native_cp" else
                                   .5+index/64 if policy=="native_cp_star" else 2+index/32)
                            rows.append({"seed":seed,"process_index":di,"rotation_index":ri,
                                "window":window,"policy":policy,"mse":mse})
                        directions.append({"seed":seed,"process_index":di,"rotation_index":ri,
                            "window":window,"native_mean_squared_alignment_when_present":.75,
                            "full_mean_squared_alignment_when_available":.9})
        result = audit.aggregate_metrics(rows,directions)
        self.assertEqual(len(rows),15104)
        self.assertEqual(len(result["equal_seed_mse_summaries"]),472)
        self.assertEqual(len(result["paired_cp_contrasts"]),896)
        primary = result["primary_identity_whole_paired_contrasts"]
        self.assertEqual(len(primary),112)
        self.assertEqual({r["window"] for r in primary},{"whole"})
        def target(result):
            return next(r for r in result["primary_identity_whole_paired_contrasts"] if
                        r["process_index"]==2 and r["target"]=="native_cp" and r["comparator"]=="common_kalman")
        expected = np.array([1+i/64 for i in range(32)])
        self.assertEqual(target(result)["effect"]["mean"],expected.mean())
        self.assertAlmostEqual(target(result)["effect"]["standard_error"],expected.std(ddof=1)/math.sqrt(32))
        self.assertEqual(target(result)["effect"]["positive_seeds"],32)
        cross = [r for r in primary if r["process_index"]==2 and r["comparator"] in audit.CP_POLICIES]
        self.assertEqual(len(cross),2)
        self.assertEqual(sorted(r["effect"]["mean"] for r in cross),[-.5,.5])
        prediction = result["strong_process_registered_alignment_prediction"]
        self.assertTrue(prediction["native_mean_alignment_exceeds_half"])
        self.assertTrue(prediction["whole_horizon_performance_is_not_part_of_this_prediction"])
        for row in rows:
            if row["process_index"]==2 and row["rotation_index"]==0 and row["policy"]=="common_kalman":
                row["mse"] = 0.
        result = audit.aggregate_metrics(rows,directions)
        self.assertLess(target(result)["effect"]["mean"],0)
        self.assertTrue(result["strong_process_registered_alignment_prediction"]["native_mean_alignment_exceeds_half"])
        rows.remove(next(r for r in rows if r["seed"]==19000 and r["process_index"]==2 and r["rotation_index"]==0
                         and r["window"]=="whole" and r["policy"]=="common_kalman"))
        result = audit.aggregate_metrics(rows,directions)
        self.assertIsNone(target(result)["effect"]["mean"])
        directions.remove(next(r for r in directions if r["seed"]==19000 and r["process_index"]==2
                               and r["rotation_index"]==0 and r["window"]=="late"))
        self.assertIsNone(audit.aggregate_metrics(rows,directions)["strong_process_registered_alignment_prediction"]["native_mean_alignment_exceeds_half"])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            audit.aggregate_metrics(rows+[rows[0]],directions)

    def test_rotation_summary_is_same_seed_effect_not_extra_n(self):
        rows = [{"seed":seed,"process_variance":.1,"window":"whole","policies":[
            {"policy":"native_cp_star","rotated_minus_identity_mse":i/32}]} for i,seed in enumerate(audit.SEEDS)]
        result = next(r for r in audit.summarize_rotations(rows) if r["process_variance"]==.1 and
                      r["window"]=="whole" and r["policy"]=="native_cp_star")
        self.assertEqual(result["rotated_minus_identity_mse"]["n_independent_seeds"],32)
        self.assertEqual(result["rotated_minus_identity_mse"]["mean"],15.5/32)
        self.assertTrue(result["rotation_is_not_an_independent_seed"])

    def test_npz_exact_members_no_pickle_and_artifact_hash(self):
        arrays,metadata = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root,path = Path(temporary),Path(temporary)/"fixture.npz"
            np.savez(path,**arrays)
            instance = audit.Audit()
            self.assertEqual(list(audit.load_arrays(path,metadata["policy_names"],instance)),list(arrays))
            record = {"path":"fixture.npz","size":path.stat().st_size,"sha256":audit.sha256(path)}
            self.assertEqual(audit.verify_file(root,record,"fixture.npz",instance),path)
            record["sha256"] = "0"*64
            self.assertIsNone(audit.verify_file(root,record,"fixture.npz",instance))
            with zipfile.ZipFile(path,"a") as archive:
                archive.writestr("extra.npy",b"unexpected")
            self.assertIsNone(audit.load_arrays(path,metadata["policy_names"],instance))
            arrays["canonical"] = np.array([[object()]],dtype=object)
            np.savez(path,**arrays)
            with self.assertRaises(ValueError):
                audit.load_arrays(path,metadata["policy_names"],audit.Audit())

    def test_failed_prefix_retained_without_survivor_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic_root(root)
            instance = audit.Audit()
            with mock.patch.object(audit,"verify_sources",return_value={"synthetic":True}):
                result = audit.audit_root(root,"a"*40,instance)
            self.assertEqual(instance.errors,[])
            self.assertEqual((result["completed_streams"],len(result["missing_streams"])),(1,191))
            self.assertFalse(result["all_scientific_streams_complete"])
            self.assertTrue(all(r["mse"]["mean"] is None for r in result["equal_seed_mse_summaries"]))
            self.assertIsNone(result["strong_process_registered_alignment_prediction"]["native_mean_alignment_exceeds_half"])

    def test_complete_192_roster_tiny_zero_fixture_envelope(self):
        # Two observations per stream, fixed zeros and unused canonical[0,2].
        # This is a file/provenance fixture, never an experimental seed draw.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"streams").mkdir()
            attempt = {"schema":"i19_tracking_attempt_v1","created_utc":"synthetic",
                "root":str(root),"frozen_commit":"a"*40,"sources":[],"seeds":list(audit.SEEDS),
                "process_variances":[0.,.01,.1],"rotations":[0.,math.pi/4],"horizon":2,
                "cooperative_seconds":1600,"array_limit_bytes":2*1024**3,"root_limit_bytes":3*1024**3,
                "pid":123,"service_invocation_id":None,"python":"fixture","numpy":"fixture","torch":"fixture",
                "cpu_threads":1,"platform":"fixture","cuda_visible_devices":"","paid_spend_usd":0,"paid_reserved_usd":0}
            json_file(root/"attempt.json",attempt)
            cells = {(di,ri):fixture(2,process,ri,zero=True) for di,process in enumerate((0.,.01,.1)) for ri in (0,1)}
            inventory = []
            for seed in audit.SEEDS:
                for di,process in enumerate((0.,.01,.1)):
                    for ri in (0,1):
                        arrays,core = cells[di,ri]
                        identity = f"{seed}-p{di}-r{ri}"
                        metadata = {"schema":"i19_tracking_stream_v1","id":identity,"seed":seed,"process_index":di,
                            "rotation_index":ri,"process_variance":process,"rotation":[0.,math.pi/4][ri],"horizon":2,
                            "canonical_sha256":hashlib.sha256(arrays["canonical"].tobytes()).hexdigest(),
                            "frozen_commit":"a"*40,"elapsed_seconds":0.,"core":core}
                        np.savez(root/"streams"/(identity+".npz"),**arrays)
                        json_file(root/"streams"/(identity+".json"),metadata)
                        record = {"id":identity}
                        for role,suffix in (("array","npz"),("metadata","json")):
                            path = root/"streams"/(identity+"."+suffix)
                            record[role] = {"path":str(path.relative_to(root)),"size":path.stat().st_size,"sha256":audit.sha256(path)}
                        inventory.append(record)
            completion = {"schema":"i19_tracking_completion_v1","status":"complete","created_utc":"synthetic",
                "frozen_commit":"a"*40,"attempt_sha256":audit.sha256(root/"attempt.json"),"expected_streams":192,
                "completed_streams":192,"completed_observations":384,"elapsed_seconds":1.,"max_rss_kib":1000,
                "array_bytes":sum(r["array"]["size"] for r in inventory),
                "root_bytes_before_completion":(root/"attempt.json").stat().st_size+sum(
                    r[role]["size"] for r in inventory for role in ("array","metadata")),"failure":None,"streams":inventory}
            json_file(root/"completion.json",completion)
            tiny_windows = tuple((name,1,2) for name,_,_ in audit.WINDOWS)
            # Default window arguments are frozen at definition, so wrap only
            # the read-only metric adapters while the envelope uses tiny H.
            original_metrics,original_rotation = audit.stream_metrics,audit.rotation_check
            def metrics(*args):
                return original_metrics(*args,windows=tiny_windows)
            def rotations(*args):
                return original_rotation(*args,windows=tiny_windows)
            original_validate = audit.validate_arrays
            def validate(*args):
                return original_validate(*args,horizon=2)
            instance = audit.Audit()
            with mock.patch.multiple(audit,N_STEPS=2,WINDOWS=tiny_windows),mock.patch.object(
                    audit,"verify_sources",return_value={"synthetic":True}),mock.patch.object(
                    audit,"stream_metrics",side_effect=metrics),mock.patch.object(
                    audit,"rotation_check",side_effect=rotations),mock.patch.object(
                    audit,"validate_arrays",side_effect=validate):
                result = audit.audit_root(root,"b"*40,instance)
            self.assertEqual(instance.errors,[])
            self.assertTrue(result["all_scientific_streams_complete"])
            self.assertEqual(result["missing_streams"],[])
            self.assertEqual(len(result["array_audit_streams"]),192)
            self.assertEqual(len(result["per_seed_window_metrics"]),15104)
            self.assertEqual(len(result["primary_identity_whole_paired_contrasts"]),112)
            self.assertTrue(all(r["mse"]["mean"]==0 for r in result["equal_seed_mse_summaries"]))

    def test_canonical_hash_failure_survives_outer_rebinding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _,completion = synthetic_root(root)
            record = completion["streams"][0]["metadata"]
            path = root/record["path"]
            metadata = audit.read_json(path)
            metadata["canonical_sha256"] = "0"*64
            json_file(path,metadata)
            old_size = record["size"]
            record.update(size=path.stat().st_size,sha256=audit.sha256(path))
            completion["root_bytes_before_completion"] += record["size"]-old_size
            json_file(root/"completion.json",completion)
            instance = audit.Audit()
            with mock.patch.object(audit,"verify_sources",return_value={}):
                with self.assertRaisesRegex(ValueError,"saved array integrity failed"):
                    audit.audit_root(root,"a"*40,instance)
            self.assertTrue(any("raw shared canonical bytes/hash differ" in e for e in instance.errors))

    def test_envelope_failure_blocks_array_load(self):
        # Construct a valid saved fixture once, then mutate only its envelope;
        # each root remains artificial and source checks are covered separately.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_attempt,original_completion = synthetic_root(root)
            for mutation in ("attempt","complete_short","extra_file","extra_directory","reordered_id","accounting","old_cap"):
                with self.subTest(mutation=mutation):
                    attempt,completion = copy.deepcopy(original_attempt),copy.deepcopy(original_completion)
                    extra = None
                    if mutation=="attempt":
                        attempt["horizon"] = 4001
                    elif mutation=="complete_short":
                        completion["status"],completion["failure"] = "complete",None
                    elif mutation=="extra_file":
                        extra = root/"streams"/"unexpected"
                        extra.write_text("unexpected")
                    elif mutation=="extra_directory":
                        extra = root/"streams"/"empty"
                        extra.mkdir()
                    elif mutation=="reordered_id":
                        completion["streams"][0]["id"] = "19001-p0-r0"
                    elif mutation=="old_cap":
                        attempt["array_limit_bytes"] = 1024**3
                    else:
                        completion["array_bytes"] += 1
                    json_file(root/"attempt.json",attempt)
                    json_file(root/"completion.json",completion)
                    instance = audit.Audit()
                    with mock.patch.object(audit,"verify_sources",return_value={}),mock.patch.object(audit,"load_arrays") as load:
                        with self.assertRaises(ValueError):
                            audit.audit_root(root,"a"*40,instance)
                        load.assert_not_called()
                    self.assertTrue(instance.errors)
                    if extra is not None:
                        extra.rmdir() if extra.is_dir() else extra.unlink()

    def test_source_closure_acquisition_analysis_commits_current_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            paths = ("native.py","core.py","runner.py")
            frozen = {}
            for name in paths+("audit_stochastic_tracking.py","test_audit_stochastic_tracking.py"):
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
                self.assertEqual(len(audit.verify_sources(attempt,"b"*40,instance)["analysis_sources"]),2)
                self.assertEqual(instance.errors,[])
                (repo/"core.py").write_text("# changed")
                instance = audit.Audit()
                audit.verify_sources(attempt,"b"*40,instance)
                self.assertTrue(any("size/hash differs" in e for e in instance.errors))
                frozen["audit_stochastic_tracking.py"] = b"different frozen bytes"
                instance = audit.Audit()
                audit.verify_sources(attempt,"b"*40,instance)
                self.assertTrue(any("analysis source not frozen" in e for e in instance.errors))

    def test_import_does_not_load_torch_native_core_or_call_rng(self):
        code = ("import importlib.util,sys,numpy as np; "
            "np.random.seed=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('RNG called')); "
            "s=importlib.util.spec_from_file_location('auditor',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; "
            "assert not any('tracking_core' in key or 'spectral_filter' in key for key in sys.modules)")
        result = subprocess.run([sys.executable,"-c",code,str(HERE/"audit_stochastic_tracking.py")],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"test.json"
            for content in ('{"x":1,"x":2}','{"x":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    audit.read_json(path)


if __name__=="__main__":
    unittest.main()
