"""Synthetic-only tests for independent I19 saved-output report arithmetic."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_i19_report_test",HERE/"check_stochastic_report.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def expected_stats(values):
    ordered = list(values.values())
    return {"per_seed":{str(seed):value for seed,value in values.items()},"available":True,
        "n_independent_seeds":len(values),"mean":statistics.mean(ordered),
        "standard_error":statistics.stdev(ordered)/math.sqrt(len(values)),
        "positive_seeds":sum(v>0 for v in ordered),"negative_seeds":sum(v<0 for v in ordered),
        "zero_seeds":sum(v==0 for v in ordered),"no_survivor_averaging":True}


def scalar_arrays(names,horizon=4):
    t = np.arange(horizon,dtype=np.float64)
    signal = np.column_stack((t,-t))
    g = signal+np.array([.5,-1.])
    output = np.empty((horizon,len(names),2),dtype=np.float64)
    for j,name in enumerate(names):
        output[:,j] = g if name=="raw" else signal+np.array([j/16,-j/32])
        output[0,j] = g[0]
    arrays = {name:np.zeros(1,dtype=np.float64) for name in check.ARRAY_KEYS}
    arrays.update(g=g,s=signal,output=output)
    return arrays


def scalar_expected(arrays,names,windows):
    result = {}
    for window,first,last in windows:
        for column,name in enumerate(names):
            errors = [[float(arrays["output"][t,column,j]-arrays["s"][t,j]) for j in range(2)] for t in range(first,last)]
            n = len(errors)
            mean = [sum(row[j] for row in errors)/n for j in range(2)]
            mse = sum(sum(x*x for x in row) for row in errors)/n
            squared_mean = sum(x*x for x in mean)
            dispersion = sum(sum((row[j]-mean[j])**2 for j in range(2)) for row in errors)/n
            result[window,name] = {"observations":n,"mse":mse,"mean_error_vector":mean,
                "squared_mean_error":squared_mean,"temporal_error_dispersion":dispersion,
                "mse_decomposition_residual":mse-squared_mean-dispersion,
                "dispersion_is_not_independent_sample_uncertainty":True}
    return result


def summary_from_raw(raw):
    rows,grouped = [],{}
    for (seed,di,ri,window,policy),metrics in raw.items():
        rows.append({"seed":seed,"process_index":di,"rotation_index":ri,"window":window,"policy":policy,**metrics})
        grouped.setdefault((di,ri,window,policy),{})[seed] = metrics["mse"]
    means = [{"process_index":di,"rotation_index":ri,"window":window,"policy":policy,
              "policy_class":check.policy_class(policy),"mse":expected_stats(values)}
             for (di,ri,window,policy),values in grouped.items()]
    paired = []
    for (di,ri,window,policy),values in grouped.items():
        for target in ("native_cp","native_cp_star"):
            if policy==target:
                continue
            effects = {seed:value-grouped[di,ri,window,target][seed] for seed,value in values.items()}
            paired.append({"process_index":di,"rotation_index":ri,"window":window,"target":target,"comparator":policy,
                "comparison":"comparator_mse_minus_target_mse","positive_favors":target,
                "comparator_class":check.policy_class(policy),"effect":expected_stats(effects)})
    return {"per_seed_window_metrics":rows,"equal_seed_mse_summaries":means,"paired_cp_contrasts":paired,
            "primary_identity_whole_paired_contrasts":[copy.deepcopy(row) for row in paired if row["rotation_index"]==0 and row["window"]=="whole"]}


def fabricated_raw():
    result = {}
    for i,seed in enumerate(check.SEEDS):
        for di,process in enumerate((0.,.01,.1)):
            for ri in (0,1):
                for window,first,last in check.WINDOWS:
                    for policy in check.policies(process):
                        value = (1+i/64 if policy=="native_cp" else .5+i/64 if policy=="native_cp_star" else 2+i/32)
                        result[seed,di,ri,window,policy] = {"observations":last-first,"mse":value,
                            "mean_error_vector":[0.,0.],"squared_mean_error":0.,"temporal_error_dispersion":value,
                            "mse_decomposition_residual":0.,"dispersion_is_not_independent_sample_uncertainty":True}
    return result


def json_file(path,value):
    path.write_text(json.dumps(value,allow_nan=False,sort_keys=True))


def envelope_fixture(root,horizon=2):
    """Full 192-stream inventory with artificial two-row arrays, not science."""
    (root/"streams").mkdir()
    windows = tuple((name,0,horizon) for name,_,_ in check.WINDOWS)
    commit = "a"*40
    provenance = {"acquisition_commit":commit,"acquisition_sources":[],"analysis_commit":"b"*40,"analysis_sources":[]}
    attempt = {"schema":"i19_tracking_attempt_v1","root":str(root),"frozen_commit":commit,"sources":[],
        "seeds":list(check.SEEDS),"process_variances":[0.,.01,.1],"rotations":[0.,math.pi/4],"horizon":horizon,
        "cpu_threads":1,"cuda_visible_devices":"","array_limit_bytes":2*1024**3,"root_limit_bytes":3*1024**3,
        "paid_spend_usd":0,"paid_reserved_usd":0}
    json_file(root/"attempt.json",attempt)
    records,raw = [],{}
    for seed in check.SEEDS:
        for di,process in enumerate((0.,.01,.1)):
            names = check.policies(process)
            arrays = scalar_arrays(names,horizon)
            metrics = scalar_expected(arrays,names,windows)
            for ri in (0,1):
                identity = f"{seed}-p{di}-r{ri}"
                metadata = {"schema":"i19_tracking_stream_v1","id":identity,"seed":seed,"process_index":di,
                    "rotation_index":ri,"process_variance":process,"rotation":[0.,math.pi/4][ri],"horizon":horizon,
                    "canonical_sha256":"0"*64,"frozen_commit":commit,"elapsed_seconds":0.,
                    "core":{"schema":"i19_stochastic_tracking_stream_metadata_v1","array_schema":"i19_stochastic_tracking_arrays_v1",
                        "horizon":horizon,"process_variance":process,"rotation_radians":[0.,math.pi/4][ri],
                        "policy_names":names,"policy_count":len(names),"array_order":list(check.ARRAY_KEYS)}}
                np.savez(root/"streams"/(identity+".npz"),**arrays)
                json_file(root/"streams"/(identity+".json"),metadata)
                record = {"id":identity}
                for role,suffix in (("array","npz"),("metadata","json")):
                    path = root/"streams"/(identity+"."+suffix)
                    record[role] = {"path":str(path.relative_to(root)),"size":path.stat().st_size,"sha256":check.sha(path)}
                records.append(record)
                for (window,policy),value in metrics.items():
                    raw[seed,di,ri,window,policy] = value
    completion = {"schema":"i19_tracking_completion_v1","status":"complete","failure":None,"frozen_commit":commit,
        "attempt_sha256":check.sha(root/"attempt.json"),"expected_streams":192,"completed_streams":192,
        "completed_observations":192*horizon,"streams":records,"array_bytes":sum(r["array"]["size"] for r in records),
        "root_bytes_before_completion":(root/"attempt.json").stat().st_size+sum(r[role]["size"] for r in records for role in ("array","metadata"))}
    json_file(root/"completion.json",completion)
    summary = summary_from_raw(raw)
    summary.update({"schema":"i19_tracking_summary_v1","audit_status":"pass","artifact_root":str(root),
        "input_provenance":provenance,"acquisition_status":"complete","all_scientific_streams_complete":True,
        "completed_streams":192,"expected_streams":192,"missing_streams":[],
        "attempt_sha256":check.sha(root/"attempt.json"),"completion_sha256":check.sha(root/"completion.json")})
    # Accepted inputs live outside the acquisition root's exact file inventory.
    return summary,windows


class StochasticReportTests(unittest.TestCase):
    def test_stats_mean_se_signs_and_missing(self):
        values = {1:-1.,2:0.,3:4.}
        observed = check.stats(values,seeds=(1,2,3))
        self.assertEqual(observed["mean"],1.)
        self.assertAlmostEqual(observed["standard_error"],math.sqrt(7/3))
        self.assertEqual((observed["positive_seeds"],observed["negative_seeds"],observed["zero_seeds"]),(1,1,1))
        with self.assertRaisesRegex(ValueError,"incomplete"):
            check.stats({1:1.},seeds=(1,2))
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            check.stats({1:float("nan"),2:1.},seeds=(1,2))

    def test_compare_stats_detects_scalar_and_seed_mutations(self):
        values = {1:1.,2:3.}
        row = expected_stats(values)
        check.compare_stats(row,values,check.Compare(),"fixture",seeds=(1,2))
        for name in ("mean","standard_error","positive_seeds","available","per_seed"):
            changed = copy.deepcopy(row)
            if name=="per_seed":
                changed[name]["1"] = 7.
            else:
                changed[name] = False if name=="available" else 7
            with self.assertRaisesRegex(ValueError,"differs"):
                check.compare_stats(changed,values,check.Compare(),"fixture",seeds=(1,2))

    def test_cancellation_residual_scales_by_components_not_near_zero_result(self):
        compare = check.Compare()
        compare.number(1e-10,0.,"large-component residual",scale=10000.)
        with self.assertRaisesRegex(ValueError,"differs"):
            compare.number(.01,0.,"material residual",scale=10000.)
        with self.assertRaisesRegex(ValueError,"differs"):
            compare.number(1e-10,0.,"ordinary near-zero metric")

    def test_scalar_array_metrics_and_load_only_g_s_output(self):
        names = check.policies(.1)
        arrays = scalar_arrays(names)
        windows = (("whole",0,4),("last",2,4))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"fixture.npz"
            # Invalid object payloads in unread state members demonstrate that
            # report corroboration does not deserialize scientific state.
            arrays["mu"] = np.array([object()],dtype=object)
            np.savez(path,**arrays)
            observed = check.load_metrics(path,names,4,windows)
            expected = scalar_expected(arrays,names,windows)
            self.assertEqual(observed,expected)
            self.assertEqual(observed["whole","raw"]["mse"],1.25)
            arrays["output"][0,1,0] += .01
            np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,"first policy"):
                check.load_metrics(path,names,4,windows)

    def test_npz_roster_dtype_and_raw_column_rejections(self):
        names = check.policies(0.)
        original = scalar_arrays(names)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"fixture.npz"
            for mutation in ("dtype","nan","raw","extra","duplicate"):
                arrays = copy.deepcopy(original)
                if mutation=="dtype":
                    arrays["s"] = arrays["s"].astype(np.float32)
                elif mutation=="nan":
                    arrays["g"][2,0] = np.nan
                elif mutation=="raw":
                    arrays["output"][2,0,0] += 1.
                np.savez(path,**arrays)
                if mutation in ("extra","duplicate"):
                    with zipfile.ZipFile(path,"a") as archive:
                        archive.writestr("extra.npy" if mutation=="extra" else "g.npy",b"unexpected")
                with self.assertRaises(ValueError):
                    check.load_metrics(path,names,4,(("whole",0,4),))

    def test_complete_scalar_matrix_and_all_112_primaries(self):
        raw = fabricated_raw()
        summary = summary_from_raw(raw)
        compare = check.Compare()
        result = check.compare_arithmetic(summary,raw,compare)
        self.assertEqual((result["per_seed_window_metrics_checked"],result["equal_seed_mse_summaries_checked"],
                          result["paired_cp_contrasts_checked"],result["primary_contrasts_checked"]),(15104,472,896,112))
        self.assertGreater(compare.count,150000)
        cross = [r for r in result["primary_identity_whole_paired_contrasts"] if r["process_index"]==2 and r["comparator"] in check.TARGETS]
        self.assertEqual(sorted(r["effect"]["mean"] for r in cross),[-.5,.5])
        self.assertEqual({r["window"] for r in result["primary_identity_whole_paired_contrasts"]},{"whole"})

    def test_mutations_of_all_summary_layers_rejected(self):
        raw = fabricated_raw()
        original = summary_from_raw(raw)
        for mutation in ("mse","mean_error","mean","se","effect","sign","label","primary","duplicate","missing"):
            summary = copy.deepcopy(original)
            if mutation=="mse":
                summary["per_seed_window_metrics"][0]["mse"] += .1
            elif mutation=="mean_error":
                summary["per_seed_window_metrics"][0]["mean_error_vector"][0] += .1
            elif mutation in ("mean","se"):
                summary["equal_seed_mse_summaries"][0]["mse"]["mean" if mutation=="mean" else "standard_error"] += .1
            elif mutation in ("effect","sign","label"):
                row = summary["paired_cp_contrasts"][0]
                if mutation=="effect":
                    row["effect"]["mean"] += .1
                elif mutation=="sign":
                    row["effect"]["positive_seeds"] -= 1
                else:
                    row["positive_favors"] = "raw"
            elif mutation=="primary":
                summary["primary_identity_whole_paired_contrasts"][0]["effect"]["mean"] += .1
            elif mutation=="duplicate":
                summary["paired_cp_contrasts"].append(summary["paired_cp_contrasts"][0])
            else:
                summary["per_seed_window_metrics"].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                check.compare_arithmetic(summary,raw,check.Compare())

    def test_source_closure_hash_commit_and_current_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            names = ("native.py","core.py")
            frozen = {}
            for name in names:
                (repo/name).write_text("# synthetic "+name)
                frozen[name] = (repo/name).read_bytes()
            records = [{"path":name,"size":(repo/name).stat().st_size,"sha256":check.sha(repo/name)} for name in names]
            def show(command,**kwargs):
                self.assertEqual(command[:2],["git","show"])
                return frozen[command[2].split(":",1)[1]]
            with mock.patch.object(check,"REPO",repo),mock.patch.object(check.subprocess,"check_output",side_effect=show):
                check.verify_source_set(records,names,"a"*40)
                with self.assertRaisesRegex(ValueError,"closure"):
                    check.verify_source_set(list(reversed(records)),names,"a"*40)
                frozen["core.py"] = b"different"
                with self.assertRaisesRegex(ValueError,"frozen source"):
                    check.verify_source_set(records,names,"a"*40)
                (repo/"core.py").write_text("changed")
                with self.assertRaisesRegex(ValueError,"file differs"):
                    check.verify_source_set(records,names,"a"*40)

    def test_file_hash_size_path_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"value").write_text("fixed")
            row = {"path":"value","size":5,"sha256":check.sha(root/"value")}
            check.verify_file(root,row,"value")
            for changed in (dict(row,size=6),dict(row,sha256="0"*64),dict(row,path="../value")):
                with self.assertRaises(ValueError):
                    check.verify_file(root,changed,"value")
            (root/"link").symlink_to(root/"value")
            with self.assertRaises(ValueError):
                check.verify_file(root,dict(row,path="link"),"link")

    def test_full_envelope_success_and_preload_closure_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory/"artifacts"
            root.mkdir()
            summary,windows = envelope_fixture(root)
            summary_path,audit_path = directory/"summary.json",directory/"audit.json"
            json_file(summary_path,summary)
            accepted = {"schema":"i19_tracking_audit_v1","status":"pass","errors":[],"summary_sha256":check.sha(summary_path),
                "artifact_root":str(root),"input_provenance":summary["input_provenance"],
                "attempt_sha256":summary["attempt_sha256"],"completion_sha256":summary["completion_sha256"]}
            json_file(audit_path,accepted)
            pins = check.sha(summary_path),check.sha(audit_path)
            original_arithmetic = check.compare_arithmetic
            def arithmetic(*args):
                return original_arithmetic(*args,windows=windows)
            with mock.patch.multiple(check,HORIZON=2,WINDOWS=windows),mock.patch.object(
                    check,"verify_sources",return_value={"synthetic":True}),mock.patch.object(
                    check,"compare_arithmetic",side_effect=arithmetic):
                result = check.corroborate(root,summary_path,audit_path,*pins,"c"*40)
                self.assertEqual(result["status"],"pass")
                self.assertEqual(result["streams_checked"],192)
                self.assertEqual(result["primary_contrasts_checked"],112)
                for mutation in ("accepted_hash","extra_file","last_file_hash","source"):
                    extra = None
                    if mutation=="extra_file":
                        extra = root/"streams"/"unexpected"
                        extra.write_text("unexpected")
                    elif mutation=="last_file_hash":
                        path = root/"streams/19031-p2-r1.npz"
                        altered = bytearray(path.read_bytes())
                        altered[-1] ^= 1
                        path.write_bytes(altered)
                    with self.subTest(mutation=mutation),mock.patch.object(check,"load_metrics") as load:
                        if mutation=="source":
                            with mock.patch.object(check,"verify_sources",side_effect=ValueError("source mismatch")):
                                with self.assertRaises(ValueError):
                                    check.corroborate(root,summary_path,audit_path,*pins,"c"*40)
                        else:
                            args_pins = ("0"*64,pins[1]) if mutation=="accepted_hash" else pins
                            with self.assertRaises(ValueError):
                                check.corroborate(root,summary_path,audit_path,*args_pins,"c"*40)
                        load.assert_not_called()
                    if extra is not None:
                        extra.unlink()

    def test_full_source_linkage_has_separate_acquisition_analysis_report_commits(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            acquisition = tuple(f"acquisition_{i}.py" for i in range(7))
            analysis,report = ("analysis.py","test_analysis.py"),("report.py","test_report.py")
            frozen = {}
            records = {}
            for names,commit in ((acquisition,"a"*40),(analysis,"b"*40),(report,"c"*40)):
                rows = []
                for name in names:
                    (repo/name).write_text("# synthetic "+name)
                    frozen[commit+":"+name] = (repo/name).read_bytes()
                    rows.append({"path":name,"size":(repo/name).stat().st_size,"sha256":check.sha(repo/name)})
                records[commit] = rows
            attempt = {"frozen_commit":"a"*40,"sources":records["a"*40]}
            provenance = {"acquisition_commit":"a"*40,"acquisition_sources":records["a"*40],
                          "analysis_commit":"b"*40,"analysis_sources":records["b"*40]}
            def show(command,**kwargs):
                return frozen[command[2]]
            with mock.patch.multiple(check,REPO=repo,ACQUISITION_SOURCES=acquisition,ANALYSIS_SOURCES=analysis,REPORT_SOURCES=report),mock.patch.object(
                    check.subprocess,"check_output",side_effect=show):
                result = check.verify_sources(attempt,provenance,"c"*40)
                self.assertEqual(result["report_sources"],records["c"*40])
                self.assertEqual(result["report_commit"],"c"*40)
                changed = copy.deepcopy(provenance)
                changed["acquisition_commit"] = "d"*40
                with self.assertRaisesRegex(ValueError,"linkage"):
                    check.verify_sources(attempt,changed,"c"*40)
                frozen["c"*40+":report.py"] = b"wrong frozen report"
                with self.assertRaisesRegex(ValueError,"frozen source"):
                    check.verify_sources(attempt,provenance,"c"*40)

    def test_exclusive_output_preserves_existing_directory_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory/"report-check-001"
            with mock.patch.multiple(check,HERE=directory,OUTPUT=output):
                check.reserve_output()
                marker = output/"preserved"
                marker.write_text("fixture")
                with self.assertRaises(FileExistsError):
                    check.reserve_output()
                self.assertEqual(marker.read_text(),"fixture")
            other = directory/"other"
            other.mkdir()
            link = other/"report-check-001"
            link.symlink_to(output,target_is_directory=True)
            with mock.patch.multiple(check,HERE=other,OUTPUT=link),self.assertRaisesRegex(ValueError,"exclusive"):
                check.reserve_output()

    def test_import_is_safe_and_json_is_strict(self):
        code = ("import importlib.util,sys,numpy as np; "
            "np.random.seed=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('RNG called')); "
            "s=importlib.util.spec_from_file_location('report',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; "
            "assert not any('tracking_core' in k or 'spectral_filter' in k or 'audit_stochastic' in k for k in sys.modules)")
        result = subprocess.run([sys.executable,"-c",code,str(HERE/"check_stochastic_report.py")],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"value.json"
            for content in ('{"a":1,"a":2}','{"a":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    check.read_json(path)


if __name__=="__main__":
    unittest.main()
