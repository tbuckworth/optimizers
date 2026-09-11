#!/usr/bin/env python3
"""Independent I19 report arithmetic from saved g/s/output; no state replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SUMMARY = HERE/"analysis-001/summary.json"
AUDIT = HERE/"analysis-001/audit.json"
OUTPUT = HERE/"report-check-001"
SEEDS = tuple(range(19000,19032))
PROCESS_VARIANCES = (0.,.01,.1)
ROTATIONS = (0,1)
HORIZON = 4000
WINDOWS = (("whole",0,4000),("startup",0,100),("transition",100,1000),("late",1000,4000))
TARGETS = ("native_cp","native_cp_star")
RELATIVE_TOLERANCE = 5e-13
ACQUISITION_SOURCES = ("spectral_filter.py",)+tuple(str((HERE/name).relative_to(REPO)) for name in (
    "stochastic_tracking_core.py","test_stochastic_tracking_core.py","run_stochastic_tracking.py",
    "test_stochastic_tracking_runner.py","protocol.md","best-practices-check.md"))
ANALYSIS_SOURCES = tuple(str((HERE/name).relative_to(REPO)) for name in
                         ("audit_stochastic_tracking.py","test_audit_stochastic_tracking.py"))
REPORT_SOURCES = tuple(str((HERE/name).relative_to(REPO)) for name in
                       ("check_stochastic_report.py","test_stochastic_report.py"))
ARRAY_KEYS = tuple("""canonical noise epsilon g s mu A native_delivery native_h full_moment full_action full_gap
full_basis_present full_basis_V basis_present basis_V basis_S output ema_fixed_state dema_first_state dema_second_state
ema_common_state common_kalman_state common_kalman_prior_weight common_kalman_posterior_variance
useful_oracle_kalman_state_base useful_oracle_kalman_prior_weight useful_oracle_kalman_posterior_variance
oracle_useful_fast_state oracle_useful_slow_state oracle_useful_star_fast_state scalar_buffer native_i17_buffer
native_cp_buffer native_cp_star_buffer native_action_raw_error ema_fixed_response_residual dema_first_response_residual
dema_second_response_residual ema_common_response_residual common_kalman_response_residual common_kalman_variance_residual
useful_oracle_kalman_response_residual useful_oracle_kalman_variance_residual oracle_useful_fast_response_residual
oracle_useful_slow_response_residual oracle_useful_star_fast_response_residual scalar_response_residual
native_i17_response_residual native_cp_response_residual native_cp_star_response_residual full_moment_residual
native_action_idempotence_error native_basis_orthogonality_error native_useful_squared_alignment
full_useful_squared_alignment""".split())


def need(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key,value in items:
            need(key not in result,"duplicate JSON key: "+key)
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle,object_pairs_hook=pairs,
                         parse_constant=lambda value:(_ for _ in ()).throw(ValueError("nonfinite JSON: "+value)))


def policies(process):
    result = ["raw","ema_q0p9","ema_q0p99","ema_q0p999","dema_q0p9","dema_q0p99","dema_q0p999"]
    if process:
        result.append("ema_common_steady_oracle")
    return result+["common_kalman","useful_oracle_kalman","scalar_k0","scalar_k0p5","scalar_k0p9","scalar_k1",
                   "oracle_useful","oracle_nuisance","native_i17","native_cp","native_cp_star","oracle_useful_star"]


def policy_class(name):
    if name.startswith("native_"):
        return "native_learned_direction"
    if name.startswith("oracle_") or name=="useful_oracle_kalman":
        return "generating_direction_oracle"
    if name in ("common_kalman","ema_common_steady_oracle"):
        return "model_based_common_control"
    if name.startswith("scalar_"):
        return "scalar_mixture"
    if name.startswith("dema_"):
        return "signed_uniform_DEMA"
    return "raw" if name=="raw" else "fixed_uniform_EMA"


class Compare:
    def __init__(self):
        self.count,self.maximum = 0,0.

    def number(self,observed,expected,label,*,scale=0.):
        need(type(observed) in (int,float) and type(expected) in (int,float)
             and math.isfinite(observed) and math.isfinite(expected) and math.isfinite(scale),label+": nonfinite/non-numeric")
        difference = abs(observed-expected)
        self.count += 1
        self.maximum = max(self.maximum,difference)
        need(difference<=RELATIVE_TOLERANCE*max(1.,abs(expected),abs(scale)),label+": differs")

    def exact(self,observed,expected,label):
        self.count += 1
        need(type(observed) is type(expected) and observed==expected,label+": differs")


def stats(values,seeds=SEEDS):
    """Scalar fsum calculation, separate from the primary NumPy aggregator."""
    need(set(values)==set(seeds) and len(seeds)>=2,"incomplete seed roster")
    ordered = [values[seed] for seed in seeds]
    need(all(type(value) in (int,float) and math.isfinite(value) for value in ordered),"nonfinite seed metric")
    n = len(ordered)
    mean = math.fsum(ordered)/n
    variance = math.fsum((value-mean)**2 for value in ordered)/(n-1)
    return {"per_seed":{str(seed):values[seed] for seed in seeds},"available":True,"n_independent_seeds":n,
        "mean":mean,"standard_error":math.sqrt(variance/n),"positive_seeds":sum(value>0 for value in ordered),
        "negative_seeds":sum(value<0 for value in ordered),"zero_seeds":sum(value==0 for value in ordered),
        "no_survivor_averaging":True}


def compare_stats(observed,values,compare,label,seeds=SEEDS):
    expected = stats(values,seeds)
    compare.exact(set(observed),set(expected),label+" fields")
    compare.exact(set(observed["per_seed"]),set(expected["per_seed"]),label+" seed roster")
    for seed,value in expected["per_seed"].items():
        compare.number(observed["per_seed"][seed],value,label+" seed "+seed)
    for name,value in expected.items():
        if name=="per_seed":
            continue
        (compare.number if name in ("mean","standard_error") else compare.exact)(observed[name],value,label+" "+name)
    return expected


def verify_file(root,record,expected):
    need(type(record) is dict and set(record)=={"path","size","sha256"}
         and record["path"]==expected and type(record["size"]) is int and record["size"]>=0
         and type(record["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}",record["sha256"]),
         "invalid file record: "+expected)
    path = root/expected
    need(path.is_file() and not path.is_symlink() and path.stat().st_size==record["size"]
         and sha(path)==record["sha256"],"file differs: "+expected)
    return path


def verify_source_set(records,names,commit):
    need(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}",commit),"full source commit required")
    need(type(records) is list and len(records)==len(names)
         and [record.get("path") for record in records if type(record) is dict]==list(names),"source closure differs")
    for record,name in zip(records,names):
        verify_file(REPO,record,name)
        frozen = subprocess.check_output(["git","show",commit+":"+name],cwd=REPO)
        need(hashlib.sha256(frozen).hexdigest()==record["sha256"],"frozen source differs: "+name)


def verify_sources(attempt,provenance,report_commit):
    need(set(provenance)=={"acquisition_commit","acquisition_sources","analysis_commit","analysis_sources"}
         and provenance["acquisition_commit"]==attempt["frozen_commit"]
         and provenance["acquisition_sources"]==attempt["sources"],"accepted source linkage differs")
    verify_source_set(attempt["sources"],ACQUISITION_SOURCES,attempt["frozen_commit"])
    verify_source_set(provenance["analysis_sources"],ANALYSIS_SOURCES,provenance["analysis_commit"])
    records = [{"path":name,"size":(REPO/name).stat().st_size,"sha256":sha(REPO/name)} for name in REPORT_SOURCES]
    verify_source_set(records,REPORT_SOURCES,report_commit)
    return {"acquisition_and_analysis":provenance,"report_commit":report_commit,"report_sources":records}


def load_metrics(path,names,horizon,windows):
    """Read only the three scalar-report arrays, never the saved observer state."""
    with zipfile.ZipFile(path) as archive:
        records = archive.infolist()
        need([r.filename for r in records]==[name+".npy" for name in ARRAY_KEYS]
             and all(0<=r.file_size<=16*1024**2 for r in records)
             and sum(r.file_size for r in records)<=32*1024**2,"NPZ roster or expanded size differs")
    with np.load(path,allow_pickle=False,max_header_size=10000) as archive:
        output,signal,g = archive["output"],archive["s"],archive["g"]
    need(output.shape==(horizon,len(names),2) and signal.shape==g.shape==(horizon,2)
         and all(value.dtype==np.float64 and np.isfinite(value).all() for value in (output,signal,g)),
         "report arrays shape/dtype/finiteness differs")
    need(np.array_equal(output[:,names.index("raw")],g),"raw output column differs from saved g")
    need(np.array_equal(output[0],np.broadcast_to(g[0],(len(names),2))),"first policy outputs differ from g1")
    result = {}
    for window,first,last in windows:
        need(0<=first<last<=horizon,"invalid report window")
        error = output[first:last]-signal[first:last,None,:]
        squared = (error*error).sum(axis=2)
        mse = squared.sum(axis=0)/(last-first)
        means = error.sum(axis=0)/(last-first)
        bias = (means*means).sum(axis=1)
        centered = error-means[None,:,:]
        dispersion = (centered*centered).sum(axis=(0,2))/(last-first)
        for column,name in enumerate(names):
            result[(window,name)] = {"observations":last-first,"mse":float(mse[column]),
                "mean_error_vector":means[column].tolist(),"squared_mean_error":float(bias[column]),
                "temporal_error_dispersion":float(dispersion[column]),
                "mse_decomposition_residual":float(mse[column]-bias[column]-dispersion[column]),
                "dispersion_is_not_independent_sample_uncertainty":True}
    return result


def compare_arithmetic(summary,raw,compare,*,seeds=SEEDS,processes=PROCESS_VARIANCES,rotations=ROTATIONS,windows=WINDOWS):
    expected_keys = {(seed,di,ri,window,name) for seed in seeds for di,process in enumerate(processes)
                     for ri in rotations for window,_,_ in windows for name in policies(process)}
    need(set(raw)==expected_keys,"raw scalar roster incomplete")
    indexed = {(r["seed"],r["process_index"],r["rotation_index"],r["window"],r["policy"]):r
               for r in summary["per_seed_window_metrics"]}
    need(len(indexed)==len(summary["per_seed_window_metrics"]) and set(indexed)==expected_keys,"per-seed summary roster differs")
    grouped = {}
    for key,metrics in raw.items():
        observed = indexed[key]
        for name in ("mse","squared_mean_error","temporal_error_dispersion","mse_decomposition_residual"):
            # A subtraction residual near zero inherits roundoff at the scale
            # of its MSE components; it is not a separate tiny-scale estimate.
            compare.number(observed[name],metrics[name],str(key)+" "+name,
                           scale=metrics["mse"] if name=="mse_decomposition_residual" else 0.)
        for name in ("observations","dispersion_is_not_independent_sample_uncertainty"):
            compare.exact(observed[name],metrics[name],str(key)+" "+name)
        compare.exact(len(observed["mean_error_vector"]),2,str(key)+" mean vector dimension")
        for j in range(2):
            compare.number(observed["mean_error_vector"][j],metrics["mean_error_vector"][j],str(key)+" error vector")
        seed,di,ri,window,name = key
        grouped.setdefault((di,ri,window,name),{})[seed] = metrics["mse"]
    means = {(r["process_index"],r["rotation_index"],r["window"],r["policy"]):r for r in summary["equal_seed_mse_summaries"]}
    need(len(means)==len(summary["equal_seed_mse_summaries"]) and set(means)==set(grouped),"equal-seed mean roster differs")
    for key,values in grouped.items():
        compare.exact(means[key]["policy_class"],policy_class(key[-1]),"mean policy class")
        compare_stats(means[key]["mse"],values,compare,"mean "+str(key),seeds)
    expected_paired = {(di,ri,window,target,name) for di,ri,window,name in grouped for target in TARGETS if name!=target}
    paired = {(r["process_index"],r["rotation_index"],r["window"],r["target"],r["comparator"]):r
              for r in summary["paired_cp_contrasts"]}
    need(len(paired)==len(summary["paired_cp_contrasts"]) and set(paired)==expected_paired,"paired contrast roster differs")
    derived = {}
    for key in sorted(expected_paired):
        di,ri,window,target,name = key
        row = paired[key]
        for field,value in (("comparison","comparator_mse_minus_target_mse"),("positive_favors",target),
                            ("comparator_class",policy_class(name))):
            compare.exact(row[field],value,"paired "+str(key)+" "+field)
        effects = {seed:grouped[di,ri,window,name][seed]-grouped[di,ri,window,target][seed] for seed in seeds}
        effect = compare_stats(row["effect"],effects,compare,"paired "+str(key),seeds)
        derived[key] = {"process_index":di,"rotation_index":ri,"window":window,"target":target,"comparator":name,
            "comparison":"comparator_mse_minus_target_mse","positive_favors":target,
            "comparator_class":policy_class(name),"effect":effect}
    expected_primary = {key for key in expected_paired if key[1]==0 and key[2]=="whole"}
    primary = {(r["process_index"],r["rotation_index"],r["window"],r["target"],r["comparator"]):r
               for r in summary["primary_identity_whole_paired_contrasts"]}
    need(len(primary)==len(summary["primary_identity_whole_paired_contrasts"]) and set(primary)==expected_primary,
         "primary identity-whole roster differs")
    for key,row in primary.items():
        compare.exact(row,paired[key],"primary row must equal registered full-matrix row")
    return {"per_seed_window_metrics_checked":len(raw),"equal_seed_mse_summaries_checked":len(means),
        "paired_cp_contrasts_checked":len(paired),"primary_contrasts_checked":len(primary),
        "primary_identity_whole_paired_contrasts":[derived[key] for key in sorted(expected_primary)]}


def corroborate(root,summary_path,audit_path,summary_sha,audit_sha,report_commit,*,check_budget=lambda:None):
    root = Path(root)
    need(root.is_absolute() and root.resolve()==root and not root.is_symlink(),"direct canonical root required")
    for path,digest in ((summary_path,summary_sha),(audit_path,audit_sha)):
        need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}",digest),"explicit accepted hash required")
        need(path.is_file() and not path.is_symlink() and sha(path)==digest,"accepted input hash differs")
    summary,accepted = read_json(summary_path),read_json(audit_path)
    need(summary["schema"]=="i19_tracking_summary_v1" and accepted["schema"]=="i19_tracking_audit_v1"
         and summary["audit_status"]==accepted["status"]=="pass" and accepted["errors"]==[]
         and accepted["summary_sha256"]==summary_sha and summary["artifact_root"]==accepted["artifact_root"]==str(root)
         and summary["input_provenance"]==accepted["input_provenance"] and summary["acquisition_status"]=="complete"
         and summary["all_scientific_streams_complete"] is True and summary["completed_streams"]==summary["expected_streams"]==192
         and summary["missing_streams"]==[],"accepted evidence envelope differs")
    attempt_path,completion_path = root/"attempt.json",root/"completion.json"
    for path,key in ((attempt_path,"attempt_sha256"),(completion_path,"completion_sha256")):
        need(path.is_file() and not path.is_symlink() and sha(path)==summary[key]==accepted[key],"terminal hash differs")
    attempt,completion = read_json(attempt_path),read_json(completion_path)
    provenance = verify_sources(attempt,summary["input_provenance"],report_commit)
    ids = [f"{seed}-p{di}-r{ri}" for seed in SEEDS for di in range(3) for ri in ROTATIONS]
    records = completion["streams"]
    need(attempt["schema"]=="i19_tracking_attempt_v1" and attempt["root"]==str(root)
         and attempt["seeds"]==list(SEEDS) and attempt["process_variances"]==list(PROCESS_VARIANCES)
         and attempt["rotations"]==[0.,math.pi/4] and attempt["horizon"]==HORIZON
         and attempt["cpu_threads"]==1 and attempt["cuda_visible_devices"]==""
         and attempt["array_limit_bytes"]==2*1024**3 and attempt["root_limit_bytes"]==3*1024**3
         and attempt["paid_spend_usd"]==attempt["paid_reserved_usd"]==0
         and completion["schema"]=="i19_tracking_completion_v1" and completion["status"]=="complete"
         and completion["failure"] is None and completion["expected_streams"]==completion["completed_streams"]==len(ids)
         and completion["completed_observations"]==len(ids)*HORIZON
         and completion["frozen_commit"]==attempt["frozen_commit"] and completion["attempt_sha256"]==summary["attempt_sha256"]
         and [record["id"] for record in records]==ids,"terminal schema/roster differs")
    expected_files = {"attempt.json","completion.json"}|{f"streams/{identity}.{suffix}" for identity in ids for suffix in ("npz","json")}
    files,dirs,total_bytes = set(),set(),0
    for current,directories,filenames in os.walk(root,followlinks=False):
        for name in directories:
            path = Path(current)/name
            need(not path.is_symlink(),"symlink artifact directory")
            dirs.add(str(path.relative_to(root)))
        for name in filenames:
            path = Path(current)/name
            need(path.is_file() and not path.is_symlink(),"nonregular artifact file")
            files.add(str(path.relative_to(root)))
            total_bytes += path.stat().st_size
    need(files==expected_files and dirs=={"streams"},"physical artifact roster differs")
    need(total_bytes<=3*1024**3 and completion["root_bytes_before_completion"]==total_bytes-completion_path.stat().st_size
         and completion["array_bytes"]==sum(r["array"]["size"] for r in records)<=2*1024**3,"artifact accounting differs")
    # Admit every source, metadata record and file hash before loading any array.
    admitted = []
    for record in records:
        check_budget()
        need(set(record)=={"id","array","metadata"},"stream record topology differs")
        identity = record["id"]
        seed,di,ri = map(int,re.fullmatch(r"([0-9]+)-p([0-2])-r([01])",identity).groups())
        path = verify_file(root,record["array"],f"streams/{identity}.npz")
        meta_path = verify_file(root,record["metadata"],f"streams/{identity}.json")
        meta = read_json(meta_path)
        names = policies(PROCESS_VARIANCES[di])
        need(set(meta)=={"schema","id","seed","process_index","rotation_index","process_variance","rotation","horizon",
                        "canonical_sha256","frozen_commit","elapsed_seconds","core"}
             and meta["schema"]=="i19_tracking_stream_v1" and meta["id"]==identity and meta["seed"]==seed
             and meta["process_index"]==di and meta["rotation_index"]==ri and meta["process_variance"]==PROCESS_VARIANCES[di]
             and meta["rotation"]==[0.,math.pi/4][ri] and meta["horizon"]==HORIZON and meta["frozen_commit"]==attempt["frozen_commit"]
             and re.fullmatch(r"[0-9a-f]{64}",meta["canonical_sha256"])
             and meta["core"]["schema"]=="i19_stochastic_tracking_stream_metadata_v1"
             and meta["core"]["array_schema"]=="i19_stochastic_tracking_arrays_v1"
             and meta["core"]["horizon"]==HORIZON and meta["core"]["process_variance"]==PROCESS_VARIANCES[di]
             and meta["core"]["rotation_radians"]==[0.,math.pi/4][ri]
             and meta["core"]["policy_names"]==names and meta["core"]["policy_count"]==len(names)
             and meta["core"]["array_order"]==list(ARRAY_KEYS),"stream scalar metadata differs: "+identity)
        admitted.append((seed,di,ri,path,meta_path,names,record))
    raw = {}
    for seed,di,ri,path,meta_path,names,record in admitted:
        check_budget()
        values = load_metrics(path,names,HORIZON,WINDOWS)
        for (window,policy),metrics in values.items():
            raw[seed,di,ri,window,policy] = metrics
        need(sha(path)==record["array"]["sha256"] and sha(meta_path)==record["metadata"]["sha256"],"artifact changed during read")
    compare = Compare()
    arithmetic = compare_arithmetic(summary,raw,compare)
    check_budget()
    need(sha(summary_path)==summary_sha and sha(audit_path)==audit_sha and sha(attempt_path)==summary["attempt_sha256"]
         and sha(completion_path)==summary["completion_sha256"],"accepted input changed during corroboration")
    need(verify_sources(attempt,summary["input_provenance"],report_commit)==provenance,"source closure changed during corroboration")
    return {"schema":"i19_stochastic_report_corroboration_v1","status":"pass","artifact_root":str(root),
        "input_sha256":{"summary":summary_sha,"audit":audit_sha,"attempt":summary["attempt_sha256"],"completion":summary["completion_sha256"]},
        "input_provenance":provenance,"streams_checked":len(admitted),"registered_scalar_comparisons":compare.count,
        "maximum_absolute_difference":compare.maximum,"tolerance":RELATIVE_TOLERANCE,
        "tolerance_rule":"relative*max(1,abs(expected)); decomposition residual additionally scales by its MSE",
        **arithmetic,"scope":["Independent NumPy/stdlib arithmetic from saved g/s/output only; no observer, RNG, optimizer or native/core/audit import.",
            "All four fixed windows, both rotations, all 32 independent seeds and both CP responses retained; no survivor means.",
            "The 112 identity-whole comparisons are primary; oracles remain labeled as oracles, not learned methods.",
            "This corroborates report arithmetic and provenance, not another full-state audit or a multiplicity-adjusted inference."]}


def reserve_output():
    need(OUTPUT.parent==HERE and OUTPUT.name=="report-check-001" and not OUTPUT.is_symlink(),"fixed exclusive report output required")
    OUTPUT.mkdir(mode=0o700,exist_ok=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--frozen-commit",required=True)
    parser.add_argument("--summary-sha256",required=True)
    parser.add_argument("--audit-sha256",required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and all(os.environ.get(name)=="1" for name in
         ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")),"hidden CUDA / one numerical thread required")
    root = args.root.resolve(strict=True)
    need(args.root.absolute()==root and root.parent==Path("/tmp/spectral-experiment-artifacts") and root.name.startswith("spectral-i19-001.")
         and os.path.ismount("/private-artifacts/storage") and root.stat().st_dev==Path("/private-artifacts/storage").stat().st_dev,
         "direct I19 root on intended mounted volume required")
    reserve_output()
    started = time.monotonic()
    def budget():
        need(time.monotonic()-started<280,"report cooperative time cap reached")
    try:
        result = corroborate(root,SUMMARY,AUDIT,args.summary_sha256,args.audit_sha256,args.frozen_commit,check_budget=budget)
    except Exception as exc:
        result = {"schema":"i19_stochastic_report_corroboration_v1","status":"fail","artifact_root":str(root),
            "requested_input_sha256":{"summary":args.summary_sha256,"audit":args.audit_sha256},
            "requested_report_commit":args.frozen_commit,"error":{"type":type(exc).__name__,"message":str(exc)}}
    result["runtime"] = {"elapsed_seconds":time.monotonic()-started,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform(),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threads":{name:os.environ.get(name) for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}}
    with (OUTPUT/"report-audit.json").open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status":result["status"],"comparisons":result.get("registered_scalar_comparisons"),"output":str(OUTPUT)}))
    return 0 if result["status"]=="pass" else 1


if __name__=="__main__":
    raise SystemExit(main())
