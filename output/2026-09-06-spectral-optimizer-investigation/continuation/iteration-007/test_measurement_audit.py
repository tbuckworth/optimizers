"""CPU-only, dataset-free fixtures built from an analytically solvable MLP.

No producer/state/response/assembly/loss imports. Hidden weights are zero;
loss and gradient are derived directly from two bias logits and label counts.
Native32 CE fields below are explicitly synthetic scalar outputs, not GPU data.
"""

import copy
import hashlib
import itertools
import json
import math
import os
import struct
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' explicitly")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np

import measurement_audit as subject

BRANCHES = ("raw", "current", "lagged", "restored", "reciprocal", "zero")
PROBES = ("batch_noisy", "train_probe_noisy", "train_probe_clean", "auxiliary_clean")
COEFF = {
    "ordering": {"lagged": 1, "current": -1},
    "direction_at_current_norm": {"restored": 1, "current": -1},
    "direction_at_lagged_norm": {"lagged": 1, "reciprocal": -1},
    "norm_at_current_direction": {"current": 1, "reciprocal": -1},
    "norm_at_lagged_direction": {"restored": 1, "lagged": -1},
    "interaction": {"restored": 1, "lagged": -1, "current": -1, "reciprocal": 1},
    "current_minus_raw": {"current": 1, "raw": -1}, "lagged_minus_raw": {"lagged": 1, "raw": -1},
    "restored_minus_raw": {"restored": 1, "raw": -1}, "reciprocal_minus_raw": {"reciprocal": 1, "raw": -1},
    "raw_minus_zero": {"raw": 1, "zero": -1}, "current_minus_zero": {"current": 1, "zero": -1},
    "lagged_minus_zero": {"lagged": 1, "zero": -1}, "restored_minus_zero": {"restored": 1, "zero": -1},
    "reciprocal_minus_zero": {"reciprocal": 1, "zero": -1},
}
U, H = 2.**-53, 2.**-1022


def gamma(n):
    return n*U/(1-n*U)


def rsum(x):
    return 2*gamma(2*len(x)+4)*sum(abs(v) for v in x)+4*len(x)*H


def rdot(x, y):
    return 2*gamma(2*len(x)+4)*float(np.abs(x*y).sum())+4*len(x)*H


def vsum(x):
    return 2*gamma(2*len(x)+4)*np.sum(np.abs(x), axis=0)+4*len(x)*H


def norm(x):
    return float(np.linalg.norm(x))


def nround(x):
    n = norm(x)
    return 0. if n == 0 else rdot(x, x)/(2*n)+8*U*n


def digest(x):
    return hashlib.sha256(x.astype(x.dtype.newbyteorder("<"), copy=False).tobytes()).hexdigest()


def vector(x):
    v = np.array(x, dtype=np.float64, copy=True)
    return dict(value=v, shape=[26], dtype="float64", device="cpu", sha256=digest(v))


def concordance(cpu, native, ceiling=None):
    ceiling = 5e-6*max(1., abs(cpu)) if ceiling is None else ceiling
    error = abs(cpu-native)
    return dict(abs_discrepancy=error, descriptive_ceiling=ceiling,
                status="discordant" if error > ceiling else "within_scale")


def geometry(left, right, mask):
    if not all(mask.values()):
        return dict(defined=False, defined_mask=mask, reason="domain_undefined_required_branch",
                    distance=None, cosine=None, cosine_reason=None, left_norm=None, right_norm=None)
    left, right = left.astype(np.float64), right.astype(np.float64)
    a, b = norm(left), norm(right)
    return dict(defined=True, defined_mask=mask, reason=None, distance=norm(left-right),
                cosine=None if a*b == 0 else float(left@right/(a*b)), cosine_reason="zero_norm" if a*b == 0 else None,
                left_norm=a, right_norm=b)


def closed_form(parameters, labels):
    logits = [float(v) for v in parameters["2.bias"]]
    top = max(logits)
    e = [math.exp(v-top) for v in logits]
    logsum = math.log(sum(e))+top
    per = [logsum-logits[int(y)] for y in labels]
    q = np.zeros(26, np.float64)
    q[-2:] = [e[j]/sum(e)-float(np.mean(labels == j)) for j in range(2)]
    return float(sum(per)/len(per)), q, per


def combine(mapping, coefficients):
    result = np.zeros(26, np.float64)
    for b, c in coefficients.items():
        result += c*mapping[b]
    return result


def record(coeff, components, direct, ceiling):
    total = sum(coeff[b]*components[b] for b in coeff)
    error = abs(direct-total)
    return dict(component_values=components, direct_value=direct, component_sum_value=total,
                identity_abs_discrepancy=error, identity_rounding_ceiling=ceiling,
                identity_status="pass" if error <= ceiling else "fatal_validation")


def fixture(*, domain=False, weak=False, native_offset=0., zero_endpoints=False):
    before = {"0.weight": np.zeros((4, 3), np.float32), "0.bias": np.zeros(4, np.float32),
              "2.weight": np.zeros((2, 4), np.float32), "2.bias": np.array([.125, -.25], np.float32)}
    e0, e1 = np.eye(26, dtype=np.float32)[:2]
    current = np.zeros(26, np.float32) if domain in (True, "current_zero", "both_zero") else e0
    lagged = np.zeros(26, np.float32) if domain in ("lagged_zero", "both_zero") else e0.copy() if weak else .4*e1
    nc, nl = norm(current.astype(np.float64)), norm(lagged.astype(np.float64))
    restored = None if nc > 0 and nl == 0 else np.zeros(26, np.float32) if nc == 0 else (lagged.astype(np.float64)/nl*nc).astype(np.float32)
    reciprocal = None if nc == 0 and nl > 0 else np.zeros(26, np.float32) if nl == 0 else (current.astype(np.float64)/nc*nl).astype(np.float32)
    candidates = dict(raw=e0+e1, current=current, lagged=lagged, restored=restored, reciprocal=reciprocal,
                      zero=np.zeros(26, np.float32))
    endpoints = {}
    for b, scale in zip(BRANCHES, (.6, 1., -.3, -.75, .4, .05)):
        endpoint = copy.deepcopy(before)
        if not zero_endpoints:
            endpoint["2.bias"] += np.array([.01, -.003], np.float32)*np.float32(scale)
        endpoints[b] = None if candidates[b] is None else endpoint
    rng = np.random.default_rng(710074)
    train = rng.uniform(0, 1, (4, 3)).astype(np.float32)
    probes = {
        "batch_noisy": dict(inputs=rng.uniform(0, 1, (4, 3)).astype(np.float32), labels=np.array([0, 1, 0, 0], np.int64)),
        "train_probe_noisy": dict(inputs=train.copy(), labels=np.array([1, 1, 0, 1], np.int64)),
        "train_probe_clean": dict(inputs=train.copy(), labels=np.array([0, 0, 0, 1], np.int64)),
        "auxiliary_clean": dict(inputs=rng.uniform(0, 1, (10, 3)).astype(np.float32), labels=np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1], np.int64)),
    }
    def chunk_meta(probe, j):
        return dict(chunk_index=j, start=j, end=j+1, count=1,
                    input_sha256=digest(probes[probe]["inputs"][j:j+1]), label_sha256=digest(probes[probe]["labels"][j:j+1]))
    shared = dict(probe_order=list(PROBES), probes={})
    for key, probe in probes.items():
        ce, q, per = closed_form(before, probe["labels"])
        native = float(np.float32(ce))+native_offset
        chunks = []
        if key == "auxiliary_clean":
            chunks = [dict(chunk_meta(key, j), cpu64_before_ce=ce_j,
                           native32_before_ce=float(np.float32(ce_j))+native_offset,
                           concordance=concordance(ce_j, float(np.float32(ce_j))+native_offset)) for j, ce_j in enumerate(per)]
        shared["probes"][key] = dict(sample_count=len(probe["labels"]),
            sample_identity_sha256=hashlib.sha256(probe["inputs"].tobytes()+probe["labels"].tobytes()).hexdigest(),
            cpu64=dict(before_ce=ce, q=vector(q), q_norm=norm(q)), native32=dict(before_ce=native),
            concordance=concordance(ce, native), auxiliary_chunks=chunks)
    theta = np.concatenate([x.ravel().astype(np.float64) for x in before.values()])
    d, dd, edata, branches = {}, {}, {}, {}
    z = (.001*.01)*theta
    defined = {b: endpoints[b] is not None for b in BRANCHES}
    for b, endpoint in endpoints.items():
        if endpoint is None:
            branches[b] = None
            continue
        d[b] = np.concatenate([x.ravel().astype(np.float64) for x in endpoint.values()])-theta
        dd[b] = d[b]+z
        edata[b] = 2*gamma(4)*(np.abs(d[b])+np.abs(z))+8*H
        displacement = dict(delta=vector(d[b]), delta_data=vector(dd[b]), delta_norm=norm(d[b]),
                            delta_squared_norm=float(d[b]@d[b]), delta_data_norm=norm(dd[b]),
                            delta_data_squared_norm=float(dd[b]@dd[b]), delta_delta_data=geometry(d[b], dd[b], {b: True}))
        measurements = {}
        for key, probe in probes.items():
            base = shared["probes"][key]
            B, q = base["cpu64"]["before_ce"], base["cpu64"]["q"]["value"]
            A, _, per = closed_form(endpoint, probe["labels"])
            Y, D, Ddata = A-B, float(q@d[b]), float(q@dd[b])
            an, bn = float(np.float32(A))+native_offset, base["native32"]["before_ce"]
            chunks = []
            if key == "auxiliary_clean":
                for j, aj in enumerate(per):
                    bj = base["auxiliary_chunks"][j]["cpu64_before_ce"]
                    ajn = float(np.float32(aj))+native_offset
                    bjn = base["auxiliary_chunks"][j]["native32_before_ce"]
                    chunks.append(dict(chunk_meta(key, j), cpu64_after_ce=aj, cpu64_Y=aj-bj,
                        native32_after_ce=ajn, native32_Y=ajn-bjn, concordance=dict(after=concordance(aj, ajn),
                        Y=concordance(aj-bj, ajn-bjn, 5e-6*(max(1., abs(aj))+max(1., abs(bj)))+
                                      max(rsum([aj, -bj]), rsum([ajn, -bjn]))))))
            measurements[key] = dict(before_binding=dict(measurement_before_artifact_id="fixture-before-container",
                probe_key=key, before_ce_cpu64_sha256=hashlib.sha256(struct.pack("<d", B)).hexdigest(), q_sha256=base["cpu64"]["q"]["sha256"]),
                cpu64=dict(after_ce=A, Y=Y, D=D, Ddata=Ddata, R=Y-D), native32=dict(after_ce=an, Y=an-bn),
                concordance=dict(after=concordance(A, an), Y=concordance(Y, an-bn, 5e-6*(max(1., abs(A))+max(1., abs(B)))+
                                                                      max(rsum([A, -B]), rsum([an, -bn])))),
                auxiliary_chunks=chunks)
        branches[b] = dict(displacement=displacement, probes=measurements)
    pairs = {}
    for a, b in itertools.combinations(BRANCHES, 2):
        mask = {a: defined[a], b: defined[b]}
        valid = all(mask.values())
        distance_error = abs(norm(d[a]-d[b])-norm(dd[a]-dd[b])) if valid else None
        bound = (norm(edata[a]+edata[b]+vsum([d[a], -d[b]])+vsum([dd[a], -dd[b]]))+
                 nround(d[a]-d[b])+nround(dd[a]-dd[b])) if valid else None
        pairs[a+"__"+b] = dict(defined=valid, defined_mask=mask, reason=None if valid else "domain_undefined_required_branch",
            delivered_gradient=geometry(candidates[a], candidates[b], mask), delta=geometry(d.get(a), d.get(b), mask),
            delta_data=geometry(dd.get(a), dd.get(b), mask), full_data_distance_discrepancy=distance_error,
            full_data_rounding_ceiling=bound, rounding_status="pass" if valid else "domain_undefined")
    contrasts = {}
    for name, coefficients in COEFF.items():
        required = ["direction", "norm"] if name == "interaction" else ["direction"] if name.startswith("direction_") else ["norm"] if name.startswith("norm_") else []
        factor = dict(direction="unavailable" if nc*nl == 0 else "qualified" if norm(current/nc-lagged/nl) > 2e-6 else "weak",
                      norm="unavailable" if max(nc, nl) == 0 else "qualified" if abs(nc-nl)/max(nc, nl) > 2e-6 else "weak")
        valid = all(defined[b] for b in coefficients)
        row = dict(coefficients=coefficients.copy(), branch_defined_mask=defined.copy(), defined=valid,
                   reason=None if valid else "domain_undefined_required_branch", factor_requirements=required,
                   factor_leverage={k: factor[k] if valid and k in required else "unavailable" if k in required else "not_required"
                                    for k in ("direction", "norm")}, vector=None, probes=None)
        contrasts[name] = row
        if not valid:
            continue
        w, wd = combine(d, coefficients), combine(dd, coefficients)
        ew, ewd = vsum([coefficients[b]*d[b] for b in coefficients]), vsum([coefficients[b]*dd[b] for b in coefficients])
        ec = sum(abs(coefficients[b])*edata[b] for b in coefficients)
        row["vector"] = {key: dict(representation="derived_from_bound_branch_vectors", canonical_sha256=digest(value),
            norm=norm(value), squared_norm=float(value@value), component_count=26) for key, value in (("delta", w), ("delta_data", wd))}
        row["vector"]["full_data_agreement"] = dict(difference_norm=norm(w-wd),
            rounding_ceiling=norm(ew+ewd+ec+vsum([w, -wd]))+nround(w-wd), status="pass")
        def y_record(key, native=False, chunk=None):
            kind = "native32" if native else "cpu64"
            B = shared["probes"][key][kind]["before_ce"] if chunk is None else shared["probes"][key]["auxiliary_chunks"][chunk][kind+"_before_ce"]
            blocks = {b: branches[b]["probes"][key][kind] if chunk is None else branches[b]["probes"][key]["auxiliary_chunks"][chunk] for b in coefficients}
            A = {b: blocks[b]["after_ce" if chunk is None else kind+"_after_ce"] for b in coefficients}
            Y = {b: blocks[b]["Y" if chunk is None else kind+"_Y"] for b in coefficients}
            ceiling = rsum([coefficients[b]*A[b] for b in coefficients])+rsum([coefficients[b]*Y[b] for b in coefficients])+sum(abs(coefficients[b])*rsum([A[b], -B]) for b in coefficients)
            return record(coefficients, Y, sum(coefficients[b]*A[b] for b in coefficients), ceiling), A
        row["probes"] = {}
        for key in PROBES:
            yr, A = y_record(key)
            yn, _ = y_record(key, True)
            cpu, native = dict(Y=yr), dict(Y=yn)
            q = shared["probes"][key]["cpu64"]["q"]["value"]
            D = {b: branches[b]["probes"][key]["cpu64"]["D"] for b in coefficients}
            DD = {b: branches[b]["probes"][key]["cpu64"]["Ddata"] for b in coefficients}
            R = {b: branches[b]["probes"][key]["cpu64"]["R"] for b in coefficients}
            ED = rdot(q, w)+sum(abs(coefficients[b])*rdot(q, d[b]) for b in coefficients)+rsum([coefficients[b]*D[b] for b in coefficients])+float(np.abs(q)@ew)
            EDD = rdot(q, wd)+sum(abs(coefficients[b])*rdot(q, dd[b]) for b in coefficients)+rsum([coefficients[b]*DD[b] for b in coefficients])+float(np.abs(q)@(ewd+ec))
            dp = float(q@w)
            ER = yr["identity_rounding_ceiling"]+ED+rsum([yr["direct_value"], -dp])+sum(abs(coefficients[b])*rsum([yr["component_values"][b], -D[b]]) for b in coefficients)+rsum([coefficients[b]*R[b] for b in coefficients])
            cpu.update(D=record(coefficients, D, dp, ED), Ddata=record(coefficients, DD, float(q@wd), EDD),
                       R=record(coefficients, R, yr["direct_value"]-dp, ER))
            chunks = []
            if key == "auxiliary_clean":
                for j in range(10):
                    yc, aj = y_record(key, chunk=j)
                    ycn, ajn = y_record(key, True, j)
                    chunks.append(dict(chunk_meta(key, j), cpu64=dict(Y=yc), native32=dict(Y=ycn),
                        concordance=dict(Y=concordance(yc["direct_value"], ycn["direct_value"], sum(abs(coefficients[b])*5e-6*max(1., abs(aj[b])) for b in coefficients)+
                            max(rsum([coefficients[b]*aj[b] for b in coefficients]), rsum([coefficients[b]*ajn[b] for b in coefficients]))))))
            _, AN = y_record(key, True)
            row["probes"][key] = dict(cpu64=cpu, native32=native,
                concordance=dict(Y=concordance(yr["direct_value"], yn["direct_value"], sum(abs(coefficients[b])*5e-6*max(1., abs(A[b])) for b in coefficients)+
                    max(rsum([coefficients[b]*A[b] for b in coefficients]), rsum([coefficients[b]*AN[b] for b in coefficients])))), auxiliary_chunks=chunks)
    return before, endpoints, candidates, probes, dict(measurement_before=shared, branches=branches,
                                                     comparisons=dict(branch_pairs=pairs, contrasts=contrasts))


class MeasurementAuditTests(unittest.TestCase):
    def test_complete_analytic_fixture_and_counts(self):
        report = subject.audit(*fixture())
        self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:10])
        self.assertTrue(report["audit_complete"])
        self.assertEqual(report["completion_counts"], dict(before_probes=4, branch_probes=24, defined_branches=6,
            pairs=15, contrasts=15, defined_contrasts=15, auxiliary_before_chunks=10, auxiliary_branch_chunks=60,
            auxiliary_contrast_chunks=150))
        value = report["comparison_audits"]["contrasts.interaction.auxiliary_clean.Y.independent"]
        self.assertEqual(value["audit_status"], "pass")

    def test_q_and_after_loss_are_independently_recomputed(self):
        args = fixture()
        args[-1]["branches"]["current"]["probes"]["auxiliary_clean"]["cpu64"]["after_ce"] += .001
        report = subject.audit(*args)
        self.assertEqual(report["overall_status"], "fatal_validation")
        self.assertTrue(any("after_ce" in item["path"] for item in report["fatal_failures"]))
        args = fixture()
        q = args[-1]["measurement_before"]["probes"]["batch_noisy"]["cpu64"]["q"]
        q["value"][0] = .001
        q["sha256"] = digest(q["value"])
        report = subject.audit(*args)
        self.assertTrue(any("q_values" in item["path"] for item in report["fatal_failures"]))

    def test_permitted_domain_nulls_preserve_other_contrasts(self):
        report = subject.audit(*fixture(domain=True))
        self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:3])
        self.assertEqual(report["domain_undefined_branches"], ["reciprocal"])
        self.assertTrue(report["audit_complete"])
        self.assertEqual(report["completion_counts"]["defined_branches"], 5)
        self.assertIn("contrasts.ordering.auxiliary_clean.Y.independent", report["comparison_audits"])
        for case, nulls in (("lagged_zero", ["restored"]), ("both_zero", [])):
            report = subject.audit(*fixture(domain=case))
            self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:3])
            self.assertEqual(report["domain_undefined_branches"], nulls)

    def test_weak_leverage_and_finite_native_discordance_are_not_fatal(self):
        report = subject.audit(*fixture(weak=True, native_offset=.01))
        self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:3])
        self.assertIn("interaction", report["weak_factor_contrasts"])
        self.assertGreater(len(report["native_discordant_paths"]), 0)
        self.assertTrue(report["audit_complete"])

    def test_shared_baseline_cancels_from_contrast_ceiling(self):
        args = fixture()
        report = subject.audit(*args)
        item = report["comparison_audits"]["contrasts.ordering.auxiliary_clean.Y.independent"]
        self.assertLess(item["fixed_ceiling"], 2.3e-10)
        self.assertGreater(item["fixed_ceiling"], 2.1e-10)
        self.assertEqual(item["resolution_status"], "resolved_positive")

    def test_corrupt_bound_hash_field_mask_and_input_identity_fail(self):
        mutations = [
            lambda a: a[-1]["comparisons"]["contrasts"]["ordering"]["probes"]["auxiliary_clean"]["cpu64"]["Y"].__setitem__("identity_rounding_ceiling", 1.),
            lambda a: a[-1]["branches"]["raw"]["displacement"]["delta"].__setitem__("sha256", "0"*64),
            lambda a: a[-1]["measurement_before"].__setitem__("unexpected", 1),
            lambda a: a[-1]["comparisons"]["contrasts"]["ordering"]["branch_defined_mask"].__setitem__("current", False),
            lambda a: a[3]["train_probe_clean"]["inputs"].__setitem__((0, 0), np.float32(.99)),
        ]
        for mutate in mutations:
            args = fixture()
            mutate(args)
            self.assertEqual(subject.audit(*args)["overall_status"], "fatal_validation")

    def test_impossible_candidate_null_and_positive_target_failure(self):
        args = fixture()
        args[2]["restored"] = None
        self.assertEqual(subject.audit(*args)["overall_status"], "fatal_validation")
        args = fixture()
        args[2]["restored"][:] = 0
        self.assertEqual(subject.audit(*args)["overall_status"], "fatal_validation")

    def test_nonfinite_values_fail_and_guard_is_called(self):
        args = fixture()
        args[-1]["measurement_before"]["probes"]["batch_noisy"]["native32"]["before_ce"] = float("nan")
        self.assertEqual(subject.audit(*args)["overall_status"], "fatal_validation")
        labels = []
        report = subject.audit(*fixture(), guard=labels.append)
        self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:3])
        self.assertTrue(any("chunk" in label for label in labels))
        self.assertEqual(labels[-1], "measurement_audit.complete")

    def test_guard_value_error_propagates_and_hostile_finite_q_has_finite_report(self):
        def guard(label):
            if "chunk" in label:
                raise ValueError("synthetic resource stop")
        with self.assertRaisesRegex(ValueError, "synthetic resource stop"):
            subject.audit(*fixture(), guard=guard)
        args = fixture()
        q = args[-1]["measurement_before"]["probes"]["batch_noisy"]["cpu64"]["q"]
        q["value"][:] = 1e308
        q["sha256"] = digest(q["value"])
        report = subject.audit(*args)
        self.assertEqual(report["overall_status"], "fatal_validation")
        self.assertFalse(report["audit_complete"])
        json.dumps(report, allow_nan=False)

    def test_bounded_ascii_id_and_exact_integer_coefficients(self):
        for invalid in ("a"*129, "../bad", "bad id", "nonascii-é"):
            args = fixture()
            args[-1]["branches"]["raw"]["probes"]["batch_noisy"]["before_binding"]["measurement_before_artifact_id"] = invalid
            self.assertEqual(subject.audit(*args)["overall_status"], "fatal_validation")
        args = fixture()
        args[-1]["comparisons"]["contrasts"]["ordering"]["coefficients"]["lagged"] = 1.
        self.assertEqual(subject.audit(*args)["overall_status"], "fatal_validation")

    def test_zero_responses_are_retained_with_unresolved_numerical_sign(self):
        report = subject.audit(*fixture(zero_endpoints=True))
        self.assertEqual(report["overall_status"], "pass", report["fatal_failures"][:3])
        records = [row for path, row in report["comparison_audits"].items() if path.endswith(".independent")]
        self.assertEqual(len(records), 390)
        self.assertTrue(all(row["resolution_status"] == "unresolved_numerical" for row in records))

    def test_cancellation_small_data_norm_has_unresolved_cosine_interval(self):
        engine = subject._Audit("fixture_tiny_mlp_cpu_v1", None)
        engine.p = 26
        delta, theta = np.zeros(26), np.zeros(26)
        delta[0], theta[0] = -3., 300000.
        z = (.001*.01)*theta
        data = delta+z
        self.assertGreater(norm(data), 0.)
        error = 2*gamma(4)*(np.abs(delta)+np.abs(z))+8*H
        engine.geometry(geometry(delta, data, {"raw": True}), delta, data, {"raw": True},
                        "fixture.cancellation", right_error=error)
        self.assertEqual(engine.report["overall_status"], "pass")
        self.assertEqual(engine.report["unresolved_geometry_paths"], ["fixture.cancellation.cosine"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
