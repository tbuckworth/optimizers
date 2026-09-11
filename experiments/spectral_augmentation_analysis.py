"""Inert, pure endpoint contrasts for the fixed augmentation protocol.

No files, model, dataset or accelerator are accessed. Callers verify acquisition
receipts/logits separately. The three paired seeds, not branches, are units.
"""
import math

SEEDS = (202609131, 202609132, 202609133)
CELLS = ("clean", "shared", "sham")
POLICIES = ("raw", "native32")
MODES = ("none", "random", "targeted", "opposite")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite non-boolean numerical metric required")
    return float(value)


def descriptive(values):
    if len(values) != len(SEEDS):
        raise ValueError("exactly three seed values required")
    values = [_finite(value) for value in values]
    return {"values": values, "mean": math.fsum(values) / 3,
            "min": min(values), "max": max(values),
            "positive_count": sum(value > 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "zero_count": sum(value == 0 for value in values)}


def metrics(evaluation):
    """Flatten prespecified outcomes; retain both cue-rate components.

    Accuracy is higher-is-better and CE is lower-is-better. Cue reliance is not
    an unconditional utility measure; absolute competence must accompany it.
    """
    result = {}
    for source in ("heldout_unpatched", "heldout_patched", "train_true"):
        for group in ("rare", "majority_macro", "balanced_total"):
            for measure in ("accuracy", "ce"):
                result[f"{source}/{group}/{measure}"] = _finite(evaluation[source][group][measure])
        for digit in range(10):
            for measure in ("accuracy", "ce"):
                result[f"{source}/class{digit}/{measure}"] = _finite(
                    evaluation[source]["per_class"][str(digit)][measure])
    for measure in ("accuracy", "ce"):
        result[f"train_assigned/{measure}"] = _finite(evaluation["train_assigned"][measure])
    for group in ("majority_nonzero", "rare", "all_nonzero"):
        for measure in ("unpatched_target0_rate", "patched_target0_rate", "patch_excess"):
            result[f"cue/{group}/{measure}"] = _finite(evaluation["cue"][group][measure])
    # Absent actually-changed training rows in Clean are explicit None, not zero.
    changed = evaluation["train_actually_changed"]
    for measure in ("accuracy", "ce"):
        value = changed[measure]
        if changed["count"] == 0:
            if value is not None:
                raise ValueError("empty changed-label group must have undefined metrics")
            result[f"train_actually_changed/{measure}"] = None
        else:
            result[f"train_actually_changed/{measure}"] = _finite(value)
    return result


def _subtract(left, right):
    if set(left) != set(right):
        raise ValueError("metric sets differ")
    result = {}
    for key in left:
        if left[key] is None or right[key] is None:
            if left[key] is not None or right[key] is not None:
                raise ValueError("metric definition differs across contrast")
            result[key] = None
        else:
            result[key] = left[key] - right[key]
    return result


def _summarize_maps(maps):
    if len(maps) != 3 or any(set(m) != set(maps[0]) for m in maps):
        raise ValueError("three matching metric sets required")
    result = {}
    for key in maps[0]:
        values = [m[key] for m in maps]
        if all(value is None for value in values):
            result[key] = None
        else:
            result[key] = descriptive(values)
    return result


def summarize(rows):
    expected = {(s, c, p, a) for s in SEEDS for c in CELLS for p in POLICIES for a in MODES}
    roster = {}
    for row in rows:
        key = (row["seed"], row["cell"], row["policy"], row["augmentation"])
        if key not in expected or key in roster:
            raise ValueError("unknown or repeated trajectory")
        if row["endpoint"]["step"] != 2000 or row["warmup"]["step"] != 100:
            raise ValueError("fixed endpoint and warmup steps required")
        roster[key] = (metrics(row["endpoint"]), metrics(row["warmup"]))
    if set(roster) != expected:
        raise ValueError("complete 72-trajectory roster required")
    # Check each cell has identical warmup metrics across modes and policies.
    for seed in SEEDS:
        for cell in CELLS:
            start = roster[seed, cell, "raw", "none"][1]
            if any(roster[seed, cell, p, a][1] != start for p in POLICIES for a in MODES):
                raise ValueError("warmup outcomes differ within a cell's full-state forks")
    endpoint = lambda s, c, p, a: roster[s, c, p, a][0]
    augmentation_delta = lambda s, c, p, a: _subtract(endpoint(s, c, p, a), endpoint(s, c, p, "none"))
    groups, from_warmup, policies, augmentation, interactions, location = {}, {}, {}, {}, {}, {}
    for cell in CELLS:
        for mode in MODES:
            policies[f"{cell}/{mode}"] = _summarize_maps([
                _subtract(endpoint(s, cell, "native32", mode), endpoint(s, cell, "raw", mode)) for s in SEEDS])
            for policy in POLICIES:
                key = f"{cell}/{policy}/{mode}"
                groups[key] = _summarize_maps([endpoint(s, cell, policy, mode) for s in SEEDS])
                from_warmup[key] = _summarize_maps([
                    _subtract(*roster[s, cell, policy, mode]) for s in SEEDS])
                if mode != "none":
                    augmentation[key] = _summarize_maps([augmentation_delta(s, cell, policy, mode) for s in SEEDS])
            if mode != "none":
                interactions[f"{cell}/{mode}"] = _summarize_maps([
                    _subtract(augmentation_delta(s, cell, "native32", mode),
                              augmentation_delta(s, cell, "raw", mode)) for s in SEEDS])
        for policy in POLICIES:
            location[f"{cell}/{policy}"] = _summarize_maps([
                _subtract(endpoint(s, cell, policy, "targeted"), endpoint(s, cell, policy, "opposite"))
                for s in SEEDS])
    q = lambda s, p, a: (endpoint(s, "shared", p, a)["cue/majority_nonzero/patch_excess"]
                         - endpoint(s, "sham", p, a)["cue/majority_nonzero/patch_excess"])
    dq = lambda s, p, a: q(s, p, a) - q(s, p, "none")
    cue = {
        "Q": {f"{p}/{a}": descriptive([q(s, p, a) for s in SEEDS]) for p in POLICIES for a in MODES},
        "delta_Q": {f"{p}/{a}": descriptive([dq(s, p, a) for s in SEEDS])
                    for p in POLICIES for a in MODES if a != "none"},
        "I_Q": {a: descriptive([dq(s, "native32", a) - dq(s, "raw", a) for s in SEEDS])
                for a in MODES if a != "none"},
        "targeted_minus_opposite_Q": {p: descriptive([q(s, p, "targeted") - q(s, p, "opposite") for s in SEEDS])
                                      for p in POLICIES},
    }
    return {"schema": "spectral_augmentation_analysis_v1", "seeds": list(SEEDS),
            "endpoint_step": 2000, "independent_units": "three paired seeds",
            "uncertainty": "mean and observed seed range; no confidence interval or significance test",
            "conventions": {"policy_contrasts": "native32 minus raw",
                            "augmentation_contrasts": "named mode minus none",
                            "interactions": "native32 augmentation change minus raw augmentation change",
                            "location_contrasts": "targeted minus opposite",
                            "cue_warning": "Negative I_Q is relative; absolute native reduction needs delta_Q(native32)<0"},
            "groups": groups, "change_from_warmup": from_warmup, "policy_contrasts": policies,
            "augmentation_contrasts": augmentation, "interactions": interactions,
            "location_contrasts": location, "cue_association": cue}
