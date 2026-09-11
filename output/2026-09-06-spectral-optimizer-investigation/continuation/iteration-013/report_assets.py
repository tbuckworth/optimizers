#!/usr/bin/env python3
"""Build independently weighted I13 scalar tables and the complete clean-curve chart."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RAW = HERE / "raw-results"
I10_RAW = HERE.parent / "iteration-010" / "raw-results"
ANALYSIS = HERE / "analysis-001"
SEEDS, PARENTS = (100, 101, 102), (1500, 2000)
OBJECTIVES = ("fixed", "soft", "redraw")
POLICIES = ("raw", "current32", "mean32", "leak01_32")
NEW_POLICIES = ("mean32", "leak01_32")
HORIZONS = (0, 1, 10, 50, 100, 250, 500)
COMPONENTS = ("raw_gradient", "native_projection", "outside_raw_gradient",
              "outside_post_mean", "outside_pre_mean", "mean32_delivery",
              "leak01_32_delivery", "selected_delivery")
ALIGNMENTS = ("train_fixed", "train_soft", "train_clean", "aux_clean",
              "fixed_minus_soft")
INTERVENTIONS = ("current32", "mean32", "leak01_32",
                 "mean_to_current_data_norm", "current_to_mean_data_norm",
                 "leak_to_current_data_norm", "current_to_leak_data_norm")
LOSS_NAMES = ("train_fixed", "train_soft", "train_clean", "aux_clean", "aux_soft")
FROZEN_COMMIT = "d91dba8"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_gzip(directory: Path, original_name: str) -> dict[str, Any]:
    return json.loads(gzip.decompress((directory / (original_name + ".gz")).read_bytes()))


def seed_first(values: dict[tuple[int, int], float | None]) -> dict[str, Any]:
    """Average parents within seed, then seeds; any missing parent propagates."""
    per_seed = []
    for seed in SEEDS:
        pair = [values.get((seed, parent)) for parent in PARENTS]
        available = all(value is not None for value in pair)
        per_seed.append({"seed": seed,
                         "parent_values": {str(parent): values.get((seed, parent))
                                           for parent in PARENTS},
                         "available": available,
                         "seed_first_parent_mean": statistics.mean(pair) if available else None})
    seed_values = [row["seed_first_parent_mean"] for row in per_seed]
    available = all(value is not None for value in seed_values)
    return {"per_seed": per_seed, "available": available,
            "all_three_seed_values": seed_values,
            "mean": statistics.mean(seed_values) if available else None,
            "no_survivor_averaging": True}


def branch_fraction(steps: list[dict[str, Any]], kind: str, basis: str) -> float | None:
    outside, total = 0.0, 0.0
    for row in steps:
        saved = row["displacement"][kind + "_leakage"][basis]
        squared = float(saved["squared_norm"])
        assert math.isfinite(squared) and squared >= 0
        assert math.isclose(squared, float(row["displacement"][kind + "_norm"]) ** 2,
                            rel_tol=2e-6, abs_tol=1e-14)
        if saved["reason"] == "zero_displacement":
            assert saved["fraction"] is None and saved["outside_squared_norm"] == 0
            continue
        assert saved["reason"] is None
        outer = float(saved["outside_squared_norm"])
        fraction = float(saved["fraction"])
        assert math.isclose(outer, squared * fraction, rel_tol=2e-12, abs_tol=1e-18)
        outside += outer
        total += squared
    return None if total == 0 else outside / total


def level(branch: dict[str, Any], horizon: int, split: str, metric: str) -> float | None:
    row = next((item for item in branch["curve"] if item["horizon"] == horizon), None)
    return None if row is None else float(row[split][metric])


def utility_value(branch: dict[str, Any], horizon: int, split: str, metric: str) -> float | None:
    value = level(branch, horizon, split, metric)
    if value is None:
        return None
    return -value if metric == "clean_ce" else value


def verify_collection() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    collection = json.loads((RAW / "collection.json").read_text())
    assert collection["status"] == "complete" and collection["json_count"] == 53
    assert len(collection["files"]) == 53
    rows = {}
    for row in collection["files"]:
        assert row["original"] not in rows
        archived = (RAW / row["archive"]).read_bytes()
        assert len(archived) == row["archive_bytes"] and sha(archived) == row["archive_sha256"]
        original = gzip.decompress(archived)
        assert len(original) == row["original_bytes"] and sha(original) == row["original_sha256"]
        rows[row["original"]] = row
    assert sha(gzip.decompress((RAW / "completion.json.gz").read_bytes())) \
        == collection["source_completion_sha256"]
    return collection, rows


def verify_sources(manifest: dict[str, Any]) -> None:
    assert manifest["git_commit"].startswith(FROZEN_COMMIT)
    assert len(manifest["source_hashes"]) == 23
    for name, expected in manifest["source_hashes"].items():
        assert sha((ROOT / name).read_bytes()) == expected
        committed = subprocess.check_output(["git", "show", FROZEN_COMMIT + ":" + name], cwd=ROOT)
        assert sha(committed) == expected


def load_branches(i13_rows: dict[str, dict[str, Any]]) -> tuple[
        dict[tuple[int, int, str, str], dict[str, Any]], dict[str, Any], int]:
    index = read_gzip(RAW, "branches.json")
    parent_inputs = read_gzip(RAW, "parent-inputs.json")
    references = read_gzip(RAW, "baseline-references.json")
    assert len(index["entries"]) == 36 and len(references["entries"]) == 36
    assert index["science_complete"] and index["numerical_failures"] == 0
    branches = {}
    for entry in index["entries"]:
        record = entry["branch_artifact"]
        assert i13_rows[record["name"]]["original_sha256"] == record["sha256"]
        branch = read_gzip(RAW, record["name"])
        key = (entry["seed"], entry["parent_step"], entry["objective"], entry["policy"])
        assert key not in branches and branch["status"] == "complete"
        assert [row["horizon"] for row in branch["curve"]] == list(HORIZONS)
        assert len(branch["steps"]) == 500
        branches[key] = branch

    i10_collection = json.loads((I10_RAW / "collection.json").read_text())
    i10_rows = {row["original"]: row for row in i10_collection["files"]}
    reference_roundtrips = 0
    for entry in references["entries"]:
        record = entry["branch_artifact"]
        archived = (I10_RAW / (record["name"] + ".gz")).read_bytes()
        row = i10_rows[record["name"]]
        assert len(archived) == row["archive_bytes"] and sha(archived) == row["archive_sha256"]
        original = gzip.decompress(archived)
        assert len(original) == row["original_bytes"] and sha(original) == row["original_sha256"]
        assert row["original_sha256"] == record["sha256"] \
            == parent_inputs["i10_hashes"][record["name"]]
        branch = json.loads(original)
        key = (entry["seed"], entry["parent_step"], entry["objective"], entry["policy"])
        assert key not in branches and [item["horizon"] for item in branch["curve"]] == list(HORIZONS)
        assert len(branch["steps"]) == 500
        branches[key] = branch
        reference_roundtrips += 1
    expected = {(seed, parent, objective, policy) for seed in SEEDS for parent in PARENTS
                for objective in OBJECTIVES for policy in POLICIES}
    assert set(branches) == expected and len(branches) == 72
    return branches, index, reference_roundtrips


def scalar_levels(branches: dict[tuple[int, int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for objective in OBJECTIVES:
        for policy in POLICIES:
            exemplar = branches[(100, 1500, objective, policy)]["curve"][0]
            for horizon in HORIZONS:
                for split in ("train", "auxiliary", "validation"):
                    for metric in exemplar[split]:
                        values = {(seed, parent): level(branches[(seed, parent, objective, policy)],
                                                        horizon, split, metric)
                                  for seed in SEEDS for parent in PARENTS}
                        rows.append({"objective": objective, "policy": policy,
                                     "horizon": horizon, "split": split, "metric": metric,
                                     **seed_first(values)})
    return rows


def clean_progress(branches: dict[tuple[int, int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for objective in OBJECTIVES:
        for policy in POLICIES:
            for split in ("auxiliary", "validation"):
                for metric in ("clean_ce", "clean_accuracy"):
                    for horizon in HORIZONS:
                        values, utility = {}, {}
                        for seed in SEEDS:
                            for parent in PARENTS:
                                branch = branches[(seed, parent, objective, policy)]
                                start, end = level(branch, 0, split, metric), level(
                                    branch, horizon, split, metric)
                                values[(seed, parent)] = None if start is None or end is None else end - start
                                utility[(seed, parent)] = (None if start is None or end is None else
                                    (start - end if metric == "clean_ce" else end - start))
                        rows.append({"objective": objective, "policy": policy, "split": split,
                                     "metric": metric, "horizon": horizon,
                                     "raw_metric_change_from_h0": seed_first(values),
                                     "clean_utility_progress_from_h0": seed_first(utility)})
    return rows


def clean_contrasts(branches: dict[tuple[int, int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = (("mean32", "current32"), ("mean32", "leak01_32"),
             ("mean32", "raw"), ("leak01_32", "current32"),
             ("leak01_32", "raw"), ("current32", "raw"))
    rows = []
    for objective in OBJECTIVES:
        for split in ("auxiliary", "validation"):
            for metric in ("clean_ce", "clean_accuracy"):
                for left, right in pairs:
                    for horizon in HORIZONS:
                        values = {}
                        for seed in SEEDS:
                            for parent in PARENTS:
                                lhs = utility_value(branches[(seed, parent, objective, left)],
                                                    horizon, split, metric)
                                rhs = utility_value(branches[(seed, parent, objective, right)],
                                                    horizon, split, metric)
                                values[(seed, parent)] = None if lhs is None or rhs is None else lhs - rhs
                        rows.append({"objective": objective, "split": split, "metric": metric,
                                     "left": left, "right": right, "horizon": horizon,
                                     "positive_means": "left policy has higher clean utility",
                                     **seed_first(values)})
    return rows


def primary_rows(contrasts: list[dict[str, Any]], independent: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    rows, discrepancy = [], 0.0
    for objective in ("soft", "redraw"):
        for metric_name, metric in (("ce", "clean_ce"), ("accuracy", "clean_accuracy")):
            row = next(item for item in contrasts if
                       (item["objective"], item["split"], item["metric"], item["left"],
                        item["right"], item["horizon"])
                       == (objective, "auxiliary", metric, "mean32", "current32", 500))
            prior = next(item for item in independent if item["objective"] == objective
                         and item["metric"] == metric_name)
            assert row["available"] == prior["available"]
            for left, right in zip(row["all_three_seed_values"], prior["all_three_seed_values"]):
                discrepancy = max(discrepancy, abs(left - right))
            discrepancy = max(discrepancy, abs(row["mean"] - prior["mean"]))
            rows.append({"objective": objective, "metric": metric_name,
                         "estimand": "U(mean32)-U(current32), auxiliary clean, h500",
                         "positive_means": "mean32 is better", **{key: row[key] for key in
                         ("per_seed", "available", "all_three_seed_values", "mean",
                          "no_survivor_averaging")}})
    assert discrepancy <= 1e-15
    return rows, discrepancy


def fixed_training(branches: dict[tuple[int, int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("fixed_minus_soft_ce", "fixed_ce", "soft_ce", "clean_ce",
               "clean_accuracy", "fixed_accuracy", "mean_max_probability",
               "mean_true_label_probability")
    rows = []
    for policy in POLICIES:
        for metric in metrics:
            for horizon in HORIZONS:
                levels, changes = {}, {}
                for seed in SEEDS:
                    for parent in PARENTS:
                        branch = branches[(seed, parent, "fixed", policy)]
                        start, end = level(branch, 0, "train", metric), level(
                            branch, horizon, "train", metric)
                        levels[(seed, parent)] = end
                        changes[(seed, parent)] = None if start is None or end is None else end - start
                rows.append({"policy": policy, "horizon": horizon, "metric": metric,
                             "interpretation": ("realization residual R_zeta=L_fixed-L_soft"
                                                if metric == "fixed_minus_soft_ce" else None),
                             "level": seed_first(levels), "change_from_h0": seed_first(changes)})
    return rows


def leakage_and_components(branches: dict[tuple[int, int, str, str], dict[str, Any]]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    leakages, components = [], []
    for objective in OBJECTIVES:
        for policy in POLICIES:
            for kind in ("total", "data"):
                for basis in ("frozen_basis", "current_basis"):
                    values = {(seed, parent): branch_fraction(
                        branches[(seed, parent, objective, policy)]["steps"], kind, basis)
                              for seed in SEEDS for parent in PARENTS}
                    leakages.append({"objective": objective, "policy": policy, "kind": kind,
                                     "basis": basis,
                                     "definition": "within-branch energy fraction, then parents within seed, then seeds",
                                     **seed_first(values)})
            if policy in NEW_POLICIES:
                for component in COMPONENTS:
                    values = {}
                    for seed in SEEDS:
                        for parent in PARENTS:
                            steps = branches[(seed, parent, objective, policy)]["steps"]
                            values[(seed, parent)] = statistics.mean(
                                row["mean_diagnostics"]["components"][component]["norm"] for row in steps)
                    components.append({"objective": objective, "policy": policy,
                                       "component": component,
                                       "definition": "arithmetic step-norm mean within branch, then parents within seed, then seeds",
                                       **seed_first(values)})
    return leakages, components


def paired_tables(index: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    descriptors = index["paired_step_probes"]
    assert len(descriptors) == 18
    records = {(row["seed"], row["parent_step"], row["objective"]): row["record"]
               for row in descriptors}
    assert len(records) == 18
    levels = []
    for objective in OBJECTIVES:
        for intervention in INTERVENTIONS:
            for loss in LOSS_NAMES:
                values = {}
                for seed in SEEDS:
                    for parent in PARENTS:
                        changes = records[(seed, parent, objective)]["interventions"] \
                            [intervention]["loss_changes"]
                        values[(seed, parent)] = None if changes is None else changes[loss]
                levels.append({"objective": objective, "intervention": intervention,
                               "loss": loss, "quantity": "one-step finite loss change",
                               **seed_first(values)})
            for quantity in ("total_norm", "data_norm"):
                values = {(seed, parent): records[(seed, parent, objective)]
                          ["interventions"][intervention][quantity]
                          for seed in SEEDS for parent in PARENTS}
                levels.append({"objective": objective, "intervention": intervention,
                               "loss": None, "quantity": quantity, **seed_first(values)})
    pairs = (("mean32", "current32", "actual mean-current"),
             ("leak01_32", "current32", "actual leak-current"),
             ("mean_to_current_data_norm", "current32", "mean-current at current data-step norm"),
             ("mean32", "current_to_mean_data_norm", "mean-current at mean data-step norm"),
             ("leak_to_current_data_norm", "current32", "leak-current at current data-step norm"),
             ("leak01_32", "current_to_leak_data_norm", "leak-current at leak data-step norm"))
    contrasts = []
    for objective in OBJECTIVES:
        for left, right, label in pairs:
            for loss in LOSS_NAMES:
                values = {}
                for seed in SEEDS:
                    for parent in PARENTS:
                        interventions = records[(seed, parent, objective)]["interventions"]
                        lhs, rhs = interventions[left]["loss_changes"], interventions[right]["loss_changes"]
                        values[(seed, parent)] = None if lhs is None or rhs is None else lhs[loss] - rhs[loss]
                contrasts.append({"objective": objective, "left": left, "right": right,
                                  "comparison": label, "loss": loss,
                                  "negative_means": "left produces the smaller one-step loss increase",
                                  **seed_first(values)})
    return levels, contrasts


def alignment_tables(index: dict[str, Any]) -> list[dict[str, Any]]:
    records: dict[tuple[int, int, str, str, int], dict[str, Any]] = {}
    for descriptor in index["common_h0_alignments"]:
        records[(descriptor["seed"], descriptor["parent_step"], "common", "common", 0)] = descriptor["record"]
    for descriptor in index["baseline_h500_alignments"]:
        records[(descriptor["seed"], descriptor["parent_step"], descriptor["objective"],
                 descriptor["policy"], 500)] = descriptor["record"]
    for entry in index["entries"]:
        descriptor = next(item["record"] for item in entry["alignment_probes"]
                          if item["horizon"] == 500)
        records[(entry["seed"], entry["parent_step"], entry["objective"],
                 entry["policy"], 500)] = descriptor["record"]
    assert len(records) == 6 + 72
    rows = []
    cohorts = [("common", "common", 0)] + [(objective, policy, 500)
        for objective in OBJECTIVES for policy in POLICIES]
    for objective, policy, horizon in cohorts:
        for alignment in ALIGNMENTS:
            for quantity in ("dot", "cosine", "gradient_norm"):
                values = {(seed, parent): records[(seed, parent, objective, policy, horizon)]
                          ["alignments"][alignment][quantity]
                          for seed in SEEDS for parent in PARENTS}
                rows.append({"objective": objective, "policy": policy, "horizon": horizon,
                             "alignment": alignment, "quantity": quantity,
                             "signed": quantity in ("dot", "cosine"), **seed_first(values)})
        values = {(seed, parent): records[(seed, parent, objective, policy, horizon)]
                  ["complement_mean_norm"] for seed in SEEDS for parent in PARENTS}
        rows.append({"objective": objective, "policy": policy, "horizon": horizon,
                     "alignment": None, "quantity": "complement_mean_norm",
                     "signed": False, **seed_first(values)})
    return rows


def plot_curves(levels: list[dict[str, Any]], target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert not target.exists()
    colors = {"raw": "#111827", "current32": "#0072b2",
              "mean32": "#009e73", "leak01_32": "#d55e00"}
    labels = {"raw": "raw", "current32": "native current32",
              "mean32": "mean32", "leak01_32": "leak01_32"}
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10), sharex=True)
    fig.patch.set_facecolor("white")
    positions = list(range(len(HORIZONS)))
    for row_index, objective in enumerate(OBJECTIVES):
        for column, metric in enumerate(("clean_ce", "clean_accuracy")):
            ax = axes[row_index, column]
            for policy in POLICIES:
                selected = [row for row in levels if (row["objective"], row["policy"],
                    row["split"], row["metric"]) == (objective, policy, "auxiliary", metric)]
                selected.sort(key=lambda row: row["horizon"])
                ax.plot(positions, [row["mean"] for row in selected],
                        marker="o", markersize=3.5, linewidth=1.8, color=colors[policy],
                        label=labels[policy])
            ax.set_title(f"{objective.capitalize()} · held-out aux "
                         + ("clean CE" if metric == "clean_ce" else "clean accuracy"))
            ax.set_ylabel("Clean CE ↓" if metric == "clean_ce" else "Clean accuracy ↑")
            ax.set_xticks(positions, [f"h{horizon}" for horizon in HORIZONS])
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=.22)
            if row_index == 2:
                ax.set_xlabel("Scheduled checkpoints (unequal update intervals)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("I13 complete auxiliary-clean continuation curves (equal seed/parent weighting)",
                 fontsize=13)
    fig.tight_layout(rect=(0, .045, 1, .965))
    fig.savefig(target, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> int:
    analysis_audit = json.loads((ANALYSIS / "audit.json").read_text())
    analysis_summary = json.loads((ANALYSIS / "summary.json").read_text())
    assert analysis_audit["status"] == "pass"
    collection, collection_rows = verify_collection()
    manifest = read_gzip(RAW, "manifest.json")
    completion = read_gzip(RAW, "completion.json")
    assert completion["status"] == "complete" and completion["science_complete"]
    assert completion["numerical_failures"] == 0 and manifest["cloud_spend_usd"] == 0
    verify_sources(manifest)
    branches, index, i10_roundtrips = load_branches(collection_rows)

    levels = scalar_levels(branches)
    progress = clean_progress(branches)
    contrasts = clean_contrasts(branches)
    primaries, primary_discrepancy = primary_rows(
        contrasts, analysis_summary["primary_four_h500_mean32_minus_current32"])
    leakage, components = leakage_and_components(branches)
    paired_levels, paired_contrasts = paired_tables(index)
    alignments = alignment_tables(index)
    tables = {
        "schema": "i13_mean_report_tables_v1", "status": "complete",
        "science_complete": True, "numerical_failures": 0,
        "evidence_scope": "three reused seeds, two current-trained parent states per seed",
        "weighting": "arithmetic parent mean within seed, then arithmetic mean of three seeds",
        "primary_four_h500_mean32_minus_current32": primaries,
        "all_scalar_levels": levels,
        "all_horizon_clean_policy_contrasts": contrasts,
        "absolute_auxiliary_and_validation_clean_progress": progress,
        "fixed_target_training_risk_diagnostics": fixed_training(branches),
        "within_branch_then_seed_first_leakage": leakage,
        "new_policy_component_norms": components,
        "h0_actual_and_matched_step_levels": paired_levels,
        "h0_actual_and_matched_step_loss_contrasts": paired_contrasts,
        "signed_alignment_h0_and_h500": alignments,
        "safety_budget": {"last_known_spend_usd": 0, "authorized_budget_usd": 100,
                          "source": "I13 manifest plus user authorization"},
    }
    report_audit = {
        "schema": "i13_mean_report_audit_v1", "status": "pass",
        "i13_archive_roundtrips": collection["json_count"],
        "i10_reference_archive_roundtrips": i10_roundtrips,
        "frozen_source_files_verified_in_worktree_and_commit": len(manifest["source_hashes"]),
        "frozen_commit": manifest["git_commit"], "expected_frozen_commit_prefix": FROZEN_COMMIT,
        "scalar_branches": len(branches), "primary_max_abs_discrepancy": primary_discrepancy,
        "table_counts": {key: len(value) for key, value in tables.items() if type(value) is list},
        "analysis_inputs": {"audit_sha256": sha((ANALYSIS / "audit.json").read_bytes()),
                            "summary_sha256": sha((ANALYSIS / "summary.json").read_bytes())},
        "scope_limits": [
            "No model/data forward, optimizer update, training, GPU operation or tensor audit was rerun.",
            "All report estimates use retained JSON scalars; the prior immutable analysis performs tensor checks.",
            "The six h0 alignment records are common across objective/policy and are not treated as independent repeats.",
            "These are conditional saved-state results, not an independent test-set replication or causal mediation proof.",
        ],
        "safety_budget": tables["safety_budget"],
    }
    for path, payload in ((ANALYSIS / "report-tables.json", tables),
                          (ANALYSIS / "report-audit.json", report_audit)):
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    plot_curves(levels, ANALYSIS / "mean-curves.png")
    print(json.dumps({"status": "pass", "i13_roundtrips": collection["json_count"],
                      "i10_references": i10_roundtrips, "branches": len(branches),
                      "primary_max_abs_discrepancy": primary_discrepancy,
                      "output": str(ANALYSIS)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
