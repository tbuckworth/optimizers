"""Fabricated arithmetic/rendering tests plus exact pinned-audit binding."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("component_utility_plots", Path(__file__).with_name("make_plots.py"))
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def fixture():
    parents = []
    for si, seed in enumerate(plots.SEEDS):
        for mi, mode in enumerate(plots.MODES):
            for ti, step in enumerate(plots.STEPS):
                endpoints = []
                for batch in (0, 1):
                    for policy in ("raw", "native", "decay"):
                        for fi, (fid, fraction) in enumerate(plots.FRACTIONS):
                            # Tenth values deliberately are not full / 10.
                            base = -(si + 1) * (mi + 1) * (ti + 1) * (batch + 1) * (fi + 1) * 1e-4
                            signed = (-1 if si == 1 else 1) * (fi + 1) * 1e-6
                            value = base + (signed if policy == "native" else 0)
                            effects = {"H_O": {"finite": value}, "C": {"finite": value / 100}}
                            endpoints.append({"batch": batch, "policy": policy, "fraction_id": fid,
                                              "fraction": fraction, "effects": effects})
                parents.append({"seed": seed, "augmentation": mode, "step": step,
                                "parent_id": f"s{seed}-{mode}-h{step:05d}", "endpoints": endpoints})
    cells = []
    for mode in plots.MODES:
        for step in plots.STEPS:
            for fid, fraction in plots.FRACTIONS:
                rows = []
                for seed in plots.SEEDS:
                    parent = next(p for p in parents if (p["seed"], p["augmentation"], p["step"]) == (seed, mode, step))
                    lookup = {(e["batch"], e["policy"], e["fraction_id"]): e for e in parent["endpoints"]}
                    objectives = {}
                    for objective in plots.OBJECTIVES:
                        raw = [lookup[b, "raw", fid]["effects"][objective]["finite"] for b in (0, 1)]
                        native = [lookup[b, "native", fid]["effects"][objective]["finite"] for b in (0, 1)]
                        objectives[objective] = {"raw_finite": math.fsum(raw) / 2,
                                                 "native_finite": math.fsum(native) / 2,
                                                 "contrast_finite": math.fsum(native[b] - raw[b] for b in (0, 1)) / 2}
                    rows.append({"seed": seed, "objectives": objectives})
                cells.append({"augmentation": mode, "step": step, "fraction_id": fid,
                              "fraction": fraction, "seed_rows": rows})
    primary = copy.deepcopy(next(c for c in cells if (c["augmentation"], c["step"], c["fraction_id"]) == ("translate", 56304, "full")))
    return {"schema": "spectral_component_utility_audit_v1", "status": "PASS", "errors": [],
            "checked_parents": parents, "independent_summary": {"primary": primary, "cells": cells}}


class PlotTests(unittest.TestCase):
    def test_order_no_pool_and_separate_actual_tenth_values(self):
        data = plots.plot_data(fixture())
        self.assertEqual(len(data["all_parents"]), 12)
        self.assertEqual([(r["augmentation"], r["step"], r["seed"]) for r in data["all_parents"]],
                         [(m, t, s) for m in plots.MODES for t in plots.STEPS for s in plots.SEEDS])
        self.assertEqual([r["seed"] for r in data["primary"]], list(plots.SEEDS))
        for parent in data["all_parents"]:
            full = parent["paths"]["full"]["objectives"]["H_O"]["contrast_finite"]
            tenth = parent["paths"]["tenth"]["objectives"]["H_O"]["contrast_finite"]
            self.assertNotEqual(tenth, full / 10)
        self.assertEqual(data["scalar_mean_and_primary_checks"], 162)

    def test_signed_contrasts_do_not_erase_negative_absolute_changes(self):
        data = plots.plot_data(fixture())
        for index, row in enumerate(data["primary"]):
            for objective in plots.OBJECTIVES:
                values = row["objectives"][objective]
                self.assertLess(values["raw_finite"], 0)
                self.assertLess(values["native_finite"], 0)
                self.assertEqual(values["contrast_finite"] > 0, index != 1)
                self.assertAlmostEqual(values["native_finite"] - values["raw_finite"], values["contrast_finite"])

    def test_roster_path_status_and_scalar_mutations_rejected(self):
        changes = [lambda a: a.update(status="FAIL"), lambda a: a["checked_parents"].reverse(),
                   lambda a: a["checked_parents"][0]["endpoints"][0].update(fraction=0.1),
                   lambda a: a["independent_summary"]["cells"].reverse(),
                   lambda a: a["independent_summary"]["cells"][0]["seed_rows"].reverse(),
                   lambda a: a["independent_summary"]["cells"][0]["seed_rows"][0]["objectives"]["H_O"].update(contrast_finite=9),
                   lambda a: a["independent_summary"]["primary"]["seed_rows"][0]["objectives"]["C"].update(raw_finite=float("nan"))]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                audit = fixture()
                change(audit)
                with self.assertRaises(ValueError):
                    plots.plot_data(audit)

    def test_tiny_contrasts_marked_without_rounding(self):
        data = plots.plot_data(fixture())
        for parent in data["all_parents"]:
            for path in parent["paths"].values():
                for obj in path["objectives"].values():
                    self.assertEqual(obj["tiny_contrast"], abs(obj["contrast_finite"]) <= 1e-8)
                    self.assertNotEqual(obj["contrast_finite"], 0)

    def test_hash_rejection_before_json_decode(self):
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(Path, "is_symlink", return_value=False), \
                mock.patch.object(Path, "stat", return_value=mock.Mock(st_size=2)), \
                mock.patch.object(Path, "read_bytes", return_value=b"{}"):
            with self.assertRaisesRegex(ValueError, "hash"):
                plots.load_pinned_audit()

    def test_fabricated_render_uses_exact_bar_and_scatter_values(self):
        data = plots.plot_data(fixture())
        fig, axes = plots.render_primary(data)
        plt, _, _ = plots._plotting()
        try:
            for col, obj in enumerate(plots.OBJECTIVES):
                expected = [r["objectives"][obj]["raw_finite"] for r in data["primary"]]
                expected += [r["objectives"][obj]["native_finite"] for r in data["primary"]]
                self.assertEqual([bar.get_height() for bar in axes[0, col].patches], expected)
                # Bottom's first three patches are exactly the three contrast bars.
                self.assertEqual([bar.get_height() for bar in axes[1, col].patches[:3]],
                                 [r["objectives"][obj]["contrast_finite"] for r in data["primary"]])
        finally:
            plt.close(fig)
        fig, axes = plots.render_all_parents(data)
        try:
            for row, mode in enumerate(plots.MODES):
                for col, obj in enumerate(plots.OBJECTIVES):
                    wanted = [p["paths"][fid]["objectives"][obj]["contrast_finite"]
                              for p in data["all_parents"] if p["augmentation"] == mode for fid, _ in plots.FRACTIONS]
                    plotted = []
                    for collection in axes[row, col].collections:
                        if collection.get_sizes()[0] == 44:
                            plotted.append(float(collection.get_offsets()[0, 0]))
                    self.assertEqual(plotted, wanted)
                    self.assertEqual(axes[row, col].get_xscale(), "symlog")
                    limits = axes[row, col].get_xlim()
                    self.assertTrue(all(limits[0] <= value <= limits[1] for value in wanted))
        finally:
            plt.close(fig)

    def test_exact_pinned_audit_and_quantitative_manifest_values(self):
        audit = plots.load_pinned_audit()
        data = plots.plot_data(audit)
        self.assertEqual([r["objectives"]["H_O"]["contrast_finite"] > 0 for r in data["primary"]], [True, False, False])
        self.assertTrue(all(r["objectives"]["C"][field] < 0 for r in data["primary"]
                            for field in ("raw_finite", "native_finite")))
        for parent in data["all_parents"]:
            if parent["step"] == 100:
                self.assertLess(parent["paths"]["full"]["objectives"]["H_O"]["contrast_finite"], 0)
                self.assertGreater(parent["paths"]["full"]["objectives"]["C"]["contrast_finite"], 0)

    def test_generated_manifest_and_images_match_pinned_audit_and_plotter(self):
        manifest = Path(__file__).with_name("plots-manifest.json")
        self.assertTrue(manifest.is_file(), "Generate the two plots before this final binding test")
        saved = json.loads(manifest.read_text())
        self.assertEqual(saved["audit_sha256"], plots.AUDIT_SHA256)
        self.assertEqual(saved["data"], plots.plot_data(plots.load_pinned_audit()))
        self.assertEqual(saved["plotter_sha256"], hashlib.sha256(Path(plots.__file__).read_bytes()).hexdigest())
        self.assertEqual([r["path"] for r in saved["figures"]], list(plots.FILENAMES))
        for row in saved["figures"]:
            payload = manifest.with_name(row["path"]).read_bytes()
            self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(row["size_bytes"], len(payload))
            self.assertEqual(row["dpi"], 150)
            self.assertEqual(row["background"], "white")


if __name__ == "__main__":
    unittest.main()
