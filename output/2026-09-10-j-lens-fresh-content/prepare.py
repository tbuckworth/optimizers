"""Import-inert fit-only CPU preparation. Real preparation is an explicit stage."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INPUT_PINS = {
    "output/2026-09-10-j-lens-completions/dataset.json": "b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6",
    "output/2026-09-10-j-lens-completions/features.npz": "96e936942fd5cde63c9c482ef19a004ba3fee5adbc46676373772fa5e4123b61",
    "output/2026-09-10-j-lens-completions/analysis_arrays.npz": "cbbe651ed084b36ed468a5932afcbc7e488c81ccdcc0fcd04bccf2fe9f6e7033",
    "experiments/j_lens_completions.py": "ae04db8770538d278fca32b58542e8bb653f42785de39c2f2bf76d7a8fffe238",
    "scripts/analyze_j_lens_completions.py": "21286ebae694095ee160cef98d95fa04902b6ffa4b1d472f9d5642435913b0c4",
}
TOPICS = ["astronomy", "cooking", "football", "programming"]
EXPECTED_IDS = [f"{topic}-{i}-{style}" for topic in TOPICS
                for i in range(6) for style in ["plain", "note"]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pins():
    for relative, expected in INPUT_PINS.items():
        require(hash_file(ROOT / relative) == expected, f"Input hash mismatch: {relative}")


def load_inputs():
    import numpy as np
    verify_pins()
    source = ROOT / "output/2026-09-10-j-lens-completions"
    rows = json.loads((source / "dataset.json").read_text())
    with np.load(source / "features.npz", allow_pickle=False) as archive:
        h = archive["activation_11"]
    with np.load(source / "analysis_arrays.npz", allow_pickle=False) as archive:
        mean = archive["activation_11_mean"]
        basis = archive["activation_11_basis"]
    verify_pins()
    return rows, h, mean, basis


def normalize32(vector):
    import numpy as np
    vector = np.asarray(vector, dtype=np.float64)
    require(np.isfinite(vector).all(), "Nonfinite direction")
    norm = np.linalg.norm(vector)
    require(np.isfinite(norm) and norm > 0, "Zero or invalid direction norm")
    result = (vector / norm).astype(np.float32)
    require(abs(float(np.linalg.norm(result.astype(np.float64))) - 1) <= 1e-6,
            "Float32 direction not unit length within tolerance")
    return result


def compute(rows, h, mean, basis):
    """No PCA, topic-based selection, held-out projections or model calls."""
    import numpy as np
    require(isinstance(rows, list) and len(rows) == 48, "Expected 48 ordered rows")
    require([r["id"] for r in rows] == EXPECTED_IDS, "Dataset ID order mismatch")
    fit_indices = []
    for i, row in enumerate(rows):
        topic, pair_index, style = row["id"].split("-")
        split = "fit" if int(pair_index) < 4 else "heldout"
        require(row["split"] == split and row["style"] == style and
                row["group"] == topic and row["pair"] == f"{topic}-{pair_index}",
                "Dataset split/style/pair mismatch")
        require(isinstance(row["prefix"], str) and bool(row["prefix"]), "Invalid prefix")
        if split == "fit":
            fit_indices.append(i)
    for array, shape, dtype, label in [
        (h, (48, 1024), np.dtype("float32"), "h"),
        (mean, (1024,), np.dtype("float64"), "mean"),
        (basis, (1024, 4), np.dtype("float64"), "basis"),
    ]:
        require(isinstance(array, np.ndarray) and array.shape == shape and array.dtype == dtype,
                f"Malformed {label}")
        require(np.isfinite(array).all(), f"Nonfinite {label}")
    require(len(fit_indices) == 32, "Expected 32 fit rows")
    fit_h = h[fit_indices].astype(np.float64)
    require(np.allclose(fit_h.mean(axis=0), mean, rtol=1e-10, atol=1e-10), "Stored mean disagrees with fit rows")
    require(np.allclose(basis.T @ basis, np.eye(4), rtol=0, atol=1e-8), "Stored basis not orthonormal")
    u32 = np.stack([normalize32(basis[:, axis]) for axis in range(4)], axis=1)
    deviation = np.linalg.norm(u32.astype(np.float64) - basis, axis=0)
    require((deviation <= 1e-6).all(), "Realized directions depart from stored basis")
    scores = (fit_h - mean) @ u32.astype(np.float64)
    require(np.isfinite(scores).all(), "Nonfinite fit scores")
    fit_rows = [rows[i] for i in fit_indices]
    axes, selected = [], set()
    for axis in range(4):
        hi = min(range(32), key=lambda j: (-float(scores[j, axis]), fit_rows[j]["id"]))
        lo = min(range(32), key=lambda j: (float(scores[j, axis]), fit_rows[j]["id"]))
        poles = {}
        for sign, index in [("+", hi), ("-", lo)]:
            row = fit_rows[index]
            selected.add(row["id"])
            poles[sign] = {"id": row["id"], "prefix": row["prefix"],
                           "pair": row["pair"], "style": row["style"],
                           "fit_score": float(scores[index, axis])}
        axes.append({"axis": f"PC{axis+1}", "poles": poles,
                     "same_content_extrema": fit_rows[hi]["pair"] == fit_rows[lo]["pair"],
                     "same_row_extrema": hi == lo,
                     "fit_mean_score": float(scores[:, axis].mean()),
                     "fit_score_std_ddof1": float(scores[:, axis].std(ddof=1)),
                     "fit_score_min": float(scores[lo, axis]),
                     "fit_score_max": float(scores[hi, axis]),
                     "u32_to_v64_l2": float(deviation[axis])})
    exemplar_ids = sorted(selected)
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    exemplars = np.stack([normalize32(h[by_id[row_id]].astype(np.float64)) for row_id in exemplar_ids])
    pc_names = [f"PC{i+1}{sign}" for i in range(4) for sign in ["+", "-"]]
    pc_inputs = np.stack([sign * u32[:, i] for i in range(4) for sign in [1, -1]])
    require(pc_inputs.dtype == np.float32 and exemplars.dtype == np.float32, "Decoder input dtype mismatch")
    arrays = {"source_v64": basis.copy(), "source_mean64": mean.copy(), "u32": u32,
              "signed_pc_inputs32": pc_inputs, "exemplar_inputs32": exemplars,
              "fit_scores64": scores}
    metadata = {"feature": "activation_11", "fit_ids": [r["id"] for r in fit_rows],
                "pc_input_names": pc_names, "exemplar_input_ids": exemplar_ids,
                "axes": axes, "fit_rows": 32, "fit_content_pairs": 16,
                "direction_recipe": "normalize saved V64 columns in float64, then cast float32",
                "score_recipe": "(h.astype(float64)-source_mean64) @ u32.astype(float64)",
                "individual_recipe": "normalize selected raw h in float64, then cast float32; never -h",
                "selection": "fit max/min only; lexical ID tie-break; repetitions retained",
                "old_analysis_scores_used": False, "heldout_scores_computed": False}
    return arrays, metadata


def prepare(target=OUT / "preparation", attempt=OUT / "preparation-attempt.json"):
    for path in [target, attempt]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Consumed/existing stage path: {path}")
    source_hash = hash_file(Path(__file__))
    with attempt.open("x") as f:
        json.dump({"stage": "fit-only CPU preparation", "status": "attempt started; receipt required for completion",
                   "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hash,
                   "target": str(target), "automatic_retry": False}, f, indent=2)
        f.write("\n")
    arrays, metadata = compute(*load_inputs())
    import numpy as np
    target.mkdir()
    with (target / "directions.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    with (target / "selection.json").open("x") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        f.write("\n")
    import sys
    import contextlib
    import io
    import platform
    build = io.StringIO()
    with contextlib.redirect_stdout(build):
        np.show_config()
    receipt = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": source_hash, "interpreter": sys.executable,
               "python_version": sys.version, "numpy_version": np.__version__,
               "numpy_build": build.getvalue(), "platform": platform.platform(),
               "command_argv": [sys.executable, *sys.argv],
               "input_pins": [{"path": p, "sha256": h} for p, h in INPUT_PINS.items()],
               "outputs": [{"path": p.name, "sha256": hash_file(p)} for p in
                           [target / "directions.npz", target / "selection.json"]],
               "new_pca": False, "model_execution": False, "heldout_scoring": False}
    with (target / "receipt.json").open("x") as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Prepared canonical directions and fit-only exemplars; no PCA, held-out scoring or model execution.")


def fixture():
    import numpy as np
    import tempfile
    from unittest.mock import patch
    rows = []
    h = np.zeros((48, 1024), dtype=np.float32)
    for i, row_id in enumerate(EXPECTED_IDS):
        topic, pair_index, style = row_id.split("-")
        split = "fit" if int(pair_index) < 4 else "heldout"
        rows.append({"id": row_id, "group": topic, "pair": f"{topic}-{pair_index}",
                     "style": style, "split": split, "prefix": f"fabricated row {i}"})
        j = i // 2
        h[i, :4] = (j, j % 4, -j, 1 + j % 3) if split == "fit" else (1e9, -1e9, 1e9, -1e9)
    fit = [i for i, r in enumerate(rows) if r["split"] == "fit"]
    mean = h[fit].astype(np.float64).mean(0)
    basis = np.eye(1024, 4, dtype=np.float64)
    basis[:2, :2] = [[1 / np.sqrt(3), np.sqrt(2 / 3)], [np.sqrt(2 / 3), -1 / np.sqrt(3)]]
    arrays, meta = compute(rows, h, mean, basis)
    require(arrays["fit_scores64"].shape == (32, 4), "Fixture fit shape")
    require(not np.array_equal(arrays["u32"].astype(np.float64), basis), "Fixture missed float32 rounding")
    require(np.array_equal(arrays["fit_scores64"], (h[fit].astype(np.float64) - mean) @ arrays["u32"].astype(np.float64)), "Scores do not use realized U32")
    require(all(r["id"] in meta["fit_ids"] for a in meta["axes"] for r in a["poles"].values()), "Held-out leakage")
    require(meta["axes"][0]["poles"]["-"]["id"] == "astronomy-0-note", "Lexical tie-break failed")
    require(meta["axes"][0]["poles"]["-"]["id"] == meta["axes"][2]["poles"]["+"]["id"], "Repeated IDs replaced")
    require(np.array_equal(arrays["signed_pc_inputs32"][1], -arrays["signed_pc_inputs32"][0]), "PC sign mismatch")
    for j, row_id in enumerate(meta["exemplar_input_ids"]):
        expected = normalize32(h[EXPECTED_IDS.index(row_id)].astype(np.float64))
        require(np.array_equal(arrays["exemplar_inputs32"][j], expected), "Individual is not raw h")
        require(arrays["exemplar_inputs32"][j, 3] > 0, "Negative-endpoint h was negated")
    changed = h.copy()
    changed[[i for i in range(48) if i not in fit]] *= -2
    other, other_meta = compute(rows, changed, mean, basis)
    require(all(np.array_equal(arrays[k], other[k]) for k in arrays) and meta == other_meta, "Held-out values affected preparation")
    flat = h.copy()
    flat[fit, 3] = 1
    _, flat_meta = compute(rows, flat, flat[fit].astype(np.float64).mean(0), basis)
    require(flat_meta["axes"][3]["same_content_extrema"] and flat_meta["axes"][3]["same_row_extrema"], "Same-content/row extrema were replaced")
    for kind in ["order", "split", "shape", "dtype", "nonfinite", "meaninf", "basisinf", "badmean", "badbasis"]:
        r, x, m, v = json.loads(json.dumps(rows)), h.copy(), mean.copy(), basis.copy()
        if kind == "order": r[0], r[1] = r[1], r[0]
        elif kind == "split": r[0]["split"] = "heldout"
        elif kind == "shape": x = x[:, :-1]
        elif kind == "dtype": x = x.astype(np.float64)
        elif kind == "nonfinite": x[0, 0] = np.nan
        elif kind == "meaninf": m[0] = np.inf
        elif kind == "basisinf": v[0, 0] = np.inf
        elif kind == "badmean": m[0] += 1
        elif kind == "badbasis": v[:, 0] = 0
        try: compute(r, x, m, v)
        except ValueError: pass
        else: raise AssertionError(f"Accepted malformed {kind}")
    with tempfile.TemporaryDirectory(prefix="jlens-preparation-fixture-") as scratch:
        folder = Path(scratch)
        for name in ["target-exists", "dangling-target", "attempt-exists", "dangling-attempt"]:
            target, attempt = folder / name, folder / f"{name}.json"
            path = target if "target" in name else attempt
            if name.startswith("dangling"): path.symlink_to(folder / "absent")
            else: path.touch()
            with patch(__name__ + ".load_inputs") as load:
                try: prepare(target, attempt)
                except FileExistsError: pass
                else: raise AssertionError("Existing stage accepted")
                load.assert_not_called()
        target, attempt = folder / "failure", folder / "failure.json"
        with patch(__name__ + ".load_inputs", side_effect=RuntimeError("fabricated load failure")) as load:
            try: prepare(target, attempt)
            except RuntimeError: pass
            require(attempt.exists() and not target.exists(), "No pre-read attempt sentinel")
            try: prepare(target, attempt)
            except FileExistsError: pass
            else: raise AssertionError("Automatic retry accepted")
            require(load.call_count == 1, "Retry reached scientific loader")
    print("PASS fabricated preparation: U32/norm/signs, fit-only ties/repetition, held-out exclusion, malformed inputs, existing/dangling guards and pre-read attempt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fixture", "prepare"])
    {"fixture": fixture, "prepare": prepare}[parser.parse_args().stage]()
