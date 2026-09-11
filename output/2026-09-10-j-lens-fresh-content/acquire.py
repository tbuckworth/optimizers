"""Import-inert, once-only reference decoding then fresh-prefix acquisition."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare import hash_file, require

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MAIN = Path("output/2026-09-10-jlens-fresh-content")
UPSTREAM = ROOT.parent / "jacobian-lens"
UPSTREAM_REV = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
HUB = Path("/private-artifacts/storage/cache/huggingface/hub")
MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_REV = "2fc06364715b967f1860aea9cf38778875588b17"
LENS_REV = "0731326edff4ae730ffc5356fe1a4728c748b3a6"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
LENS_HASH = "aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48"
LAYER, WIDTH = 11, 1024
PINS = {
    OUT / "preparation/directions.npz": "47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa",
    OUT / "preparation/selection.json": "aac75625b67d9a90ba351d979f3e1a77cd75f69706a8b3113e071630de023f4d",
    OUT / "preparation/receipt.json": "0ff288e04f61fb0d33db15439a091dd111b92e6a43e20418dd186e6aa64a0dbc",
    MAIN / "protocol.md": "56b7d85fa14438f06cd6a23996c858f4f3cd908f45eed3a44c96cf7b5a5f9980",
    MAIN / "dataset.json": "c655a5531af3215a29e332ddc6b23ac4854b0bffdf91b8ad5e8a045748ff7cd9",
    MAIN / "pairs.json": "8fa981cd7700b1385502e5b96ce451eebb9c34f6b93e05037203e9f314c1347d",
    OUT / "prepare.py": "5bfa5cfadd22ad54ffa6c4073e570860faf3e9590e7a474a50da0d1b657413c4",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def absent(paths):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Consumed or existing stage path: {path}")


def claim(attempt, target):
    absent([attempt, target])
    write_json(attempt, {"status": "attempt started; completion receipt required", "started_utc": utc(),
                         "target": str(target), "automatic_retry": False,
                         "source_sha256": hash_file(Path(__file__))})


def verify_pins():
    for path, expected in PINS.items():
        require(hash_file(path) == expected, f"Source/input hash mismatch: {path}")


def load_inputs():
    import numpy as np
    verify_pins()
    selection = json.loads((OUT / "preparation/selection.json").read_text())
    with np.load(OUT / "preparation/directions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in
                  ["source_mean64", "u32", "signed_pc_inputs32", "exemplar_inputs32"]}
    names = selection["pc_input_names"] + selection["exemplar_input_ids"]
    require(selection["pc_input_names"] == [f"PC{i}{s}" for i in range(1, 5) for s in ["+", "-"]], "PC input order")
    require(len(selection["exemplar_input_ids"]) == len(set(selection["exemplar_input_ids"])) == 7, "Expected seven frozen exemplars")
    for key, shape, dtype in [("source_mean64", (WIDTH,), np.float64), ("u32", (WIDTH, 4), np.float32),
                              ("signed_pc_inputs32", (8, WIDTH), np.float32), ("exemplar_inputs32", (7, WIDTH), np.float32)]:
        value = arrays[key]
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), f"Malformed {key}")
    for i in range(4):
        require(np.array_equal(arrays["signed_pc_inputs32"][2*i], arrays["u32"][:, i]), "Positive PC input differs")
        require(np.array_equal(arrays["signed_pc_inputs32"][2*i+1], -arrays["u32"][:, i]), "Negative PC input differs")
    inputs = np.concatenate([arrays["signed_pc_inputs32"], arrays["exemplar_inputs32"]])
    require(np.allclose(np.linalg.norm(inputs.astype(np.float64), axis=1), 1, rtol=0, atol=1e-6), "Input norms")
    rows = json.loads((MAIN / "dataset.json").read_text())
    pairs = json.loads((MAIN / "pairs.json").read_text())
    expected_ids = [f"{topic}-new-{i}" for topic in ["astronomy", "cooking", "football", "programming"] for i in range(6)]
    require([r["id"] for r in rows] == expected_ids and len({r["prefix"] for r in rows}) == 24, "Fresh ID/text roster")
    require([p["id"] for p in pairs] == [f"P{i:02d}" for i in range(1, 13)], "Pair roster")
    require(sorted([p[k] for p in pairs for k in ["left", "right"]]) == sorted(expected_ids), "Fresh pair coverage")
    verify_pins()
    return {"inputs": inputs, "names": names, "u32": arrays["u32"], "mean64": arrays["source_mean64"],
            "selection": selection}, rows, pairs


def parameter_state(hf):
    require(all(not m.training for m in hf.modules()), "Model not fully in eval mode")
    require(all(not p.requires_grad and p.grad is None for p in hf.parameters()), "Parameters not frozen/gradient-free")
    return [(id(p), p._version) for p in hf.parameters()]


def load_runtime():
    require(os.environ.get("JLENS_FRESH_GPU_RELEASE") == "1", "Fresh parent GPU/memory admission required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    import transformers
    import huggingface_hub
    versions = {"numpy": np.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                "huggingface_hub": huggingface_hub.__version__}
    require(versions == {"numpy": "1.26.4", "torch": "2.11.0+cu128", "transformers": "5.5.0", "huggingface_hub": "1.8.0"}, "Runtime versions changed")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip() == UPSTREAM_REV, "Upstream revision changed")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=UPSTREAM, text=True).strip(), "Upstream source dirty")
    sys.path.insert(0, str(UPSTREAM))
    import jlens
    require(Path(jlens.__file__).resolve().is_relative_to(UPSTREAM / "jlens"), "Wrong jlens import")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), "CUDA unavailable; do not substitute another run")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
    model_snapshot = HUB / "models--Qwen--Qwen3.5-0.8B/snapshots" / MODEL_REV
    weight_path = model_snapshot / "model.safetensors-00001-of-00001.safetensors"
    weight_hash = hash_file(weight_path)
    require(weight_hash == "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696", "Cached model shard changed")
    lens_path = HUB / "models--neuronpedia--jacobian-lens/snapshots" / LENS_REV / LENS_FILE
    require(hash_file(lens_path) == LENS_HASH, "Cached lens changed")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, cache_dir=str(HUB), local_files_only=True, trust_remote_code=False)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, cache_dir=str(HUB), local_files_only=True,
                                                        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="eager").cuda()
    hf.eval()
    hf.requires_grad_(False)
    model = jlens.from_hf(hf, tok)
    require(model.d_model == WIDTH and model.n_layers == 24, "Model layout changed")
    lens = jlens.JacobianLens.load(str(lens_path))
    baseline = parameter_state(hf)
    import contextlib
    import io
    build = io.StringIO()
    with contextlib.redirect_stdout(build): np.show_config()
    provenance = {"interpreter": sys.executable, "python_version": sys.version, "packages": versions,
                  "numpy_build": build.getvalue(), "command_argv": [sys.executable, *sys.argv],
                  "model": MODEL, "model_revision": MODEL_REV, "model_weight_sha256": weight_hash,
                  "lens_revision": LENS_REV, "lens_sha256": LENS_HASH, "lens_n_prompts": lens.n_prompts,
                  "upstream": str(UPSTREAM), "upstream_revision": UPSTREAM_REV, "cache": str(HUB),
                  "model_config_sha256": hash_file(model_snapshot / "config.json"),
                  "tokenizer_sha256": hash_file(model_snapshot / "tokenizer.json"),
                  "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": total,
                  "invocation_id": os.environ.get("INVOCATION_ID"), "no_network": True}
    return hf, model, tok, lens, baseline, provenance


def decode(model, tokenizer, lens, inputs, names, device):
    import numpy as np
    import torch
    used, readouts = [], []
    with torch.no_grad():
        for name, direction in zip(names, inputs, strict=True):
            tensor = torch.tensor(direction, device=device, dtype=torch.float32)
            actual = tensor.cpu().numpy().copy()
            require(np.array_equal(actual, direction), "Decoder inputs changed in transport to device")
            logits = model.unembed(lens.transport(tensor, LAYER)).float()
            require(logits.ndim == 1 and torch.isfinite(logits).all(), "Invalid readout logits")
            top = logits.topk(12)
            ids = top.indices.cpu().tolist()
            readouts.append({"name": name, "token_ids": ids, "tokens": [tokenizer.decode([i]) for i in ids],
                             "scores": top.values.cpu().tolist()})
            used.append(actual)
    return np.stack(used), readouts


def capture(model, tokenizer, rows, device):
    import numpy as np
    import torch
    activations, records = [], []
    with torch.no_grad():
        for row in rows:
            encoded = tokenizer(row["prefix"], add_special_tokens=False, truncation=False)
            ids = encoded["input_ids"]
            require(isinstance(ids, list) and 0 < len(ids) <= 96 and all(type(i) is int and i >= 0 for i in ids), "Invalid tokenized prefix; no dropping/truncation")
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            captured = []
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                require(tuple(h.shape) == (1, len(ids), WIDTH), "Unexpected residual shape")
                captured.append(h[0, -1].detach().float().cpu().numpy().copy())
                # Returning None preserves the module output unchanged.
            handle = model.layers[LAYER].register_forward_hook(hook)
            try:
                result = model.forward(tensor)
                del result
            finally:
                handle.remove()
            require(len(captured) == 1 and np.isfinite(captured[0]).all(), "Missing/repeated/nonfinite capture")
            activations.append(captured[0])
            records.append({"id": row["id"], "prefix": row["prefix"], "input_ids": ids,
                            "captured_position": len(ids)-1, "layer": LAYER})
    return np.stack(activations), records


def complete_stage(directory, files, extra):
    receipt = {"status": "complete", "completed_utc": utc(),
               "outputs": [{"path": p.name, "sha256": hash_file(p)} for p in files], **extra}
    write_json(directory / "receipt.json", receipt)
    return receipt


def interpretation_export(rankings, selection):
    by_name = {r["name"]: r["tokens"] for r in rankings}
    require(len(by_name) == len(rankings), "Duplicate reference name")
    require([a["axis"] for a in selection["axes"]] == [f"PC{i}" for i in range(1, 5)], "Reference axis order")
    axes = []
    for axis in selection["axes"]:
        entry = {"axis": axis["axis"], "A": {}, "B": {}, "C": {}}
        for sign, label in [("+", "positive"), ("-", "negative")]:
            pole = axis["poles"][sign]
            entry["A"][label] = by_name[axis["axis"] + sign]
            entry["B"][label] = by_name[pole["id"]]
            entry["C"][label] = pole["prefix"]
        axes.append(entry)
    return {"schema": "jlens_fresh_references_v1", "axes": axes}


def check_reference_lock(directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    require(receipt["status"] == "complete", "Reference stage incomplete")
    require([r["path"] for r in receipt["outputs"]] == ["decoder-inputs.npz", "readouts.json", "interpretations.json"], "Reference receipt inventory")
    for record in receipt["outputs"]:
        require(hash_file(directory / record["path"]) == record["sha256"], "Reference artifact changed")
    return hash_file(directory / "receipt.json")


def run_stages(directory, hf, model, tokenizer, lens, baseline, prepared, rows, pairs, device, provenance):
    import numpy as np
    references, fresh = directory / "references", directory / "fresh-features"
    claim(directory / "references-attempt.json", references)
    used, rankings = decode(model, tokenizer, lens, prepared["inputs"], prepared["names"], device)
    require(parameter_state(hf) == baseline, "Parameter mutation during reference decoding")
    references.mkdir()
    with (references / "decoder-inputs.npz").open("xb") as f: np.savez_compressed(f, inputs32=used)
    write_json(references / "readouts.json", {"layer": LAYER, "method": "jlens", "readouts": rankings})
    write_json(references / "interpretations.json", interpretation_export(rankings, prepared["selection"]))
    complete_stage(references, [references / "decoder-inputs.npz", references / "readouts.json", references / "interpretations.json"],
                   {"source_sha256": hash_file(Path(__file__)), "provenance": provenance, "parameters_unchanged": True})
    reference_hash = check_reference_lock(references)
    claim(directory / "fresh-features-attempt.json", fresh)
    h, records = capture(model, tokenizer, rows, device)
    require(parameter_state(hf) == baseline, "Parameter mutation during fresh forwards")
    require(check_reference_lock(references) == reference_hash, "Reference lock changed")
    scores = (h.astype(np.float64) - prepared["mean64"]) @ prepared["u32"].astype(np.float64)
    indices = {r["id"]: i for i, r in enumerate(rows)}
    gaps = np.stack([scores[indices[p["left"]]] - scores[indices[p["right"]]] for p in pairs])
    require(h.dtype == np.float32 and np.isfinite(scores).all() and np.isfinite(gaps).all(), "Invalid fresh arrays")
    fresh.mkdir()
    with (fresh / "features.npz").open("xb") as f: np.savez_compressed(f, activation_11=h, scores64=scores, gaps64=gaps)
    write_json(fresh / "inputs.json", records)
    write_json(fresh / "gaps.json", {"axes": [f"PC{i}" for i in range(1, 5)],
        "orientation": "left score minus right score", "pairs": [{**p, "gaps64": gap.tolist()} for p, gap in zip(pairs, gaps, strict=True)]})
    write_json(fresh / "scores.json", {"schema": "jlens_fresh_scores_v1", "scores": [
        {"axis": f"PC{axis+1}", "values": {row["id"]: float(scores[i, axis]) for i, row in enumerate(rows)}} for axis in range(4)]})
    complete_stage(fresh, [fresh / "features.npz", fresh / "inputs.json", fresh / "gaps.json", fresh / "scores.json"],
                   {"reference_receipt_sha256_before_forwards": reference_hash, "prefix_count": len(rows),
                    "parameters_unchanged": True, "no_losses_gradients_generation": True})


def acquire():
    started = time.monotonic()
    require(os.environ.get("JLENS_FRESH_GPU_RELEASE") == "1", "Fresh parent GPU/memory admission required")
    absent([OUT / name for name in ["acquisition-attempt.json", "acquisition-receipt.json", "references-attempt.json",
                                    "references", "fresh-features-attempt.json", "fresh-features"]])
    claim(OUT / "acquisition-attempt.json", OUT / "acquisition-receipt.json")
    prepared, rows, pairs = load_inputs()
    hf, model, tok, lens, baseline, provenance = load_runtime()
    provenance["input_pins"] = [{"path": str(p), "sha256": h} for p, h in PINS.items()]
    run_stages(OUT, hf, model, tok, lens, baseline, prepared, rows, pairs, "cuda", provenance)
    require(parameter_state(hf) == baseline, "Final parameter invariant failed")
    import torch
    write_json(OUT / "acquisition-receipt.json", {"status": "complete", "completed_utc": utc(),
        "source_sha256": hash_file(Path(__file__)), "provenance": provenance, "reference_count": 15, "fresh_prefixes": 24,
        "reference_receipt_sha256": hash_file(OUT / "references/receipt.json"),
        "fresh_receipt_sha256": hash_file(OUT / "fresh-features/receipt.json"),
        "elapsed_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_memory_limit_scope": "8 GiB PyTorch allocator fraction; not a total-process GPU-memory hard cap",
        "parameters_unchanged": True,
        "pca_refit": False, "old_fit_forwards": 0, "semantic_grading": False})
    print("Completed one reference stage then 24 fresh-prefix forwards; no PCA, old-fit forwards, losses or grading.")


def fixture():
    import numpy as np
    import torch
    import tempfile
    from unittest.mock import patch
    torch.set_num_threads(1)
    class Tokenizer:
        def __call__(self, text, **kwargs):
            require(kwargs == {"add_special_tokens": False, "truncation": False}, "Tokenizer flags")
            return {"input_ids": list(range(1, len(text.split()) + 1))}
        def decode(self, ids): return f"fabricated-{ids[0]}"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
            self.before_forward = lambda: None
            self.calls = 0
        def forward(self, ids):
            self.before_forward()
            self.calls += 1
            h = ids.float()[..., None] + torch.arange(WIDTH)[None, None, :]
            output = self.layers[LAYER](h)
            require(torch.equal(output, h), "Hook altered output")
            return output
        def unembed(self, x): return x
    class Lens:
        def transport(self, x, layer):
            require(layer == LAYER, "Wrong layer")
            return x
    model = Model().eval()
    baseline = parameter_state(model)
    rows = [{"id": "one", "prefix": "alpha beta"}, {"id": "two", "prefix": "gamma delta epsilon"}]
    names = [f"PC{i}{sign}" for i in range(1, 5) for sign in ["+", "-"]] + ["fit-one", "fit-two"]
    prepared = {"inputs": np.eye(WIDTH, dtype=np.float32)[:10], "names": names,
                "mean64": np.zeros(WIDTH), "u32": np.eye(WIDTH, 4, dtype=np.float32),
                "selection": {"axes": [{"axis": f"PC{i}", "poles": {
                    "+": {"id": "fit-one", "prefix": "fabricated fit first"},
                    "-": {"id": "fit-two", "prefix": "fabricated fit second"}}} for i in range(1, 5)]}}
    with tempfile.TemporaryDirectory(prefix="jlens-acquisition-fixture-") as scratch:
        directory = Path(scratch)
        model.before_forward = lambda: check_reference_lock(directory / "references")
        run_stages(directory, model, model, Tokenizer(), Lens(), baseline, prepared, rows,
                   [{"id": "pair", "left": "one", "right": "two"}], "cpu", {"kind": "fabricated"})
        require(model.calls == 2, "Unexpected forward count")
        with np.load(directory / "fresh-features/features.npz", allow_pickle=False) as arrays:
            require(np.array_equal(arrays["activation_11"][:, 0], [2, 3]), "Captured wrong prefix position")
            require(np.array_equal(arrays["gaps64"], -np.ones((1, 4))), "Gap order or U32 scoring")
        with np.load(directory / "references/decoder-inputs.npz", allow_pickle=False) as arrays:
            require(np.array_equal(arrays["inputs32"], prepared["inputs"]), "Actual decoder inputs not preserved")
        rankings = json.loads((directory / "references/readouts.json").read_text())["readouts"]
        require([r["name"] for r in rankings] == prepared["names"] and all(len(r["tokens"]) == 12 for r in rankings), "Readout order/ranks")
        exported = json.loads((directory / "references/interpretations.json").read_text())
        require(exported["schema"] == "jlens_fresh_references_v1" and len(exported["axes"]) == 4, "Reference export schema")
        require(exported["axes"][0]["A"]["positive"] == rankings[0]["tokens"] and exported["axes"][0]["B"]["negative"] == rankings[9]["tokens"], "Reference export joins")
        require(exported["axes"][0]["C"]["negative"] == "fabricated fit second", "Exact C prefix")
        exported_scores = json.loads((directory / "fresh-features/scores.json").read_text())
        require(exported_scores["schema"] == "jlens_fresh_scores_v1" and exported_scores["scores"][0]["values"] == {"one": 2., "two": 3.}, "Score export schema/values")
        for existing in [directory / "references", directory / "references-attempt.json"]:
            try: claim(directory / "unclaimed.json", existing)
            except FileExistsError: pass
            else: raise AssertionError("Existing stage accepted")
        dangling = directory / "dangling"
        dangling.symlink_to(directory / "absent")
        try: claim(directory / "unclaimed.json", dangling)
        except FileExistsError: pass
        else: raise AssertionError("Dangling target accepted")
        with patch(__name__ + ".capture") as fresh_call:
            try: run_stages(directory, model, model, Tokenizer(), Lens(), baseline, prepared, rows, [], "cpu", {})
            except FileExistsError: pass
            else: raise AssertionError("Repeated stage accepted")
            fresh_call.assert_not_called()
    model.before_forward = lambda: None
    for prefix in ["", "word " * 97]:
        try: capture(model, Tokenizer(), [{"id": "invalid", "prefix": prefix}], "cpu")
        except ValueError: pass
        else: raise AssertionError("Invalid input was silently accepted")
    require(model.calls == 2 and parameter_state(model) == baseline, "Invalid prefix forwarded or parameters changed")
    with torch.no_grad(): model.weight.add_(1)
    require(parameter_state(model) != baseline, "Parameter mutation was not detected")
    print("PASS fabricated acquisition: reference lock before forwards, exact decoder inputs/ranks, unmodified post-block last position, U32 gap order, token limits, guards, frozen-state checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fixture", "acquire"])
    {"fixture": fixture, "acquire": acquire}[parser.parse_args().stage]()
