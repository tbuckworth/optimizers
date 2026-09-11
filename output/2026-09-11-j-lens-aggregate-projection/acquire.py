"""Inert, once-only acquisition; no tokenizer, covariance fit, lens or decoder.

Input: jlens_aggregate_panel_v1 with rows {id, article_id, role, batch,
input_ids[40], attention_mask[40]}; extra source provenance is preserved.
Production is exactly 448 articles x 4 rows: fit1024/query256/reference512.
Launch requires reviewed source/panel/protocol hashes and explicit release.
Outer unit must enforce CPU1, MemoryMax8GiB, MemorySwapMax0, RuntimeMaxSec5400,
Restart=no. The 8GiB Torch allocator limit is not a whole-process GPU cap.

Contract follows experiments/j_lens_completions.py, not its entrypoint.
API checks: docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html,
torch.nn.Module.html; huggingface.co/docs/transformers/main_classes/model.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
PROTOCOL = Path('output/2026-09-11-jlens-aggregate-projection/protocol.md')
PROTOCOL_SHA = 'c933bf833b2e6bda0eacd0b3a52597adf1eda2ea05b7e69122064892a8d75b8e'
REV = '2fc06364715b967f1860aea9cf38778875588b17'
MODEL = 'Qwen/Qwen3.5-0.8B'
SNAPSHOT = Path('/private-artifacts/storage/cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots')/REV
ADAPTER = OUT.parents[2]/'jacobian-lens/jlens/hf.py'
QWEN = Path('/private-artifacts/.local/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')
PINS = {
    ADAPTER: '228cf078e4586a7b7f61a6f5064403b8960de337afd19256efa56f04d53e3222',
    QWEN: 'aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048',
    SNAPSHOT/'config.json': 'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204',
    SNAPSHOT/'model.safetensors.index.json': 'd8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e',
    SNAPSHOT/'model.safetensors-00001-of-00001.safetensors': '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',
}
PYTHON = '3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]'
NUMERICAL = {'numpy': '1.26.4', 'torch': '2.11.0+cu128', 'transformers': '5.5.0', 'huggingface_hub': '1.8.0'}
PACKAGES = {**{p: '3.12.3-1ubuntu0.17' for p in ('python3.12', 'python3.12-minimal',
    'libpython3.12-minimal', 'libpython3.12-stdlib', 'libpython3.12t64')},
    'libc6': '2.39-0ubuntu8.9', 'libc-bin': '2.39-0ubuntu8.9'}
BINARIES = {'/usr/bin/python3.12': 'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f',
    '/lib/x86_64-linux-gnu/libc.so.6': '3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf'}
COUNTS = {'fit': 1024, 'query': 256, 'reference': 512}
LAYER, WIDTH, VOCAB = 11, 1024, 248320


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def read_json(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024**2, 'JSON path/size')
    def pairs(items):
        obj = {}
        for k, v in items:
            require(k not in obj, 'duplicate JSON key'); obj[k] = v
        return obj
    def bad(value): raise ValueError('nonfinite JSON constant')
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=bad)
    def finite(x):
        if isinstance(x, float): require(math.isfinite(x), 'nonfinite JSON number')
        elif isinstance(x, dict):
            for v in x.values(): finite(v)
        elif isinstance(x, list):
            for v in x: finite(v)
    finite(value)
    return value


def validate_panel(panel):
    require(type(panel) is dict and panel.get('schema') == 'jlens_aggregate_panel_v1', 'panel schema')
    rows = panel.get('rows')
    require(type(rows) is list and len(rows) == sum(COUNTS.values()), 'exact row count')
    articles, ids, sequences, counts, batches = {}, set(), set(), Counter(), Counter()
    article_rows = Counter()
    for row in rows:
        require(type(row) is dict and {'id', 'article_id', 'role', 'batch', 'input_ids', 'attention_mask'} <= row.keys(), 'row fields')
        name, article, role, batch = (row[k] for k in ('id', 'article_id', 'role', 'batch'))
        require(type(name) is str and name and name not in ids, 'duplicate/invalid ID'); ids.add(name)
        require(type(article) is str and article and role in COUNTS, 'article/role')
        require((role == 'fit' and batch is None) or (role != 'fit' and type(batch) is int and
                0 <= batch < (16 if role == 'query' else 2)), 'batch roster')
        require(article not in articles or articles[article] == (role, batch), 'article crosses role/batch')
        articles[article] = (role, batch); article_rows[article] += 1
        tokens, mask = row['input_ids'], row['attention_mask']
        require(type(tokens) is list and len(tokens) == 40 and all(type(t) is int and 0 <= t < VOCAB for t in tokens), 'exact forty token IDs')
        require(type(mask) is list and len(mask) == 40 and all(type(t) is int and t == 1 for t in mask), 'unpadded all-one mask')
        require(tuple(tokens) not in sequences, 'duplicate token window'); sequences.add(tuple(tokens))
        counts[role] += 1; batches[(role, batch)] += 1
    require(counts == COUNTS and all(n == 4 for n in article_rows.values()), 'four rows per article/role count')
    require(batches == Counter({('fit', None): 1024, **{('query', i): 16 for i in range(16)},
                              **{('reference', i): 256 for i in range(2)}}), 'complete fixed batch counts')
    return rows


def verify(expected):
    for path, digest in expected.items():
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) and sha(path) == digest,
                'input/source hash mismatch: '+str(path))


def runtime():
    import numpy, torch, transformers, huggingface_hub
    versions = {m.__name__: m.__version__ for m in (numpy, torch, transformers, huggingface_hub)}
    raw = subprocess.check_output(['dpkg-query', '-W', '-f=${Package}\t${Version}\n',
                                  *[p+':amd64' for p in PACKAGES]], text=True, timeout=10)
    packages = dict(line.split('\t') for line in raw.splitlines())
    binaries = {p: sha(p) for p in BINARIES}
    require(sys.version == PYTHON and sys.executable == '/usr/bin/python3' and
            Path(sys.executable).resolve() == Path('/usr/bin/python3.12'), 'Python runtime changed')
    require(versions == NUMERICAL and packages == PACKAGES and binaries == BINARIES, 'runtime pins changed')
    require(Path(sys.modules['transformers.models.qwen3_5.modeling_qwen3_5'].__file__).resolve() == QWEN
            if 'transformers.models.qwen3_5.modeling_qwen3_5' in sys.modules else True, 'Qwen module path')
    return {'python': sys.version, 'interpreter': sys.executable, 'packages': versions,
            'distro_packages': packages, 'binary_sha256s': binaries}


def load_model():
    import torch
    from transformers import AutoModelForCausalLM
    require(torch.cuda.is_available(), 'CUDA unavailable; no fallback')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    raw = ADAPTER.read_bytes(); require(hashlib.sha256(raw).hexdigest() == PINS[ADAPTER], 'adapter changed')
    spec = importlib.util.spec_from_file_location('aggregate_hf_adapter', ADAPTER)
    adapter = importlib.util.module_from_spec(spec); sys.modules[spec.name] = adapter
    exec(compile(raw, str(ADAPTER), 'exec'), adapter.__dict__)
    hf = AutoModelForCausalLM.from_pretrained(str(SNAPSHOT), local_files_only=True, token=False,
        trust_remote_code=False, use_safetensors=True, weights_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').cuda()
    hf.eval(); hf.requires_grad_(False)
    model = adapter.from_hf(hf, None, force_bos=False)
    require(model.d_model == WIDTH and model.n_layers == 24 and model._logit_softcap is None, 'model contract')
    require(all(p.dtype == torch.bfloat16 for p in hf.parameters()), 'parameter precision')
    return hf, model


def parameter_state(hf):
    require(all(not m.training for m in hf.modules()), 'eval required')
    require(all(not p.requires_grad and p.grad is None for p in hf.parameters()), 'frozen gradient-free parameters required')
    return [(name, id(p), p._version) for name, p in hf.named_parameters()]


def capture_one(model, ids, width=WIDTH, layer=LAYER):
    """One forward + one residual VJP. Return h/g at31, eight FP32 losses."""
    import torch
    require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), 'gradient acquisition needs grad mode')
    require(ids.shape == (1, 40) and ids.dtype == torch.long, 'one exact forty-token row')
    captured = []
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        require(h.shape == (1, 40, width) and torch.isfinite(h).all().item(), 'post-block shape/nonfinite')
        require(not captured, 'block invoked more than once')
        leaf = h.detach().requires_grad_(True); captured.append(leaf)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf
    handle = model.layers[layer].register_forward_hook(hook)
    try:
        result = model.forward(ids)  # Pinned adapter: use_cache=False; implicit all-valid mask.
        require(len(captured) == 1, 'missing capture')
        # result is already final-normalized; do not apply model.unembed/RMSNorm again.
        logits = model._lm_head(result.last_hidden_state[:, 31:39]).float()
        losses = torch.nn.functional.cross_entropy(logits[0], ids[0, 32:40], reduction='none')
        require(losses.shape == (8,) and torch.isfinite(losses).all().item(), 'loss nonfinite/shape')
        mean_loss = losses.mean()
        gradient, = torch.autograd.grad(mean_loss, captured[0], retain_graph=False, create_graph=False)
        h = captured[0][0, 31].detach().float().cpu().numpy().copy()
        g = gradient[0, 31].detach().float().cpu().numpy().copy()
        require(torch.isfinite(gradient).all().item(), 'nonfinite gradient')
        return h, g, losses.detach().float().cpu().numpy().copy(), mean_loss.detach().float().cpu().numpy().copy()
    finally:
        handle.remove()


def claim(target, source_sha):
    target.mkdir()  # Atomic no-clobber claim; existing directories/files/dangling links fail.
    write_json(target/'attempt.json', {'schema': 'jlens_aggregate_attempt_v1',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_sha,
        'invocation_id': os.environ.get('INVOCATION_ID'), 'automatic_retry': False})


def run(panel_path, panel_sha, protocol_path, protocol_sha, source_sha, target):
    require(os.environ.get('JLENS_AGGREGATE_ACQUIRE_RELEASE') == '1', 'explicit parent release required')
    require(sha(Path(__file__)) == source_sha, 'reviewed source pin required')
    claim(target, source_sha)
    started, completed = time.monotonic(), 0
    prior = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame): raise InterruptedError('outer unit terminated acquisition')
    signal.signal(signal.SIGTERM, terminate)
    try:
        require(protocol_path == PROTOCOL and protocol_sha == PROTOCOL_SHA, 'frozen protocol required')
        expected = {**PINS, panel_path: panel_sha, protocol_path: protocol_sha, Path(__file__): source_sha}
        verify(expected); panel = read_json(panel_path); rows = validate_panel(panel)
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN',
                     'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            require(os.environ.get(name) == '1', 'release environment: '+name)
        env = runtime()
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20260923)
        write_json(target/'inputs.json', panel); (target/'rows').mkdir()
        hf, model = load_model(); baseline = parameter_state(hf)
        require(runtime() == env, 'runtime changed during model load')
        features = defaultdict(list); artifacts = []
        for i, row in enumerate(rows):
            require(time.monotonic()-started < 5400, '90-minute stage cap')
            ids = torch.tensor([row['input_ids']], dtype=torch.long, device='cuda')
            h, g, losses, mean_loss = capture_one(model, ids)
            require(parameter_state(hf) == baseline, 'parameter state changed')
            values = {'activation32': h, 'gradient32': g, 'losses32': losses, 'mean_losses32': mean_loss}
            path = target/'rows'/f'{i:04}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, **values); f.flush(); os.fsync(f.fileno())
            artifacts.append({'index': i, 'id': row['id'], 'path': str(path.relative_to(target)), 'sha256': sha(path)})
            for key, value in values.items(): features[key].append(value)
            completed += 1
            if completed % 64 == 0: print(f'Completed {completed}/{len(rows)} rows', flush=True)
        arrays = {k: np.stack(v) for k, v in features.items()}
        require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in arrays.values()), 'saved precision/finite')
        arrays['ids'] = np.array([row['id'] for row in rows], dtype=np.str_)
        with (target/'features.npz').open('xb') as f: np.savez_compressed(f, **arrays)
        write_json(target/'row-artifacts.json', artifacts)
        verify(expected); require(runtime() == env and parameter_state(hf) == baseline, 'final state changed')
        outputs = [{'path': n, 'sha256': sha(target/n), 'size_bytes': (target/n).stat().st_size}
                   for n in ('inputs.json', 'features.npz', 'row-artifacts.json')]
        write_json(target/'receipt.json', {'schema': 'jlens_aggregate_acquisition_v1', 'status': 'complete',
            'source_sha256': source_sha, 'input_pins': {str(p): d for p, d in expected.items()}, 'outputs': outputs,
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
            'runtime': env, 'model': MODEL, 'revision': REV, 'forward_count': completed, 'backward_count': completed,
            'role_counts': COUNTS, 'layer': LAYER, 'capture_index': 31, 'context_tokens': 32, 'target_tokens': 8,
            'loss': 'mean FP32 CE of logits[31:39] against input_ids[32:40]',
            'parameters_unchanged': True, 'parameter_gradients_absent': True, 'tokenizer_loaded': False,
            'pca_fit': False, 'lens_loaded': False, 'gpu': torch.cuda.get_device_name(0),
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'memory_scope': '8GiB Torch allocator envelope; not whole-process GPU cap'})
    except BaseException as error:
        write_json(target/'failure.json', {'status': 'FAILED', 'error_type': type(error).__name__,
            'completed_rows': completed, 'automatic_retry': False, 'partial_outputs_preserved': True})
        raise
    finally:
        signal.signal(signal.SIGTERM, prior)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'protocol', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('panel-sha', 'protocol-sha', 'source-sha'): parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    run(a.panel, a.panel_sha, a.protocol, a.protocol_sha, a.source_sha, a.output)
