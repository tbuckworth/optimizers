"""Import-inert independent-content token preflight and forwards only; no lens/decoding."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MAIN = Path('output/2026-09-11-jlens-independent-content')
OLD = ROOT/'output/2026-09-10-j-lens-fresh-content'
HUB = Path('/private-artifacts/storage/cache/huggingface/hub')
MODEL, REV = 'Qwen/Qwen3.5-0.8B', '2fc06364715b967f1860aea9cf38778875588b17'
SNAPSHOT = HUB/'models--Qwen--Qwen3.5-0.8B/snapshots'/REV
UPSTREAM = ROOT.parent/'jacobian-lens'
UPSTREAM_REV = '581d398613e5602a5af361e1c34d3a92ea82ba8e'
WEIGHT_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
PINS = {
    MAIN/'protocol.md': 'f7314607d2846707a2ee2e8b64c8f2f1fb3098a2366bb1f0d5b0bc0afdcbea3b',
    OLD/'preparation/directions.npz': '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa',
    OLD/'references/interpretations.json': '0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c',
    SNAPSHOT/'config.json': 'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204',
    SNAPSHOT/'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',  # gitleaks:allow -- independently verified public model-file SHA256, not a credential
    SNAPSHOT/'tokenizer_config.json': '49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c',  # gitleaks:allow -- independently verified public model-file SHA256, not a credential
    SNAPSHOT/'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    SNAPSHOT/'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
    SNAPSHOT/'model.safetensors.index.json': 'd8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e',
}
TOPICS = ['astronomy', 'cooking', 'football', 'programming']
IDS = [f'{topic}-wiki-{i}' for topic in TOPICS for i in range(6)]
LAYER, WIDTH, VOCAB = 11, 1024, 248320
DATASET_SHA = '7391a1fa6c02872cdeabb2e8dffc1e63181b6379d80e2042f33b38a61308ff02'
PAIRS_SHA = 'b38badeebcfe3c54a4315bd60dbaf8b5ace85975a8554e0074c2bd286d301ecc'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def read_json(path):
    import math
    require(not path.is_symlink() and path.stat().st_size <= 1024**2, 'invalid JSON file')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value): raise ValueError('nonfinite JSON constant')
    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(x):
        if type(x) is float: require(math.isfinite(x), 'nonfinite JSON number')
        elif type(x) is dict:
            for v in x.values(): finite(v)
        elif type(x) is list:
            for v in x: finite(v)
    finite(value)
    return value


def pins(dataset_sha, pairs_sha):
    require(all(type(s) is str and re.fullmatch('[0-9a-f]{64}', s) for s in [dataset_sha, pairs_sha]), 'explicit dataset/pair pins required')
    require((dataset_sha, pairs_sha) == (DATASET_SHA, PAIRS_SHA), 'frozen collection pins required')
    return {**PINS, MAIN/'excerpts-resumed/dataset.json': dataset_sha, MAIN/'excerpts-resumed/pairs.json': pairs_sha}


def verify(expected):
    for path, digest in expected.items(): require(sha(path) == digest, f'Input hash mismatch: {path}')


def claim(root, name):
    target = root/name
    if target.exists() or target.is_symlink(): raise FileExistsError(f'Consumed/existing stage: {target}')
    target.mkdir()
    write_json(target/'attempt.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    return target


def finish(target, expected, outputs, extra):
    verify(expected)
    require(read_json(target/'attempt.json')['source_sha256'] == sha(Path(__file__)), 'source changed during stage')
    write_json(target/'receipt.json', {'schema': 'jlens_independent_'+target.name.replace('-', '_')+'_receipt_v1',
        'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
        'input_pins': {str(p): s for p, s in expected.items()},
        'outputs': [{'path': p.name, 'sha256': sha(p)} for p in outputs], **extra})


def failure(target, error):
    write_json(target/'failure.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'automatic_retry': False})


def validate_rows(rows, pairs):
    require(type(rows) is list and [r['id'] for r in rows] == IDS, 'exact 24-row ordering required')
    require(len({r['prefix'] for r in rows}) == 24, 'duplicate prefix')
    for row in rows:
        require(set(row) == {'id', 'topic', 'prefix'} and row['topic'] == row['id'].split('-')[0], 'row metadata')
        text = row['prefix']
        require(type(text) is str and len(text.split()) == 16 and text == ' '.join(text.split()), 'exact 16-word prefix required')
    require(pairs == [{'id': f'P{i+1:02}', 'left': IDS[2*i], 'right': IDS[2*i+1]} for i in range(12)], 'fixed within-category pairs required')


def runtime():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    import transformers
    import huggingface_hub
    versions = {'numpy': np.__version__, 'torch': torch.__version__, 'transformers': transformers.__version__, 'huggingface_hub': huggingface_hub.__version__}
    require(versions == {'numpy': '1.26.4', 'torch': '2.11.0+cu128', 'transformers': '5.5.0', 'huggingface_hub': '1.8.0'}, 'runtime changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return {'python': sys.version, 'interpreter': sys.executable, 'packages': versions,
            'command': [sys.executable, *sys.argv], 'model': MODEL, 'revision': REV, 'no_network': True,
            'invocation_id': os.environ.get('INVOCATION_ID')}


def tokenize(rows, tokenizer):
    records = []
    for row in rows:
        encoded = tokenizer(row['prefix'], add_special_tokens=False, truncation=False, return_attention_mask=True)
        ids, mask = encoded['input_ids'], encoded['attention_mask']
        require(type(ids) is list and 1 <= len(ids) <= 96 and all(type(i) is int and 0 <= i < VOCAB for i in ids), 'token count/ID failure; no dropping or truncation')
        require(type(mask) is list and len(mask) == len(ids) and all(type(x) is int and x == 1 for x in mask), 'expected unpadded all-valid mask')
        records.append({**row, 'input_ids': ids, 'attention_mask': mask, 'captured_position': len(ids)-1, 'layer': LAYER})
    return records


def preflight(dataset_sha, pairs_sha, root=OUT):
    require(os.environ.get('JLENS_INDEPENDENT_PREFLIGHT_RELEASE') == '1', 'parent preflight release required')
    target = claim(root, 'token-preflight')
    started = time.monotonic()
    try:
        expected = pins(dataset_sha, pairs_sha); verify(expected)
        rows, pairs = read_json(MAIN/'excerpts-resumed/dataset.json'), read_json(MAIN/'excerpts-resumed/pairs.json')
        validate_rows(rows, pairs)
        env = runtime()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REV, cache_dir=str(HUB), local_files_only=True, trust_remote_code=False, use_fast=True)
        require(tokenizer.is_fast and tokenizer.bos_token_id is None, 'tokenizer/BOS configuration changed')
        records = tokenize(rows, tokenizer)
        write_json(target/'tokens.json', {'schema': 'jlens_independent_tokens_v1', 'records': records})
        finish(target, expected, [target/'tokens.json'], {'runtime': env, 'count': 24, 'elapsed_seconds': time.monotonic()-started,
            'model_loaded': False, 'scientific_array_loaded': False, 'token_limit': 96})
    except Exception as error:
        failure(target, error); raise


def frozen_tokens(root, receipt_sha, expected, rows):
    path = root/'token-preflight/receipt.json'
    require(type(receipt_sha) is str and re.fullmatch('[0-9a-f]{64}', receipt_sha) is not None and sha(path) == receipt_sha, 'preflight receipt pin')
    receipt = read_json(path)
    require(receipt['schema'] == 'jlens_independent_token_preflight_receipt_v1' and receipt['status'] == 'complete'
            and receipt['source_sha256'] == sha(Path(__file__)) and receipt['input_pins'] == {str(p): s for p, s in expected.items()}, 'preflight source/input binding')
    require(len(receipt['outputs']) == 1 and receipt['outputs'][0]['path'] == 'tokens.json', 'token output inventory')
    tokens = root/'token-preflight/tokens.json'
    require(sha(tokens) == receipt['outputs'][0]['sha256'], 'token bytes changed')
    payload = read_json(tokens)
    require(payload['schema'] == 'jlens_independent_tokens_v1' and len(payload['records']) == 24, 'token schema/roster')
    for record, row in zip(payload['records'], rows, strict=True):
        require({k: record[k] for k in row} == row and set(record) == {*row, 'input_ids', 'attention_mask', 'captured_position', 'layer'}, 'token row identity')
        ids = record['input_ids']
        require(type(ids) is list and 1 <= len(ids) <= 96 and all(type(i) is int and 0 <= i < VOCAB for i in ids), 'frozen token IDs')
        mask = record['attention_mask']
        require(type(mask) is list and len(mask) == len(ids) and all(type(x) is int and x == 1 for x in mask)
                and record['captured_position'] == len(ids)-1 and record['layer'] == LAYER, 'frozen position/mask')
    return payload['records'], receipt


def validate_directions(u, mean):
    import numpy as np
    for value, shape, dtype in [(u, (WIDTH, 4), np.float32), (mean, (WIDTH,), np.float64)]:
        require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), 'invalid canonical array')
    require(np.allclose(np.linalg.norm(u.astype(np.float64), axis=0), 1, rtol=0, atol=1e-6), 'canonical unit norms')


def parameter_state(hf):
    require(all(not m.training for m in hf.modules()), 'model must be eval')
    require(all(not p.requires_grad and p.grad is None for p in hf.parameters()), 'parameters must be frozen/gradient-free')
    return [(id(p), p._version) for p in hf.parameters()]


def load_model():
    import torch
    from transformers import AutoModelForCausalLM
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip() == UPSTREAM_REV, 'adapter revision changed')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=UPSTREAM, text=True).strip(), 'adapter source dirty')
    sys.path.insert(0, str(UPSTREAM))
    import jlens
    require(Path(jlens.__file__).resolve().is_relative_to(UPSTREAM/'jlens'), 'wrong adapter import')
    require(torch.cuda.is_available(), 'CUDA unavailable; no alternate run')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    require(sha(SNAPSHOT/'model.safetensors-00001-of-00001.safetensors') == WEIGHT_SHA, 'model weight pin mismatch')
    hf = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, cache_dir=str(HUB), local_files_only=True,
                                           trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').cuda()
    hf.eval(); hf.requires_grad_(False)
    # Tokens are already frozen. This adapter only locates/forwards the identical text module.
    model = jlens.from_hf(hf, None, force_bos=False)
    require(model.d_model == WIDTH and model.n_layers == 24, 'model layout changed')
    return hf, model


def capture(model, records, device):
    import numpy as np
    import torch
    activations = []
    with torch.no_grad():
        for record in records:
            ids = record['input_ids']
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            captured = []
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                require(tuple(h.shape) == (1, len(ids), WIDTH), 'unexpected post-block output')
                captured.append(h[0, -1].detach().float().cpu().numpy().copy())
                # None return: observe, never replace the block output.
            handle = model.layers[LAYER].register_forward_hook(hook)
            try:
                result = model.forward(tensor)  # Original implicit all-valid mask; no padding/cache.
                del result
            finally: handle.remove()
            require(len(captured) == 1 and np.isfinite(captured[0]).all(), 'missing/repeated/nonfinite capture')
            activations.append(captured[0])
    return np.stack(activations)


def score_export(h, u, mean, rows, pairs):
    import numpy as np
    validate_directions(u, mean)
    require(h.shape == (24, WIDTH) and h.dtype == np.float32 and np.isfinite(h).all(), 'invalid saved activation shape/dtype')
    scores = (h.astype(np.float64)-mean) @ u.astype(np.float64)
    index = {r['id']: i for i, r in enumerate(rows)}
    gaps = np.stack([scores[index[p['left']]]-scores[index[p['right']]] for p in pairs])
    require(np.isfinite(scores).all() and np.isfinite(gaps).all(), 'nonfinite geometric outputs')
    export = {'schema': 'jlens_independent_content_scores_v1', 'scores': [
        {'axis': f'PC{a+1}', 'values': {r['id']: float(scores[i, a]) for i, r in enumerate(rows)}} for a in range(4)]}
    return scores, gaps, export


def forwards(dataset_sha, pairs_sha, preflight_sha, root=OUT):
    require(os.environ.get('JLENS_INDEPENDENT_GPU_RELEASE') == '1', 'new parent GPU admission required')
    target = claim(root, 'forwards'); started = time.monotonic()
    try:
        expected = pins(dataset_sha, pairs_sha); verify(expected)
        rows, pairs = read_json(MAIN/'excerpts-resumed/dataset.json'), read_json(MAIN/'excerpts-resumed/pairs.json')
        validate_rows(rows, pairs)
        records, token_receipt = frozen_tokens(root, preflight_sha, expected, rows)
        env = runtime()
        require(env['packages'] == token_receipt['runtime']['packages'] and env['python'] == token_receipt['runtime']['python'], 'preflight runtime differs')
        import numpy as np
        import torch
        with np.load(OLD/'preparation/directions.npz', allow_pickle=False) as archive:
            u, mean = archive['u32'], archive['source_mean64']
        validate_directions(u, mean); verify(expected)
        hf, model = load_model(); baseline = parameter_state(hf)
        h = capture(model, records, 'cuda')
        require(parameter_state(hf) == baseline, 'model parameter state changed')
        scores, gaps, export = score_export(h, u, mean, rows, pairs)
        with (target/'features.npz').open('xb') as f: np.savez_compressed(f, activation_11=h, scores64=scores, gaps64=gaps)
        write_json(target/'inputs.json', {'schema': 'jlens_independent_inputs_v1', 'records': records,
                   'mask_application': 'No padding; all-one saved masks are implicit in the unchanged adapter forward.'})
        write_json(target/'scores.json', export)
        write_json(target/'gaps.json', {'axes': [f'PC{i}' for i in range(1, 5)], 'orientation': 'left minus right',
            'pairs': [{**p, 'gaps64': g.tolist()} for p, g in zip(pairs, gaps, strict=True)]})
        frozen_tokens(root, preflight_sha, expected, rows)
        finish(target, expected, [target/n for n in ['features.npz', 'inputs.json', 'scores.json', 'gaps.json']],
            {'runtime': env, 'preflight_receipt_sha256': preflight_sha, 'model_weight_sha256': WEIGHT_SHA,
             'adapter_revision': UPSTREAM_REV, 'parameters_unchanged': True, 'forward_count': 24,
             'reference_decodes': 0, 'pca_refit': False, 'tokenizer_loaded': False, 'elapsed_seconds': time.monotonic()-started,
             'gpu': torch.cuda.get_device_name(0), 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'memory_scope': '8 GiB PyTorch allocator bound, not a total-process GPU cap'})
    except Exception as error:
        failure(target, error); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'forwards'])
    parser.add_argument('--dataset-sha', required=True)
    parser.add_argument('--pairs-sha', required=True)
    parser.add_argument('--preflight-sha')
    args = parser.parse_args()
    if args.stage == 'preflight': preflight(args.dataset_sha, args.pairs_sha)
    else: forwards(args.dataset_sha, args.pairs_sha, args.preflight_sha)
