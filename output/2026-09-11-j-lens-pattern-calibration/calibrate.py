"""Inert calibration-only producer: 64 frozen forwards, four patterns, eight decodes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import types

OUT = Path(__file__).resolve().parent
MAIN = Path('output/2026-09-11-jlens-pattern-calibration')
PREFLIGHT_SOURCE_SHA = 'f484e4b022b835f4f38d1e3ffdce4722f5a1694e50297bf05d3db7e94f100a3d'
PREFLIGHT_RECEIPT_SHA = 'd4cba9b4376098720a9fa8111f7f7d7dcf7f634e9e66413cb591860ef0c25ec4'
CALIBRATION_TOKENS_SHA = '460965950596e03661eb4bc574423780be8b1baad4346e43e79045deaea5398a'  # gitleaks:allow -- verified committed token-record file SHA256, not a credential
PATTERNS_SHA = '524efc43e54c9549e66586f8a8dcc9db8542979637a56d840c3739f1c156a789'
OLD = OUT.parent/'2026-09-10-j-lens-fresh-content'
DIRECTIONS = OLD/'preparation/directions.npz'
DIRECTIONS_SHA = '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa'
ORIGINAL = OLD/'references/interpretations.json'
ORIGINAL_SHA = '0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c'
LENS_REV = '0731326edff4ae730ffc5356fe1a4728c748b3a6'
LENS_FILE = 'qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt'
LENS_SHA = 'aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48'
ADAPTER_PINS = {'jlens/hf.py': '228cf078e4586a7b7f61a6f5064403b8960de337afd19256efa56f04d53e3222',
    'jlens/lens.py': 'e231e7d3a6c8e8f7791b53705a34342d0bba376a127a82376eaf6ec30ca11808',
    'jlens/__init__.py': 'bd7bb6302565a712583b05b91d48c8638d9f5d2ace00e52db0db2fd300f7fe3a'}
QWEN_SOURCE = Path('/private-artifacts/.local/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')
QWEN_SOURCE_SHA = 'aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048'
WIDTH, VOCAB, LAYER = 1024, 248320, 11
AXES = [f'PC{i}' for i in range(1, 5)]
NAMES = [a+s for a in AXES for s in ('+', '-')]
IDS = [f'C{i:02}-{s}' for i in range(1, 33) for s in ('L', 'R')]


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2); handle.write('\n')


def import_pinned(path, digest):
    require(path.is_file() and not path.is_symlink(), 'source absent/symlink')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'source pin: '+str(path))
    module = types.ModuleType('pinned_'+path.stem); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(sha(path) == digest, 'source drift during import')
    return module


def input_pins(p, f):
    # Deliberately NOT p.frozen_pins(): that metadata map also names evaluation files.
    return {OUT/'preflight.py': PREFLIGHT_SOURCE_SHA, p.BASE: p.BASE_SHA, p.HOST_HELPER: p.HOST_HELPER_SHA,
        MAIN/'protocol.md': p.PROTOCOL_SHA, MAIN/'patterns.py': PATTERNS_SHA,
        MAIN/'selection/manifest.json': p.MANIFEST_SHA,
        OUT/'excerpts/receipt.json': p.COLLECTION_SHA,
        **{OUT/'excerpts'/n: p.COLLECTION_OUTPUTS[n] for n in ('calibration-dataset.json', 'calibration-pairs.json')},
        OUT/'token-preflight/receipt.json': PREFLIGHT_RECEIPT_SHA,
        OUT/'token-preflight/calibration-tokens.json': CALIBRATION_TOKENS_SHA,
        DIRECTIONS: DIRECTIONS_SHA, ORIGINAL: ORIGINAL_SHA,
        **{f.SNAPSHOT/n: f.PINS[f.SNAPSHOT/n] for n in (*p.CACHE_NAMES, 'model.safetensors.index.json')},
        f.SNAPSHOT/'model.safetensors-00001-of-00001.safetensors': f.WEIGHT_SHA,
        f.HUB/'models--neuronpedia--jacobian-lens/snapshots'/LENS_REV/LENS_FILE: LENS_SHA,
        **{f.UPSTREAM/n: d for n, d in ADAPTER_PINS.items()}, QWEN_SOURCE: QWEN_SOURCE_SHA}


def verify(expected):
    require(not any('evaluation-' in p.name for p in expected), 'evaluation file in calibration read set')
    for path, digest in expected.items(): require(sha(path) == digest, 'input hash mismatch: '+str(path))


def load_calibration(p, f):
    receipt = f.read_json(OUT/'token-preflight/receipt.json')
    require(receipt['schema'] == 'jlens_pattern_calibration_token_preflight_receipt_v1' and
            receipt['status'] == 'complete' and receipt['source_sha256'] == PREFLIGHT_SOURCE_SHA and
            receipt['base_helper_sha256'] == p.BASE_SHA and receipt['host_helper_sha256'] == p.HOST_HELPER_SHA,
            'preflight source/schema binding')
    # Comparing a metadata map does not open any referenced evaluation file.
    require(receipt['input_pins'] == {str(k): v for k, v in p.frozen_pins(f).items()}, 'preflight input bindings')
    require(receipt['count'] == 96 and receipt['role_counts'] == {'calibration': 64, 'evaluation': 32} and
            receipt['token_limit'] == 96 and receipt['capture_locations'] == ['prefix_end'] and
            receipt['model_loaded'] is False and receipt['scientific_array_loaded'] is False and
            receipt['reference_decodes'] == 0, 'whole-panel preflight scope')
    outputs = receipt['outputs']
    require(len(outputs) == 2 and [r['path'] for r in outputs] == ['calibration-tokens.json', 'evaluation-tokens.json'] and
            outputs[0]['sha256'] == CALIBRATION_TOKENS_SHA and
            outputs[0]['size_bytes'] == (OUT/'token-preflight/calibration-tokens.json').stat().st_size,
            'calibration token output binding')
    rows = f.read_json(OUT/'excerpts/calibration-dataset.json')
    pairs = f.read_json(OUT/'excerpts/calibration-pairs.json')
    payload = f.read_json(OUT/'token-preflight/calibration-tokens.json')
    require(type(rows) is list and len(rows) == 64 and [r['id'] for r in rows] == IDS and
            len({r['prefix'] for r in rows}) == 64, 'exact calibration-only rows')
    require(type(pairs) is list and len(pairs) == 32, 'exact calibration pairs')
    manifest = f.read_json(MAIN/'selection/manifest.json')
    require(manifest['schema'] == 'jlens_pattern_calibration_candidate_manifest_v1' and
            manifest['protocol_sha256'] == p.PROTOCOL_SHA and manifest['seed'] == '20260921', 'manifest binding')
    candidates = {c['candidate_id']: c for c in manifest['candidates']}
    require(len(candidates) == 86, 'candidate manifest scope')
    ranks = []
    for i, pair in enumerate(pairs):
        require(set(pair) == {'id', 'left', 'right', 'topic', 'candidate_id', 'role'} and
                pair['id'] == f'C{i+1:02}' and pair['left'] == IDS[2*i] and pair['right'] == IDS[2*i+1] and
                pair['role'] == 'calibration', 'calibration pair identity')
        candidate = candidates[pair['candidate_id']]
        require(candidate['role'] == 'calibration' and candidate['topic'] == pair['topic'] ==
                rows[2*i]['topic'] == rows[2*i+1]['topic'], 'calibration candidate role/topic')
        ranks.append(int(pair['candidate_id'][1:]))
    require(ranks == sorted(set(ranks)), 'calibration acceptance order')
    require(payload['schema'] == 'jlens_pattern_calibration_tokens_v1' and payload['role'] == 'calibration' and
            len(payload['records']) == 64, 'calibration token role/roster')
    for row, record in zip(rows, payload['records'], strict=True):
        require(set(row) == {'id', 'topic', 'prefix'} and row['topic'] in f.TOPICS, 'row fields/topic')
        require(type(row['prefix']) is str and len(row['prefix'].split()) == 16 and
                row['prefix'] == ' '.join(row['prefix'].split()), 'exact sixteen-word prefix')
        require(record == f.token_record(row, record), 'frozen IDs/mask/offset/last-position identity')
    return rows, pairs, payload['records'], receipt


def original_references(f):
    raw = ORIGINAL.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA, 'original reference bytes changed')
    refs = f.read_json(ORIGINAL)
    require(refs['schema'] == 'jlens_fresh_references_v1' and [r['axis'] for r in refs['axes']] == AXES,
            'original axis roster')
    for axis in refs['axes']:
        for pole in ('positive', 'negative'):
            require(type(axis['C'][pole]) is str and type(axis['A'][pole]) is list and
                    len(axis['A'][pole]) == 12 and all(type(t) is str for t in axis['A'][pole]), 'original C/A schema')
    return raw, refs


def load_directions(f):
    import numpy as np
    require(sha(DIRECTIONS) == DIRECTIONS_SHA, 'direction archive pin before load')
    with np.load(DIRECTIONS, allow_pickle=False) as archive:
        u, mean = archive['u32'], archive['source_mean64']
    f.validate_directions(u, mean)
    require(sha(DIRECTIONS) == DIRECTIONS_SHA, 'direction archive pin after load')
    u.setflags(write=False); mean.setflags(write=False)
    return u, mean


def lens_state(lens):
    import torch
    require(lens.d_model == WIDTH and LAYER in lens.source_layers and type(lens.n_prompts) is int and lens.n_prompts > 0,
            'lens dimensions/source layers')
    states = []
    for layer, matrix in lens.jacobians.items():
        require(matrix.shape == (WIDTH, WIDTH) and matrix.dtype == torch.float32 and
                torch.isfinite(matrix).all() and not matrix.requires_grad, 'invalid FP32 frozen lens')
        states.append((layer, id(matrix), matrix._version))
    return states


def load_decoder(p, f, model):
    import torch
    import jlens
    require(Path(jlens.__file__).resolve().is_relative_to(f.UPSTREAM/'jlens'), 'wrong lens package')
    require(model._lm_head.weight.dtype == torch.bfloat16 and model._logit_softcap is None, 'BF16/no-softcap head required')
    path = f.HUB/'models--neuronpedia--jacobian-lens/snapshots'/LENS_REV/LENS_FILE
    require(sha(path) == LENS_SHA, 'lens checksum')
    lens = jlens.JacobianLens.load(str(path))  # Pinned method: weights_only=True, map_location='cpu'.
    lens_state(lens)
    tokenizer = p.load_tokenizer(f)  # Decode IDs only; never call it on text in this stage.
    return lens, tokenizer


def decode(model, tokenizer, lens, signed, device):
    import numpy as np
    import torch
    require(signed.shape == (8, WIDTH) and signed.dtype == np.float32 and np.isfinite(signed).all(), 'eight signed inputs')
    require(np.array_equal(signed[1::2], -signed[0::2]) and
            np.allclose(np.linalg.norm(signed.astype(np.float64), axis=1), 1, rtol=0, atol=1e-6), 'signed unit display convention')
    actuals, full, ids_all, values_all, readouts = [], [], [], [], []
    with torch.no_grad():
        for i, (name, direction) in enumerate(zip(NAMES, signed, strict=True)):
            tensor = torch.tensor(direction, dtype=torch.float32, device=device)
            actual = tensor.cpu().numpy().copy()
            require(np.array_equal(actual, direction), 'actual decoder input changed')
            transported = lens.transport(tensor, LAYER)
            require(transported.dtype == torch.float32 and transported.shape == (WIDTH,) and
                    torch.isfinite(transported).all(), 'nonfinite or non-FP32 transport')
            logits = model.unembed(transported).float()
            require(logits.shape == (VOCAB,) and torch.isfinite(logits).all(), 'full finite vocabulary logits')
            top = logits.topk(12)  # Exactly once per pole; preserve library order including ties.
            ids, scores = top.indices.cpu().numpy().copy(), top.values.cpu().numpy().copy()
            cutoff = top.values[-1]
            tokens = [tokenizer.decode([int(j)]) for j in ids]
            require(all(type(t) is str for t in tokens), 'decoded token string type')
            readouts.append({'name': name, 'axis': AXES[i//2], 'pole': 'positive' if i % 2 == 0 else 'negative',
                'token_ids': ids.tolist(), 'tokens': tokens, 'scores': scores.tolist(),
                'cutoff_logit': float(cutoff.item()), 'strictly_above_cutoff_count': int((logits > cutoff).sum().item()),
                'cutoff_tie_count': int((logits == cutoff).sum().item()),
                'selected_at_cutoff_count': int((top.values == cutoff).sum().item())})
            actuals.append(actual); full.append(logits.cpu().numpy().copy()); ids_all.append(ids); values_all.append(scores)
    return {'inputs32': np.stack(actuals), 'logits32': np.stack(full),
            'top_ids64': np.stack(ids_all), 'top_logits32': np.stack(values_all)}, readouts


def reference_export(original, readouts):
    require([r['name'] for r in readouts] == NAMES and all(len(r['tokens']) == 12 for r in readouts), 'eight complete readouts')
    arms = {}
    for arm in ('U', 'P'):
        axes = []
        for i, old in enumerate(original['axes']):
            row = {'axis': old['axis']}
            for j, pole in enumerate(('positive', 'negative')):
                row[pole+'_reference'] = {'example_prefix': old['C'][pole],
                    'direction_tokens': list(old['A'][pole] if arm == 'U' else readouts[2*i+j]['tokens'])}
            axes.append(row)
        arms[arm] = {'axes': axes}
    return {'schema': 'jlens_pattern_calibration_references_v1', 'arms': arms}


def save_npz(path, **arrays):
    import numpy as np
    with path.open('xb') as handle: np.savez_compressed(handle, **arrays)


def calculate(target, f, patterns, hf, model, records, pairs, u, mean, refs, raw_refs, decoder_factory, device):
    import numpy as np
    baseline = f.parameter_state(hf)
    h = f.capture(model, records, device)
    require(h.shape == (64, 1, WIDTH) and h.dtype == np.float32 and np.isfinite(h).all(), '64 calibration post-block captures')
    save_npz(target/'features.npz', activation_11=h, u32=u, source_mean64=mean)
    require(f.parameter_state(hf) == baseline, 'model parameters changed in capture')
    index = {r['id']: i for i, r in enumerate(records)}
    require(list(index) == IDS and len(index) == 64, 'calibration-only capture roster before squeeze')
    joined = [(index[p['left']], index[p['right']]) for p in pairs]
    fitted = patterns.fit_pair_patterns(h[:, 0, :], u, joined)
    save_npz(target/'patterns.npz', **fitted)
    lens, tokenizer = decoder_factory()
    lens_baseline = lens_state(lens)
    arrays, readouts = decode(model, tokenizer, lens, fitted['signed_inputs32'], device)
    save_npz(target/'decoder.npz', **arrays)
    write(target/'readouts.json', {'schema': 'jlens_pattern_calibration_readouts_v1', 'layer': LAYER,
        'method': 'jlens', 'readouts': readouts, 'tie_policy': 'One torch.topk(12); no sorting, filtering or retry.'})
    with (target/'original-interpretations.json').open('xb') as handle: handle.write(raw_refs)
    require(sha(target/'original-interpretations.json') == ORIGINAL_SHA, 'original reference copy changed')
    write(target/'references.json', reference_export(refs, readouts))
    require(f.parameter_state(hf) == baseline and lens_state(lens) == lens_baseline, 'model/lens mutation during decoding')
    return {'forward_count': 64, 'calibration_pairs': 32, 'reference_decodes': 8, 'topk_calls': 8,
            'lens_n_prompts': lens.n_prompts, 'lens_source_layers': lens.source_layers,
            'parameters_unchanged': True, 'lens_parameters_unchanged': True}


def run(source_sha):
    require(os.environ.get('JLENS_PATTERN_CALIBRATION_GPU_RELEASE') == '1', 'root calibration GPU admission required')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')) and
            os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') ==
            os.environ.get('HF_HUB_DISABLE_IMPLICIT_TOKEN') == '1', 'offline one-thread envelope')
    require(type(source_sha) is str and re.fullmatch('[0-9a-f]{64}', source_sha) and sha(Path(__file__)) == source_sha,
            'exact reviewed calibration source required')
    target = OUT/'calibration'
    if target.exists() or target.is_symlink(): raise FileExistsError('calibration stage consumed')
    target.mkdir(); started = time.monotonic(); phase = 'inputs'
    write(target/'attempt.json', {'schema': 'jlens_pattern_calibration_calibration_attempt_v1',
        'source_sha256': source_sha, 'started_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False})
    try:
        p = import_pinned(OUT/'preflight.py', PREFLIGHT_SOURCE_SHA)
        f, a = import_pinned(p.BASE, p.BASE_SHA), import_pinned(p.HOST_HELPER, p.HOST_HELPER_SHA)
        expected = input_pins(p, f); verify(expected)
        rows, pairs, records, preflight = load_calibration(p, f)
        raw_refs, refs = original_references(f)
        env = f.runtime(); host = a.host_identity(); p.validate_runtime(env, host, a)
        require(env['python'] == preflight['runtime']['python'] and env['packages'] == preflight['runtime']['packages'] and
                host == preflight['host'], 'runtime differs from completed preflight')
        patterns = import_pinned(MAIN/'patterns.py', PATTERNS_SHA)
        u, mean = load_directions(f)
        write(target/'inputs.json', {'schema': 'jlens_pattern_calibration_calibration_inputs_v1',
            'role': 'calibration', 'records': records, 'pairs': pairs, 'capture_locations': ['prefix_end'],
            'mask_application': 'No padding; all-one masks implicit in unchanged adapter forward.'})
        verify(expected); require(a.host_identity() == host, 'runtime drift before model')
        phase = 'model_capture_fit_decode'
        hf, model = f.load_model()
        scope = calculate(target, f, patterns, hf, model, records, pairs, u, mean, refs, raw_refs,
                          lambda: load_decoder(p, f, model), 'cuda')
        verify(expected); require(load_calibration(p, f) == (rows, pairs, records, preflight), 'calibration metadata drift')
        require(a.host_identity() == host and {n: sys.modules[n].__version__ for n in a.NUMERICAL} == a.NUMERICAL,
                'runtime drift after decoding')
        require(sha(Path(__file__)) == source_sha, 'source changed during stage')
        import torch
        names = ['inputs.json', 'features.npz', 'patterns.npz', 'decoder.npz', 'readouts.json',
                 'original-interpretations.json', 'references.json']
        write(target/'receipt.json', {'schema': 'jlens_pattern_calibration_calibration_receipt_v1', 'status': 'complete',
            'source_sha256': source_sha, 'input_pins': {str(k): v for k, v in expected.items()},
            'outputs': [{'path': n, 'sha256': sha(target/n), 'size_bytes': (target/n).stat().st_size} for n in names],
            'runtime': env, 'host': host, 'model': f.MODEL, 'model_revision': f.REV, 'model_weight_sha256': f.WEIGHT_SHA,
            'adapter_revision': f.UPSTREAM_REV, 'lens_revision': LENS_REV, 'lens_sha256': LENS_SHA,
            'preflight_receipt_sha256': PREFLIGHT_RECEIPT_SHA, 'capture_locations': ['prefix_end'],
            'evaluation_rows_loaded': 0, 'evaluation_forwards': 0, 'prefix_tokenizations': 0, 'pca_refit': False,
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
            'gpu': torch.cuda.get_device_name(0), 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'memory_scope': '8 GiB PyTorch allocator bound, not total-process GPU cap', **scope})
    except Exception as error:
        write(target/'failure.json', {'status': 'FAILED', 'phase': phase, 'error_type': type(error).__name__,
            'reason': str(error)[:500], 'automatic_retry': False, 'evaluation_rows_loaded': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    run(parser.parse_args().source_sha)
