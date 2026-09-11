"""Inert aggregate readout only: no text forward, tokenization, fit or update.

One actual topk(12) per defined display, preserving library ties/order.
Undefined directions retain masked zero placeholders and no decoder/topk call.
Root owns release and an exclusive new output directory; failures never retry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
MAIN = Path('output/2026-09-11-jlens-aggregate-projection')
ACQUIRE_SHA = 'd73ebdc22165196bbeb0b8d2960f107833d17f3645626abb4b47e4ec1d884120'
ADDENDUM_SHA = 'a6bd2ccc029be981b99604616a0055695c94434158bd478f07cbeaf5b329bb08'
LENS_REV = '0731326edff4ae730ffc5356fe1a4728c748b3a6'
LENS = Path('/private-artifacts/storage/cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots')/LENS_REV/'qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt'
LENS_SHA = 'aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48'
LENS_SOURCE_SHA = 'e231e7d3a6c8e8f7791b53705a34342d0bba376a127a82376eaf6ec30ca11808'
TOKEN_PINS = {
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',  # gitleaks:allow -- verified public tokenizer SHA256, not a credential
    'tokenizer_config.json': '49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c',  # gitleaks:allow -- verified public tokenizer configuration SHA256, not a credential
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
}
ARMS = ('raw', 'centered', 'second_moment')
REFERENCES = ('reference0', 'reference1', 'reference_pooled')
SUFFIXES = [f'query{b:02}/{arm}' for b in range(16) for arm in ARMS]+[
    f'query_pooled/{arm}' for arm in ARMS]+list(REFERENCES)+['fit_mean']
NAMES = [family+'/'+suffix for family in ('gradient', 'activation') for suffix in SUFFIXES]


def require(ok, message):
    if not ok: raise ValueError(message)


def helper():
    path = OUT/'acquire.py'; raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ACQUIRE_SHA, 'immutable acquisition helper changed')
    spec = importlib.util.spec_from_file_location('aggregate_decode_helpers', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(module.sha(path) == ACQUIRE_SHA, 'helper source drift')
    return module


def validate_inputs(inputs, raw, records, width=1024):
    import numpy as np
    require(inputs.shape == raw.shape == (110, width) and inputs.dtype == np.float32 and raw.dtype == np.float64,
            'exact110 raw64/input32 shape and precision')
    require(np.isfinite(inputs).all() and np.isfinite(raw).all(), 'nonfinite display input')
    require(type(records) is list and len(records) == 110 and [r.get('name') for r in records] == NAMES, 'exact display roster/order')
    for record, direction, actual in zip(records, raw, inputs, strict=True):
        family = record['name'].split('/')[0]; sign = -1 if family == 'gradient' else 1
        norm = float(np.linalg.norm(direction))
        require(set(record) == {'name', 'family', 'sign', 'norm', 'defined'} and record['family'] == family
                and type(record['sign']) is int and record['sign'] == sign and record['norm'] == norm
                and type(record['defined']) is bool and record['defined'] == (norm > 0), 'display metadata/sign/norm')
        expected = (sign*direction/norm if norm else np.zeros_like(direction)).astype(np.float32)
        require(np.array_equal(expected, actual), 'actual input differs from saved signed normalization')


def decode_all(model, tokenizer, jacobian, inputs, records, device, width=1024, vocab=248320):
    import numpy as np
    import torch
    require(jacobian.shape == (width, width) and jacobian.dtype == torch.float32 and
            not jacobian.requires_grad and torch.isfinite(jacobian).all().item(), 'finite FP32 frozen J')
    actuals, transported, full, all_ids, all_values, readouts = [], [], [], [], [], []
    with torch.no_grad():
        for direction, record in zip(inputs, records, strict=True):
            actual = torch.tensor(direction, dtype=torch.float32, device=device)
            require(np.array_equal(actual.cpu().numpy(), direction), 'actual input changed')
            actuals.append(actual.cpu().numpy().copy())
            if not record['defined']:
                transported.append(np.zeros(width, dtype=np.float32)); full.append(np.zeros(vocab, dtype=np.float32))
                all_ids.append(np.full(12, -1, dtype=np.int64)); all_values.append(np.zeros(12, dtype=np.float32))
                readouts.append({**record, 'token_ids': [], 'tokens': [], 'scores': [], 'ties': None})
                continue
            # Exact pinned lens.transport algebra; J is cast once from checkpoint to FP32.
            z = actual@jacobian.T
            logits = model.unembed(z).float()  # Pinned adapter casts BF16 BEFORE final RMSNorm/head.
            require(z.shape == (width,) and z.dtype == torch.float32 and torch.isfinite(z).all().item()
                    and logits.shape == (vocab,) and torch.isfinite(logits).all().item(), 'decoder shape/finite')
            top = logits.topk(12)  # Exactly one call; do not sort/redecode tied indices.
            ids, values = top.indices.cpu().numpy().copy(), top.values.cpu().numpy().copy()
            tokens = [tokenizer.decode([int(i)]) for i in ids]
            require(all(type(t) is str for t in tokens), 'token strings')
            cutoff = top.values[-1]
            readouts.append({**record, 'token_ids': ids.tolist(), 'tokens': tokens, 'scores': values.tolist(),
                'ties': {'cutoff_logit': float(cutoff.item()),
                    'strictly_above_cutoff_count': int((logits > cutoff).sum().item()),
                    'cutoff_tie_count': int((logits == cutoff).sum().item()),
                    'selected_at_cutoff_count': int((top.values == cutoff).sum().item())}})
            transported.append(z.cpu().numpy().copy()); full.append(logits.cpu().numpy().copy())
            all_ids.append(ids); all_values.append(values)
    return {'inputs32': np.stack(actuals), 'transported32': np.stack(transported), 'logits32': np.stack(full),
        'top_ids64': np.stack(all_ids), 'top_logits32': np.stack(all_values),
        'defined': np.array([r['defined'] for r in records], dtype=np.bool_)}, readouts


def fidelity(arrays, readouts):
    import numpy as np
    index = {row['name']: i for i, row in enumerate(readouts)}
    require(len(index) == 110 and list(index) == NAMES, 'fidelity roster')
    for i, row in enumerate(readouts):
        row['reference_fidelity'] = {}
        for name in REFERENCES:
            j = index[row['family']+'/'+name]
            value = {'full_logit_cosine': None, 'top12_overlap_count': None, 'top12_overlap_fraction': None}
            if row['defined'] and readouts[j]['defined']:
                left, right = arrays['logits32'][i].astype(np.float64), arrays['logits32'][j].astype(np.float64)
                denom = float(np.linalg.norm(left)*np.linalg.norm(right))
                overlap = len(set(row['token_ids']) & set(readouts[j]['token_ids']))
                value = {'full_logit_cosine': float(left@right/denom) if denom else None,
                         'top12_overlap_count': overlap, 'top12_overlap_fraction': overlap/12}
            row['reference_fidelity'][name] = value


def run(analysis_dir, arrays_sha, displays_sha, receipt_sha, source_sha, target):
    require(os.environ.get('JLENS_AGGREGATE_DECODE_RELEASE') == '1', 'explicit decoder release required')
    a = helper(); require(a.sha(Path(__file__)) == source_sha, 'reviewed decoder source required')
    a.claim(target, source_sha); started = time.monotonic()
    try:
        expected = {**a.PINS, OUT/'acquire.py': ACQUIRE_SHA, Path(__file__): source_sha,
            a.PROTOCOL: a.PROTOCOL_SHA, MAIN/'readout-addendum.md': ADDENDUM_SHA,
            a.ADAPTER.with_name('lens.py'): LENS_SOURCE_SHA, LENS: LENS_SHA,
            **{a.SNAPSHOT/name: digest for name, digest in TOKEN_PINS.items()},
            analysis_dir/'arrays.npz': arrays_sha, analysis_dir/'displays.json': displays_sha,
            analysis_dir/'receipt.json': receipt_sha}
        a.verify(expected)
        receipt = a.read_json(analysis_dir/'receipt.json')
        require(receipt['status'] == 'complete' and receipt['protocol_sha256'] == a.PROTOCOL_SHA
                and receipt['artifacts']['arrays.npz'] == arrays_sha and
                receipt['artifacts']['displays.json'] == displays_sha, 'analysis receipt bindings')
        expected[MAIN/'analyze.py'] = receipt['source_sha256']; a.verify(expected)
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN',
                     'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            require(os.environ.get(name) == '1', 'release environment: '+name)
        env = a.runtime()
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        with np.load(analysis_dir/'arrays.npz', allow_pickle=False) as archive:
            inputs, raw = archive['display_inputs32'], archive['display_raw64']
        records = a.read_json(analysis_dir/'displays.json'); validate_inputs(inputs, raw, records)
        a.verify(expected)
        hf, model = a.load_model(); state = a.parameter_state(hf)
        checkpoint = torch.load(LENS, map_location='cpu', weights_only=True)
        require(checkpoint['d_model'] == 1024 and checkpoint['n_prompts'] == 233 and
                11 in checkpoint['source_layers'], 'lens metadata')
        jacobian = checkpoint['J'][11].float().to('cuda'); lens_state = (id(jacobian), jacobian._version)
        tokenizer = AutoTokenizer.from_pretrained(str(a.SNAPSHOT), local_files_only=True, token=False,
            trust_remote_code=False, use_fast=True)
        require(tokenizer.is_fast and tokenizer.bos_token_id is None, 'tokenizer configuration')
        arrays, readouts = decode_all(model, tokenizer, jacobian, inputs, records, 'cuda')
        fidelity(arrays, readouts)
        with (target/'decoder.npz').open('xb') as f: np.savez_compressed(f, **arrays)
        a.write_json(target/'readouts.json', {'schema': 'jlens_aggregate_readouts_v1', 'readouts': readouts,
            'tie_policy': 'One torch.topk(12) per defined direction; no sorting/filtering/retry.',
            'fidelity_is_semantics': False, 'undefined_array_rows': 'Masked zero placeholders; top_ids=-1; not decoded.'})
        a.verify(expected)
        require(a.runtime() == env and a.parameter_state(hf) == state and
                (id(jacobian), jacobian._version) == lens_state, 'runtime/model/lens changed')
        a.write_json(target/'receipt.json', {'schema': 'jlens_aggregate_decode_v1', 'status': 'complete',
            'source_sha256': source_sha, 'input_pins': {str(p): d for p, d in expected.items()},
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
            'runtime': env, 'display_count': 110, 'decode_count': sum(r['defined'] for r in readouts),
            'topk_calls': sum(r['defined'] for r in readouts), 'model_forward_count': 0, 'tokenizer_encode_calls': 0,
            'layer': 11, 'lens_n_prompts': 233, 'parameters_unchanged': True, 'lens_unchanged': True,
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'memory_scope': '8GiB Torch allocator envelope, not whole-process GPU cap',
            'artifacts': {name: a.sha(target/name) for name in ('decoder.npz', 'readouts.json')}})
    except BaseException as error:
        a.write_json(target/'failure.json', {'status': 'FAILED', 'error_type': type(error).__name__,
            'automatic_retry': False, 'partial_outputs_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis-dir', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('arrays-sha', 'displays-sha', 'receipt-sha', 'source-sha'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    run(args.analysis_dir, args.arrays_sha, args.displays_sha, args.receipt_sha, args.source_sha, args.output)
