"""Inert, once-only same-input forward after the pinned pre-model OS-update failure.

No tokenizer or old stage entrypoints. The original producer validates its old
preflight; this wrapper records its own source and the explicit runtime change.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
BASE = OUT/'forward.py'
BASE_SHA = 'd2557036a3ebdd6ecbc61be89adf3582f16f084fda091e86f3faabe9ff39f1d0'
AMENDMENT = Path('output/2026-09-11-jlens-natural-addon/runtime-amendment.md')
AMENDMENT_SHA = '5bcc78c413c319e1ccf329066d01699a9c56bfdbb7a8bb0aa4ecc5abb9a8a72c'
PREFLIGHT_SHA = '675411e8428e5a8797101b0bed85a4604a4e59d21fe41b40c498a3e330c492fb'
FAILED = {'attempt': 'c313921dcdb033cd973dbbd8884d80feef1ca80e79362d02b453c8e9df2bf6ba',
          'failure': '7f81cc5fad6f71b7d8ee901a642100cce1040dae9ad8fc1c2d440db2616fe1c6'}
OLD_PYTHON = '3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]'
NEW_PYTHON = '3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]'
NUMERICAL = {'numpy': '1.26.4', 'torch': '2.11.0+cu128', 'transformers': '5.5.0', 'huggingface_hub': '1.8.0'}
PACKAGES = {**{p: '3.12.3-1ubuntu0.17' for p in ('python3.12', 'python3.12-minimal',
             'libpython3.12-minimal', 'libpython3.12-stdlib', 'libpython3.12t64')},
            'libc6': '2.39-0ubuntu8.9', 'libc-bin': '2.39-0ubuntu8.9'}
BINARIES = {'/usr/bin/python3.12': 'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f',
            '/lib/x86_64-linux-gnu/libc.so.6': '3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf'}
STAGE = 'forwards-runtime-amendment'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def metadata_pins():
    return {BASE: BASE_SHA, AMENDMENT: AMENDMENT_SHA,
            **{OUT/'forwards'/f'{name}.json': digest for name, digest in FAILED.items()}}


def verify_metadata(expected):
    for path, digest in expected.items():
        require(not path.is_symlink() and sha(path) == digest, f'metadata pin mismatch: {path}')
    require({p.name for p in (OUT/'forwards').iterdir()} == {'attempt.json', 'failure.json'},
            'original failed stage contains unexpected output')


def load_base():
    raw = BASE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BASE_SHA, 'base producer changed')
    spec = importlib.util.spec_from_file_location('jlens_natural_addon_base_runtime', BASE)
    module = importlib.util.module_from_spec(spec)
    # Execute the pinned source bytes, not a possibly stale bytecode cache.
    exec(compile(raw, str(BASE), 'exec'), module.__dict__)
    require(sha(BASE) == BASE_SHA and module.OUT == OUT, 'base producer source/location drift')
    return module


def host_identity():
    raw = subprocess.check_output(['dpkg-query', '-W', '-f=${Package}\t${Version}\n',
                                   *[p+':amd64' for p in PACKAGES]], text=True, timeout=10)
    versions = {}
    for line in raw.splitlines():
        fields = line.split('\t')
        require(len(fields) == 2 and fields[0] not in versions, 'package metadata format/duplicate')
        versions[fields[0]] = fields[1]
    return {'python': sys.version, 'interpreter': sys.executable,
            'resolved_interpreter': str(Path(sys.executable).resolve()),
            'package_versions': versions, 'binary_sha256s': {p: sha(p) for p in BINARIES}}


def validate_transition(old, new, host):
    require(old['python'] == OLD_PYTHON and new['python'] == host['python'] == NEW_PYTHON,
            'unapproved Python build transition')
    require(old['interpreter'] == new['interpreter'] == host['interpreter'] == '/usr/bin/python3' and
            host['resolved_interpreter'] == '/usr/bin/python3.12', 'interpreter path changed')
    require(old['packages'] == new['packages'] == NUMERICAL, 'numerical package versions changed')
    require(host['package_versions'] == PACKAGES, 'unapproved distro package versions')
    require(host['binary_sha256s'] == BINARIES, 'unapproved runtime binary hashes')
    return {'preflight_python': old['python'], 'new_python': new['python'],
            'old_preflight_source_sha256': BASE_SHA, 'package_versions': dict(host['package_versions']),
            'binary_sha256s': dict(host['binary_sha256s'])}


def run(dataset_sha, pairs_sha, preflight_sha):
    require(os.environ.get('JLENS_NATURAL_ADDON_RUNTIME_AMENDMENT_RELEASE') == '1', 'new recovery admission required')
    target = OUT/STAGE
    if target.exists() or target.is_symlink(): raise FileExistsError(f'Consumed stage: {target}')
    target.mkdir()
    started = time.monotonic()
    source_hash = sha(Path(__file__))
    write_json(target/'attempt.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_hash, 'base_producer_sha256': BASE_SHA,
        'runtime_amendment_sha256': AMENDMENT_SHA, 'automatic_retry': False})
    try:
        require(preflight_sha == PREFLIGHT_SHA, 'exact original preflight required')
        meta = metadata_pins(); verify_metadata(meta)
        f = load_base()
        expected = f.pins(dataset_sha, pairs_sha); f.verify(expected)
        rows, pairs = f.load_panel()
        records, preflight = f.frozen_tokens(OUT, preflight_sha, expected, rows)
        env = f.runtime()
        transition = validate_transition(preflight['runtime'], env, host_identity())
        import numpy as np
        import torch
        with np.load(f.OLD/'preparation/directions.npz', allow_pickle=False) as archive:
            u, mean = archive['u32'], archive['source_mean64']
        f.validate_directions(u, mean); f.verify(expected)
        verify_metadata(meta)
        require(validate_transition(preflight['runtime'], env, host_identity()) == transition, 'runtime drift before model')
        hf, model = f.load_model(); baseline = f.parameter_state(hf)
        h = f.capture(model, records, 'cuda')
        require(f.parameter_state(hf) == baseline, 'model parameter state changed')
        scores, gaps, export = f.score_export(h, u, mean, rows, pairs)
        with (target/'features.npz').open('xb') as handle:
            np.savez_compressed(handle, activation_11=h, scores64=scores, gaps64=gaps, locations=np.array(f.LOCATIONS))
        write_json(target/'inputs.json', {'schema': 'jlens_natural_addon_inputs_v1', 'records': records,
            'mask_application': 'No padding; all-one saved masks are implicit in the unchanged adapter forward.'})
        write_json(target/'scores.json', export)
        write_json(target/'gaps.json', {'locations': f.LOCATIONS, 'axes': [f'PC{i}' for i in range(1, 5)],
            'orientation': 'left minus right', 'pairs': [{**p, 'gaps64': g.tolist()} for p, g in zip(pairs, gaps, strict=True)]})
        final_records, final_preflight = f.frozen_tokens(OUT, preflight_sha, expected, rows)
        require(final_records == records and final_preflight == preflight, 'frozen inputs changed')
        f.verify(expected); verify_metadata(meta)
        require(validate_transition(preflight['runtime'], env, host_identity()) == transition, 'runtime drift after model')
        require({name: getattr(sys.modules[name], '__version__') for name in NUMERICAL} == NUMERICAL,
                'loaded numerical versions changed')
        require(sha(Path(__file__)) == source_hash, 'wrapper source changed')
        outputs = [target/name for name in ('features.npz', 'inputs.json', 'scores.json', 'gaps.json')]
        write_json(target/'receipt.json', {'schema': 'jlens_natural_addon_forwards_receipt_v1', 'status': 'complete',
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hash,
            'base_producer_sha256': BASE_SHA, 'runtime_amendment_sha256': AMENDMENT_SHA,
            'failed_admission_sha256s': dict(FAILED), 'runtime_transition': transition,
            'input_pins': {str(p): s for p, s in expected.items()},
            'outputs': [{'path': p.name, 'sha256': sha(p)} for p in outputs],
            'runtime': env, 'preflight_runtime': preflight['runtime'], 'preflight_receipt_sha256': preflight_sha,
            'model_weight_sha256': f.WEIGHT_SHA, 'adapter_revision': f.UPSTREAM_REV,
            'parameters_unchanged': True, 'forward_count': 32, 'capture_locations': f.LOCATIONS,
            'reference_decodes': 0, 'pca_refit': False, 'tokenizer_loaded': False,
            'elapsed_seconds': time.monotonic()-started, 'gpu': torch.cuda.get_device_name(0),
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'memory_scope': '8 GiB PyTorch allocator bound, not a total-process GPU cap'})
    except Exception as error:
        write_json(target/'failure.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-sha', required=True)
    parser.add_argument('--pairs-sha', required=True)
    parser.add_argument('--preflight-sha', required=True)
    args = parser.parse_args()
    run(args.dataset_sha, args.pairs_sha, args.preflight_sha)
