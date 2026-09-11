"""Import-inert, once-only decomposition of the 32 frozen fit-score rows."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TOPICS = ['astronomy', 'cooking', 'football', 'programming']
FIT_IDS = [f'{g}-{i}-{s}' for g in TOPICS for i in range(4) for s in ['plain', 'note']]
ALL_IDS = [f'{g}-{i}-{s}' for g in TOPICS for i in range(6) for s in ['plain', 'note']]
PINS = {
    'output/2026-09-10-j-lens-fresh-content/preparation/directions.npz': '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa',
    'output/2026-09-10-j-lens-fresh-content/preparation/selection.json': 'aac75625b67d9a90ba351d979f3e1a77cd75f69706a8b3113e071630de023f4d',
    'output/2026-09-10-j-lens-completions/dataset.json': 'b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def verify_pins():
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, f'Input pin mismatch: {relative}')


def load_inputs():
    import numpy as np
    verify_pins()
    prep = ROOT / 'output/2026-09-10-j-lens-fresh-content/preparation'
    selection = json.loads((prep / 'selection.json').read_text())
    rows = json.loads((ROOT / 'output/2026-09-10-j-lens-completions/dataset.json').read_text())
    # Deliberately access only this archive member: no activations/basis/PCA.
    with np.load(prep / 'directions.npz', allow_pickle=False) as archive:
        scores = archive['fit_scores64']
    verify_pins()
    return rows, selection, scores


def decompose(rows, selection, scores):
    import numpy as np
    require(isinstance(rows, list) and [r['id'] for r in rows] == ALL_IDS, 'Exact 48-row metadata order required')
    require(selection['feature'] == 'activation_11' and selection['fit_ids'] == FIT_IDS, 'Exact 32-row score order required')
    fit = []
    for row in rows:
        group, content, style = row['id'].split('-')
        expected_split = 'fit' if int(content) < 4 else 'heldout'
        require(row['group'] == group and row['pair'] == f'{group}-{content}' and row['style'] == style
                and row['split'] == expected_split, 'Metadata group/pair/style/split mismatch')
        if expected_split == 'fit':
            require(type(row['prefix']) is str and bool(row['prefix']), 'Invalid fit prefix')
            fit.append({key: row[key] for key in ['id', 'pair', 'group', 'style', 'split', 'prefix']})
    require([r['id'] for r in fit] == FIT_IDS, 'Fit metadata order differs')
    require(isinstance(scores, np.ndarray) and scores.shape == (32, 4) and scores.dtype == np.float64,
            'Expected (32,4) float64 scores')
    require(np.isfinite(scores).all(), 'Nonfinite scores')
    for i in range(0, 32, 2):
        require(fit[i+1]['prefix'] == 'A brief factual note:\n' + fit[i]['prefix'], 'Paired framing text mismatch')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        mean = scores.mean(axis=0)
        topic_means = np.stack([scores[i:i+8].mean(axis=0) for i in range(0, 32, 8)])
        pair_means = np.stack([scores[i:i+2].mean(axis=0) for i in range(0, 32, 2)])
        # Explicit row weighting implements nested population moments (ddof=0).
        topic_by_row = np.repeat(topic_means, 8, axis=0)
        pair_by_row = np.repeat(pair_means, 2, axis=0)
        total = ((scores - mean)**2).mean(axis=0)
        between = ((topic_by_row - mean)**2).mean(axis=0)
        content = ((pair_by_row - topic_by_row)**2).mean(axis=0)
        framing = ((scores - pair_by_row)**2).mean(axis=0)
        residual = total - between - content - framing
        shifts = scores[1::2] - scores[0::2]
        mean_shift = shifts.mean(axis=0)
        mean_squared_shift = (shifts**2).mean(axis=0)
        rms_shift = np.sqrt(mean_squared_shift)
        framing_residual = framing - mean_squared_shift/4
    require(np.isfinite(np.stack([total, between, content, framing, residual, mean_shift, rms_shift])).all(), 'Nonfinite output')
    require(np.all(np.abs(residual) <= 128*np.finfo(np.float64).eps*np.maximum(total, between+content+framing)),
            'Nested variance reconstruction exceeds rounding tolerance')
    require(np.all(np.abs(framing_residual) <= 128*np.finfo(np.float64).eps*np.maximum(framing, mean_squared_shift/4)),
            'Paired framing identity exceeds rounding tolerance')
    axes = []
    for i in range(4):
        parts = {'V': float(total[i]), 'B': float(between[i]), 'Q': float(content[i]), 'F': float(framing[i])}
        axes.append({'axis': f'PC{i+1}', 'mean_score': float(mean[i]), **parts,
                     'fractions': {k: parts[k]/parts['V'] if parts['V'] > 0 else None for k in ['B', 'Q', 'F']},
                     'reconstruction_residual_V_minus_B_Q_F': float(residual[i]),
                     'framing_identity_residual_F_minus_mean_squared_shift_over_4': float(framing_residual[i]),
                     'mean_note_minus_plain': float(mean_shift[i]), 'rms_note_minus_plain': float(rms_shift[i])})
    return {'schema': 'jlens_fit_geometry_v1', 'normalization': 'population moments, row-weighted, ddof=0',
            'axes': axes, 'rows': [{**row, 'scores64': scores[i].tolist()} for i, row in enumerate(fit)],
            'topic_means': [{'group': g, 'count': 8, 'scores64': topic_means[i].tolist()} for i, g in enumerate(TOPICS)],
            'pairs': [{'pair': fit[2*i]['pair'], 'group': fit[2*i]['group'], 'plain_id': fit[2*i]['id'],
                       'note_id': fit[2*i+1]['id'], 'count': 2, 'mean_scores64': pair_means[i].tolist(),
                       'note_minus_plain64': shifts[i].tolist()} for i in range(16)],
            'scope': 'All 32 fit rows only; F includes content-dependent framing effects, not a universal causal style effect.'}


def run(directory=OUT):
    started = time.monotonic()
    attempt, target = directory / 'attempt.json', directory / 'analysis'
    for p in [attempt, target]:
        if p.exists() or p.is_symlink():
            raise FileExistsError(f'Consumed/existing stage: {p}')
    source_pin = sha(Path(__file__))
    write_json(attempt, {'started_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_pin,
                         'target': str(target), 'automatic_retry': False})
    rows, selection, scores = load_inputs()
    result = decompose(rows, selection, scores)
    verify_pins()
    require(sha(Path(__file__)) == source_pin, 'Source changed during run')
    target.mkdir()
    write_json(target / 'results.json', result)
    import numpy as np
    write_json(target / 'receipt.json', {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_pin, 'test_source_sha256': sha(OUT / 'test_analyze.py'), 'input_pins': PINS,
        'runtime': {'interpreter': sys.executable, 'python': sys.version, 'numpy': np.__version__,
                    'platform': platform.platform(), 'threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}},
        'elapsed_seconds': time.monotonic()-started, 'archive_members_read': ['fit_scores64'],
        'fit_rows': 32, 'heldout_projections': 0, 'model_calls': 0, 'pca_refit': False,
        'outputs': [{'path': 'results.json', 'sha256': sha(target / 'results.json')}]})
    print('Completed fixed fit-score decomposition; no model, PCA, heldout projection or grading.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['analyze'])
    parser.parse_args()
    run()
