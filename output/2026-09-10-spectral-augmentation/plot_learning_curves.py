#!/usr/bin/env python3
"""Presentation-only rendering of audited scalar curves; no tensor/logit reads.

Reads each of the 72 curve JSON files exactly once, checks its recorded byte
receipt, and plots all 21 scheduled values. No inferential statistics or audit
rerun. Outputs are exclusive: an existing PDF/preview directory is not replaced.
"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ACQUISITION = Path('/tmp/spectral-experiment-artifacts/spectral-augmentation-20260910.ZZWKWc/acquisition-001')
SEEDS = (202609131, 202609132, 202609133)
CELLS = ('clean', 'shared', 'sham')
MODES = ('none', 'random', 'targeted', 'opposite')
POLICIES = ('raw', 'native32')
COLORS = {'raw': '#778394', 'native32': '#176B9B'}
PAGES = (
    ('majority_macro', 'accuracy', 'Common-digit accuracy', 100),
    ('rare', 'accuracy', 'Rare-digit accuracy', 100),
    ('majority_macro', 'ce', 'Common-digit cross-entropy', 1),
    ('rare', 'ce', 'Rare-digit cross-entropy', 1),
)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def receipt(path):
    payload = path.read_bytes()
    return {'path': str(path.relative_to(HERE)), 'size_bytes': len(payload), 'sha256': sha(payload)}


def main():
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError(f'Set {key}=1 before Python')
    output, preview = HERE / 'learning-curves.pdf', HERE / 'curves-preview'
    if output.exists() or preview.exists():
        raise FileExistsError('Presentation outputs already exist; no overwrite')
    complete_bytes = (HERE / 'acquisition-complete.json').read_bytes()
    complete = json.loads(complete_bytes)
    summary_bytes = (HERE / 'audit/summary.json').read_bytes()
    summary = json.loads(summary_bytes)
    audit_bytes = (HERE / 'audit/audit.json').read_bytes()
    audit = json.loads(audit_bytes)
    if complete['status'] != 'complete' or audit['status'] != 'PASS':
        raise ValueError('Completed acquisition and existing PASS audit required')
    if audit['source_completion_sha256'] != sha(complete_bytes):
        raise ValueError('Audit completion binding differs')
    if summary['schema'] != 'spectral_augmentation_analysis_v1' or summary['seeds'] != list(SEEDS):
        raise ValueError('Unexpected audited summary')
    receipts = {r['path']: r for r in complete['receipts']}
    curves, inputs = {}, []
    steps = np.arange(0, 2001, 100)
    for seed in SEEDS:
        for cell in CELLS:
            for mode in MODES:
                for policy in POLICIES:
                    name = f'curve-s{seed}-{cell}-{mode}-{policy}.json'
                    payload = (ACQUISITION / name).read_bytes()  # sole read of this input
                    expected = receipts[name]
                    if len(payload) != expected['size_bytes'] or sha(payload) != expected['sha256']:
                        raise ValueError('Curve byte receipt mismatch: ' + name)
                    record = json.loads(payload)
                    if (record['seed'], record['cell'], record['augmentation'], record['policy']) != (seed, cell, mode, policy):
                        raise ValueError('Curve identity differs: ' + name)
                    if [r['step'] for r in record['curve']] != steps.tolist():
                        raise ValueError('Incomplete scheduled curve: ' + name)
                    curves[seed, cell, mode, policy] = record['curve']
                    inputs.append(expected)
    preview.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'pdf.fonttype': 42})
    outputs = []
    with output.open('xb') as handle, PdfPages(handle, metadata={
            'Title': 'Spectral augmentation: complete learning curves',
            'Author': 'Codex — Spectral Optimizer Investigation',
            'Subject': 'Audited scalar presentation, three paired seeds, all scheduled evaluations',
            'CreationDate': None, 'ModDate': None}) as pdf:
        for page, (group, measure, title, scale) in enumerate(PAGES, 1):
            fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharex=True, sharey=True)
            fig.subplots_adjust(left=.077, right=.975, bottom=.11, top=.80, hspace=.26, wspace=.15)
            fig.suptitle(title, x=.077, y=.958, ha='left', fontsize=22, fontweight='bold')
            population = ('Digits 0–7 and 9 · n = 4,500 per seed · class-balanced common mean'
                          if group == 'majority_macro' else 'Digit 8 · n = 500 per seed · only 50 rare training examples')
            fig.text(.077, .918, 'Held-out original images, never masked · ' + population, fontsize=10.5, color='#435164')
            legend = [Line2D([0], [0], color=COLORS['raw'], lw=2.3, label='Raw AdamW — mean'),
                      Line2D([0], [0], color=COLORS['native32'], lw=2.3, label='Spectral + AdamW — mean'),
                      Line2D([0], [0], color='#435164', alpha=.35, lw=1, label='Faint lines: all three seeds'),
                      Patch(facecolor='#E7DCC8', alpha=.6, label='Unmasked raw warmup (0–100)')]
            fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.067, .902),
                       ncol=2, frameon=False, fontsize=10, columnspacing=2)
            maximum = 0.
            for row, cell in enumerate(CELLS):
                for col, mode in enumerate(MODES):
                    ax = axes[row, col]
                    ax.axvspan(0, 100, color='#E7DCC8', alpha=.6, zorder=0)
                    ax.axvline(100, color='#A78E63', lw=.8, ls=':', zorder=1)
                    for policy in POLICIES:
                        values = np.array([[r['heldout_unpatched'][group][measure]
                            for r in curves[seed, cell, mode, policy]] for seed in SEEDS], dtype=float) * scale
                        if values.shape != (3, 21) or not np.isfinite(values).all():
                            raise ValueError('Invalid scalar plotting values')
                        maximum = max(maximum, float(values.max()))
                        for seed_values in values:
                            ax.plot(steps, seed_values, color=COLORS[policy], lw=.9, alpha=.32)
                        ax.plot(steps, values.mean(axis=0), color=COLORS[policy], lw=2.2)
                    if row == 0:
                        ax.set_title(mode.capitalize(), fontsize=12, fontweight='bold', pad=10)
                    if col == 0:
                        units = 'Accuracy (%)' if measure == 'accuracy' else 'CE (nats; lower is better)'
                        ax.set_ylabel(cell.capitalize() + '\n' + units, labelpad=10)
                    if row == 2:
                        ax.set_xlabel('Training update')
                    ax.set_xlim(0, 2000)
                    ax.set_xticks([0, 500, 1000, 1500, 2000])
                    ax.grid(axis='y', color='#DCE2E8', lw=.65)
                    ax.set_axisbelow(True)
                    ax.spines[['top', 'right']].set_visible(False)
            axes[0, 0].set_ylim(0, 100 if measure == 'accuracy' else maximum * 1.05)
            fig.text(.077, .055, 'Three paired seeds, not 72 independent replicates. All 21 scheduled evaluations; no smoothing or epoch selection.',
                     fontsize=10, color='#435164')
            fig.text(.077, .031, 'Cell rows describe training conditions. Shared/Sham labels remain wrong after masking. Targeted uses known cue location.',
                     fontsize=9.5, color='#435164')
            fig.text(.975, .031, f'{page}/4', ha='right', fontsize=10, color='#435164')
            pdf.savefig(fig)
            path = preview / f'page-{page}.png'
            with path.open('xb') as image_file:
                fig.savefig(image_file, format='png', dpi=120, facecolor='white')
            outputs.append(receipt(path))
            plt.close(fig)
    outputs.insert(0, receipt(output))
    manifest = {'scope': 'Scalar presentation only; no training, logits/tensors, or new audit',
        'acquisition': str(ACQUISITION), 'completion_sha256': sha(complete_bytes),
        'existing_audit_sha256': sha(audit_bytes), 'audited_summary_sha256': sha(summary_bytes),
        'script_sha256': sha(Path(__file__).read_bytes()),
        'curve_json_reads': 72, 'logical_evaluations_shown': 1512,
        'curve_receipts_checked': inputs, 'pdf_pages': 4, 'outputs': outputs,
        'plotting': 'Arithmetic seed means and all three individual seed lines; linear axes; all 21 steps'}
    with (preview / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({'curve_json_reads': 72, 'pdf_pages': 4, 'outputs': outputs}))


if __name__ == '__main__':
    main()
