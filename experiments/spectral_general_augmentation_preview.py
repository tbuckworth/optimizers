#!/usr/bin/env python3
"""Training-only input visual validation; no model or optimizer instantiated."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import spectral_general_augmentation_data as data

RAW = Path('data/MNIST/raw')
PINS = {'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
        'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve(strict=True)
    paths = [out / 'preview.json', *(out / f'preview-{i}.png' for i in (1, 2, 3))]
    if any(p.exists() for p in paths):
        raise RuntimeError('preview artifacts already exist')
    buffers = {k: (RAW / k).read_bytes() for k in PINS}
    if any(hashlib.sha256(buffers[k]).hexdigest() != v for k, v in PINS.items()):
        raise RuntimeError('unexpected source data')
    if struct.unpack('>IIII', buffers['train-images-idx3-ubyte'][:16]) != (2051, 60000, 28, 28):
        raise RuntimeError('IDX shape')
    images = np.frombuffer(buffers['train-images-idx3-ubyte'], dtype=np.uint8, offset=16).reshape(60000, 784)
    labels = np.frombuffer(buffers['train-labels-idx1-ubyte'], dtype=np.uint8, offset=8).astype(np.int64)
    plan = data.make_plan(labels, data.SEEDS[0])
    ids = plan['train_ids']
    cpu = images[ids].astype(np.float32) / np.float32(255)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shown = [(0, 0), (-2, -2), (0, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (0, 2), (2, 2)]
    choices = []
    for sample in range(3):
        positions = np.array([digit * 500 + sample for digit in range(10)])
        choices.append(ids[positions].tolist())
        fig, axes = plt.subplots(10, 9, figsize=(13.5, 15), facecolor='white')
        for column, shift in enumerate(shown):
            view = data.translate(cpu[positions], np.tile(np.array(shift, dtype=np.int8), (10, 1)))
            for digit in range(10):
                ax = axes[digit, column]
                ax.imshow(view[digit].reshape(28, 28), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if digit == 0:
                    ax.set_title('Original' if shift == (0, 0) else f'dx={shift[0]}, dy={shift[1]}', fontsize=10)
                if column == 0:
                    ax.set_ylabel(f'{digit} | ID {ids[positions[digit]]}', fontsize=9)
        fig.suptitle(f'Ordinary translation: fixed training example {sample + 1}/class\n'
                     'Zero fill; no model evaluation or label-based exclusions', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .96))
        with (out / f'preview-{sample + 1}.png').open('xb') as handle:
            fig.savefig(handle, format='png', dpi=120)
        plt.close(fig)
    mass = cpu.sum(axis=1, dtype=np.float64)
    if not (mass > 0).all():
        raise RuntimeError('blank training image')
    coverage = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            shifted = data.translate(cpu, np.tile(np.array([dx, dy], dtype=np.int8), (len(cpu), 1)))
            lost = mass - shifted.sum(axis=1, dtype=np.float64)
            fraction = lost / mass
            coverage.append({'dx': dx, 'dy': dy, 'images': len(cpu),
                'images_with_cropped_mass': int((lost > 1e-12).sum()),
                'mean_lost_mass_fraction': float(fraction.mean()),
                'max_lost_mass_fraction': float(fraction.max()),
                'total_lost_mass_fraction': float(lost.sum() / mass.sum())})
    result = {'schema': 'spectral_general_augmentation_preview_v1', 'seed': data.SEEDS[0],
              'scope': 'training-only input validation; no outcome evaluation',
              'shown_source_ids_by_sample': choices, 'shown_shifts_dx_dy': shown,
              'cropping_all_25_translations': coverage, 'data_pins': PINS,
              'figures': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in paths[1:]]}
    with (out / 'preview.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'preview_generated', 'images': 30,
                      'max_individual_cropped_fraction': max(r['max_lost_mass_fraction'] for r in coverage),
                      'mean_occurrence_cropped_fraction': np.mean([r['mean_lost_mass_fraction'] for r in coverage])}))


if __name__ == '__main__':
    main()
