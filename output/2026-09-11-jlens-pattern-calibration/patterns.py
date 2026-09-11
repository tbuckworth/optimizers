"""Pure, model-free pair-pattern primitives; no file/network/stage entrypoint.

Callers must enforce the frozen calibration/evaluation split and source pins.
Only fabricated arrays are used until a separately reviewed calibration stage.
"""
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(features32, basis32, pairs):
    require(isinstance(features32, np.ndarray) and features32.dtype == np.float32,
            'features must be a float32 array')
    require(isinstance(basis32, np.ndarray) and basis32.dtype == np.float32,
            'basis must be a float32 array')
    require(features32.ndim == basis32.ndim == 2, 'matrix shapes required')
    n, width = features32.shape
    require(n > 0 and width > 0 and basis32.shape[0] == width and basis32.shape[1] > 0,
            'empty or mismatched shapes')
    require(np.isfinite(features32).all() and np.isfinite(basis32).all(), 'nonfinite input')
    norms = np.linalg.norm(basis32.astype(np.float64), axis=0)
    require(np.allclose(norms, 1., rtol=0., atol=1e-6), 'basis vectors not unit length')
    require(isinstance(pairs, (list, tuple)) and len(pairs)*2 == n, 'whole calibration-pair roster required')
    flat = []
    for pair in pairs:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2, 'pair must have two indices')
        require(all(type(i) is int and 0 <= i < n for i in pair), 'invalid pair index')
        flat.extend(pair)
    require(sorted(flat) == list(range(n)), 'each calibration row must occur exactly once')


def fit_pair_patterns(features32, basis32, pairs):
    """Return zero-intercept patterns, exact display inputs and diagnostics.

    No axis/pair dropping, ridge, clipping, gap threshold, score-sign adjustment
    or adaptive weighting. Near-zero positive score energy is retained.
    The m×k accumulation excludes supplied input/output archival storage.
    """
    validate(features32, basis32, pairs)
    basis = basis32.astype(np.float64)
    width, k = basis.shape
    cross = np.zeros((width, k), dtype=np.float64)
    energy = np.zeros(k, dtype=np.float64)
    trace = 0.
    gaps = []
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for left, right in pairs:
            delta = features32[left].astype(np.float64)-features32[right].astype(np.float64)
            dz = delta @ basis
            cross += delta[:, None]*dz[None, :]
            energy += dz*dz
            trace += float(delta @ delta)
            gaps.append(dz)
        require(np.isfinite(cross).all() and np.isfinite(energy).all() and np.isfinite(trace),
                'nonfinite accumulators')
        require(np.all(energy > 0), 'at least one fixed axis has zero score energy; stop entire fit')
        patterns = cross/energy[None, :]
        norms = np.linalg.norm(patterns, axis=0)
        require(np.isfinite(patterns).all() and np.isfinite(norms).all() and np.all(norms > 0),
                'invalid pattern norm')
        positive32 = (patterns/norms[None, :]).astype(np.float32)
        require(np.isfinite(positive32).all(), 'invalid float32 display input')
        signed32 = np.empty((2*k, width), dtype=np.float32)
        signed32[0::2] = positive32.T
        signed32[1::2] = -positive32.T
        require(np.allclose(np.linalg.norm(signed32.astype(np.float64), axis=1), 1., rtol=0., atol=1e-6),
                'display norm changed during cast')
        gap_array = np.stack(gaps)
        fractions = gap_array**2/energy[None, :]
        require(np.isfinite(fractions).all(), 'invalid diagnostic fractions')
    return {'patterns64': patterns, 'cross64': cross, 'score_energy64': energy,
            'signed_inputs32': signed32, 'calibration_gaps64': gap_array,
            'max_single_pair_energy_share64': fractions.max(axis=0),
            'direction_energy_fraction64': energy/trace,
            'unit_score_identity64': np.sum(basis*patterns, axis=0)}
