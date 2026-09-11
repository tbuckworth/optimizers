"""Inert NumPy helpers for the clean, all-class augmentation experiment.

No files, model libraries, global RNG state, or training are accessed. A caller
must persist the actual plan arrays and NumPy version: reproducibility assumes
the same RNG implementation, environment, and sequence of calls, not just seed.
"""

import numpy as np


SEEDS = (202609141, 202609142, 202609143)
STEPS = 4000
BATCH = 64
TRAIN_PER_CLASS = 500
EVAL_PER_CLASS = 500


def _rng(seed, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))


def make_plan(labels, seed):
    """Return independent, writable arrays for one seed's paired conditions.

    ``labels`` must be a one-dimensional integer NumPy array of digits 0..9,
    with at least 1,000 source examples per digit. Source IDs are shuffled in
    class order 0..9; each class contributes the first 500 shuffled IDs to
    training and the next 500 to evaluation. Both ID arrays are class-major.

    PCG64 SeedSequence([stream_id, seed]) streams are 0=split, 1=occurrences,
    2=shifts. Occurrences are sampled with replacement over every local training
    ID, including all ten classes from the first update. There is no curriculum
    or per-batch class balancing. All conditions reuse this same plan.

    Returns exactly ``train_ids`` int64[5000], ``eval_ids`` int64[5000],
    ``occurrences`` int64[4000,64], and ``shifts`` int8[4000,64,2]. Shifts are
    independent discrete uniforms on -2..2, with final coordinates (dx, dy).
    No input array is modified and no global RNG state is consumed.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer, not boolean")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not isinstance(labels, np.ndarray) or labels.dtype.kind not in "iu":
        raise TypeError("labels must be an integer NumPy array, not boolean")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [source_examples]")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("labels must lie in 0..9")
    required = TRAIN_PER_CLASS + EVAL_PER_CLASS
    counts = np.bincount(labels.astype(np.int64), minlength=10)
    if np.any(counts < required):
        raise ValueError(f"each of the ten classes needs at least {required} examples")

    split_rng = _rng(seed, 0)
    train_parts, eval_parts = [], []
    for digit in range(10):
        ids = np.flatnonzero(labels == digit).astype(np.int64)
        split_rng.shuffle(ids)
        train_parts.append(ids[:TRAIN_PER_CLASS])
        eval_parts.append(ids[TRAIN_PER_CLASS:required])
    return {
        "train_ids": np.concatenate(train_parts),
        "eval_ids": np.concatenate(eval_parts),
        "occurrences": _rng(seed, 1).integers(
            0, 10 * TRAIN_PER_CLASS, size=(STEPS, BATCH), dtype=np.int64),
        "shifts": _rng(seed, 2).integers(
            -2, 3, size=(STEPS, BATCH, 2), dtype=np.int8),
    }


def translate(images, shifts):
    """Translate float32 [N,784] images, copying with zero-filled boundaries.

    ``images`` must be finite and in [0,1]. ``shifts`` must be an integer NumPy
    array [N,2] in -2..2, ordered (dx, dy): positive dx moves right; positive dy
    moves down. Pixels leaving the 28x28 image are discarded. There is no wrap,
    interpolation, normalization, or label manipulation. Inputs, including
    strided or read-only arrays, are preserved. The output is a new C-contiguous
    float32 array, including for zero shifts and an empty batch.
    """
    if not isinstance(images, np.ndarray) or images.dtype != np.dtype(np.float32):
        raise TypeError("images must be a float32 NumPy array")
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError("images must have shape [N,784]")
    if not np.all(np.isfinite(images)) or np.any((images < 0) | (images > 1)):
        raise ValueError("images must be finite and lie in [0,1]")
    if not isinstance(shifts, np.ndarray) or shifts.dtype.kind not in "iu":
        raise TypeError("shifts must be an integer NumPy array, not boolean")
    if shifts.shape != (images.shape[0], 2):
        raise ValueError("shifts must have shape [N,2] matching images")
    if np.any((shifts < -2) | (shifts > 2)):
        raise ValueError("shifts must lie in -2..2")

    result = np.zeros(images.shape, dtype=np.float32, order="C")
    source = images.reshape(images.shape[0], 28, 28)
    target = result.reshape(images.shape[0], 28, 28)
    # Group occurrences by displacement; source/destination slices have exactly
    # matching extents, and advanced indexing writes only into the new array.
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            selected = np.flatnonzero((shifts[:, 0] == dx) & (shifts[:, 1] == dy))
            if selected.size == 0:
                continue
            x0, x1 = max(0, -dx), min(28, 28 - dx)
            y0, y1 = max(0, -dy), min(28, 28 - dy)
            target[selected, y0 + dy:y1 + dy, x0 + dx:x1 + dx] = (
                source[selected, y0:y1, x0:x1])
    return result
