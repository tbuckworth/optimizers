"""Inert fixed-label corruption on the frozen ordinary augmentation plans.

Labels attach to source examples, never to views or occurrences. The helper
reads no files and touches neither the global RNG nor model libraries.
"""

import numpy as np

from experiments import spectral_general_augmentation_data as ordinary


SEEDS = (202609151, 202609152, 202609153)
STEPS, BATCH = ordinary.STEPS, ordinary.BATCH
TRAIN_PER_CLASS, EVAL_PER_CLASS = ordinary.TRAIN_PER_CLASS, ordinary.EVAL_PER_CLASS
WRONG_PER_CLASS = 400
translate = ordinary.translate


def make_plan(labels, seed):
    """Return the four ordinary arrays plus four true/assigned label arrays.

    Streams 0..2 and their draw shapes are unchanged from ``ordinary.make_plan``.
    PCG64 SeedSequence([3, seed]) shuffles each class's local training IDs in
    digit order 0..9; its first 400 IDs receive corruption. An independent
    PCG64 SeedSequence([4, seed]) draws int64 offsets with
    ``integers(1, 10, size=5000, dtype=np.int64)``. For marked IDs only,
    assigned_label = (true_label + offset) % 10. Thus exactly 4,000 labels are
    wrong, with exactly 400 per true class. All other labels remain unchanged.

    Added keys: train_labels/eval_labels/assigned_labels int64[5000], and
    corruption_mask bool[5000]. Persist these arrays, source hashes, and NumPy
    version; a seed alone is not an unconditional cross-version replay promise.
    """
    plan = ordinary.make_plan(labels, seed)
    true = labels[plan["train_ids"]].astype(np.int64)
    selection_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3, int(seed)])))
    mask = np.zeros(true.shape, dtype=np.bool_)
    for digit in range(10):
        ids = np.flatnonzero(true == digit)
        selection_rng.shuffle(ids)
        mask[ids[:WRONG_PER_CLASS]] = True
    offset_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([4, int(seed)])))
    offsets = offset_rng.integers(1, 10, size=true.size, dtype=np.int64)
    assigned = true.copy()
    assigned[mask] = (true[mask] + offsets[mask]) % 10
    plan.update(train_labels=true,
                eval_labels=labels[plan["eval_ids"]].astype(np.int64),
                assigned_labels=assigned, corruption_mask=mask)
    return plan
