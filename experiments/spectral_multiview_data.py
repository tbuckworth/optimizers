"""Inert paired NumPy plans for the clean four-arm multiview experiment.

No data loading, global RNG consumption, model imports or execution on import.
The original split/occurrence/first-view algorithms and transform are reused.
"""

import numpy as np

from experiments import spectral_general_augmentation_data as ordinary


SEEDS = (202609161, 202609162, 202609163)
STEPS, BATCH = ordinary.STEPS, ordinary.BATCH
TRAIN_PER_CLASS, EVAL_PER_CLASS = ordinary.TRAIN_PER_CLASS, ordinary.EVAL_PER_CLASS
POLICIES = ("raw1", "native1", "observer4", "raw4")
VIEWS = 4
translate = ordinary.translate


def make_plan(labels, seed):
    """Return eight independent writable arrays for a fixed paired seed.

    Reuse the accepted PCG64 SeedSequence([stream, seed]) plan unchanged:
    streams 0/1/2 select the balanced disjoint splits, occurrence IDs and first
    view shifts. Add stream 3 extra_shifts int8[4000,3,64,2] and stream 4
    eval_shifts int8[5000,2], each drawn once as uniform integers -2..2, (dx,dy).
    train_labels/eval_labels are int64[5000] true labels in saved-ID order.
    Returned keys are train_ids, eval_ids, occurrences, shifts, extra_shifts,
    eval_shifts, train_labels, eval_labels. All four policies share this plan.
    Persist actual arrays and the NumPy version, not just the seed.
    """
    plan = ordinary.make_plan(labels, seed)  # Includes input/seed validation.
    extra_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3, int(seed)])))
    eval_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([4, int(seed)])))
    plan.update({
        "extra_shifts": extra_rng.integers(-2, 3, size=(STEPS, 3, BATCH, 2), dtype=np.int8),
        "eval_shifts": eval_rng.integers(-2, 3, size=(10 * EVAL_PER_CLASS, 2), dtype=np.int8),
        "train_labels": labels[plan["train_ids"]].astype(np.int64, copy=True),
        "eval_labels": labels[plan["eval_ids"]].astype(np.int64, copy=True),
    })
    return plan
