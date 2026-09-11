"""Inert NumPy plans/transforms for the fixed strong-regime augmentation study.

No file reads, model imports, RNG-global mutations, or acquisition on import.
Dimensions overrides exist only for fabricated fixtures; the runner must freeze
the default production dimensions. Save actual plan arrays and NumPy version.
"""

from dataclasses import dataclass

import numpy as np

from experiments import spectral_general_augmentation_data as ordinary


SEEDS = (202609171, 202609172, 202609173)
SOURCE_SIZE, TRAIN_SIZE, VALIDATION_SIZE, REPORTING_SIZE = 60000, 50000, 5000, 5000
EPOCHS, BATCH = 72, 64
BATCHES_PER_EPOCH = (TRAIN_SIZE + BATCH - 1) // BATCH
STEPS = EPOCHS * BATCHES_PER_EPOCH
CORRUPTION_PROBABILITY = .9
POLICIES, AUGMENTATIONS = ("raw", "native200"), ("none", "translate")
PIXEL_MEAN, PIXEL_STD = np.float32(.1307), np.float32(.3081)
PLAN_KEYS = ("train_ids", "validation_ids", "reporting_ids", "train_labels",
             "validation_labels", "reporting_labels", "corruption_mask",
             "replacement_labels", "assigned_labels", "changed_mask",
             "occurrences", "shifts", "batch_boundaries")


@dataclass(frozen=True)
class Dimensions:
    source_count: int = SOURCE_SIZE
    train_count: int = TRAIN_SIZE
    validation_count: int = VALIDATION_SIZE
    reporting_count: int = REPORTING_SIZE
    epochs: int = EPOCHS
    batch_size: int = BATCH

    def __post_init__(self):
        if not all(type(value) is int and value > 0 for value in vars(self).values()):
            raise ValueError("dimensions must be positive Python integers")
        if self.source_count != self.train_count + self.validation_count + self.reporting_count:
            raise ValueError("source count must equal the three disjoint role counts")
        if self.train_count > np.iinfo(np.int32).max:
            raise ValueError("local occurrence IDs exceed int32")

    @property
    def batches_per_epoch(self):
        return (self.train_count + self.batch_size - 1) // self.batch_size

    @property
    def updates(self):
        return self.epochs * self.batches_per_epoch


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer, not boolean")
    seed = int(seed)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0,2**63)")
    return seed


def _rng(seed, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream, seed])))


def make_plan(labels, seed, dimensions=Dimensions()):
    """Return exactly PLAN_KEYS, with writable contiguous independent arrays.

    stream0 globally permutes source IDs into train/validation/reporting order;
    no class balancing. stream1 first draws the selected-for-replacement mask
    via random(train_count)<.9, then draws replacements for ALL train positions.
    Selection can retain the true label; changed_mask is assigned!=true.
    stream2 sequentially permutes each epoch (not replacement sampling), storing
    local indices int32[epochs,train_count]. stream3 draws shifts once as
    int8[epochs,train_count,2], columns (dy,dx), in occurrence order.

    Shared per-epoch batch_boundaries includes both 0 and train_count, retaining
    the final partial batch. All labels/role IDs/boundaries are int64; masks bool.
    The production plan has 72*50000 exposures and 72*782=56304 updates.
    """
    if type(dimensions) is not Dimensions:
        raise TypeError("dimensions must be a Dimensions instance")
    seed = _seed(seed)
    if not isinstance(labels, np.ndarray) or labels.dtype.kind not in "iu":
        raise TypeError("labels must be an integer NumPy array, not boolean")
    if labels.shape != (dimensions.source_count,):
        raise ValueError("labels must contain exactly source_count entries")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("labels must be digits 0..9")
    order = _rng(seed, 0).permutation(dimensions.source_count).astype(np.int64)
    train_end = dimensions.train_count
    validation_end = train_end + dimensions.validation_count
    train_ids, validation_ids, reporting_ids = (part.copy() for part in
                                               (order[:train_end], order[train_end:validation_end], order[validation_end:]))
    train_labels = labels[train_ids].astype(np.int64, copy=True)
    corruption_rng = _rng(seed, 1)
    selected = corruption_rng.random(train_end) < CORRUPTION_PROBABILITY
    replacements = corruption_rng.integers(0, 10, size=train_end, dtype=np.int64)
    assigned = np.where(selected, replacements, train_labels)
    occurrence_rng = _rng(seed, 2)
    occurrences = np.empty((dimensions.epochs, train_end), dtype=np.int32)
    for epoch in range(dimensions.epochs):
        occurrences[epoch] = occurrence_rng.permutation(train_end).astype(np.int32)
    shifts = _rng(seed, 3).integers(-2, 3, size=(dimensions.epochs, train_end, 2), dtype=np.int8)
    boundaries = np.concatenate((np.arange(0, train_end, dimensions.batch_size, dtype=np.int64),
                                 np.array([train_end], dtype=np.int64)))
    return {
        "train_ids": train_ids, "validation_ids": validation_ids, "reporting_ids": reporting_ids,
        "train_labels": train_labels,
        "validation_labels": labels[validation_ids].astype(np.int64, copy=True),
        "reporting_labels": labels[reporting_ids].astype(np.int64, copy=True),
        "corruption_mask": selected, "replacement_labels": replacements,
        "assigned_labels": assigned, "changed_mask": assigned != train_labels,
        "occurrences": occurrences, "shifts": shifts, "batch_boundaries": boundaries,
    }


def translate(images, shifts):
    """Unchanged raw-[0,1] translation with explicit new (dy,dx) convention.

    Reverse only the columns supplied to the accepted helper, whose convention
    is (dx,dy). Positive dy moves down; positive dx moves right. Never translate
    already-standardized inputs: zero padding belongs in raw pixel space.
    """
    if not isinstance(shifts, np.ndarray):
        raise TypeError("shifts must be a NumPy array")
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("shifts must have shape [N,2]")
    return ordinary.translate(images, shifts[:, ::-1])


def normalize(images):
    """Standardize finite raw float32[N,784] AFTER optional translation.

    Caller performs uint8->float32 and division by float32(255) beforehand.
    This returns a new contiguous FP32 array and preserves the input, including
    read-only/strided arrays. Standardized black is -.1307/.3081, not zero.
    """
    if not isinstance(images, np.ndarray) or images.dtype != np.float32:
        raise TypeError("images must be a float32 NumPy array")
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError("images must have shape [N,784]")
    if not np.isfinite(images).all() or np.any((images < 0) | (images > 1)):
        raise ValueError("images must be finite raw pixels in [0,1]")
    return np.ascontiguousarray((images - PIXEL_MEAN) / PIXEL_STD)
