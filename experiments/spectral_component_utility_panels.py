"""Fixed diagnostic panel selection; import-inert and no file/model access.

Consumes only the seven role/label arrays extracted from the already pinned
strong-study plan, not its training occurrences or images. Full original-plan
receipt verification and correspondence to source labels belong to the runner.
No dimensions, seed roster or draw-count tuning options are exposed.
"""

import numpy as np


SEEDS = (202609171, 202609172, 202609173)
ORIGINAL_VIEW_INDEX = 12
INPUT_KEYS = ("train_ids", "validation_ids", "reporting_ids", "train_labels",
              "validation_labels", "reporting_labels", "assigned_labels")
OUTPUT_KEYS = ("action_positions", "action_ids", "action_true", "action_assigned",
               "action_shifts", "evaluation_positions", "evaluation_ids",
               "evaluation_true", "evaluation_assigned", "evaluation_wrong",
               "reporting_ids", "reporting_true", "baseline_ids", "baseline_true",
               "view_shifts")


def _array(value, shape, name):
    if type(value) is not np.ndarray or value.dtype != np.int64 or value.shape != shape:
        raise ValueError(f"{name} must be an exact int64 array of shape {shape}")
    return value


def view_shifts():
    """Return a fresh int8[25,2] array; dy outer, dx inner, zero at index 12."""
    return np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], dtype=np.int8)


def select_panels(role_arrays, seed):
    """Return independent contiguous arrays under the fixed streams 40 and 41.

role_arrays must contain exactly INPUT_KEYS, with original ordered roles.
Validate all role IDs as a partition of 0..59999 and all labels as digits.
No labels influence position or transformation selection; no class balancing.
All action inputs will be translated, including for unaugmented parents.
Global NumPy RNG state and every supplied array are untouched.
"""
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed must be one of the three fixed Python integer seeds")
    if type(role_arrays) is not dict or set(role_arrays) != set(INPUT_KEYS):
        raise ValueError("exact original role-array keys required")
    for role, size in (("train", 50000), ("validation", 5000), ("reporting", 5000)):
        _array(role_arrays[role + "_ids"], (size,), role + "_ids")
        labels = _array(role_arrays[role + "_labels"], (size,), role + "_labels")
        if np.any((labels < 0) | (labels > 9)):
            raise ValueError("true labels must be digits")
    assigned = _array(role_arrays["assigned_labels"], (50000,), "assigned_labels")
    if np.any((assigned < 0) | (assigned > 9)):
        raise ValueError("assigned labels must be digits")
    all_ids = np.concatenate([role_arrays[role + "_ids"] for role in
                              ("train", "validation", "reporting")])
    if not np.array_equal(np.sort(all_ids), np.arange(60000, dtype=np.int64)):
        raise ValueError("role IDs must partition the original 60000 source IDs")
    order = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([40, seed]))).permutation(50000).astype(np.int64)
    action = order[:128].reshape(2, 64).copy()
    evaluation = order[128:384].copy()
    shifts = np.random.Generator(np.random.PCG64(np.random.SeedSequence([41, seed]))).integers(
        -2, 3, size=(2, 64, 2), dtype=np.int8)
    true, ids = role_arrays["train_labels"], role_arrays["train_ids"]
    values = (action, ids[action], true[action], assigned[action], shifts,
              evaluation, ids[evaluation], true[evaluation], assigned[evaluation],
              assigned[evaluation] != true[evaluation],
              role_arrays["reporting_ids"][:128], role_arrays["reporting_labels"][:128],
              role_arrays["reporting_ids"][:500], role_arrays["reporting_labels"][:500],
              view_shifts())
    return {key: np.array(value, copy=True, order="C")
            for key, value in zip(OUTPUT_KEYS, values)}
