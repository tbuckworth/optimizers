"""Inert, occurrence-indexed augmentation plans; no data loading or training.

Insert the canonical white top-left 3x3 cue BEFORE calling ``apply_masks``.
Apply to every occurrence, including uncued and rare examples. Reuse one plan
across conditions/policies: never regenerate it from example IDs or labels.
Rectangle order is (row_start, row_stop, col_start, col_stop), half-open.
"""

from dataclasses import dataclass
import numpy as np


DEFAULT_SHAPE = (1900, 64)
EXPERIMENT_SEEDS = (202609131, 202609132, 202609133)
MODES = ("none", "random", "targeted", "opposite")


def _readonly(array):
    array.setflags(write=False)
    return array


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not boolean")
    return int(value)


@dataclass(frozen=True)
class MaskPlan:
    """Read-only-by-convention arrays; NumPy flags also reject ordinary writes."""

    gates: np.ndarray  # bool [steps, batch]
    centers: np.ndarray  # int16 [steps, batch, row/col]


def make_mask_plan(seed, shape=DEFAULT_SHAPE):
    """PCG64 streams [seed,6] for p=.5 gates, [seed,7] for centers.

    Nondefault shapes are intended only for fabricated fixtures. Gates consume
    float64 uniforms; centers consume int16 integers in [0,28). No global RNG.
    """
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape must be a two-element tuple")
    shape = tuple(_integer(v, "shape dimension") for v in shape)
    if any(v <= 0 for v in shape):
        raise ValueError("shape dimensions must be positive")
    gate_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 6])))
    center_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 7])))
    return MaskPlan(
        _readonly(gate_rng.random(shape) < 0.5),
        _readonly(center_rng.integers(0, 28, size=shape + (2,), dtype=np.int16)),
    )


def _validate_plan(plan):
    if not isinstance(plan, MaskPlan):
        raise TypeError("expected MaskPlan")
    g, c = plan.gates, plan.centers
    if not isinstance(g, np.ndarray) or g.dtype != np.bool_:
        raise TypeError("gates must be a boolean NumPy array")
    if g.ndim != 2 or any(d == 0 for d in g.shape):
        raise ValueError("gates must have nonempty [steps,batch] shape")
    if not isinstance(c, np.ndarray) or c.dtype != np.int16:
        raise TypeError("centers must be an int16 NumPy array")
    if c.shape != g.shape + (2,) or np.any((c < 0) | (c >= 28)):
        raise ValueError("centers must match gates and lie in [0,28)")


def _center_rectangles(centers):
    r, c = centers[..., 0], centers[..., 1]
    return np.stack((np.maximum(r - 4, 0), np.minimum(r + 4, 28),
                     np.maximum(c - 4, 0), np.minimum(c + 4, 28)), axis=-1)


def rectangles_for_mode(plan, mode):
    """Return int16 [steps,batch,4]; inactive rectangles are exactly all -1."""
    _validate_plan(plan)
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rectangles = np.full(plan.gates.shape + (4,), -1, dtype=np.int16)
    if mode == "random":
        rectangles[plan.gates] = _center_rectangles(plan.centers)[plan.gates]
    elif mode == "targeted":
        rectangles[plan.gates] = (0, 8, 0, 8)
    elif mode == "opposite":
        rectangles[plan.gates] = (20, 28, 20, 28)
    return _readonly(rectangles)


def _validate_rectangles(rectangles):
    if not isinstance(rectangles, np.ndarray) or rectangles.dtype.kind not in "iu":
        raise TypeError("rectangles must be an integer NumPy array")
    if rectangles.ndim < 2 or rectangles.shape[-1] != 4:
        raise ValueError("rectangles must have shape [...,4]")
    flat = rectangles.reshape(-1, 4)
    inactive = np.all(flat == -1, axis=1)
    active = flat[~inactive]
    if np.any(active > 28) or np.any(active < 0):
        raise ValueError("active bounds must lie in [0,28]; inactive rows must be all -1")
    if np.any(active[:, 0] >= active[:, 1]) or np.any(active[:, 2] >= active[:, 3]):
        raise ValueError("active rectangles must have positive width and height")
    return flat.astype(np.int64), ~inactive


def apply_masks(images, rectangles):
    """Copy floating [N,784] images in [0,1], then erase occurrence-wise.

    No cue insertion is performed here. Repeated examples can receive distinct
    masks. Input arrays are never modified; floating dtype is preserved.
    """
    if not isinstance(images, np.ndarray) or images.dtype.kind != "f":
        raise TypeError("images must be a floating NumPy array")
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError("images must have shape [N,784]")
    if not np.all(np.isfinite(images)) or np.any((images < 0) | (images > 1)):
        raise ValueError("images must be finite and lie in [0,1]")
    flat, active = _validate_rectangles(rectangles)
    if rectangles.shape != (images.shape[0], 4):
        raise ValueError("rectangles must have shape [N,4] matching images")
    result = images.copy()
    pixels = result.reshape(-1, 28, 28)
    for i in np.flatnonzero(active):
        r0, r1, c0, c1 = flat[i]
        pixels[i, r0:r1, c0:c1] = 0
    return result


def coverage_summary(rectangles):
    """Geometry counts, not counts of changed nonzero pixels or cued examples.

    Cue footprint is [0,3)x[0,3). Denominator includes inactive occurrences.
    ``erased_area_counts[a]`` counts occurrences with geometric erased area a.
    """
    flat, active = _validate_rectangles(rectangles)
    r0, r1, c0, c1 = flat.T
    areas = np.where(active, (r1 - r0) * (c1 - c0), 0)
    return {
        "occurrences": int(len(flat)),
        "active": int(active.sum()),
        "any_cue_coverage": int(np.sum(active & (r0 < 3) & (c0 < 3))),
        "full_cue_coverage": int(np.sum(active & (r0 == 0) & (c0 == 0)
                                         & (r1 >= 3) & (c1 >= 3))),
        "erased_area_total": int(areas.sum()),
        "erased_area_counts": _readonly(np.bincount(areas, minlength=785)),
    }


def exact_random_geometry():
    """Enumerate all 784 centers, row-major, conditional on an active gate.

    Returns read-only int16 centers/rectangles and the coverage summary. Divide
    counts by 784 for conditional probabilities, then multiply by .5 for the
    unconditional masking probabilities. No sampling or file I/O.
    """
    centers = np.indices((28, 28), dtype=np.int16).reshape(2, -1).T.copy()
    rectangles = _center_rectangles(centers)
    return {"centers": _readonly(centers), "rectangles": _readonly(rectangles),
            "summary": coverage_summary(rectangles)}
