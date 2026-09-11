"""Deterministic I14 train-only MNIST planning; import performs no data read."""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "i14_cross_optimizer_data_plan_v1"
DATA = Path("data/MNIST/raw")
CALIBRATION_SEEDS = (190, 191)
CONFIRMATION_SEEDS = (200, 201, 202)
STEPS = 2000
HORIZONS = (0, 100, 250, 500, 1000, 1500, 2000)
BATCH = 64
EXAMPLE_COUNT = 60_000
SPLIT_SIZE = 5_000
CALIBRATION_POOL_SIZE = 15_000
DATA_KEYS = ("x", "clean", "noisy", "vx", "vy", "ax", "ay")
PLAN_KEYS = (
    "schema", "phase", "seed", "initialization_seed", "train_indices",
    "validation_indices", "auxiliary_indices", "replacement_mask",
    "replacement_digits", "training_batches",
)
ARRAY_DTYPES = {
    "train_indices": "int64",
    "validation_indices": "int64",
    "auxiliary_indices": "int64",
    "replacement_mask": "bool",
    "replacement_digits": "int64",
    "training_batches": "int64",
}
ARRAY_SHAPES = {
    "train_indices": (SPLIT_SIZE,),
    "validation_indices": (SPLIT_SIZE,),
    "auxiliary_indices": (SPLIT_SIZE,),
    "replacement_mask": (SPLIT_SIZE,),
    "replacement_digits": (SPLIT_SIZE,),
    "training_batches": (STEPS, BATCH),
}

_GLOBAL_ORDER = np.random.default_rng(
    np.random.SeedSequence([20260907, 14, 0, 0])
).permutation(EXAMPLE_COUNT)
_GLOBAL_ORDER.setflags(write=False)


class DataPlanError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DataPlanError(message)


def _phase_seed(phase: str, seed: int) -> None:
    _need(type(phase) is str and phase in ("calibration", "confirmation"),
          "phase must be calibration or confirmation")
    _need(type(seed) is int and seed in (
        CALIBRATION_SEEDS if phase == "calibration" else CONFIRMATION_SEEDS
    ), "seed is not registered for phase")


def _pool(phase: str) -> np.ndarray:
    if phase == "calibration":
        return _GLOBAL_ORDER[:CALIBRATION_POOL_SIZE]
    return _GLOBAL_ORDER[CALIBRATION_POOL_SIZE:]


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([20260907, 14, seed, stream])
    )


def make_plan(seed: int, phase: str) -> dict[str, Any]:
    """Construct one exact plan without consulting data or global RNG state."""
    _phase_seed(phase, seed)
    order = _rng(seed, 0).permutation(_pool(phase))
    return {
        "schema": SCHEMA,
        "phase": phase,
        "seed": seed,
        "initialization_seed": int(_rng(seed, 1).integers(
            0, 2**31, dtype=np.int64)),
        "train_indices": order[:SPLIT_SIZE].astype(np.int64, copy=True),
        "validation_indices": order[SPLIT_SIZE:2 * SPLIT_SIZE].astype(
            np.int64, copy=True),
        "auxiliary_indices": order[2 * SPLIT_SIZE:3 * SPLIT_SIZE].astype(
            np.int64, copy=True),
        "replacement_mask": (_rng(seed, 2).random(SPLIT_SIZE) < .9),
        "replacement_digits": _rng(seed, 3).integers(
            0, 10, SPLIT_SIZE, dtype=np.int64),
        "training_batches": _rng(seed, 4).integers(
            0, SPLIT_SIZE, (STEPS, BATCH), dtype=np.int64),
    }


def _array(value: Any, name: str) -> np.ndarray:
    dtype = np.bool_ if ARRAY_DTYPES[name] == "bool" else np.int64
    _need(type(value) is np.ndarray, name + " must be an exact ndarray")
    _need(value.dtype == dtype, name + " dtype differs")
    _need(value.shape == ARRAY_SHAPES[name], name + " shape differs")
    _need(value.flags.c_contiguous, name + " must be C contiguous")
    return value


def validate_plan(plan: Any) -> None:
    """Validate structure, phase membership, and exact deterministic contents."""
    _need(type(plan) is dict and tuple(plan) == PLAN_KEYS,
          "plan keys/order differ")
    _need(type(plan["schema"]) is str and plan["schema"] == SCHEMA,
          "plan schema differs")
    phase, seed = plan["phase"], plan["seed"]
    _phase_seed(phase, seed)
    _need(type(plan["initialization_seed"]) is int
          and 0 <= plan["initialization_seed"] < 2**31,
          "initialization_seed is invalid")

    arrays = {name: _array(plan[name], name) for name in ARRAY_DTYPES}
    splits = np.concatenate(tuple(arrays[name] for name in (
        "train_indices", "validation_indices", "auxiliary_indices"
    )))
    _need(np.unique(splits).size == 3 * SPLIT_SIZE,
          "data splits are not mutually disjoint and unique")
    _need(bool(((splits >= 0) & (splits < EXAMPLE_COUNT)).all()),
          "data split index is out of range")
    membership = np.zeros(EXAMPLE_COUNT, dtype=np.bool_)
    membership[_pool(phase)] = True
    _need(bool(membership[splits].all()), "data split is outside phase pool")
    _need(bool(((arrays["replacement_digits"] >= 0)
                & (arrays["replacement_digits"] < 10)).all()),
          "replacement digit is out of range")
    _need(bool(((arrays["training_batches"] >= 0)
                & (arrays["training_batches"] < SPLIT_SIZE)).all()),
          "training batch index is out of range")

    expected = make_plan(seed, phase)
    _need(plan["initialization_seed"] == expected["initialization_seed"],
          "initialization_seed differs from deterministic plan")
    for name in ARRAY_DTYPES:
        _need(np.array_equal(arrays[name], expected[name]),
              name + " differs from deterministic plan")


def _json_int(value: Any, label: str) -> int:
    _need(type(value) is int and -(2**63) <= value < 2**63,
          label + " must contain signed-64-bit exact JSON integers")
    return value


def _json_vector(value: Any, length: int, label: str,
                 *, boolean: bool = False) -> list[Any]:
    _need(type(value) is list and len(value) == length,
          label + " JSON shape differs")
    if boolean:
        _need(all(type(item) is bool for item in value),
              label + " must contain exact JSON booleans")
    else:
        for item in value:
            _json_int(item, label)
    return value


def plan_from_json(payload: Any) -> dict[str, Any]:
    """Reconstruct exact array dtypes from a parsed JSON plan, then validate."""
    _need(type(payload) is dict and tuple(payload) == PLAN_KEYS,
          "JSON plan keys/order differ")
    _need(type(payload["schema"]) is str and type(payload["phase"]) is str,
          "JSON plan metadata types differ")
    _json_int(payload["seed"], "seed")
    _json_int(payload["initialization_seed"], "initialization_seed")
    result: dict[str, Any] = {
        "schema": payload["schema"],
        "phase": payload["phase"],
        "seed": payload["seed"],
        "initialization_seed": payload["initialization_seed"],
    }
    for name in ("train_indices", "validation_indices", "auxiliary_indices",
                 "replacement_digits"):
        values = _json_vector(payload[name], SPLIT_SIZE, name)
        result[name] = np.asarray(values, dtype=np.int64)
    result["replacement_mask"] = np.asarray(_json_vector(
        payload["replacement_mask"], SPLIT_SIZE, "replacement_mask",
        boolean=True), dtype=np.bool_)
    rows = payload["training_batches"]
    _need(type(rows) is list and len(rows) == STEPS,
          "training_batches JSON row count differs")
    for row in rows:
        _json_vector(row, BATCH, "training_batches")
    result["training_batches"] = np.asarray(rows, dtype=np.int64)
    result = {key: result[key] for key in PLAN_KEYS}
    validate_plan(result)
    return result


def json_tree(value: Any) -> Any:
    """Convert the supported plan tree into JSON-safe exact basic values."""
    if type(value) is np.ndarray:
        _need(value.dtype in (np.dtype(np.int64), np.dtype(np.bool_)),
              "unsupported ndarray dtype")
        return value.tolist()
    if type(value) is dict:
        _need(all(type(key) is str for key in value),
              "JSON tree dict keys must be exact strings")
        return {key: json_tree(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [json_tree(item) for item in value]
    if isinstance(value, np.generic):
        return json_tree(value.item())
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        _need(math.isfinite(value), "JSON tree contains a nonfinite float")
        return value
    raise DataPlanError("JSON tree contains an unsupported value")


def read_training(data_dir: str | Path = DATA) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only the two MNIST training IDX files, matching the I9 semantics."""
    root = Path(data_dir)
    image_bytes = (root / "train-images-idx3-ubyte").read_bytes()
    _need(len(image_bytes) >= 16, "training image IDX header is truncated")
    magic, count, rows, cols = struct.unpack(">IIII", image_bytes[:16])
    _need((magic, count, rows, cols) == (2051, EXAMPLE_COUNT, 28, 28),
          "training image IDX header differs")
    _need(len(image_bytes) == 16 + EXAMPLE_COUNT * 784,
          "training image IDX length differs")
    x = torch.from_numpy(np.frombuffer(
        image_bytes, np.uint8, offset=16
    ).copy().reshape(EXAMPLE_COUNT, 784))

    label_bytes = (root / "train-labels-idx1-ubyte").read_bytes()
    _need(len(label_bytes) >= 8, "training label IDX header is truncated")
    _need(struct.unpack(">II", label_bytes[:8]) == (2049, EXAMPLE_COUNT),
          "training label IDX header differs")
    _need(len(label_bytes) == 8 + EXAMPLE_COUNT,
          "training label IDX length differs")
    y = torch.from_numpy(np.frombuffer(
        label_bytes, np.uint8, offset=8
    ).copy()).to(dtype=torch.int64)
    _need(bool(((y >= 0) & (y < 10)).all()),
          "training label IDX contains an invalid class")
    return x, y


def data_for_plan(x: torch.Tensor, y: torch.Tensor, plan: Any,
                  device: str | torch.device) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Materialize one validated plan on an explicit target device."""
    validate_plan(plan)
    _need(type(x) is torch.Tensor and x.device.type == "cpu"
          and x.dtype == torch.uint8 and tuple(x.shape) == (EXAMPLE_COUNT, 784)
          and x.layout == torch.strided and x.is_contiguous(),
          "training images must be a contiguous CPU uint8 IDX tensor")
    _need(type(y) is torch.Tensor and y.device.type == "cpu"
          and y.dtype == torch.int64 and tuple(y.shape) == (EXAMPLE_COUNT,)
          and y.layout == torch.strided and y.is_contiguous(),
          "training labels must be a contiguous CPU int64 IDX tensor")
    _need(bool(((y >= 0) & (y < 10)).all()), "training labels are invalid")
    target_device = torch.device(device)

    ti = torch.from_numpy(plan["train_indices"])
    vi = torch.from_numpy(plan["validation_indices"])
    ai = torch.from_numpy(plan["auxiliary_indices"])
    clean = y[ti]
    mask = torch.from_numpy(plan["replacement_mask"])
    digits = torch.from_numpy(plan["replacement_digits"])
    noisy = torch.where(mask, digits, clean)
    data = {
        "x": x[ti].to(dtype=torch.float32).div_(255).to(target_device),
        "clean": clean.to(target_device),
        "noisy": noisy.to(target_device),
        "vx": x[vi].to(dtype=torch.float32).div_(255).to(target_device),
        "vy": y[vi].to(target_device),
        "ax": x[ai].to(dtype=torch.float32).div_(255).to(target_device),
        "ay": y[ai].to(target_device),
    }
    _need(tuple(data) == DATA_KEYS, "materialized data key order differs")
    stats = {
        "replaced_count": int(mask.sum().item()),
        "incorrect_count": int((noisy != clean).sum().item()),
        "train_count": SPLIT_SIZE,
    }
    return data, stats
