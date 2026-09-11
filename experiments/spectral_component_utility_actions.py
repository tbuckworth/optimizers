"""One raw/native pair on private copies of caller-owned strong-model state.

No file access, model creation, RNG draw or science on import. This module
does not compute the action gradient: both branches receive its identical
caller-supplied FP32 vector. No trajectory continues after these endpoints.
"""

import copy

import torch

from experiments import spectral_strong_augmentation_core as core


PARAMETER_NAMES = ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
PARAMETER_SHAPES = ((256, 784), (256,), (128, 256), (128,), (10, 128), (10,))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu(value):
    return value.detach().cpu().clone()


def _moments(optimizer, parameters, name):
    return torch.cat([optimizer.state[p][name].detach().reshape(-1)
                      for p in parameters]).clone()


def paired_actions(parent, raw_gradient):
    """Return two CPU tensor records, raw then native, without mutating parent.

The parent must be a fully initialized strong native state at step100 or56304.
Joint deepcopy preserves model/optimizer/tracker ownership. Process branches
sequentially; never copy a previous endpoint to initialize a later branch.
FP32 endpoints/decay are saved verbatim; scalar/dot readouts belong elsewhere.
Counters record the unused raw branch observer as unchanged, not falsely
advanced. Only native observes the supplied gradient and projects once.
"""
    _need(type(parent) is tuple and len(parent) == 3, "parent must be (model, optimizer, tracker)")
    model, optimizer, tracker = parent
    _need(tracker is not None, "native parent tracker required")
    parameters, counts = core._validate_ownership(*parent)
    _need(type(model) is torch.nn.Sequential and len(model) == 5
          and [type(module) for module in model] ==
          [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear]
          and [(model[i].in_features, model[i].out_features) for i in (0, 2, 4)] ==
          [(784, 256), (256, 128), (128, 10)]
          and tuple(dict(model.named_parameters())) == PARAMETER_NAMES
          and tuple(tuple(parameter.shape) for parameter in parameters) == PARAMETER_SHAPES
          and getattr(model, "_strong_augmentation_spec", None) == core.MODEL_SPEC
          and sum(p.numel() for p in parameters) == core.PARAMETER_COUNT,
          "exact strong-model topology required")
    _need(counts[0] in (100, 56304), "only the fixed native parent stages are admitted")
    _need(type(raw_gradient) is torch.Tensor and raw_gradient.dtype == torch.float32
          and raw_gradient.shape == (core.PARAMETER_COUNT,)
          and raw_gradient.device == parameters[0].device
          and bool(torch.isfinite(raw_gradient).all()), "finite matching FP32 raw gradient required")
    theta = core.flat_params(model)
    _need(bool(torch.isfinite(theta).all()), "nonfinite parent parameters")
    for name in ("exp_avg", "exp_avg_sq"):
        values = _moments(optimizer, parameters, name)
        _need(bool(torch.isfinite(values).all()) and
              (name != "exp_avg_sq" or bool((values >= 0).all())), "invalid inherited Adam moments")
    records = []
    for policy in ("raw", "native"):
        private_model, private_optimizer, private_tracker = copy.deepcopy(parent)
        private_parameters, private_counts = core._validate_ownership(
            private_model, private_optimizer, private_tracker)
        _need(private_counts == counts, "private Adam clocks differ")
        before = core.flat_params(private_model)
        _need(torch.equal(before, theta), "private parent parameters differ")
        record = {"policy": policy, "theta_before": _cpu(before),
                  "decay_endpoint": _cpu(before.clone().mul_(1 - core.LR * core.WD)),
                  "gradient_raw": _cpu(raw_gradient),
                  "m_before": _cpu(_moments(private_optimizer, private_parameters, "exp_avg")),
                  "v_before": _cpu(_moments(private_optimizer, private_parameters, "exp_avg_sq")),
                  "adam_steps_before": torch.tensor(counts, dtype=torch.int64),
                  "observer_steps_before": int(private_tracker.step_count)}
        core.set_grad(private_model, raw_gradient.detach())
        if policy == "native":
            private_tracker.filter_grad()
        delivered = core.flat_grad(private_model)
        private_optimizer.step()
        after_counts = core._counts(private_optimizer, private_parameters)
        _need(after_counts == [count + 1 for count in counts], "Adam must advance once")
        expected_observer = counts[0] + (policy == "native")
        _need(private_tracker.step_count == expected_observer, "observer clock mismatch")
        after = core.flat_params(private_model)
        _need(bool(torch.isfinite(after).all()), "nonfinite action endpoint")
        m_after = _moments(private_optimizer, private_parameters, "exp_avg")
        v_after = _moments(private_optimizer, private_parameters, "exp_avg_sq")
        _need(bool(torch.isfinite(m_after).all()) and bool(torch.isfinite(v_after).all())
              and bool((v_after >= 0).all()), "invalid resulting Adam moments")
        record.update({"theta_after": _cpu(after), "gradient_delivered": _cpu(delivered),
                       "m_after": _cpu(m_after), "v_after": _cpu(v_after),
                       "adam_steps_after": torch.tensor(after_counts, dtype=torch.int64),
                       "observer_steps_after": int(private_tracker.step_count),
                       "post_basis": _cpu(private_tracker.V) if policy == "native" and
                           private_tracker.V is not None else None,
                       "post_singular_values": _cpu(private_tracker.S) if policy == "native" and
                           private_tracker.S is not None else None})
        if record["post_basis"] is not None:
            _need(record["post_basis"].shape[0] == core.PARAMETER_COUNT
                  and record["post_basis"].shape[1] <= 200
                  and bool(torch.isfinite(record["post_basis"]).all()), "invalid post-native basis")
        records.append(record)
        del private_model, private_optimizer, private_tracker, private_parameters
    return records
