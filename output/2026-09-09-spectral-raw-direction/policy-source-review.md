# Independent source review: raw-direction policy

9 September 2026. **PASS for use by the separately reviewed runner; no source
fix is required.** This is a bounded source-and-synthetic-fixture review, not
acquisition authorization, a runner review or an experiment result.

Reviewed:

- `experiments/grokking_raw_direction_policy.py`, SHA-256
  `d5c10674bc20aaf16adf9b7d3f9591a86d152e6d7fdee1f3c1767dc339611ec3`;
- `tests/test_grokking_raw_direction_policy.py`, SHA-256
  `dc099d31694cecc2cd36284a23829026b0fbf19117a57590706db45e2e328719`;
- the prospective discriminator, independent design review and implementation
  check identified in the parent task.

## Findings

No correctness, intervention-integrity or observability defect was found in the
reviewed policy source and fixtures.

The target is the exact legacy native action in the incoming gradient dtype and
historical parenthesization, `native = V @ (V.T @ g)`. Its float64 norm is then
matched along the detached raw gradient before the result is cast back to the
original gradient dtype. Thus the delivered nondegenerate action implements
`||V(Vᵀg)|| g/||g||` before the caller's ordinary AdamW step. The inherited
`filter_grad()` updates the unchanged legacy observer exactly once before
calling this action substitution. A shared-input fixture confirms identical
`V`, `S`, `grad_mean`, step count and inherited diagnostics against the frozen
legacy filter.

The frozen `1e-30` denominator floor and imported ten-epsilon/minimum-relative
tolerance convention are preserved. A positive target with exactly zero raw
direction fails closed. A zero target yields an explicit zero tensor even when
the raw gradient is nonzero; two zeros likewise yield an explicit zero tensor,
with zero-vector cosines recorded as `None`. The carried-moment fixture then
executes AdamW normally and confirms that its step counter, moment decay,
decoupled weight decay and parameter movement remain active. Positive sub-floor
inputs are clamped and explicitly reported as approximate rather than silently
accepted as exact.

The result keeps `raw_gradient`, `native`, unscaled `orthogonal`, projected
`norm_matched` and delivered `raw_norm_matched` as distinct tensors. Diagnostics
include all action norms, raw-to-action and projected-to-raw-action cosines,
projected-over-raw norm fraction, numerical rank, clamp/degeneracy flags and
post-cast mismatch/tolerance. When basis retention is enabled, raw-action
coordinates are labeled as incomplete outside the retained span. The pure
action fixture confirms that `V`, `g` and CPU RNG state are unchanged; the
source contains no random operation. The wrapper's intended mutations are only
the inherited estimator update, diagnostic snapshots and replacement of
parameter gradients.

## Frozen-source integrity

The four frozen files have no worktree diff and their current SHA-256 values
match the accepted prior action-analysis receipts:

- `spectral_filter.py`:
  `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`;
- `experiments/grokking_action_policy.py`:
  `d3555d2e6816c031acc51f4939316eddba7e8b3fafcfaef47b0169310960ca1b`;
- `experiments/grokking_action_intervention.py`:
  `95200567f9c7eb50b76b563a554305f3ea7ccd03b6ab3525adf5d1dad388eb37`;
- `experiments/grokking_confirmation.py`:
  `31bd8c9694e882c62b8184ed5cbf2d0f895ef67f255266dcf33abf0e2d0b92b6`.

## Synthetic verification

Command:

```text
python3 -m unittest discover -s tests -p 'test_grokking_raw_direction_policy.py' -v
```

Result: **9 tests passed** in 0.786 seconds on CPU with installed
PyTorch `2.11.0+cu128`. `pytest` was not installed, so the standard-library
runner was used; it executed all nine test methods. No checkpoint was loaded,
no inference or training ran, and no GPU, cloud resource, prior analysis or
experimental arm was started.

This pass covers only the policy module and its synthetic fixtures. Runner,
protocol, receipts, resource guards and first-step/acquisition outputs require
their own independent review before launch.
