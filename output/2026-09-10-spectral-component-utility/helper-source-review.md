# Component-utility helpers: independent source review

10 September 2026 — **PASS for inert-helper acceptance only.** No unresolved
correctness issue was found in the reviewed versions. This is not acquisition
admission, a real-parent restoration check, or a scientific result.

## Scope and finding

Read the complete [protocol](protocol.md), [design decision](design-decision.md),
[source inventory](source-inventory.md), all four new helpers and their four
test files. Inspected the relevant unchanged strong-model, canonical flat/tree,
and spectral-filter dependencies. Used the review-changes and best-practices
guidance for evidence-backed findings and versioned library checks. Only
fabricated CPU values and temporary fabricated serialization files were used;
no real images, plans, models, checkpoints, logits or scientific outcomes were
opened. No source/test edits were made by this reviewer.

**Low severity, resolved — disconnected graph lifetime.** The original
`spectral_component_utility_objectives.py:170` promised that the last gradient
call releases “the graph.” At the actual loop (now lines 189–191), only the graph
reachable from the final objective is released. Separate earlier I-panel
forward graphs can remain alive while the caller holds their objectives/logits.
A fabricated one-parameter `I=sin(p)`, `R=cos(p)` check confirmed that I remained
differentiable after the helper returned, whereas R raised the freed-graph error.
This is not an inherent persistent leak, but the original contract could cause
avoidable graph retention in the future runner.

Main corrected the docstring at lines 170–173: **detach/store needed values,
then drop every original objective and forward-logit graph reference immediately
after baseline derivatives, before private actions/endpoints.** The added
composed fixture follows this lifecycle. No numerical change or extra backward
pass was required. The correction has been reread and tested.

## Checks completed

| Helper | Source-review conclusion |
|---|---|
| [Objectives](../../experiments/spectral_component_utility_objectives.py) | FP32 logits are promoted before differentiable class-0 offset removal. S uses the fixed `.1 true + .9 uniform` law; F includes all realized labels; C differentiates both view and mean terms. L is independent direct per-view assigned CE. S/F/C/direct-L scalar and gradient identities agree with separate references. H_T and accuracy average per-view outcomes, not mean-logit predictions. Empty actually-wrong subsets have explicit zero counts and unavailable means. |
| [Panels](../../experiments/spectral_component_utility_panels.py) | Stream 40 supplies one local 50,000-position permutation: two ordered action batches, then disjoint I. Stream 41 supplies action shifts independently of labels. R and the baseline check retain the original reporting order. IDs must partition all 60,000 source IDs. The 25 shifts are dy-outer/dx-inner, with original view 12. Returned arrays are independent copies; global NumPy state is unchanged. |
| [Restoration](../../experiments/spectral_component_utility_restore.py) | Exact strong-schema keys/order, six parameter shapes, model modes, Adam IDs/configuration/moments/counts and tracker configuration are checked. Reconstruction preserves parameter ownership aliases without sharing mutable input storage. Basis/mean move to the target; singular values remain CPU FP64. Recorded RNG is validated/preserved as payload rather than installed globally; CPU construction is RNG-isolated. Exact round trips and failure paths pass. Restricted loading hashes the same bounded regular-file descriptor before and after CPU `weights_only=True` loading, with no permissive fallback. |
| [Actions](../../experiments/spectral_component_utility_actions.py) | Raw and native start from separate joint copies of the same admitted native parent and receive identical supplied gradients. Native calls canonical observe/project once before actual inherited AdamW; raw leaves the observer unchanged. All Adam clocks advance exactly once. Exact topology/stages, moments and finite outputs are checked. CPU records do not alias the parent, and repeated pairs do not continue earlier endpoints. |

Path accounting preserves the actual FP32 full endpoint. The 0.1 point uses
explicit FP32 subtraction/multiplication/addition; utility then subtracts the
stored points in FP64. Decay is the rounded multiplicative endpoint, not a
zero-gradient Adam step. Data displacement uses the matching fractional decay
point. Tests include cancellation-sensitive endpoints, rounded-away fractional
movement, and compensated signed-dot summation.

The canonical-action fixtures cover warmup/projection and scheduled-repair
boundaries using fabricated rank-two states. The composed fixture at
`tests/test_spectral_component_utility_actions.py:122` checks
restore → common FP32 action gradient → raw/native pair → component derivatives
and fractional endpoint accounting, including unchanged parent state.

## Verification receipt

Final independent command, exit 0:

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 60s python3 -m unittest discover -s tests -p 'test_spectral_component_utility_*.py' -v
Ran 48 tests in 2.745s
OK
```

Environment: PyTorch `2.11.0+cu128`, NumPy `1.26.4`; CUDA hidden, no accelerator
work performed. An earlier independent 47-test run passed before the composed
fixture was added. `git diff --check` also passed.

Reviewed SHA-256 values:

```text
objectives.py       4e8f9890773e9eab9b6eae073cde343f9171667052b70011cffb7fb334d7bb7c
restore.py          690cdc74ea08118972f166b8ef7e96c55669bf921e14d9663dc2a4401bd5097e
panels.py           aa73f9eaed0ec26260c346714f8ba8ce6897f1877078358250f1e7d887ef2112
actions.py          a02af0da34854de7368d3140acde05b8f4cdeb103bf20f020aecde506be7a168
test_objectives.py  0d92c6662761f490568ccbfb112cab6519422cbd38420f823e1cef9657ed3e23
test_restore.py     2e0fe76fc2c6f28cada7abeaf6e78b39813c184644d9a0096182b3c768c280e2
test_panels.py      fd0dc9740d4011c9fe8473b8b6db9e6a921b1d01bfb6ab323b5e16003df01ac0
test_actions.py     98064860f246be2c88886ffff4aa3f410d177133b7a71ee3aff4d168d1139f7e
```

Names abbreviate the corresponding `spectral_component_utility_` source/test
prefixes, not different files.

## Official API alignment and remaining boundaries

Versioned PyTorch 2.11 documentation confirms that
[autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html)
returns gradients without accumulating `.grad`, and graph retention applies to
the traversed graph. The explicit schema/order checks address
[AdamW's order-based state loading](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html#torch.optim.AdamW.state_dict),
which does not itself verify parameter identity. CPU-only construction uses
[fork_rng with an empty device list](https://docs.pytorch.org/docs/2.11/random.html#torch.random.fork_rng).
[Restricted CPU loading](https://docs.pytorch.org/docs/2.11/generated/torch.load.html)
is appropriate here but is not a resource/security sandbox.

Future runner review must still establish exact parent/plan receipt binding,
the recorded 500-example baseline-logit equality, complete panel/action/readout
coverage, source pins, graph-reference cleanup, one-parent/action-at-a-time
lifetime, archive caps and external memory/time limits. GPU restoration, actual
full-rank memory/runtime and device-specific CUDA RNG semantics were not tested.
The adapter accepts additional counts for fabricated boundary tests; the real
parent roster must remain exactly h100/final56304. These are explicit remaining
acquisition boundaries, not findings that this source-only PASS has resolved.
