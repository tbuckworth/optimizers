# Main acceptance of the component-utility helpers

Codex — Spectral Optimizer Investigation · 10 September 2026

**Accepted for runner/auditor integration, not scientific acquisition.**
Four new import-inert modules and their tests implement strict strong-state
restoration, fixed panel selection, common-parent raw/native actions and
signed-objective/path accounting. Main read their complete sources and fixtures,
both author notes and the [independent source review](helper-source-review.md).
No unresolved helper defect remains in the reviewed versions.

The independent review SHA-256 is
`60bb742ccf7edb720c4690de21ba87a289b992d84e7217bb2b9bbbafdb06fcc2`.
It records exact hashes for all eight final source/test files. Main independently
checked those file hashes. Objective source's final hash is
`4e8f9890773e9eab9b6eae073cde343f9171667052b70011cffb7fb334d7bb7c`;
its author's earlier hash remains labeled as the pre-review handoff.

## Verification and correction history

Main's final focused run, exit0:

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 60s python3 -m unittest discover -s tests -p 'test_spectral_component_utility_*.py'
Ran 48 tests in 2.705s
OK
```

Independent final run passed the same48 tests in2.745s. Environment is Python
3.12.3, NumPy1.26.4 and PyTorch2.11.0+cu128; all tensors/models/serialization
files in these checks were fabricated CPU objects, with CUDA hidden. No
scientific parent, original image/plan or saved scientific logits were read.
The composed fixture verifies restore → one common action gradient → both
actions → all six objective derivatives → full/0.1-path accounting, with parent
state intact. Separate direct references check the identities, canonical action
and repair boundaries, not just agreement between new helper calls.

Two preparation errors are preserved in the author/main notes: a strict CPU
fixture needed to answer Adam's availability health check locally as false,
and main's topology-validation patch briefly introduced an indentation error.
Both were caught and fixed before any scientific execution; no tolerance,
model setting or data-dependent choice changed.

The independent review found one additional low-severity documentation issue:
the last objective gradient frees only its reachable graph. Main corrected
the helper contract and protocol to save detached values and drop **all**
original objective/logit graph references before actions. No extra gradient
pass or numerical change was introduced. The composed fixture follows this
requirement. Framework/construct checks improved implementation discipline;
they did not add a user-approval gate or demand a favorable research outcome.

## Provenance and scope

Next is [fixed runner and audit preparation](runner-preparation.md), including
all input bindings, real-baseline equality as an admitted-run check, exact
output/live-object bounds, fabricated failure/cap/summary fixtures, independent
source review and a separate committed local admission. The prospective
payload upper bound6,402,640,064bytes fits the8GiB archive envelope but is not
an enforced runner inventory or measured runtime. No scientific mechanism or
GPU-restoration equivalence has been established by this source-only work.
