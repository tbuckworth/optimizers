# CPU component implementation decision

Codex parent, Spectral Optimizer Investigation, 6 September 2026.

The [scientific design](common-state-design.md) has completed two independent
challenge rounds. Parent has now read and checked the complete
[analysis contract](analysis-spec.md), anchor schema (artifact not distributed in this public snapshot) and
resource contract (artifact not distributed in this public snapshot), including installed AdamW group
membership, observer None-basis behavior, native post-branch moments and exact
pre/post phase counters. No scientific plan arrays or MNIST outcomes were read.

Approve implementation and **dataset-free CPU tests only** for two separable
components whose contracts are now explicit:

1. Six delivered-gradient constructions, factor leverage, fixed contrast
   coefficients and complete-mask aggregation. Norm/direction gates are the
   fixed 1e-6 tolerances; these do not depend on future loss-audit tolerances.
2. Exact model/AdamW/observer/RNG core capture, owned primitive/tensor encoding,
   restoration, restricted serialization and tiny CPU replay fixtures.

This permits incremental engineering while the independent numerical-loss
contract is finalized. It does **not** waive that contract: loss measurement,
cross-device numerical-audit implementation and a scientific runner still
require parent acceptance of their separate tolerances and schemas. This is a
component-level approval, not a statement that the full implementation or any
neural experiment is ready. Final integration and native pilot remain required.

Required parent checks before accepting code: read complete implementation and
tests, run the CPU tests, examine injected-invalid-state coverage, verify storage
ownership and no RNG leakage, and independently inspect actual replay comparisons.
Passing fixtures is engineering evidence only. Any future GO must additionally
bind committed scientific sources, exact data plans, native environment, full
artifact accounting and observed native timing/instrumentation gates.

## Subsequent CPU numerical-contract acceptance

Parent read the complete numerical contract and independent dataset-free test
implementation, then ran its ten tests successfully. TL/Tq and K=32 are accepted
unchanged for CPU-only loss/arithmetic implementation and integration fixtures.
They remain engineering screens, not universal or CUDA-certified error bounds.
The original two components and these fixtures are reviewed in
[cpu-component-review.md](cpu-component-review.md).

Next authorized work remains dataset-free: specify exact full artifact mappings
and byte accounting, implement CPU loss/gradient and independent formula checks,
then integrate tiny-fixture capture/branch execution with a live next-step witness.
Projection and native-concordance fixture coverage is still required. No data
plan generation, MNIST access, native pilot or scientific runner execution is
approved by this acceptance. Any later pilot needs a separate reviewed GO.

The next CPU integration uses the separate 26-parameter two-layer fixture in
measurement-schema.md (artifact not distributed in this public snapshot). The original one-layer fixture
cannot exercise the defined two-layer loss; neither fixture may enter scientific
manifests. This approves only the new fixture profile and dataset-free tests.

At the subsequent [measurement checkpoint](measurement-component-review.md),
parent accepts the unchanged projection screens and native-loss-concordance
logic for CPU implementation after independent fixtures. The full-shape CPU
loss check uses synthetic examples only. Neither these tests nor the bounded
artifact-store component authorize a scientific manifest, native pilot or run.

The next scoped implementation integrates the exact measurement schema with
raw endpoints/probes, separate NumPy checks, immutable fixture serialization,
and cooperative runtime checks. The detailed `i7_assembly_roundoff_v1` formulas
are recorded before scientific input access. CPU fixture counts are explicitly
separate from the study counts. Independent audit stays CPU-only; native source
phases must use the recorded local GPU. This is implementation authority only,
not a study-root creation, data-plan generation or native-pilot GO.
