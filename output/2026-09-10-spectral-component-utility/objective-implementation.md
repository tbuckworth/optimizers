# Component-utility objectives: implementation record

The inert helper and its 19 fabricated CPU fixtures are ready for integration and source review. This is not scientific acquisition or admission: no model, checkpoint, dataset, optimizer, scientific archive, or GPU was used. The helper has no CLI, file access, sampling, or optimizer operations.

## API and numerical conventions

`experiments/spectral_component_utility_objectives.py` exposes:

- `objective_tensors(I, true, assigned, R_original=None, R_grid=None, R_true=None)`: caller-supplied FP32 logits in image/view/class order produce differentiable FP64 scalars `S`, `F`, `C`, independently evaluated assigned per-view `L`, `S_true`, and `S_uniform`; optional reporting inputs add clean `H_O` and `H_T`. Every image and view has equal weight. The soft target is exactly the prescribed 0.1 true-label plus 0.9 uniform law, not the realized wrong-label frequency.
- `metric_report(..., original_view_index=...)`: JSON-compatible original and per-view sufficient statistics for true and assigned I labels, their actually-wrong subset, and optional clean R. Counts mean image-view outcomes; image and view counts are separately named. Empty wrong subsets retain zero counts and CE sums, with unavailable/null accuracy and mean CE. No mean-logit accuracy or view vote is substituted.
- `flat_objective_gradients(objectives, parameters, retain_graph=False)`: ordered, detached FP32 flat gradients from distinct caller-supplied FP32 trainable leaves. `torch.autograd.grad` does not accumulate `.grad`; existing sentinel gradients are preserved. Unused parameters are errors. Intermediate objectives retain the graph; the final call follows the explicit retention argument.
- `materialize_path`, `actual_delta`, `path_accounting`, and `signed_utility`: actual FP32 path points, FP64 endpoint differences, same-fraction supplied decay accounting, and negative gradient–displacement products summed with ordered `math.fsum`.

Before view averaging, each FP32 logit vector is cast to FP64 and its class-0 value is subtracted **differentiably** from every class. This removes arbitrary common offsets independently for each image/view, preventing avoidable large-offset cancellation in F and C. The mean logits remain in the differentiation graph. L is separately computed from assigned per-view CE, not reconstructed as S + F + C. C is not clamped. This convention cannot recover class differences already lost when the incoming logits were rounded to FP32.

Objective reductions use explicit Torch FP64 means. Metric CE sums use CPU `math.fsum` over FP64 per-outcome losses in image-major/view-major order, so their independent reductions can differ slightly. The frozen scalar checks use absolute/relative tolerance 1e-10; gradient identities use the prescribed 1e-6 absolute and 5e-5 relative tolerances. Production aggregate/norm-law admission remains the runner/auditor's responsibility.

Fraction 1 copies the supplied full endpoint exactly; intermediate fractions explicitly perform FP32 subtraction, multiplication, and addition. Deltas are computed by casting the **materialized endpoints before subtraction**, not by scaling an ideal full displacement. The supplied decay endpoint is materialized at the same fraction. It is not an Adam zero-gradient update. Signed utilities use FP64 products and compensated summation of the actual delta.

## Verification

Executed only fabricated CPU fixtures:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= timeout 60s python3 -m unittest discover -s tests -p 'test_spectral_component_utility_objectives.py' -v
```

The initial 18 fixtures passed in 0.076 seconds. The final 19 fixtures, adding an exact 25-view/10-class grid and all six prescribed gradients, passed in 0.076 seconds. There were no fixture failures. Coverage includes independent scalar/direct-CE and analytic gradient identities, one-view C degeneracy, label-independent C, per-view common offsets up to magnitude 2²², original versus averaged-view R losses, empty wrong subsets, parameter ordering and sentinel `.grad` preservation, unchanged noncontiguous input metadata, FP32 fractional-path rounding, exact full endpoints, and a compensated-dot cancellation example.

The best-practices validator was applied before implementation. Relevant official PyTorch 2.11 contracts were checked directly: [autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html) returns gradients without accumulating them into input `.grad`; [logsumexp](https://docs.pytorch.org/docs/2.11/generated/torch.logsumexp.html) supplies the stabilized, explicit-axis reduction. The gauge and decomposition are project mathematics, not claims supplied by those API documents.

## Provenance and limits

- Helper SHA256: `36bc2a460d74fd9abb7d71214b8c8d7aaa98c9039ec746b2cdd93282eda32487`.
- Fixture SHA256: `0d92c6662761f490568ccbfb112cab6519422cbd38420f823e1cef9657ed3e23`.
- An external preservation hook committed the helper in `e5dc024`; it was preserved unchanged. The leaf made no commit. Main owns the final source freeze and integration.

The generic helper accepts small grids solely to make fabricated algebra checks practical. Exact production roster, 25 shifts, 10 classes, original-view index 12, state restoration, input/source pins, resource limits, and once-only launch admission belong to the caller. Passing these fixtures establishes neither a scientific result nor GPU bitwise equivalence.

## Main review addendum

The source/fixture hashes above describe the author's handoff. Independent
source review subsequently reproduced a lifecycle detail on fabricated tensors:
the last gradient call frees only the graph reachable from its objective.
Separate earlier I graphs may remain retained while H_T frees an R graph.
Main corrected the helper docstring to require saving detached values and
dropping all original objective/logit graph references before private actions.
No objective, derivative or numerical tolerance changed. The final main source
receipt records the post-review hash; this addendum preserves author provenance.
