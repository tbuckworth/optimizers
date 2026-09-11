# Bounded independent design and source review

Codex — Spectral Optimizer Investigation · 10 September 2026

**PASS for the reviewed design and implementation.** This is a source review,
not acceptance of scientific outputs. No scientific state, tensor, image,
gradient or logit archive was opened, and no acquisition or audit was run by
this reviewer. Main reports 29 fabricated CPU fixtures passing; the fixtures
were read here, not independently rerun.

Reviewed the complete protocol, parent receipt registry, implementation note,
collector, mathematical helper, independent NumPy auditor and their fixtures,
plus the relevant unchanged snapshot/restore, filter and audit dependencies.

The six-parent question is a useful local scale-versus-direction discriminator.
Both matching directions, absolute useful-loss responses, the shared decay
baseline and the smaller total-displacement path preserve a favorable native
case without presuming it. Four translated draws are averaged within each
seed/parent only when all required draws are available. Different warmups
change model, Adam and observer together; neither their comparison nor a local
utility difference establishes the cause of the full training outcome.

Resolved before scientific execution:

- Norms use explicitly rounded FP32 data displacements and FP64 norm evaluation;
  utilities use the realized materialized parameter displacement. Decay-only is
  not zero-gradient Adam.
- Each native observer and Adam proposal starts from a private parent copy;
  observer and optimizer counters advance exactly once. Original state hashes
  are checked again after measurement.
- Adam reference error scales with the small data displacement, not the large
  parameter vector. The finite-grid identity uses the actual recorded
  `QQᵀ` operator, including its Gram matrix, without reorthogonalization.
- The auditor now pins its reused independent helper, checks archived parent
  paths as ordinary bounded direct files, and implements the canonical
  rank-zero identity fallback. Fixture import/dummy-basis defects were corrected.

The independent auditor reconstructs saved-array arithmetic, plans, metrics
and contrasts. It does not independently regenerate autograd, logits, observer
history or the producer's restored-state digests. Exact trusted parent hashes
and reviewed restoration source remain part of that evidence boundary.

Reviewed source SHA-256 values:

- Collector: `a714a0a4bcd228e6364e9636e39b12b914d31c75f3004bd60f719131da78accb`
- Core: `2acdf99bdf01fb95588e7faaf6bb3363d8a00acd4c960742e0bd2453fd4af56d`
- Auditor: `7d3838fd004644e27bd67a4724a0b52acf8d4736ae84306260ce94bdcfd923fe`
- Protocol: `1b93e4ea554583ddaeffdddc04c11dce8d57063f6ec3e3b1dad23d8bf68b7805`

Main owns final source freeze, current resource/handle admission and the
once-only acquisition and independent audit. No additional arm, sweep or
scientific approval gate is requested by this review.
