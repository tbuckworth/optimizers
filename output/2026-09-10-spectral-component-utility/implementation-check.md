# Framework check before inert implementation

Codex — Spectral Optimizer Investigation · 10 September 2026

Scope: strict restoration plus signed-objective helpers and fabricated CPU
fixtures. This check admits neither a real parent load nor an experiment.
The best-practices-validator skill prompted explicit primary-documentation
checks; it did not add a user-approval gate or change the research question.

## Aligned choices

- PyTorch 2.11 is the original source environment; use versioned references,
  not the moving stable documentation alias. Restricted `torch.load` with
  explicit CPU mapping is appropriate for tensor/primitive snapshots. Never
  broaden to an unrestricted pickle fallback for this schema.
  [Official loading API](https://docs.pytorch.org/docs/2.11/generated/torch.load.html).
- Validate parameter identity/order/shapes before optimizer restoration:
  state loading associates saved IDs and parameters by order rather than
  verifying their identity. [AdamW state loading](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html#torch.optim.AdamW.state_dict).
- Obtain diagnostic derivatives without changing the parent's saved `.grad`:
  `autograd.grad` returns them without accumulation. Retain a shared graph
  only for the needed objectives, free it after the last derivative, and do
  not construct higher-order graphs. [Autograd API](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html).
- Cast caller-provided FP32 logits to FP64 for objective arithmetic without
  detaching their graph. Use stable log-sum-exp or log-softmax operations;
  direct exponentiation is unnecessary. [Log-sum-exp API](https://docs.pytorch.org/docs/2.11/generated/torch.logsumexp.html).

## Gaps the code and fixtures must close

Restricted loading does not solve memory denial of service; fixed input byte
checks, tensor/schema checks and process limits still matter.
[Serialization limits](https://docs.pytorch.org/docs/2.11/notes/serialization.html#torch-load-with-weights-only-true).
Our specific strong snapshot's compatibility and equality are implementation
questions, not guarantees supplied by the framework documentation.

Stable log-sum-exp does not by itself prevent cancellation when subtracting
large terms. Test large shared logit offsets, the differentiated identity and
the zero-consistency one-view case using an independently computed direct CE.
Keep mean-logit objectives distinct from mean-per-view accuracy and CE.

Test restored modes, parameter gradients, tracker ownership, RNG isolation and
all inherited Adam counters/moments. Preserve the actual rounded FP32 path
point and use its FP64-subtracted displacement. No fabricated test can certify
the old model's predictions or real GPU restoration before admitted replay.

## Selected implementation boundary

New import-inert helpers and tests only; old scientific sources remain frozen.
No CLI acquisition path, dataset read, scientific checkpoint load or paid call
is selected here. A source-reviewed runner, saved-array audit, exact output/
object-lifetime inventory and current resource admission still precede a run.

## Pure panel helper validation

Before implementation, also checked NumPy 1.26's
[generator interface](https://numpy.org/doc/1.26/reference/random/generator.html),
[permutation](https://numpy.org/doc/1.26/reference/random/generated/numpy.random.Generator.permutation.html)
and [integer sampling](https://numpy.org/doc/1.26/reference/random/generated/numpy.random.Generator.integers.html).
Use explicit PCG64/SeedSequence streams, with integer high bound exclusive
and the protocol's int8 dtype/shape fixed. The selector must not mutate global
RNG, balance labels or change the supplied role order. Role partition and
array-shape validation supplement, not replace, original-plan byte verification.

## Runner, independent audit and guard integration — 10 September, 19:20 UTC

The initial helper-only implementation boundary above is historical. The next
selected stage now adds an import-inert producer, numeric saved-array auditor
and runtime guard. Their default CLIs do no science. Main and independent
source review plus fabricated integration fixtures precede any actual launch;
no real-data smoke test is used to calibrate tolerances or drop difficult cells.

Versioned [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
does not guarantee identical results across releases/devices. Pin the original
2.11.0+cu128 environment and all flags, and test original reporting predictions
exactly at acquisition before permitting a local action. A mismatch is failure,
not permission for a different model, device, batching rule or tolerance.

The [allocator fraction API](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
limits the PyTorch caching allocator, not every CUDA context/driver allocation.
The4GiB setting is reported with that boundary. Separately check free GPU memory
and existing clients; enforce8GiB/no-swap host and one CPU through cgroup v2.
Time and output caps are enforced during serialization, with partial files
retained and an exclusive bounded failure footer. Runtime admission must use
the pinned named unit and invocation rather than a free-standing GPU shell.

[NumPy1.26 loading](https://numpy.org/doc/1.26/reference/generated/numpy.load.html)
supports explicit pickle rejection and header-size limits. The
[NPY/NPZ format](https://numpy.org/doc/1.26/reference/generated/numpy.lib.format.html)
and [1.26.4 implementation](https://github.com/numpy/numpy/blob/v1.26.4/numpy/lib/format.py)
support numeric header inspection and pickle-disabled array writing. Additional
application checks bound declared shape/payload bytes before allocation,
reject symlinks/nonregular input and use nonblocking descriptors. The guard
uses context-managed ZIP/NPY writes so failed NPZ serialization closes cleanly.
No unrestricted pickle, object arrays, compressed inflation or restart fallback.

Integration tests compare actual fabricated CPU canonical actions against the
independent NumPy Adam/projection/path checks at both registered stages, and
Torch objective metrics against independent NumPy reductions. These test
implementation agreement only, not real GPU fidelity or a scientific effect.
