# Neural clustering implementation handoff

## Fixed implementation

New files only in the isolated `clustering-anchor-pilot-20260909` worktree:
`experiments/anchor_graph_mnist.py`, `tests/test_anchor_graph_mnist.py`, this note
and `protocol.md`. The frozen synthetic pilot, canonical spectral source and I9
model/Adam/snapshot helpers remain unchanged. Main accepted the prospective
24-trajectory roster before coding assumptions were hardened.

The principal design judgment was to retain the exact synthetic pilot's
normalized-profile anchor-graph and cluster-mean action, but feed it the same
centered stable rank32 covariance factor as hard32. This is an explicitly
documented neural adaptation, not a claim to reproduce the pilot's uncentered
streaming estimator. `F=V diag(S)` uses singular values correctly. The fixed
half-identity mixture steelmans the risk that literal equal-coordinate group
updates are too restrictive; it is not selected from outcomes or norm-matched.

Main's independent launch guard supplies systemd16GiB/no-swap/1CPU/30-minute
limits. Runner checks a29-minute deadline every step, refresh and cell, caps
serialized output2GiB and GPU allocation8GiB, and maintains1GiB disk reserve.
No resume/retry/search mode exists. Real runtime remains unmeasured.

The reviewer requested and source now includes accepted original IDX hashes,
all19 membership vectors per clustering trajectory, exact first-action gradient
and observer bindings, cluster sizes/isolate fraction/effective projector rank,
refresh timings and runtime contraction/orthogonality checks. Final completion
also records host high-water RSS and peak allocated GPU bytes. Expected output
is below0.6GiB; no parameter-by-parameter matrix is constructed.

## Validation and corrected fixture issue

Only tiny synthetic CPU fixtures are allowed for this implementation handoff.
They cover deterministic plans/disjoint splits/fixed replacement, correct factor
scaling, self-adjoint/idempotent cluster projection with isolate passthrough,
half-strength complement retention, permutation-invariant stability, actual
changed-label denominators and clean-condition nulls, finite-logit rejection,
canonical hard-step equivalence and model/Adam restoration, all4 first-gradient
and3 first-observer equality, graph refresh schedule/history hashes, and exclusive
capped JSON/NPZ/tensor round-trips.

Main's first8-fixture run found a real packaging compatibility bug before any
acquisition: NumPy1.26.4 `zipfile_factory` recognizes file-like objects via a
`read` attribute. The bounded writer originally exposed only `write/flush` and
was wrongly treated as a path. We inspected that installed function and added a
minimal forwarding `read`; ZIP writes still pass through the original byte cap.
The same NPZ fixture then passed and verified contents. No cap was relaxed.
An initial leaf test invocation used a nonexistent importable `tests` package;
unittest discovery is the correct invocation and is listed below.

Final validation command:

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -m unittest discover -s tests -p test_anchor_graph_mnist.py -v
```

Nine tiny synthetic tests pass. No cached MNIST file was loaded by these tests.
The existing pilot acquisition or its old fixtures were not restarted.

## Primary-documentation implementation check

The best-practices-validator guidance influenced preservation of parameter order,
deep-cloned optimizer state and explicit `foreach=False,fused=False`. PyTorch
documents that loading optimizer state matches parameter IDs by ordered pairing,
not by additional identity verification; the fixture therefore checks actual
model/Adam/full-state hashes at forks. See
[PyTorch2.11 AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).

RNG plans are independent of branch order and all forks restore common RNG.
Deterministic algorithms and disabled TF32 are fixed. These controls do not imply
cross-platform bitwise equivalence; PyTorch explicitly cautions about different
releases and CPU/GPU execution. See
[PyTorch2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
The mathematical usefulness of cluster projection is a hypothesis, not a
documentation-supported fact.

## Source hashes handed to main

- Runner: `a2e47651cada793619e5b6e7ef1646e6c553c93ba1c8898f45a9681c3e7dcde4`
- Fixtures: `41c75156b53b7f65a6d0c5dcc935799b71f29f3c4f3dfa1408b7e5cf40f30b38`
- Protocol: `0dc04e3efa96666501975d48b33f98f4248cdfd60aa4ff8789e7e05e54e56839`
- Unchanged canonical spectral: `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`
- Unchanged synthetic graph: `2581010abe529d1b3da2af0a71106710dcc59b81ec03e58b93b56f2f4790f89e`

The runner's `source_pins()` also binds imported I9 `neural_core.py`. Main should
recompute the complete source manifest at freeze. Do not edit any pinned source
once acquisition starts. A source or fixture correction, if necessary, must be
reviewed before a fresh authorized launch, never applied to a live acquisition.
