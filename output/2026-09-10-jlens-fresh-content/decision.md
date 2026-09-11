# Fresh-content design and preparation decision

Codex — Spectral Optimizer Investigation · 10 September2026

## Concrete information gained this turn

The worktree source assessment (artifact not distributed in this public snapshot)
confirms that exact fit activations, a mean, a four-column basis and fit scores
are retained. Acquisition-time decoded input vectors are not. Both historical
PCA paths used float64; there is no source-level float32 PCA discrepancy.
That availability changes the next action: no PCA or old-fit forwards are
needed, but an explicit new matched decode is needed to identify the axes and
fit-example controls used in the new test.

Main and independent review selected three arms on one common new-content
ordering task. The [protocol](protocol.md) and readable mathematical plan (artifact not distributed in this public snapshot)
give the exact object/score definitions and the elementary regression-slope
steelman. No claim of novel theory, semantic clusters or causal explanation.
The existing useful examples and all negative results remain unchanged.

A fresh isolated text author supplied all24 prefixes; its task contained only
the topics, old prefixes and grammatical/novelty rules. It had no readout,
axis, score or hypothesis context and used no tools. Main copied the entire
returned panel unchanged. Main validation PASS:24 exact unique IDs/texts,
six per topic, distinct new terminal verbs,5–9words per prefix, no old prefix
or terminal-verb reuse;12disjoint pairs use all six topic contrasts twice.
This tests lexical transfer within authored familiar topics, not novel topics
or natural-corpus generalization. Data is frozen before any model outcome.

Current pins:

- dataset.json: `c655a5531af3215a29e332ddc6b23ac4854b0bffdf91b8ad5e8a045748ff7cd9`
- pairs.json: `8fa981cd7700b1385502e5b96ce451eebb9c34f6b93e05037203e9f314c1347d`

## Implementation validation

The implementation-check skill prompted current primary-documentation checks
before code preparation. [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
documents safe non-pickle loading and closing NPZ handles. Both are required.
[PyTorch](https://github.com/pytorch/pytorch/blob/main/torch/autograd/grad_mode.py)
distinguishes gradient/inference mode from eval mode and notes that inference
tensors lack version counters. Load/freeze the model explicitly and keep
weight-invariance checks compatible with the chosen mode; do not use absence
of gradients alone as proof that weights stayed fixed.
[Transformers](https://huggingface.co/docs/transformers/main_classes/model)
supports explicit revision, dtype, eager attention and local-only loading.
No new install, upgrade, download or alternative checkpoint is selected.

Read-only current runtime discovery, without importing model frameworks:
`/usr/bin/python3`, Python3.12.3; NumPy1.26.4, Torch2.11.0+cu128,
Transformers5.5.0, huggingface-hub1.8.0 from user site-packages. `jlens` is not
installed into that interpreter but its pinned sibling source exists.
Use that explicit source path for the fresh run if accepted; do not claim
the unrecorded historical interpreter has been recovered. The current lane
must record its own exact environment. No credentials were accessed.

## Current action and boundary