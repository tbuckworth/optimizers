# New-state-only raw-direction measurement protocol

9 September 2026. **Prospective; not a measurement launch receipt.** This is
the fixed readout specified in the [raw-direction acquisition protocol](protocol.md).
It does not change acquisition code, the policy, checkpoints, seeds or horizons.

## Admission and exact state roster

Wait for the complete five-seed new batch. Admit exactly seed 100–104,
`raw_norm_matched`, steps 1501/2000/2500: **15 new states**. Reject missing,
extra, reordered, failed or source-mismatched acquisitions. Verify the batch
manifest and completion, every seed completion and manifest, all nine artifacts
per seed and all three complete history prefixes. Check seed-100 admission
receipts, the original legacy parent/metrics/full-state identity and the
accepted archived action-batch binding. Hash every admitted file and recheck
at measurement completion. Repository HEADs may differ only when all pinned
scientific files remain identical; preserve each HEAD in provenance.

The new outer checkpoint schema names `raw_norm_matched` and its first-step
tensor receipt. The inner state retains the original `legacy` configuration,
source, split and parameter identities. Validate these separately, together
with exact legacy filter configuration/counter, finite state, CUDA topology
and the continuation's existing evaluation prefix. Never relabel the inner
state as if it had originally been trained by the new policy. Load only these
15 new checkpoint files, using the existing hash-bound, singly linked,
`weights_only=True` loader. No old model or first-step tensor is loaded for
inference; those files can be hashed for provenance.

## Unchanged measurement

Bind all imported old extraction, probe, symmetry, analysis and collection
helpers to accepted source hashes. In addition to the original representation
corpus's source contracts, verify the completed old 35-state measurement's
completion SHA-256
`fe11e1404e8ab20c792d8d441477f325f48a2bb9fdc4c868620a3ffeeafbb9ee`
and its manifest/source map. This uses receipt JSON, not old inference or a
repeat of old numerical analyses.

Use the original CUDA/Python/PyTorch/NumPy/GPU environment and recorded backend
flags; there is no CPU fallback, package change or precision adjustment.
Regenerate each seed's original split, IDs and 20 null permutations and prove
exact equality with the archived native-1500 **structural arrays only**. No
old logits, activations, fitted readouts or symmetry values are recomputed.
Retain source receipts and structural-array hashes.

For each new state, create one model copy and call the frozen extractor once,
using batch size 1024 on all 12,769 modular-addition pairs. It saves logits,
final-hidden and pre-attention activations; it does not call backward or an
optimizer. Check finite parameters/activations and unchanged extraction RNG.
Save the same structural arrays and raw activation members as the accepted
original recipe. From these new arrays compute:

- Original train/held-out CE and accuracy, checking existing step-2000/2500
  training-grid values within absolute tolerances `1e-4` for CE and `1e-6`
  for accuracy. Step 1501 has no extra training-grid reference and gets no
  replay to create one.
- All 56 frequencies, top-five fit-selected readouts, ridge 0.001 and all 20
  shuffled-row null selections on unchanged per-seed fit/evaluation halves,
  for final-hidden and pre-attention representations.
- The unchanged fixed panel `[9, 33, 32, 49, 11]`, correct-class margins,
  symmetry and training-membership excess via the original `analyze_state`.
  Undefined diagnostics remain undefined, not favorable zeros.

Persist raw NPZ, full scalar JSON and analyzed-state JSON for each of the
15 states. Bind all 45 output receipts and the manifest in the completion
record, which retains batch completion, prior recipe and source identities.
This stage produces per-state measurements only, not paired aggregate claims.
The primary comparison remains the archived norm-matched projected policy;
native/orthogonal are secondary context. No archived readout is rerun here.

## Execution envelope and failure policy

One new exclusive output beneath `/tmp/spectral-experiment-artifacts`, preferably a unique
measurement child of the completed new batch. Never reuse an existing,
partial or completed output. Reuse the frozen atomic exclusive output writer:
3 GiB maximum output plus at least 1 GiB free reserve. The uncompressed fixed
raw-array payload for all 15 states is analytically about 409 MB, before JSON
and archive framing; the 3 GiB ceiling is protective, not expected usage.

Before inference, main must verify a unique guarded service with 16 GiB RAM,
no swap, one CPU math thread / at most 100% CPU quota, and a 20-minute service
deadline. The reused collector additionally enforces a 19-minute cooperative
deadline and RSS/output/free-space checks. Bind this adapter, its fixtures,
this protocol and all unchanged imported scientific source files in a new
committed measurement snapshot. Do not mutate any acquisition source to
launch measurement.