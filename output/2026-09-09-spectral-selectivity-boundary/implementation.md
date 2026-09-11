# Selectivity-boundary implementation readiness

9 September 2026. **Source and fabricated-fixture preparation only. No scientific
acquisition, MNIST read, archived checkpoint read, CUDA workload, or experiment
launch was performed by the implementation agent.** Main owns source review,
commit/freeze, the single guarded launch, and independent saved-evidence audit.

## Sources and preparation checks

- Runner: `experiments/spectral_selectivity_boundary.py`, SHA256
  `c851cb711e85a07aacad891cfd1122d5bf1a3f7d406142d1b98cca4cb2afac0f`.
- Fixtures: `tests/test_spectral_selectivity_boundary.py`, SHA256
  `a4c2f6385747211df41d1d32d7ac711b7563af057db1de75f78e37eda98d2ccc`.
- Reviewed protocol bytes at preparation:
  `2666a79a9afd6baca82a463ae793c62eeb76e2910c57d8b9a89f01d00ddb4d0a`.
- Unchanged I9 core: `706851828b7120bea0d7a41bd0fa75474ac800b2df71087941d0140c8d1970ec`.
- Unchanged canonical filter: `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`.

The valid fabricated-only invocation was:

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -p test_spectral_selectivity_boundary.py -v
```

All **22 tests passed**, 1.002 seconds unittest runtime (2.485 seconds command
wall time). A preceding dotted-module unittest invocation failed before test
import because `tests/` is not a Python package; discovery corrected that command.
There was no failed scientific acquisition or retry. `git diff --check` passed.
Fixtures use only invented labels/pixels, a 55-parameter CPU network and temporary
fixture files; CUDA is hidden and core RNG snapshots are patched to CPU-only.

Coverage includes independent local RNG/split pairing; absent rare examples in
warmup and no rare corruption; same Shared/Sham targets; true-class/poison-balanced
sham allocation and deterministic ties; patch-copy immutability; all byte values
in the explicit CPU NumPy FP32 normalization; fixed probe ordering; finite CE and
empty/undefined semantics; majority macro versus frequency weighting; matched
nonzero patch excess; three-seed contrasts/interactions/SD/SE/signs; patched and
per-class summary coverage; exact warmup restore and initial plain-dict hashing;
raw-observe-delivery-Adam ordering; norm-law FP64 scaling/FP32 cast and zero cases;
native identity fallback; autograd.grad side-effect preservation; actual update
diagnostic evidence and finite probes; ordinary Adam moment/counter advancement
under explicit zero gradients; capped streaming writes/receipts/exclusivity;
resource rejection and explicit-execute admission.

The researcher experiment instructions informed evidence preservation and
reproducibility. User scope overrides its fresh-seed replication/fail-fast gates:
this is the fixed 36-trajectory protocol, not an additional rerun workflow.

## Fixed scientific implementation

The runner imports inertly. Its only scientific entry requires `--execute` and a
new exclusive `acquisition-001` under an otherwise empty parent on the mounted
`/private-artifacts/storage` volume. It rejects any wrong unit or limits; requires exactly
`spectral-selectivity-boundary-001.service`, Type=exec, Restart=no,
RuntimeMaxUSec=30min, KillMode=control-group, one CPU quota, 16 GiB memory and
zero swap; checks four math-thread environment variables; pins Torch
2.11.0+cu128; requires the local RTX3090 and an 8 GiB CUDA allocation ceiling.
The cooperative batch deadline is 25 minutes. Source, test and protocol bytes
must match the recorded Git HEAD; canonical/core hashes and accepted original
training IDX hashes are independently pinned. No official test file is read.

Exactly three seeds × four cells × three policies are executed, with one shared
100-update majority-only clean warmup per seed. Every fork is restored and hashed
before raw policy drops its observer. Native and norm-raw policies observe their
own raw gradient once. At update101, all three policies must share the raw
gradient hash within each cell; both observed policies must share the post-raw-
observation tracker hash. Policy order rotates deterministically, without changing
the fixed scientific roster. There is no checkpoint selector, CPU fallback,
resume, retry, tuning or outcome-dependent control path.

Pixel normalization is explicitly **CPU NumPy**:
`images[ids].astype(np.float32) / np.float32(255)`, contiguous before device
transfer. Patches set exactly the upper-left 3×3 values to FP32 one in a clone.
Source inputs are checked unchanged. This avoids depending on bitwise agreement
between CPU division and an independently implemented CUDA normalization kernel.

The norm-raw control computes the native norm on its own current observer/state,
then scales raw direction in FP64 and casts back. It does not match another
trajectory's future norm sequence or the actual Adam displacement. Explicit zero
delivery remains a real optimizer step. Raw/native/applied norms, cast mismatch,
rank and observer-step scalars are saved at every continuation update.

## Artifact schema for independent audit

Root JSON schema is `spectral_selectivity_boundary_v1`; receipts contain relative
direct-child `path`, `size_bytes`, and SHA256. `manifest.json` binds source/data,
Git commit, limits, runtime identity, roster, environment versions and byte budget.
Successful `complete.json` binds `results.json`, every earlier artifact receipt,
source/data pins, resource high-water marks and exact trajectory/diagnostic counts.
A failure preserves partial files plus a reserved bounded `failed.json`; no retry
path is implemented. Pre-output admission failures are retained in the service
journal, not represented as a completed acquisition.

Per seed:

- `plan-sSEED.npz`: train/heldout source IDs and true labels, warmup and continuation
  batches, diffuse selected mask/replacement labels/targets, poison mask,
  Shared/Sham targets and sham patch mask. Six PCG64 streams use
  `SeedSequence([seed, stream_id])` in the protocol's named order. Arrays are class-
  blocked by true digit. `plan-sSEED.json` binds each array and sham rounding rows.
- `warmup-sSEED.pt`: complete I9 model, `.grad`, Adam, observer and RNG state.
  `warmup-actions-sSEED.json` retains all 100 raw-observed warmup updates.
- `common-heldout-sSEED.npz`: saved unpatched/patched logits at steps0/100,
  reused exactly in every branch history.

Per seed/cell, `binding-sSEED-CELL.json` includes plan/warmup/common-prediction
receipts, initial model hash, full warmup tree hash, array hashes, target and patch
mask hashes, actual constructed train input hash, clean train input hash and both
heldout input hashes. Array hashing includes dtype, shape and contiguous bytes.
The binding also saves patch × poison × true × assigned counts and overlap counts.

Per seed/cell/policy:

- `logits-sSEED-CELL-POLICY.npz`: steps `[0,100,...,2000]`; `train`,
  `heldout_unpatched`, `heldout_patched`, each FP32 `[21,5000,10]`.
- `actions-...json`: all 1,900 continuation scalar rows, fixed order 101–2000.
- `final-...pt`: full endpoint state for every one of 36 trajectories.
- `curve-...json`: common binding plus 21 metric rows, sufficient counts and
  FP64 CE sums, all relevant receipts, first-action identity and runtime scope.
  Timings include evaluation and native-only diagnostics/serialization, so are
  descriptive and are **not a fair optimizer-speed comparison**.

For native only, `diagnostic-sSEED-CELL-native32-uSTEP.pt/.json` at101/500/2000
gives **36 anchors**. Tensor payloads contain full before/after states; raw/applied
training gradients; parameter order; each probe's training positions/source IDs,
actual inputs, targets/true targets, FP32 per-example gradients, and pre/post
logits. JSON summaries bind both full-state hashes and the tensor receipt, plus
true-label counts, coherence/retention, span diagnostics, actual movements,
finite CE changes, signed local utilities and explicit side-effect PASS checks.

`results.json` retains all 36 fixed endpoints and corresponding warmup rows.
Summary keys preserve rare/common/balanced unpatched CE/accuracy, distinct patched
equivalents, all ten classes for unpatched/patched/training truth, assigned/wrong
training fit, and all three cue populations. Every metric flows to per-group
mean/SD/SE/sign summaries, native-minus-raw/native-minus-norm contrasts, and change
from warmup. Absent clean wrong-example fit remains null. Primary excess and
secondary ASR interactions are explicit; the raw Shared-minus-Sham continuous
assay is retained. No p-values or selected checkpoint are introduced.

## Numerical diagnostic conventions and limits

Coherence uses FP64 accumulation of saved FP32 per-example gradients. Native
retention uses the actual canonical FP32 `V(Vᵀg)` action, including identity
fallback. For mean-action retention, the gradient mean is formed in FP32 and
its own squared norm is the denominator; the separately reported FP64 mean
energy is not silently substituted. Wrong-minus-true residuals are subtracted
in FP32 and this dtype is explicit. Zero denominators are null with reasons.

The numerical span comes from FP64 thin SVD of saved V; keep singular values
above `eps64 * max(V.shape) * sigma_max`. This Q is diagnostic only. Actual total
movement is saved-after minus saved-before. Decay first applies the AdamW
FP32 multiplication by `(1-lr*wd)`; adaptive movement is saved-after minus that
rounded intermediate, so components sum exactly in FP64. Local signed utilities
use pre-update saved probe gradients dotted with these actual movements.
Probe computation verifies exact preservation of parameters, model modes,
training `.grad`, optimizer, observer and all captured RNG state.

Wrong probes are the first 32 changed examples in class-blocked order: a fixed
convenience subset, often an early true class, **not representative** of all
wrong examples. Their true-label counts are reported. Diffuse versus Shared/Sham
coherence is not a controlled comparison of corruption severity or target mix.
Rare digit 8 difficulty and rarity remain entangled. Saved diagnostics do not
establish heldout local utilities, future subspace angles, or long-horizon
mediation; low cue response must be interpreted with actual common/rare learning.

## Conservative pre-acquisition byte inventory

| Saved component | Upper bytes |
| --- | ---: |
| Three warmups + 36 full endpoints | 296,303,592 |
| 36 diagnostic anchors, full pre/post state | 547,022,016 |
| 4,032 per-example FP32 gradients | 820,753,920 |
| Raw/applied training gradients at 36 anchors | 14,656,320 |
| Probe inputs/targets/pre/post logits | 13,031,424 |
| 36 complete three-way logit histories | 453,600,000 |
| Shared initial/warmup heldout logits | 2,400,000 |
| Plans, scalar JSON, ZIP headers and metadata allowance | 268,435,456 |
| **Total conservative upper estimate** | **2,416,202,728** |

The estimate is approximately 2.250 GiB and leaves 805,022,744 bytes under the
3 GiB ceiling. Each full snapshot allowance assumes a rank-32 observer and full
gradients, even for raw endpoints or cleared-gradient warmups where that
overcounts. The 4,032 gradient rows are 9 clean anchors × 64 rows plus 27 other
anchors × 128 rows. Extra scalar/metadata allowance is 256 MiB. Capped streaming
writers enforce actual output bytes, not merely this estimate. `RESERVE_BYTES`
is a separate 1 MiB failure/footer allowance. The **free-disk reserve is 1 GiB**,
checked at admission in addition to estimated output and during execution.
No evidence removal or resource increase is required for this implementation.
