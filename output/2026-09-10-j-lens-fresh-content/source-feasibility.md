# Fresh-content comparison: source feasibility

Assessment at worktree HEAD `0117df3a32c473d7124d589daa4b0d2ad2ac7b1c`.
No array values, PCA, model objects or scientific scores were read/computed.
Archive inspection read member names and exactly 128 header bytes for each of
the four relevant NPY members below. This is source availability evidence,
not a new experiment.

**Recommendation:** use the retained CPU-analysis basis and mean as explicitly
frozen objects for a new comparison. Once the common task is fixed, select fit
exemplars from fit geometry alone, then newly decode the eight signed PC
directions and at most eight distinct selected fit activations through the
same pinned layer-11 J-Lens pipeline. This creates a single identifiable set of
directions/readouts instead of assuming that old acquisition readouts exactly
match the later saved basis. No PCA refit or forward pass over the old fit
prefixes is needed. Fresh-content model forwards are a separate requirement
if the task needs new geometric ground truth; they have not been run.

## Objects actually retained

Paths in this table are under
[the completion acquisition](../2026-09-10-j-lens-completions/).

| Object | Archive member and header | Scope/certainty |
|---|---|---|
| Raw residual exports `h` | `features.npz/activation_11.npy`: `(48,1024)`, `<f4`, C order | Exact archived float32 exports for all 32 fit and 16 held-out rows; no complete forward/autograd state |
| Fit mean `mu` | `analysis_arrays.npz/activation_11_mean.npy`: `(1024,)`, `<f8`, C order | Exact saved **later-analysis** mean, not separately saved acquisition-time mean tensor |
| Four eigenvectors `V` | `analysis_arrays.npz/activation_11_basis.npy`: `(1024,4)`, `<f8`, C order | Exact saved **later-analysis** basis; acquisition-time basis was not persisted |
| Signed scores | `analysis_arrays.npz/activation_11_scores.npy`: `(48,4)`, `<f8`, C order | Includes all fit scores; source defines `(h.astype(float64)-mu) @ V` |

Dataset JSON (artifact not distributed in this public snapshot) fixes row order.
Fit IDs are `{astronomy,cooking,football,programming}-{0,1,2,3}-{plain,note}`,
in topic, pair-index, then plain/note order: 32 rows but 16 content pairs.
Held-out indices 4/5 are a different eight content pairs. Never infer a split
from matrix row position alone without checking this pinned ID ordering.
Inputs JSON (artifact not distributed in this public snapshot) retains the aligned
token IDs and completion boundary. `h` is the layer-11 post-block residual at
the final prefix position, not a parameter gradient or a completion average.

[Readouts JSON](../2026-09-10-j-lens-completions/readouts.json) contains the 16
held-out individual directions, one fit mean, PC1–4 both signs and four random
axes both signs, under J-Lens/plain at each saved feature kind/layer. It retains
12 token IDs/strings/scores and pre-normalization norms per direction. **No fit
individual readouts, acquisition basis, or actual lens-input tensors are saved.**
Norms and truncated rankings do not identify a 1024-dimensional direction.

## What “same PCs” does and does not establish

[Acquisition source](../../experiments/j_lens_completions.py), `pca()` and the
readout loop: PCA first casts its input to float64, centers the fit rows,
diagonalizes the 32-by-32 Gram covariance, lifts four columns, then fixes each
sign by its largest-magnitude coordinate. The readout loop concatenates those
columns, their negatives and the other controls, normalizes them in float64,
then creates float32 Torch input vectors for J-Lens.

[Analysis source](../../scripts/analyze_j_lens_completions.py) first casts the
saved float32 features to float64 and calls **that same `pca()` function**. The
second cast is not a second lossy conversion: both routes do their mean/Gram/
eigensystem arithmetic on the same float64 representations of the exports.
There is no source-level float32-versus-float64 PCA difference here.

However, acquisition and later analysis were separate processes. Their exact
NumPy/LAPACK/build/thread environment was not pinned together, and the actual
acquisition eigenvectors were never stored. Thus the source recipe agrees;
**bitwise identity is not certified, and even approximate acquisition-versus-
analysis agreement has not been directly measured**. The expected numerical
agreement is an inference. Near ties could also affect sign/order or basis
orientation; no eigenvalue-gap calculation was done in this assessment.

[Self-check source](../../scripts/check_j_lens_completions.py) and its
PASS receipt (artifact not distributed in this public snapshot) compare the
later saved subspace to an SVD and reproduce centroid predictions. Subspace
overlap is insensitive to signs and rotations and does not compare the
acquisition basis or the decoded vectors. It cannot close this identity gap.
Even reproducing the old top-12 rankings later would validate a readout outcome,
not uniquely prove equality of the input vectors.

## Minimal new decoding and strict constraints

Use only `activation_11`, the pinned mean/basis, and the fit `h` rows. For each
of four axes, a concrete exemplar rule is largest/smallest **fit** signed
projection, with an ID tie-break; retain repeated content/framing choices
rather than replace them after inspecting token quality. This selects up to
eight unique existing vectors. Their exact IDs have not been selected here.

Freeze both the float64 source basis and the **realized normalized float32
lens-input tensors**. If the ground-truth task is directional projection, use
those realized signed-axis vectors for fit/fresh projections, or explicitly
record the precision distinction; do not silently mix a new rounded direction
with old float64 scores. This is a projection, not another eigensystem fit.
Save the normalization and dtype recipe, array hashes, selected IDs, actual
input tensors and decoded outputs together. Individual exemplars use normalized
raw `h`, including the example selected at the negative PC end; never `-h`.

The minimal matched readout set is eight signed PC lists plus no more than
eight unique fit-example lists, J-Lens only, twelve ranks per list. The model
must expose the same averaged Jacobian, final normalization and unembedding;
this is new decoding, even though fit activations already exist. It requires
no fit transformer forwards or gradients. Plain-lens/mean/random expansions
are unnecessary for the specified spectral-versus-exemplar question.

Select and freeze all summaries/exemplars before fresh evaluation content is
used to choose anything. Existing held-out rows, their reviewed endpoints and
the later qualitative display are not fresh content and must not select fit
controls. Do not use topic labels to improve exemplar selection or adapt the
readout budget to observed fluency. Both arms receive the same new downstream
question and target inputs; main owns that design. A shared fit pool still
does not make an axis and an exemplar the same object.

## Available fixed resources and consumed handles

- Model: `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`; lens revision
  `0731326edff4ae730ffc5356fe1a4728c748b3a6`, lens SHA-256
  `aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48`.
- Snapshot directories exist under `/private-artifacts/storage/cache/huggingface/hub/`
  in `models--Qwen--Qwen3.5-0.8B/snapshots/<model revision>` and
  `models--neuronpedia--jacobian-lens/snapshots/<lens revision>`. Metadata checks
  confirmed the model shard and the pinned
  `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`
  targets exist; no weight/checkpoint contents were opened or rehashed.
- Sibling source `/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/jacobian-lens`
  remains at `581d398613e5602a5af361e1c34d3a92ea82ba8e`. Its `lens.py:135`
  transports row directions as `h @ J.T`; `hf.py:166` casts to the head dtype,
  applies final norm, then unembeds. Original loading requested bfloat16 weights.
  A sibling `.venv` does not exist; this check did not locate or certify the
  original interpreter. The earlier pilot's Torch version alone is insufficient.
  Subsequent read-only checks also found no `.venv` in the worktree or sibling
  source and no interpreter path in the acquisition/launch/source references.
  The exact old user-unit journal retained only systemd start/stop metadata;
  checking executable/command-name fields revealed no Python path. This is a
  recorded historical limitation, not a blocker to a freshly recorded runtime.
- Acquisition source freeze: `edccfb199daf6af0f7fc88c2ab566795105e3f5e`.
  Old launch receipt (artifact not distributed in this public snapshot):
  unit `j-lens-completions-20260910-MMxZIs.service`, PID `3392543`, invocation
  `a3b3c96cce71440ab2e9d01bdb12e4f8`, completed and garbage-collected. This handle
  is consumed, not a runnable continuation or a new resource authorization.

## SHA-256 source pins

These JSON/source hashes were checked in this assessment. Archive hashes are
inherited from the frozen packet provenance (artifact not distributed in this public snapshot),
not recomputed by reading archive payloads here; reverify before any future
authorized array load.

| Path | SHA-256 |
|---|---|
| `experiments/j_lens_completions.py` | `ae04db8770538d278fca32b58542e8bb653f42785de39c2f2bf76d7a8fffe238` |
| `scripts/analyze_j_lens_completions.py` | `21286ebae694095ee160cef98d95fa04902b6ffa4b1d472f9d5642435913b0c4` |
| `scripts/check_j_lens_completions.py` | `112b23e0eb7839574a7bc12adedca602a43578a9cdcdd1d2840847bda2484259` |
| `output/2026-09-10-j-lens-completions/dataset.json` | `b41b91381c1bfd6521ceb0b5d15c35ea28e0929d8eba57527b232a933a85cdd6` |
| `output/2026-09-10-j-lens-completions/readouts.json` | `3f37f8918cdeea67ebeae12d3a33ef283f9b55baf7153e9384a6f1e3a9ab5769` |
| `output/2026-09-10-j-lens-completions/acquisition.json` | `1568b73f777adf15dfbce0f17941678f22b90ce7a19fce257e4edfe4b5e0fd2b` |
| `output/2026-09-10-j-lens-completions/inputs.json` | `335e2c8a9edb029216ff679fc28e5d4db8e81d3be49c597bba1934a8d02bf271` |
| `output/2026-09-10-j-lens-completions/features.npz` (inherited pin) | `96e936942fd5cde63c9c482ef19a004ba3fee5adbc46676373772fa5e4123b61` |
| `output/2026-09-10-j-lens-completions/analysis_arrays.npz` (inherited pin) | `cbbe651ed084b36ed468a5932afcbc7e488c81ccdcc0fcd04bccf2fe9f6e7033` |