# Pattern calibration: design and metadata/source preparation review

11 September 2026 · no new corpus/model/decoder/reader result

## Decision

Proceed with bounded preparation for the [protocol](protocol.md), SHA
4d80a9eef24e32a475c1d5016f57ca05419cacb45f2b27504f9ca7f916fae762.
Root selected pair-difference calibration because the intended use predicts
within-topic pair scores. The score basis stays fixed; raw-text PCA is not
refitted. This is a more specific estimand than the earlier pooled-C candidate,
chosen before candidate ordering or new article content. Unit pair weighting,
zero-only variance cutoff and original one-shot topk convention are explicit.

The primitive's algebraic fit is not the scientific success criterion. Regression
can be dominated by large gaps; later credit weights each pair equally. A
held-out reader gain is unknown and necessary for the proposed utility claim.
The unbalanced captured catalogue is not general prose or four-topic balance.
No old result, reference string or model state is changed to favor this test.

## Saved-source feasibility and independent root reconstruction

Existing user-authorized Astra worker commit
0fe7170863f79db056457da3dd1b30708f363230 contains:

- `output/2026-09-11-j-lens-pattern-calibration/feasibility.md`, SHA
  da084bf217b81b6fff2855496288af8284e4f781b2c66d0e069f2efe12613ffd.
- `output/2026-09-11-j-lens-pattern-calibration/inventory.json`, SHA
  e57b194d6ed0b6b0999097d64c12dc37198f7b73d9b27bb5c2dd31e51a54f7aa.

Root read the complete feasibility note and independently rehashed all146
metadata/source bindings. It reconstructed exclusions directly from63 raw
article-request records, including one copied event, and first-root ownership
from the catalogue. The62 excluded IDs and remaining40/26/0/107 by topic match
the inventory. Scoped discovery found no unrecognized request file in main or
worker J-Lens output directories. No article response text or numerical archive
was parsed, no weights loaded, and no old inventory stage rerun.

The metadata capacity is173 unused articles/86 disjoint within-topic pairs;
technical eligibility remains unknown. The plan fixes58 calibration and28
evaluation candidate roles, targeting32/16 accepted pairs without role changes.
The source surplus is not a guarantee of96 usable leads. Calibration and
evaluation must remain article-disjoint, including technical rejects from old
studies. No candidate ordering existed when this review was written.

## Mathematical implementation and tests

`patterns.py` SHA524efc43e54c9549e66586f8a8dcc9db8542979637a56d840c3739f1c156a789
is a pure NumPy primitive, not a model/array-loading/stage entrypoint. It
requires complete pair coverage, FP32 finite features/basis and unit basis;
it accumulates FP64 cross-products and score energy, preserving every axis.
No finite positive variance threshold, regularization, clipping or adaptive
pair weighting is introduced. The future caller must validate1024×4 basis,
calibration-role IDs, exact source hashes and scientific split before invoking
it; the primitive itself does not certify those bindings.

Thirteen fabricated tests pass independently in root and Astra: dense second-
moment identity, symmetrized population, pair reversal/order, translation and
scale, unchanged score orientation, signed FP32 display, large-gap influence,
low-positive-energy retention, zero-axis failure, input/coverage validation and
nonmutation. Raw d satisfies uᵀd=1; its normalized display vector does not have
that unit-score amplitude. Root corrected an initial planner filename before
any actual stage to avoid shadowing Python's standard `select` module.

`plan_candidates.py` SHA8d97d6b8412a6dd2f21a1532713d966603debc87e2df5f240f8479726eb35b5b
independently checks metadata sources/request discovery, reconstructs ownership
and exclusions, then applies the exact fixed hash/pair/modulo3 role rules.
It has no network or model dependency. Seven fabricated tests pass in root and
Astra, covering complete coverage/unused tails, roles/IDs, input-order
invariance, exact hash recipe, ownership-before-exclusion and invalid metadata.
Astra did not invoke the actual inspection or ordering stage. Root's separate
read-only actual-inventory check passed before any selection write.

Combined root fixture command: CUDA hidden, OMP/OpenBLAS1, no bytecode writes,
`python3 -m unittest discover -s output/2026-09-11-jlens-pattern-calibration -p 'test_*.py' -v`.
All20 pass. These tests establish implementation properties, not useful model
descriptions or a held-out effect. The metadata manifest must be frozen before
the future collector is allowed to request new content.

## Decoder and documentation review

Root and Astra inspected the exact existing lens/adapter code. Float32
transport precedes a BF16 cast, final RMS normalization and the head; output
`.float()` does not restore lost precision. Original examples/tokens remain
byte-exact and are not decoded again. The pinned local lens load uses
`weights_only=True`, with no network-capable fallback. An actual future load
must verify its historical content digest, shape/layer/finiteness and cached
model identities; metadata/stat inspection is not such a load.

The best-practices-validator guidance informed a check against version-specific
primary docs: [PyTorch2.11 topk](https://docs.pytorch.org/docs/2.11/generated/torch.topk.html)
confirms tied indices are not guaranteed stable; the protocol captures one
call, retains ties and never rerolls them. [PyTorch2.11 load](https://docs.pytorch.org/docs/2.11/generated/torch.load.html)
documents restricted tensor loading and CPU map location. Future NumPy archive
reads must use `allow_pickle=False` and close the archive, as documented in
[NumPy1.26 load](https://numpy.org/doc/1.26/reference/generated/numpy.load.html).
[Python3.12 JSON](https://docs.python.org/3.12/library/json.html) supports the
duplicate-key/nonfinite guards; this metadata-only planner rejects floats.

## Independent verdict and remaining work

Astra independently read the whole protocol, mathematical source and tests,
then the planner/test source, and returned PROCEED/no material blocker for
preparation. No fresh reader or child agent was called. The read-only review
does not admit model work: new collection code, exact accepted inputs,
tokenizer, split-enforcing calibration/decoder and evaluation callers, judging
transports and saved-array checks remain to implement/review in order.

The continuation/research guidance was used for evidence-based next action,
construct validity and independent checks, without restarting the interactive
workflow or imposing a new user-approval gate. Existing J-Lens-only authority
and resource limits govern; goal/timer stay active, paid spent/reserved$0/$0.
