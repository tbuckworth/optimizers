# Observer-pathway producer readiness

## Candidate source and checks

- Runner `experiments/spectral_observer_pathway.py`:
  `263d7fb0475bec32f27a4a1f6409e1f381fae6152e631cd938aa706cbe63f63e`.
- Fixtures `tests/test_spectral_observer_pathway.py`:
  `a84cdb16232ab95e44216aae9dd9c01f6f751154bb02ed40b875d9f698f910b2`.
- New operational protocol:
  `d361f2813c74078a4326f09211c9d6c09a288f296ecb5ed4bc6f8384ea2a36a5`.
- Authoritative scientific design:
  `2fbb538f4da88680322d64ec6506e16545fbd8a4bd7e068d9cb0599af6ed21fd`.
- Main design review:
  `bbcca4942cb2e17c0fc9acd06b3002f8128848b492aea80599618b6c1390b1f8`.

The runtime also pins the unchanged canonical filter, I9 core, selectivity
helper and batch-composition helper. No imported globals are modified and
none of their acquisition/main/audit entry points is invoked.

All **14 fabricated CPU fixtures passed** in the initial invocation (1.181
seconds unittest runtime). After the prospective NumPy mean/cast clarification,
all 14 passed again in 1.167 seconds (2.521 seconds command wall time):

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -p test_spectral_observer_pathway.py -v
```

Fixtures use only invented arrays and a55-parameter CPU model. CUDA is hidden
and RNG snapshots are restricted to CPU. Coverage includes the symmetric FP64
mean/FP32 cast and exact numerical bound; failure reporting without relaxing
the tolerance; alias detection including views; zero-vector cosine semantics;
independent parent/sibling copies; `.grad`, modes, RNG, optimizer and observer
preservation around autograd.grad; the100/150/151 versus100/101 clock distinction;
pre/post native action arithmetic; explicit readout assignment; actual Adam
movement and oracle utility; nontrivial momentum/decay under zero delivery;
all-seed improvement/contrast signs and nulls; separate resource limits;
exclusive capped tensor/NPZ/JSON writing; and no-execute admission.
`git diff --check` passed. No scientific failure or retry occurred.

## Parent and input admission

The only permitted parent is the accepted H2Ow40 acquisition, completion SHA
`39b4ef091e691780b46c1d7f5dcb12e108dbfd5b3b85c6abba612a93a393b60a`,
and its full PASS audit SHA
`3640269898d6c4115adbac925586fb54a6f1d47b0000b7e511a2fa1f1fcdeb6b`.
The input reader checks the audit's756/36/72 counts, matching completion and
receipt roster, accepted source pins and exact Torch/NumPy/Python environment.
Only22 old artifacts are consumed: manifest plus each seed's warmup, original
plan, schedules, probes, common predictions and two label-cell bindings.
Every used artifact is checked by path, size and SHA before reading, and checked
again at completion. It does not rerun the previous audit or trajectory.

The preserved warmup model hash, actual normalized input hashes, assigned-target
hash and receipt cross-links bind reused step100 predictions. No fresh
before-logit extraction occurs. CPU NumPy FP32 pixel normalization and accepted
training IDX hashes are unchanged. The official test set remains out of scope.

## Fixed measurements, counters and state ownership

Exactly600 new fixed-model stream gradients are measured: three parents × two
label cells × two saved50-batch histories. Parameters and Adam remain fixed
during every measurement. Each gradient saves equal before/after envelope
hashes, the unchanged parent hash and a nonaliasing PASS. Tensor storage addresses
are checked across parent, live copies, observer siblings and saved envelopes;
snapshot equality includes `.grad`, model modes and RNG, not just parameters.

The six additional true-label oracle means are measured once per parent from
the inherited54-majority/32-rare probe memberships, then shared across label
cells. They are never fed into an observer or delivered as optimizer actions.
Oracle probe definitions are truth-informed diagnostic information, not a
deployable filter feature.

Both complete FP32 `[50,50890]` streams and exact memberships are saved before
mean admission. Each mean is constructed canonically on CPU using NumPy
`array.astype(float64).mean(axis=0, dtype=float64)`. The symmetric mean is
`((mean_interleaved+mean_grouped)/np.float64(2)).astype(float32)`; independent
copied NumPy buffers become Torch tensors. This fixes the exact reduction and
cast convention for the auditor, avoiding a Torch-versus-NumPy reduction-order
ambiguity at an FP32 rounding midpoint. Norms remain FP64; the fixed bound is
`D <=128*eps32*B +1e-10`. A failed bound is saved before stopping. The shared
`g_star` is the FP32 cast of the arithmetic average of the two FP64 stream means.
This is a block-loss gradient, not a usual next64-example gradient.

Every observer history uses an independent copy. Fifty observations take its
clock100→150 without any Adam update. O150 is preserved; another independent
copy observes `g_star` once to151 before the primary native action. Pre-inclusion
actions are descriptive only. Parent, streams, oracle means and `g_star` are
checked unchanged through observation/action/geometry work.

The new envelope is:

```text
{schema: spectral_observer_pathway_v1,
 model_adam: I9 snapshot with tracker=None,
 observer: raw observer-state dictionary or None,
 clocks: {adam:100 or101, observer:150 or151 orNone}}
```

No counter is rewritten to fake a normal continuation. Native readouts retain
observer151 while Adam100→101. Raw and zero readouts use observer=None.
Each physical readout saves three envelopes: `inherited` before action assignment,
`before` after assigning explicit delivery to `.grad`, and `after` the sole Adam
step. Comparing before with inherited permits only `.grad` assignment to differ.
Delivered gradients remain present after the step. Zero is explicit, so inherited
momentum and decay remain active. There is no additional gradient computation,
observer update, normalization, moment reset or continuation in a readout.

There are21 physical readouts and21 new train/heldout logit pairs:12 native,
six cell-specific raw and three shared zero. Each seed's zero reference is
physically generated once and receipt-linked from both label cells; its
label-dependent metrics are computed separately from identical predictions.
The24 logical case/action records do not imply24 physical optimizer steps.

## Saved schema and reporting conventions

Root schema is `spectral_observer_pathway_v1`; all artifacts have relative
direct-child path/size/SHA receipts. Per-seed oracle files and shared-zero
readouts are separate from six per-case collections:

- `stream-sSEED-CELL-SCHEDULE.pt`: memberships, all gradients, preservation
  hashes and fixed100/100 measurement clocks.
- `common-action-sSEED-CELL.pt` and `mean-admission-...json`: both FP64 means,
  common FP32 action input and immutable tolerance evidence.
- `observer-sSEED-CELL-SCHEDULE.pt/.json`: O150/O151 envelopes, pre/post native
  actions, basis diagnostics, block/oracle retention and self-inclusion changes.
- `readout-PHYSICAL_ID.pt/.npz/.json`: inherited/before/after states, delivered
  action, parameter order, new train/heldout logits, actual movement and signed
  oracle-gradient utilities. IDs encode seed/cell/action, except shared zero.
- `case-sSEED-CELL.json`: direct parent bindings, before metrics, all four
  logical action readouts, after metrics, absolute improvements and historical
  pre/post cross-schedule action comparisons.

`results.json` contains six cases and summaries. Improvements are **before−after
for CE and after−before for accuracy**, hence positive is favorable for both.
The primary contrast is Grouped-native minus Interleaved-native improvement;
both absolute improvements and each native/raw/zero comparison are retained.
Every scalar has all three seeds, mean/sample SD/sample SE/sign counts; absent
wrong-fit metrics remain null. This differs intentionally from the prior
schedule study's raw CE-change sign. Incoming retention, native gain/direction
and actual Adam displacement are reported separately; this is not norm-matched
or covariance-only causal evidence.

## Guarded limits and full conservative inventory

The new `Run` enforces480 seconds from main entry, including admission/preparation,
while the service enforces600 seconds for the process lifetime. It does not use
the prior runner's25-minute/3GiB defaults. Actual admission checks require
`spectral-observer-pathway-001.service`, Type=exec, Restart=no,
KillMode=control-group, one CPU quota/math thread,16GiB/no swap, local RTX3090
and≤8GiB GPU allocation. Source/test/protocol bytes must be committed and are
rechecked at completion. `--execute` and a new exclusive mounted large-volume
parent/acquisition-001 are mandatory. No retry or fallback exists.

| Conservative saved component | Upper bytes |
| --- | ---: |
|600 stream gradients |122,136,000 |
|24 O150/O151 full envelopes |182,334,144 |
|36 native inherited/before/after envelopes |273,501,216 |
|27 raw/zero model-Adam envelopes |23,753,952 |
|Six common actions and paired FP64 means |6,106,800 |
|Pre/post native and delivered actions |9,160,200 |
|Six oracle means and probe inputs |2,036,640 |
|21 new train/heldout logit pairs |8,400,000 |
|Membership/receipt/scalar/metadata allowance |134,217,728 |
|**Total** |**761,646,680** |

This leaves312,095,144 bytes below1GiB. Snapshot allowances overcount cleared
gradients and include64KiB padding per state. A separate1MiB failure/footer
allowance is reserved within the cap; at least1GiB free disk is required in
addition to estimated output at admission and throughout execution. Actual
streamed writes remain capped if the estimate is wrong. Failures preserve all
existing artifacts and a bounded failure footer, without retry.

The researcher experiment instructions inform preservation and reproducibility;
explicit user scope overrides automatic fresh-seed replication, early-stop and
workflow approval gates. The resulting evidence, if acquired, is an
outcome-informed local observer-channel diagnostic, not endpoint confirmation.
