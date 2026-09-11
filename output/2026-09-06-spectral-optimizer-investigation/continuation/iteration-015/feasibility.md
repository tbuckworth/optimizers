# I15 feasibility — SGDm history transport versus mean restoration

Codex — Spectral Optimizer Investigation, 7 September 2026.
Read-only source and manifest audit. I did not deserialize an I14 tensor,
evaluate a model, execute an optimizer step or inspect a new outcome.

## Verdict

**Feasible with a small new SGDm branch adapter; not feasible by directly
calling the existing I13 mean branch core.** The six required I14 SGDm h100
states are present and were saved in a complete resumable schema. The three
confirmation plans retain all 2,000 batch rows, so the exact remaining rows
can drive 1,900-step continuations. Existing I14 raw and native-current curves
can be reused as references without restarting them.

Exact resumption is currently possible but depends on bulk files that are not
in Git and are not backed up. If any of the six h100 state files under
`/tmp/spectral-experiment-artifacts/spectral-i14-001.SJWZCj/confirmation` is lost or changes,
the raw JSON archive is insufficient to reconstruct model weights, momentum,
observer state and RNG. In that case I15 must stop; an I14 restart or a soft
reconstruction would not be the same experiment.

## Available seam state

I14's `i14_optimizer_snapshot_v1` captures all state needed at the branch
seam:

- model specification, parameters, module modes and residual gradients;
- the exact SGDm parameter-group configuration and every momentum buffer;
- the complete native observer, including covariance basis, running gradient
  mean and step counter;
- Python, NumPy, CPU Torch and CUDA RNG states.

`optimizer_core.restore` rebuilds the model and SGDm optimizer, validates
parameter and optimizer-state topology, restores the observer bindings and
RNG, and can be checked by snapshotting again and comparing the tree digest.
There is no scheduler, scaler, stochastic augmentation or implicit data-loader
cursor to reconstruct. Training batches are explicit plan arrays.

All six canonical current32 h100 state files exist, each 7,346,541 bytes:

| Seed | Clean parent | Fixed parent |
|---:|---|---|
| 200 | `state-s200-sgdm-lr0.03-clean-current32-h100.pt` | `state-s200-sgdm-lr0.03-fixed-current32-h100.pt` |
| 201 | `state-s201-sgdm-lr0.03-clean-current32-h100.pt` | `state-s201-sgdm-lr0.03-fixed-current32-h100.pt` |
| 202 | `state-s202-sgdm-lr0.03-clean-current32-h100.pt` | `state-s202-sgdm-lr0.03-fixed-current32-h100.pt` |

The six curve records bind each filename to its byte count, file SHA256 and
full-state tree digest. Each current32 state digest and h100 evaluation digest
also equals its paired raw arm's digest, as required by I14. Use the current32
file as the single canonical parent while retaining both raw/current equality
records in the input manifest; do not choose between duplicate parents at run
time.

The saved confirmation plans have exactly 2,000 rows of 64 batch indices.
Relative I15 step 1 must use Python row 100, which is cumulative I14 update 101;
relative step 1,900 uses row 1,999 and ends at cumulative update 2,000. The
registered evaluation mapping is therefore:

| I15 relative horizon | 0 | 150 | 400 | 900 | 1,400 | 1,900 |
|---|---:|---:|---:|---:|---:|---:|
| I14 cumulative horizon | 100 | 250 | 500 | 1,000 | 1,500 | 2,000 |

An off-by-one here would silently destroy comparability while still producing
plausible learning curves. Slice `training_batches[100:2000]` once, validate
shape `(1900,64)`, and record both relative and cumulative horizons.

## Exact intervention

The four-cell conceptual factorial is delivery `{current, mean}` × history
`{native, projected}`. Only three cells are new because I14 already supplies
current/native:

| Cell | Gradient delivery | Old SGDm history before recurrence | Acquisition |
|---|---|---|---|
| Current / native | native post-ingest `A_t(g_t)` | unchanged | reuse I14 current32 |
| Current / projected | native post-ingest `A_t(g_t)` | `A_t(b_{t-1})` | new |
| Mean / native | `A_t(g_t)+mu_t-A_t(mu_t)` | unchanged | new |
| Mean / projected | `A_t(g_t)+mu_t-A_t(mu_t)` | `A_t(b_{t-1})` | new |

Here `A_t(x)=V_t(V_t^T x)` is the actual native post-ingest linear action and
`mu_t` is the post-ingest observer mean. The new arms are exactly
`current_projected_history`, `mean_native` and `mean_projected_history`, for
clean/fixed × three seeds: 18 new trajectories.

At every step, compute the raw gradient, call the canonical filter exactly
once, retain its literal filtered gradient as `A_t(g_t)`, then form the selected
delivery. For a projected-history arm, flatten the already initialized SGDm
momentum buffers in parameter order, replace their contents by
`A_t(b_{t-1})`, and only then call the unchanged PyTorch recurrence. This gives

\[
b_t=.9A_t(b_{t-1})+h_t.
\]

Do **not** project the completed `b_t`: doing so would erase the newly delivered
outside-mean component in `mean_projected_history` and collapse the intended
factorial. `A_t` is only approximately an orthoprojector numerically; use the
native action and homogeneous residual tolerances rather than asserting exact
idempotence.

After delivery/history handling, preserve I14's update order exactly: apply
manual decoupled parameter decay by `1-.03*.01`, then call `optimizer.step()`.
Changing this order, normalizing the momentum recurrence, or projecting the
gradient and buffer under different bases would create a different experiment.

The contrast is not a pure direction-only mediation. With a fixed ideal basis,
native mean history accumulates the outside mean through momentum, whereas
projected history removes previous outside accumulation but keeps the new mean
term. This changes temporal gain and step amplitude as well as history
orientation. Retain that limitation rather than adding an outcome-driven
rescaling arm.

## Minimal runner and provenance design

Reuse, without modification:

- I14 `data_plan.py` to parse and validate the three saved confirmation plans
  and materialize the identical train/validation/auxiliary splits and fixed
  corruption;
- I14 `optimizer_core.py` for snapshot/restore, live topology checks, manual
  decay semantics, leakage helpers and complete-state capture;
- I14's evaluation schema so the h100 seam and subsequent outcomes compare
  exactly with existing curves.

Add one I15 core containing only SGDm momentum flatten/set validation,
post-ingest current/mean delivery, optional old-history projection, the I14
decay/step order and diagnostics. Add one runner for source admission, branch
pairing, failures and bounded output. I13 `mean_core.py` should be a pinned
definition/test reference, not the live branch engine.

Before any update, the runner should:

The new frozen-source manifest should include the canonical filter, I9 neural
core, I14 optimizer core/data plan/evaluation runner, I13 mean core, every I15
source/test/protocol file and their exact Git bytes. Current inspected hashes
match the I14 acquisition manifest for the canonical filter, I9 neural core and
all three I14 production files. The I13 mean-core definition currently hashes
to `1ebba52cb85dfbea1fed1cc829c250bbe6ff02c6c046c50c8526cbeeb7b4ed4b`;
freeze it in the machine-readable manifest.
Recheck all source and input hashes at terminal. The I14 source directory must
remain read-only from I15, with source hashes checked both before and after.

For each seed/target, load the canonical parent independently for all three
arms. Do not carry live global RNG or a mutated Python state from one arm to the
next. Require identical h0 state/evaluation and, on the first update, identical
raw-gradient and post-observer digests across all three arms. The two mean arms
must also have identical delivered-gradient digests before history handling;
the two projected-history arms must have identical projected-old-buffer
digests. These are treatment-boundary checks, not empirical outcomes.

Existing I14 raw and current32 curves should be referenced by hash and parsed
for analysis, never copied as new acquisition or replayed. Preserve their
original cumulative horizons and distinguish them from the 18 new physical
branches.

## Diagnostics and checkpoints

Per step, retain only the scalars and digests needed to identify the
intervention:

- raw, native, outside-post-mean and selected-delivery norms and the raw/post-
  observer/applied-gradient digests;
- old momentum norm, `A_t(old momentum)` norm, discarded outside-history norm,
  projected-history flag and pre/post buffer digests;
- homogeneous residual bounds for native projection, mean definition, mean
  recurrence and projected-history recurrence;
- actual total/data/nominal-decay displacement norms, current-basis leakage and
  raw-gradient dot data displacement, using I14's definitions;
- observer step before/after, basis rank, cumulative update and target/policy
  identity.

Report outcome curves at the six mapped horizons with the same train,
validation and auxiliary fields as I14, including fixed CE/accuracy,
expected-soft CE, `R_zeta` and confidence. Aggregate seeds first. Reuse I14 raw
and current references without survivor averaging; a missing new arm makes its
paired factorial contrasts unavailable.

## Required tests before source freeze

1. **I14 state round trip:** construct an SGDm snapshot with initialized
   momentum and active observer, restore/resnapshot it and require exact tree
   equality, including RNG, modes and residual gradients.
2. **No-intervention equivalence:** a test-only current/native path in the I15
   adapter must reproduce I14 `train_step` exactly for multiple post-warmup
   synthetic steps—state, diagnostics and parameter displacement—not merely
   approximately.
3. **Delivery algebra:** verify the I15 mean formula against I13's post-ingest
   definition on the same synthetic tensors, including mean recurrence and
   finite backward-error bounds.
4. **History ordering:** use a buffer with known inside/outside components to
   prove the old buffer is projected before the recurrence, the incoming mean
   complement survives `mean_projected_history`, and no completed buffer is
   projected accidentally.
5. **Pair neutrality:** all three arms must leave the immutable parent object
   unchanged and share h0, first raw-gradient and post-observer digests; the two
   mean deliveries and two projected histories must match at the declared
   boundaries.
6. **Plan seam:** assert exact use of rows 100–1999, 1,900 updates and the
   relative/cumulative horizon mapping above. Test row 99/100 and terminal
   off-by-one failures explicitly.
7. **Source admission:** reject a wrong parent policy/base/rate/horizon, altered
   plan or curve, missing completion membership, changed file bytes, absent
   supplement, or a source state whose tree digest differs after authorized
   deserialization.
8. **Failure/output behavior:** numerical failures retain last-good evidence;
   structural and provenance errors abort; duplicated IDs, partial branch
   rosters, cap overrun and attempted source-root writes are rejected.

After synthetic tests, the later authorized prelaunch check should load each
real h100 parent, restore/resnapshot it and perform only a state-neutral h0
evaluation against its recorded I14 value before any scientific updates. This
current feasibility audit did not perform that check.

## Launch decision

There is no source-evidence blocker today. The parents, plans, curves and
required hashes are present, and the snapshot schema is sufficient for exact
continuation. Launch readiness still requires the new adapter/runner and the
tests above; direct reuse of I13's Adam-specific branch engine would be a
blocking implementation error. Preserve the bulk source states now. If their
hash or availability changes before launch, mark I15 blocked rather than
recreate or restart I14.
