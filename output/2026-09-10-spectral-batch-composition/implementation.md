# Batch-composition implementation readiness

## Frozen-candidate bytes and fabricated checks

- `experiments/spectral_batch_composition.py`:
  `35c4b3f59000e862dd4103bc96bb1b20beb9b21a67b7cc8dce342f0ac5cc8b1a`.
- `tests/test_spectral_batch_composition.py`:
  `ab45329b2267102dc6516264bac2a8762a159f37e13e7102b1ec099f6009f2cc`.
- Protocol at preparation:
  `e1b0efc220be2a1f67d643862f1852ba3862bb0cf71c47207cf9ecf7d8e02dbc`.
- Immutable imported selectivity helper:
  `c851cb711e85a07aacad891cfd1122d5bf1a3f7d406142d1b98cca4cb2afac0f`.
- Immutable I9 core:
  `706851828b7120bea0d7a41bd0fa75474ac800b2df71087941d0140c8d1970ec`.
- Immutable canonical filter:
  `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`.

All **18 fabricated CPU fixtures passed**, 1.273 seconds unittest runtime
(2.648 seconds command wall time), on the first invocation:

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -p test_spectral_batch_composition.py -v
```

Fixtures use invented labels/indices and a 55-parameter CPU model, with CUDA
hidden and RNG snapshots restricted to CPU. They check all/no-rare 3,200-draw
blocks, repeated occurrences, intermediate count extremes, named draw order,
rare-then-common concatenation and common slot permutation, block multiset
identity, independent RNGs, first-extremum/tie anchor rules, randomized probe
strata and true-target invariants, the fixed-parameter mean/covariance toy
identities, mean-gradient side-effect preservation, native versus numerical
span distinction, zero/undefined geometry, nonfinite rejection, exact fork
restoration, real first-step raw/observer hashes across policies, norm-law cast
tolerance, complete mean-diagnostic evidence and actual Adam movement, absent
wrong probes, summary contrast/interaction signs and nulls, resource inventory
and explicit-execute admission. The tiny synthetic optimizer-state preparation
is not a reproduction of the scientific warmup distribution.

`git diff --check` passed. The accepted filter/core/helper sources were not
modified. Their previously accepted norm-zero and capped-writer semantics are
reused through immutable imports, not copied into a changed algorithm. The
researcher experiment instructions supply evidence-preservation discipline;
the user’s fixed roster, no-restart and autonomous scope override its generic
fresh-replication and early-stop gates.

## Exact schedules and probe membership

The complete original `make_plan` recipe is called on the three fresh seeds,
including unused poison/sham arrays for reproducibility. Only Clean/Diffuse
targets and unpatched inputs enter this study. One majority-only clean warmup
per seed is restored identically for all 12 branches; initial and warmup
training/heldout logits are generated once and reused.

For each original block, stream6 permutes occurrence positions (not unique IDs).
The permuted sequence is separated into rare and nonrare lists. Stream7 supplies
the shared batch-position permutation. Interleaved assigns quotient/remainder
counts; Grouped fills up to64 rare slots along that permutation. Chronological
batches consume each shared list without extra sampling, concatenate the rare
slice **before** the common slice, then apply the shared stream8 slot permutation.
Every block verifies original and both new sorted multisets exactly, all realized
rare counts, at most one partial Grouped batch and Interleaved spread at most1.

At blocks0/7/37, first chronological argmax is high; first argmin excluding high
is low. Events are sorted by absolute update and retain labels and an equal-count
flag. Their updates are data/schedule-selected prospectively, not loss-selected.
The last event need not be update2000: endpoint state must not be equated with
the last saved diagnostic state unless their actual updates coincide.

Probe stream9/group0 is consumed across ascending majority digit strata, six
positions each; group1 selects32 rare positions; group2 selects up to32 randomly
permuted actually changed Diffuse positions. Majority and rare probes always use
true labels, including Diffuse. Assigned-target metadata remains separate.
Wrong-assigned/corrected probes share the exact same inputs/positions. Clean
wrong groups are absent. No first32 class-blocked convenience subset is reused.

## Schema and arithmetic

Root schema: `spectral_batch_composition_v1`. Standard receipts are direct-child
path, byte size and SHA256. Per seed:

- `plan-sSEED.npz`: complete immutable accepted plan arrays.
- `schedules-sSEED.npz`: both `[1900,64]` batch arrays and `[1900]` rare-count
  arrays; occurrence permutations `[38,3200]`, count-position permutations
  `[38,50]`, within-batch permutations `[38,50,64]`.
- `probes-sSEED.npz`: `majority`, `rare8`, `wrong` training-position arrays.
- `plan-sSEED.json`: all array hashes, block multiset checks/count variances,
  both anchor lists, generator/version metadata and unused original sham rows.
- Full `warmup-sSEED.pt`, all100 warmup action scalars, and
  `common-predictions-sSEED.npz` with exact steps0/100 train/heldout logits.

Each cell/schedule binding records the full warmup tree hash, initial model
hash, all input/plan/schedule/probe receipts, assigned-label hash, changed/selected
counts, prospective anchors, and actual train/heldout FP32 input hashes.
Normalization remains CPU NumPy uint8→FP32 division by FP32(255), then transfer.

Every policy branch saves `curve`, `actions`, `logits` and full `final` artifacts
with identifier `sSEED-CELL-SCHEDULE-POLICY`. Curves have21 evaluations; NPZ keys
are `steps`, `train`, `heldout`, with both logits `[21,5000,10]`. Training action
rows cover101–2000, include actual rare counts and raw/native/applied norms, and
retain the original norm-law cast guard. First-step raw hashes must match all
three policies within each cell/schedule; post-observation tracker hashes must
match native/norm controls. Different schedules do not require matching gradients.

Native-only `diagnostic-IDENTIFIER-uSTEP.pt/.json` records72 events. Payloads save
full before/after model, `.grad`, Adam, tracker and RNG; raw/applied training
gradients; parameter order; anchor identity; and groups
`majority`, `rare8`, `wrong_assigned`, `wrong_corrected`. Each defined group has
inputs, training/source IDs, true/assigned/evaluation targets, pre/post logits
and one FP32 `mean_gradient` from autograd.grad of mean CE. **There are no
per-example gradients or coherence estimates.** Mean-difference vectors are
explicit FP32 rare−majority and wrong-assigned−corrected subtraction.

JSON reports true-label counts, finite CE changes, native mean-action retention,
separate numerical-span retention, mean-difference retention, rounded FP32 decay
and actual adaptive/total movement, off-span energies/fractions and signed local
utilities. FP64 thin SVD and its inherited tolerance define diagnostic Q only;
Q never replaces the canonical action. Native energy need not equal orthogonal
span energy. Zero denominators remain null; basis-unavailable native identity
does not fabricate a numerical span. Exact snapshots check neutrality around
pre/post probes and geometry. The audit will validate saved arithmetic, not
rerun gradient computation or infer unobserved long-horizon mediation.

`results.json` retains36 endpoints/warmups and all metrics. Summary contains
`per_group`, `change_from_warmup`, `schedule_contrasts`
(`cell/policy/grouped_minus_interleaved`), `policy_contrasts`
(`cell/schedule/native32_minus_CONTROL`) and `schedule_policy_interactions`
(`cell/POLICY_minus_raw`). Every scalar carries all three seed values, mean,
sample SD/SE and signs; absent wrong-fit metrics stay null. Interaction is
`(Grouped−Interleaved)policy − (Grouped−Interleaved)raw`: positive accuracy and
negative CE are favorable, with no composite or noninferiority threshold.

## Admission and conservative output bound

Only an explicit `--execute --output-dir NEWPARENT/acquisition-001` entry reads
science data. Actual cgroup/service checks require exactly
`spectral-batch-composition-001.service`, non-restarting Type=exec, hard30min,
one CPU quota,16GiB/no swap. CUDA remains local RTX3090, allocation≤8GiB,
fixed Torch2.11.0+cu128 and deterministic settings. Immutable inherited storage
enforces the same25-minute deadline and3GiB cap; new and inherited bound constants
must match. Source/helper/test/protocol bytes must be committed and unchanged;
accepted dataset hashes are checked before and after. No old main/acquire is
called or monkeypatched. No resume/retry/CPU fallback exists.

| Conservative component | Upper bytes |
| --- | ---: |
| Three warmups +36 full endpoint states | 296,303,592 |
| 144 complete diagnostic pre/post states | 1,094,044,032 |
| 72 events ×(four means +two differences +two actions) | 117,250,560 |
| Probe inputs, labels/IDs, pre/post logits | 35,164,800 |
| 36 complete train/heldout logit histories | 302,400,000 |
| Shared initial/warmup predictions | 2,400,000 |
| Plans, schedules, scalar JSON and metadata allowance | 268,435,456 |
| **Total** | **2,115,998,440** |

This is approximately1.971GiB, leaving1,105,227,032 bytes under3GiB. It overcounts
Clean absent means and raw/cleared-gradient snapshots. Actual writers remain
capped even if this estimate is wrong. A separate1MiB failure/footer allowance
is reserved, and at least1GiB free disk is enforced both at admission beyond the
estimate and during execution. Partial failed evidence is retained without retry.
There is no paid spend or evidence-inventory adjustment needed.
