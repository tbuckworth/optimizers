# Independent source review: raw-direction acquisition runner

9 September 2026. **PASS for commit and subsequent one-shot guarded launch;
no scientific or launch-blocking source defect was found.** This review does
not itself authorize or perform acquisition. The required prelaunch commit,
real input/source admission, exclusive output allocation and service receipt
remain future operational acts.

Reviewed source SHA-256:

- `experiments/grokking_raw_direction.py`:
  `dbf5ad6d33c75f27c3a016499d60fa9a36e503543b946dcd313a707d91804f5a`;
- `experiments/run_grokking_raw_direction_batch.py`:
  `0cd2f527a3a2c8a7dd22ce9c2b2504b86264ab04496caf921667c7f89f6d8b25`;
- `tests/test_grokking_raw_direction.py`:
  `cfbab6b1c1458f19f65c96ce486e3237f1600ba9c35af20abf782333e5c84bab`;
- `output/2026-09-09-spectral-raw-direction/guarded_launch.py`:
  `53f0ad44143cb54fa568831bd08488369c2cc69f5e48c052220cbdce69d53036`;
- `tests/test_grokking_raw_direction_guard.py`:
  `14e6c229f1031a7654ca0d18fbf0f17a3e33b751642c2b4fb2f1dbec8871b383`.

Also reviewed the protocol (`a361ca5d...`), analytic resource plan
(`ede64210...`) and runner implementation note (`ae7add2d...`), plus the
previously passed policy source and fixtures. The policy runner's live source
map has 19 entries and now includes the guard and guard fixture.

## Scientific execution path

Each update in the sole loop calls the frozen `train_step` exactly once.
That helper performs one forward/backward pass, one inherited
`tracker.filter_grad()` call and one AdamW step. The inherited call increments
the observer counter once and advances the legacy observer once before the raw
action is delivered. `adam_diagnostic`, snapshots, evaluations and local
counterfactual action construction do not call the optimizer or observer.
The synthetic call-count/state fixture confirms one training call, observer
step 1500 to 1501 and one carried-Adam counter increment.

The runner admits only seeds 100--104, policy `raw_norm_matched`, updates
1501--2500 and captures 1501/2000/2500. The batch constructs commands only for
the new runner and launches the five seeds sequentially. It cannot select an
extra seed or invoke the old action runner. Seed 100 completion is checked only
for fixed roster, source/environment identity, finite histories, elapsed bound
and artifact integrity; no accuracy, loss threshold or other scientific outcome
appears in admission. The mocked batch fixture confirms the exact ordered
roster `[100,101,102,103,104]` after seed-100 admission, with no retry or
replacement path.

## Parent and source identity

Before scientific checkpoint loading, `corpus()` verifies the audited original
15-run summary and metrics hashes. The runner selects exactly the original
legacy step-1500 receipt for the requested fixed seed. It separately verifies
the accepted action-batch completion hash, all-five accepted roster, per-seed
completion hash, exact parent receipt, parent metrics hash and the complete old
source map. The checkpoint is then hash-bound, singly linked, regular-file
loaded with `weights_only=True`; its schema/config/step/split/source/parameter
and evaluation prefix are checked against reconstructed seed data and current
frozen sources. Finally, its scientific-state hash must equal the accepted
action fork and the restored model/optimizer/filter/RNG snapshot must be
bitwise identical under the same scientific hash. No archived action-branch
tensor is loaded or executed.

The new source map includes runner, batch, policy, all three new fixture files,
the guarded launcher, protocol and every imported frozen scientific helper,
while also requiring equality to the old action receipt's source map. The
guard refuses launch unless every mapped working-tree hash equals the same file
at current `HEAD`. The four explicitly frozen files remain byte-identical to
their accepted receipt hashes and have no worktree diff.

## First action and longitudinal observability

Before the first update the runner saves the complete restored model,
optimizer, legacy-filter and CPU/CUDA RNG state. The first post-update bundle
contains the corresponding complete state, the raw gradient, retained float64
Q, native action, unscaled orthogonal projection, projected norm-matched action,
delivered raw norm-matched action and full post-step Adam diagnostic. That
diagnostic contains actual displacement, adaptive and decay movements, post-step
moments/adaptive direction, and residual norms; the residual tensor is exactly
reconstructible from the three stored movement tensors. Only the raw-matched
action is present in parameter gradients at the optimizer step. Basis retention
is disabled immediately after this capture.

Every one of the 1,000 updates contributes one ordered action-history row.
Rows contain raw/native/delivered/projected norms, numerical rank/k/singular
values, action cosines, clamp/exactness/post-cast mismatch fields, actual,
adaptive and decay movement norms and decomposition residuals. Histories at
1501, 2000 and 2500 bind their exact checkpoint receipts; completion verifies
the full integer step ranges, exact nine-artifact roster, hashes, sizes,
containment and absence of extra files. The final 2500 history therefore binds
all 1,000 rows. Evaluation runs only on the original 50-step post-update grid
and explicitly checks CPU and all CUDA RNG states before and after. Capturing
1501 precedes any new evaluation.

## Writes, failures and resource controls

The batch parent must be a fresh empty directory directly under
`/tmp/spectral-experiment-artifacts`; seed directories are exclusively created with fixed
names. JSON and tensor writes use exclusive temporary files, fsync and hard-link
publication, never overwrite. The shared writer accounts across the entire
batch, rejects symlinks/escapes, prospectively sizes every JSON/tensor payload,
enforces the 20 GiB aggregate budget and 1 GiB free reserve, then checks actual
post-serialization usage. Completion and batch receipts hash all subordinate
artifacts. Failure identity, last completed step and already written partial
captures are preserved; failures are re-raised, so there is no hidden repair or
automatic restart.

The runner itself sets PyTorch intra-op and inter-op threads to one, checks disk
on every step/write, cooperatively stops before three hours, and the sequential
batch gives each child a hard subprocess timeout within its twelve-hour budget.
The external one-shot service is correctly required for the controls a Python
runner cannot robustly self-impose: exact 16 GiB cgroup memory, zero swap,
CPU quota at most 400%, 12-hour service deadline, `Type=exec`,
`KillMode=control-group` and `Restart=no`. The guard also verifies all four
single-thread environment variables, exact `/usr/bin/python3.12`, its intended
unit/cgroup, empty large-volume output and committed source equality before
`exec` of the batch. The actual service unit, PID, invocation/ExecStart and
effective values still need to be recorded in the launch receipt; absence of
that future receipt would block launch admission, but it is not a source defect.
A no-concurrent-research-GPU preflight likewise remains an operational launch
condition.

The analytic storage arithmetic is internally consistent: O(Pk) V/Q and O(P)
vectors, no P-by-P allocation, under-10-GiB expected payload with slack below
the enforced 20-GiB ceiling. Forecasts remain estimates rather than outcome or
completion guarantees.

## Synthetic verification

Commands:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -m unittest discover -s tests -p 'test_grokking_raw_direction.py' -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -m unittest discover -s tests -p 'test_grokking_raw_direction_guard.py' -v
```

Results: **9/9 runner fixtures passed** in 4.275 seconds and **1/1 guard
fixture passed**. These used only tiny CPU tensors, mocks and temporary files.
No scientific checkpoint, inference, acquisition, GPU, cloud resource, prior
analysis or old branch was loaded or run.

## Non-blocking documentation note

`runner-implementation.md` records runner hash `0367b536...` from the
pre-guard implementation pass. The current runner hash is `dbf5ad6d...` after
adding the guard and guard fixture to `source_pins()`. This is stale historical
provenance text, not an executable or scientific defect: the runtime source map
and guard use the current full hashes and compare them to committed `HEAD`.
Updating that note before commit would improve clarity, but launch integrity
does not depend on the prose hash.

Verdict boundaries: commit the reviewed snapshot, allocate the fresh output,
instantiate and record the exact guarded service, and allow its real parent and
environment checks to pass. Do not bypass the guard. This review gives no
permission to rerun completed branches, alter the fixed roster, retry a failure,
or proceed if any source, parent, resource or receipt check fails.
