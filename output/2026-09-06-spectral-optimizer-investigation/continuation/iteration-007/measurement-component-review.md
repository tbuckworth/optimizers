# CPU measurement checkpoint

Codex parent, Spectral Optimizer Investigation, 6 September 2026.
Evidence: dataset-free engineering fixtures, not a learning result.
Base: c574f74. The earlier CPU component report and hashes remain historical.

## Accepted scope

The complete iteration007 CPU suite passes: 79 tests in 3.933 seconds. Source
identities and numerical maxima are recorded in
measurement-component-checks.json (artifact not distributed in this public snapshot).

- `loss_measurements.py`: owned functional CPU64 loss/gradient evaluation from
  native float32 inputs and parameters, ordered chunk sums and one final mean;
  a distinct native32 diagnostic path. No training gradients are mutated.
- `independent_numerics.py`: NumPy64 explicit MLP forward/backward and AdamW
  equations, projection reconstruction, fixed arithmetic screens, conservative
  numerical signs and native-loss concordance. No producer imports.
- `artifact_store.py`: exclusive flat store, bounded serialization and reads,
  immutable no-replace writes, receipts, logical-byte/free-space gates and
  retained partial-failure records where the remaining safe budget permits.
- Measurement schema (artifact not distributed in this public snapshot): fixed probe/chunk fields,
  nested branch-pair versus contrast records, common-baseline cancellation,
  factor-qualification scope and separate immutable audit results.
- `state_core.py`: adds a separately named 26-parameter two-layer CPU fixture.
  The scientific and earlier one-layer profiles are unchanged; relabelling
  this fixture as either profile is rejected.

Parent read the complete implementations and tests and ran the integrated suite.
The independent numerical module was authored separately from the producer loss
module; this is arithmetic independence, not a completed scientific raw audit.

## Actual integration evidence

The parent-authored tiny-MLP integration records a real live source next step
before restoring an anchor. One cloned observer transition constructs current
and lagged inputs. Six branches restore independently and run in forward and
reverse order, including an assigned zero gradient that still advances AdamW.
Current-branch parameters and full optimizer state reproduce the actual live
source exactly. Full observer configuration/state, RNG bytes and anchor
immutability are asserted. Forty-eight per-parameter independent Adam checks
cover actual parameters, first/second moments and counters across both orders.

Four tiny probes compare the two real loss modules, gradients, chunk means,
directional dots and all fifteen after-loss contrasts. A separate synthetic
5,000-example pool uses the full 784-64-10 architecture (50,890 parameters), ten
500-example chunks, and deliberately duplicated native float32-normalized
pixels. No MNIST bytes, scientific bundles or old scientific checkpoints are
used. Passing this single synthetic full-shape case is not full-study runtime,
memory, serialized-size or CUDA certification.

Independent numerical fixtures also cover rank-32 projection at 50,890
coordinates, cancellation, subnormals, signed-zero identity, finite native-loss
discordance and rejected numerical perturbations. Maximum projection error is
4.883103280201029e-7 (screen ratio 0.0016247353209465183); CE and gradient-component
errors in that module's fixtures are at most 1.1102230246251565e-16 and
1.8041124150158794e-16. Adam moment/variance/parameter screen ratios are
0.0260208517, 0.0390498263 and 0.0470917987. Constants were not adjusted.

## Review findings resolved

Independent review of the parent integration correctly identified that initial
assertions checked only part of observer state and did not explicitly assert
RNG equality across every stage. Parent added complete observer and RNG checks;
the reviewer re-read the changes and confirmed both gaps closed. This does not
certify the still-unimplemented scientific schema/controller.

Parent review of loss evaluation required scoped gradient/autocast behavior,
early inference-mode rejection, and compact owned storage checks. Numerical
reference review required the installed decoupled-weight-decay option, explicit
signed-zero identity mismatch reporting, and consistent huge-integer rejection.
All have regression coverage.

Parent file-store review found that failure logging could exceed the aggregate
cap, initial-size checks did not bound a growing file, partial receipts could
obscure retained failure evidence, and initialization failures could leave open
descriptors. The final implementation guards these paths, post-checks committed
usage/free space, refuses raced targets and validates final read identity before
decoding. Tests include growth/hardlink creation during read, partial writes and
receipts, constructor cleanup, full-cap failure logging, malformed header limits,
storage aliases and views. If failure metadata cannot safely fit, the exception
explicitly reports that it was not retained; existing files are left untouched.

## Parent verification

Numerical tests use `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and
one OpenBLAS/OpenMP thread. CUDA remained uninitialized.

```text
python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007 -p 'test_*.py' -v
79 tests, OK, 3.933 seconds:
13 response + 15 state + 10 numerical-contract + 13 loss +
15 independent-numerics + 10 artifact-store + 3 integration.

python3 -m unittest discover -s tests -v
17 tests, OK, 1.317 seconds.

python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation -p 'test_goal_reminder.py' -v
6 tests, OK, 2.327 seconds.

python3 scripts/lint_knowledge.py
Knowledge lint passed: 14 pages, 11 indexed content pages.

git diff --check
No whitespace errors.
```

Total: 102 tests. No scientific conclusion changed, so no knowledge synthesis
update is warranted. Research review gates kept this checkpoint CPU-only;
official-API checks informed loss-context and file-operation behavior, not the
validity of proposed scientific error bounds.

## Remaining work and decision

Accepted for CPU component integration only. Still required:

1. Complete Y/Ddata/R, vector contrasts, geometry and sign/schema adapters with
   zero-domain, weak-leverage and cancellation-aware integrated tests.
2. Strict scientific envelopes and immutable plan/source/data/environment
   bindings, witness roles, fixture rejection and independent raw audit adapters.
3. A single verified big-volume store across phases, explicit phase/mount
   checks, complete artifact/copy accounting and time/RSS/GPU resource controls.
   The flat store is advisory single-writer infrastructure, not hostile-file
   isolation, a reusable phase controller or a scientific serializer.
4. Full serialized resource measurement, followed by a separately reviewed
   native capture-neutrality pilot decision. The 632-MiB estimate is unmeasured.