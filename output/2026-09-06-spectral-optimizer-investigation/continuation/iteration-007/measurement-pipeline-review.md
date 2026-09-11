# Saved CPU measurement pipeline checkpoint

Codex, Spectral Optimizer Investigation, 6 September 2026. Base commit 4d60d68.
Evidence is dataset-free engineering, not a learning result or launch approval.
Historical component reports and their hashes are preserved.

## Outcome

The tiny-MLP saved-output pipeline passes end to end: an actual live source
step is captured before restoration, six independent AdamW endpoints are
generated, raw values are saved/reloaded, complete measurements are assembled
and saved/reloaded, then a separate NumPy implementation checks those recovered
values. The audit report is also saved and reloaded separately. All 115 I7
tests, 17 repository tests and six reminder tests pass: **138 total**.
Current bindings are in measurement-pipeline-checks.json (artifact not distributed in this public snapshot).

## Accepted components and evidence

- `measurement_assembly.py` emits exact shared, branch and comparison mappings:
  four probes, six branch slots, 15 unordered pairs and 15 fixed contrasts.
  It checks complete delivered-candidate bindings, native endpoint membership,
  scoped evaluation modes, owned storage and prospective roundoff identities.
  Failed numerical identities raise with bounded failure evidence; they cannot
  return a normally completed result marked fatal.
- `measurement_audit.py` imports NumPy and `independent_numerics.py`, not Torch
  or producer arithmetic. It recomputes CPU64 CE/q from native raw values and
  independently validates displacements, geometry, scalar identities, masks,
  factor qualifications and contrast signs. Exact completion counts distinguish
  an early fatal return from a completed audit. Native32 scalar arithmetic and
  concordance are checked, but native32 forward evaluation is not reproduced.
- `runtime_guard.py` separates elapsed wall/CPU, lifetime peak process RSS,
  allocated/reserved CUDA memory, terminal failures and bounded metadata.
  Scientific source phases require the recorded initialized local GPU;
  independent audit stays CPU-only. Mount identity checks target, device,
  UUID, major/minor and stat metadata without creating a study root.
- Loss callbacks now bracket unchanged existing chunk reductions in both
  implementations. Tests require byte-identical results and RNG preservation,
  stop on callback failure and preserve the caller's evaluation context.
  The file store accepts the correctly named tiny-MLP fixture profile without
  relaxing scientific limits or existing no-alias/no-overwrite checks.

The main all-defined integration requires four independent q checks, 24
independent after-CE checks and exactly 390 independent contrast scalar checks
(15 contrasts times four probes times Y/D/Ddata/R, plus 15 times ten auxiliary
chunk Y checks). It also audits 24 parameter-level Adam updates from recovered
endpoints/moments/counters, including a zero-input branch that still moves.
Full observer/RNG equality and anchor immutability hold across construction,
branches, serialization and audit; branch replay does not ingest another
observer sample. Existing reverse-order replay tests remain in the suite.

A second saved fixture explicitly forces a zero lagged direction. This is a
synthetic domain intervention, not a native old-basis outcome. The restored
branch is null, five branches and ten contrasts remain defined, all membership
is retained, and unaffected measurements pass independently. A separately
serialized deliberate CE corruption fails. Component fixtures also retain weak
leverage, finite native discordance, zero unresolved signs and cancellation-
limited cosine intervals without turning them into exclusions or false signs.

## Review findings resolved

Parent read the implementations and tests. Independent roles authored producer,
auditor and runtime components; a separate reviewer checked the parent pipeline
and storage count. Review found incomplete candidate metadata validation,
fatal statuses returned as normal output, bare assertions in the fixture,
missing native-concordance summation terms, incorrect relative-to-small-result
comparison of cancellation-sensitive identities, incomplete factor-null
metadata, loosely typed coefficients/IDs and nonfinite hostile-input reports.
These paths now have explicit validation, bounded retained failures or fixed
roundoff propagation and regression coverage. Guard exceptions propagate;
the final review moved callbacks outside the numerical exception handler so
even a callback FloatingPointError is not relabelled as an MLP failure.
The scientific TL/Tq, K=32, leverage, native projection and sign thresholds
were not tuned. New assembly roundoff formulas (artifact not distributed in this public snapshot) were
written prospectively before scientific data access.

## Storage correction and remaining work

The old 28-MiB-per-anchor / 632-MiB total estimate omitted required duplicated
value fields. Parent and reviewer independently obtained the same fixed-tensor
lower bound: 45.816 MiB per group. Eighteen groups plus the old 128-MiB shared
allowance give **952.679 MiB**, leaving only 71.321 MiB below the unchanged
1-GiB cap. This is not measured serialization or a complete forecast.
The exact ledger (artifact not distributed in this public snapshot) records omissions and unresolved envelope
choices. No cap increase, evidence omission or compression benefit is assumed.

Complete scientific envelopes and source/plan/data/environment bindings,
file-to-audit adapters, capture proofs, phase transitions and single-root
all-phase accounting remain unimplemented. The flat store and mount/runtime
helpers do not jointly certify those properties. A real read-only big-volume
identity check passed; it cannot certify later writes or full-study resource
feasibility. Full-width serialized size, native timing, occupancy/configuration
and capture neutrality remain unmeasured and require separate decisions.

## Parent verification and scope

With CUDA explicitly hidden, bytecode disabled and one OpenBLAS/OpenMP thread:

```text
python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007 -p 'test_*.py'
115 tests, OK, 6.705 seconds.
python3 -m unittest discover -s tests -v
17 tests, OK, 1.416 seconds.
python3 -m unittest discover -s output/2026-09-06-spectral-optimizer-investigation -p 'test_goal_reminder.py' -v
6 tests, OK, 2.022 seconds.
python3 scripts/lint_knowledge.py
Passed: 14 pages, 11 indexed content pages.
git diff --check
Passed.
```

Private monitoring and recovery details are omitted from this public edition.
