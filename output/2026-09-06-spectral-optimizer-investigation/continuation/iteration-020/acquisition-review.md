# Independent I20 acquisition review

8 September 2026. **Pre-acquisition review; no I20 root or attempt existed at
review time.** I read the complete protocol, response theory, numerical core,
runner and both fixture suites. I did not open a parent NPZ, import or execute
the native observer, draw randomness, or run a scientific stream.

## Verdict

The settled source is internally consistent with the frozen protocol and is
cleared for the single bounded saved-stream acquisition, conditional on first
committing the six operational source files and launching under the stated
external 600-second/4-GiB systemd limits. The runner itself fail-closes at 540
seconds and enforces the storage caps. A separate independent numerical audit
is still required after acquisition; this review does not accept future output
or a scientific result.

## Admission and provenance

- `expected_cells()` contains exactly 192 distinct cells in the required
  seed-major, process-major, rotation-minor order: seeds19000--19031, three
  process indices and two rotation indices. The parent inventory must have
  exactly the same ordered IDs and exactly two physical files per ID; every
  recorded size and SHA-256 is checked.
- I independently rehashed the five immutable parent JSON pins in the runner;
  all match. Without opening an NPZ, I also verified all 11 source bindings
  represented by those pinned records: seven acquisition, two analysis and
  two report sources. Each current file matches both its recorded hash/size
  and the byte content at its recorded frozen commit.
- The runner validates the exact parent ZIP member order before numeric load,
  requires stored rather than compressed entries, bounds each expanded NPY
  member by its metadata-derived size, disables pickle, and then checks the
  eight selected parent fields for exact shape, dtype and finiteness. Parent
  file hashes are checked immediately before load.
- The I20 source manifest contains the core, both fixture suites, runner,
  protocol and numerical-API review. It compares current bytes with a full
  40-character Git commit before creating the attempt and repeats the complete
  source and parent validation before allowing `status=complete`.

## Core and response review

The NumPy core has no file I/O, RNG, native-observer dependency or mutable
hidden state. Non-oracle estimation uses only saved `g`, saved post-ingest
`mu`, native/full actions and their masks. Process variance and latent signal
are not passed to it. The privileged useful axis affects only the two labeled
oracles and alignment diagnostics; the added axis-variation fixture confirms
that the first four estimators' actions, moments and 24 outputs are unchanged.

The six estimators, two rho values and three responses produce the exact 36
estimator-major policies. The core implements zero-start `.99` and `.999`
weighted moments and masses, normalization before eigendecomposition, the
frozen relative-gap rule, identity fallback, masked alignment, and action
change. All 10 returned arrays have their exact registered order, shape and
float64/bool dtype and are checked again before an exclusive write.

The CP, `rec99` and `rec9` recurrences match the reviewed algebra. Every first
output is `g_1`; the `rho=.9, gamma=.9` branch is exactly the scalar `.9` EMA;
fixed-projector CP and `rec99` agree; moving projectors can differ. Before each
stream is written, 11 fail-closed parent closures are checked: both native CP
outputs, three available fixed-oracle CP outputs and the `.9` EMA output for
all six direction estimators.

## Failure and resource behavior

Attempt, stream NPZ, stream metadata and completion writes use exclusive
creation. Invalid arrays or metadata are rejected before their output path is
created. A caught stream failure records a failed completion and the completed
prefix; a second acquisition cannot reuse the root. The runner requires the
approved mounted large-volume location, an empty nonsymlink root, the correct
device, at least 10 GiB free, hidden CUDA and one numerical thread.

The cooperative clock is checked before load, after reconstruction/closures,
after every write and after final source/parent validation, so an over-540s run
cannot be marked complete. The hard 600-second limit, 5-second stop, 4-GiB
memory limit and `Restart=no` remain launch-service properties rather than
in-process assertions and must be verified on the actual invocation.

One deterministic `T=4000` review fixture returned all 10 arrays and 36
policies in 0.43 seconds wall time, with 40,420 KiB maximum RSS and exactly
4,248,000 raw output bytes. Across 192 streams that is 815,616,000 raw bytes,
well below the 2-GiB array and 3-GiB root caps; NPZ/JSON/completion overhead is
also protected by prospective and actual checks plus a 1-MiB reserve.

## Defects found and resolved before this verdict

1. The first runner draft loaded parent outputs but did not enforce the
   protocol's algebra closures. The final runner checks all 11 controls before
   every write, and the corruption fixture exercises each one.
2. Two initial core fixtures used inconsistent floating-point expressions for
   `.1` and the first EW mass. The implementation was correct; the fixtures
   now construct expectations with the literal registered recurrence and pass.
3. The first settled runner could finish final validation after 540 seconds
   and still report success. Post-write and post-final-validation checks now
   make cooperative expiry a failed attempt.
4. Privileged-axis separation was initially established only by source
   inspection. A direct axis-variation fixture now protects the non-oracle
   actions, moments and outputs.

Final local checks: 24 deterministic core/runner fixtures pass, the exact
rational response-theory checker passes, and `git diff --check` is clean.
No scientific outcome was inspected or inferred.
