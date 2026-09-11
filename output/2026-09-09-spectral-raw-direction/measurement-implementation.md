# Raw-direction measurement adapter implementation

9 September 2026. Source and tiny synthetic fixtures only. **No real checkpoint
was loaded, no inference or measurement was launched, and no saved-data
scientific calculation was rerun by this leaf.** Main owns source review,
commit and eventual guarded admission.

New files:

- `experiments/measure_grokking_raw_direction_states.py`
- `tests/test_grokking_raw_direction_measurement.py`
- `output/2026-09-09-spectral-raw-direction/measurement-protocol.md`

No existing source, including the now-frozen acquisition/guard/protocol/tests,
was modified. The adapter imports the old `Output`, resource/source/environment
guards, structural-array recipe checks, extraction, Fourier probes and
`analyze_state` unchanged. It never invokes an old runner/main and never
changes another module's globals. New code supplies the 15-state raw-policy
roster, complete batch/history admission, explicit new-envelope validation
and a one-extraction-per-new-state collection loop.

The imported old measurement source map is additionally bound to its accepted
completion/manifest, not merely hashed in its current state. The inherited
legacy configuration and the new outer policy remain distinct. Every seed's
nine artifacts, complete history prefixes, source/parent and seed-100 admission
identities are checked. The adapter retains the original recipe and grid
reproduction tolerances, saves raw/scalar/analyzed outputs and hashes them in
its completion. No old model/logit/probe computation is hidden in admission:
the only archived arrays inspected are unchanged structural IDs/splits/nulls.

## Synthetic checks

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -p test_grokking_raw_direction_measurement.py -v
```

**8 tests passed in 0.101 seconds.** They cover the exact 15-state roster,
unchanged helper identities/recipe, complete 52-file input receipt roster,
changed-history/source/missing-batch/failure rejection, distinct outer/inner
identities, filter/device/finiteness/evaluation-prefix guards, original grid
tolerances, accepted old-source binding and rejection of every old policy
before checkpoint loading or extraction. Tiny synthetic tensor objects are
used only for envelope validation; checkpoint-like fixture files contain JSON
and are never loaded as model states.

`git diff --check` passed. Source SHA-256 at this implementation check:

```text
adapter 0580e6db123dd3b5a8786afd2ac4f66585041e88f8b72cbe57f250986031fcb3
tests   dce7d371b55e2d1eeca3b41cf1aa4f01e489abdb93d3ce00276026c3737c99e2
```

These fixtures validate admission and collection plumbing, not successful
CUDA acquisition or any new empirical optimizer result. Service/cgroup
verification, real receipt admission and the actual measurement pass remain
main-agent responsibilities after the complete training batch is terminal.

Main final-freeze addendum: after adding operational guard and guard-fixture
entries to the measurement source map, the final collectorSHA256 is
`a6d01790fc54ddb1f913819a0d53fbbc5961e2913ea5ac54b820219d73ae9dd8`.
Independent collector/guard source reviewPASS; ten synthetic measurement/guard
fixturesPASS. Main read all source, tests, protocol and review. No measurement
has run. An unrequested external WIP commit47803ea captured the two initial
collector/test files while source preparation was underway; neither main nor
the implementation leaf issued that commit. It was not acceptance or a launch.
The final reviewed snapshot supersedes that WIP state without rewriting history.
