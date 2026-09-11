# Independent I17 analysis source review

Status: PASS for acquisition admission after the remaining physical launch gates.

RESOURCE-AMENDMENT REVIEW: after the consumed synthetic smoke exceeded the
agent-set1800s forecast gate, main read the entire analyzer/test diff for the
explicit pre-scientific2400s resource amendment. Final analyzerSHA
d3916b6bf9f43dcbfbe3689b5f051d2828ace27c8f6be1061083202b955ae8bf;
testSHAb8ed45f0f1a68ac5fa40b5491a0a5a0a32f33ad3a4c5fb6ab48d8c09e9f5729b.
It pins the original smoke commit/completion/root, verifies separate24/27-file
closures with identical ordered scientific prefix, binds attempts per phase,
and records the failed1800s gate alongside the2400s allowance. Scientific
reductions/checkpoints/failures are unchanged. Main combined53a405PASS48/48
in3.860s (17analysis+8collector+5resourcewrapper+7core+11runner), synthetic CPU
only. Current_report independently reviewed the wrapper,5testsPASS; read
resource-amendment-review.md. Original review below is preserved chronology,
not a claim the earlier gate passed or the original analyzer is still final.

8 September 2026. Main read all972 lines of analyse_gain.py and all515 lines
of test_analyse_gain.py, plus the reused root inventory, artifact cap,
trajectory, checkpoint and CPU state-digest helpers. The independent author
i17_analysis did not implement acquisition core/runner. Main also compared
the analyzer's phase/step/branch schema and forecast to the frozen runner and
the I17 protocol. No scientific analysis or tensor load was invoked by review.

Analysis SHA256:
8b5e83cc59cb0725ebe37e59917ff9c199323556c92edcf09b382ce9d2187d97

Tests SHA256:
00c39a72ab30f8883952704f8f98152e9a9631b2927b10aa268b8d1fe8d51e42

Main combined synthetic CPU suite69e9a8 passes42/42 in3.929s:16analysis,
8collector,7core and11runner tests. CUDA hidden, numerical threads1. Unlike
the historical I16 module-shadowing invocation, these42 are all I17 tests;
the terminal lists their four unique module names. Tiny synthetic training is
only a fixture, not scientific acquisition or replay of any existing run.

Review covers exact24-file acquisition closure and four analysis sources;
phase/attempt/root/JSON/artifact membership; three-seed/eight-primary family
selection, tie rules and auxiliary blindness; no survivor selection/averaging;
all30 logical trajectories,16 endpoints,32 per-k selected contrasts and960
scheduled aggregates; full/late exact-window path and energy; normalized
ideal-vs-actual delivery and non-enforced native-action defects. The analyzer
imports no acquisition code and constructs no model. It never reruns an old
analysis entrypoint. Only hash-bound six used h100 parents, two synthetic
parents and each new terminal/failure state are CPU-loaded for tree digests.
Old terminal checkpoints are hash-checked, not deserialized or forwarded.

Accepted I16 pins and direct six k0 reference JSONs are bound transitively to
the original parent/plan evidence. Unaccepted provenance aborts before parent
loading; malformed new evidence prevents a selected mean. Complete and typed
numerical-failure states have distinct coverage rules. Native nonorthogonality
is measured, not quietly repaired or rejected merely for being nonzero.

Synthetic tests exercise the tiny real core's four policy records, scalar
reductions/mutations, a complete20-artifact synthetic smoke with10state digests,
hash/digest/runtime/source tampering and mocked historical provenance. They do
not claim a full real-data confirmation audit has already passed. One actual
independent analysis remains required after new acquisition finishes.

Fresh resource check3d8272: /dev/RECONFIGURE_FOR_LOCAL_STORAGE mounted at/private-artifacts/storage with740GiB free,
host50,611MiB available, RTX3090 used537MiB; only unrelated GNOME/Stremio PIDs
2101/8861 visible and untouched. Old unrelated swap is full; new units must
enforce MemorySwapMax0. Both proposed I17 service names are not-found/MainPID0.
No I17 root or attempt was created during source review. Exact source freeze,
exclusive root and bounded unit arguments are recorded separately at launch.
