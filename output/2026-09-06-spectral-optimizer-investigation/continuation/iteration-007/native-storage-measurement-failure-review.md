# One-shot CPU storage check: failed setup, no scientific result

Codex / Spectral Optimizer Investigation — 7 September 2026.

The actual check failed without retaining a measurement candidate. I missed a
dependency-visibility mismatch between the test interpreter and the isolated
production interpreter. The previous tests and reviews did not establish that
the actual isolated command could load its numerical dependencies.

## Observed outcome

The fixed service was launched once from clean, immutable revision
`1b71ae3767c0d0138bc06fd300ded671bd02fec1`. Its invocation was
`a687faef2db249af8c9ce558e857f401`; controller PID 1578740 exited with status 2
after 0.148873 seconds. The service recorded `Result=exit-code` and a kernel
charged-memory peak of 22,843,392 bytes. This is not an RSS measurement.
The separate terminal observer, session 33398, exited successfully after
capturing the failed service. Its exit 0 must not be confused with service success.

The reserved measurement file (artifact not distributed in this public snapshot) is empty,
mode 0600, and remains exactly as produced. An empty file is not a schema-valid
failed candidate, an authenticated source observation, or storage-fit evidence.
It consumes the one-shot slot. A remains absent; the controller process and
service cgroup are gone. The failed unit remains loaded. No retry, native
permission, training, or optimizer-performance finding followed.

The external observation (artifact not distributed in this public snapshot)
retains the unit properties, timestamps, source expectation, file identity,
observer handles and discovery checks. This report does not replace M.

## Hypotheses and read-only diagnosis

After seeing the empty reservation, the plausible classes were a post-reservation
OS failure, a controller ownership/cgroup-observation failure, or an early worker
bootstrap failure. We did not reproduce the measurement: the consumed slot and
frozen code prohibit that. Source inspection and module discovery distinguished
a definite launch-blocking defect from the still-unobserved historical throw.

The frozen controller's `_worker_command` invokes `/usr/bin/python3.12 -I -B`.
The inserted module directory contains only I7 code. The worker's
`_production_dependencies()` first imports `identity_codec`, which imports NumPy
and Torch at module scope. A read-only `importlib.util.find_spec` check found
neither package in the isolated interpreter. The normal interpreter found both
under `/private-artifacts/.local/lib/python3.12/site-packages`. Neither discovery check
imported a numerical library or invoked a worker/recipe/measurement API.

Python's isolated mode excludes the user site; this is expected interpreter
behavior, not a library regression. See the official
[Python 3.12 command-line documentation](https://docs.python.org/3.12/using/cmdline.html#cmdoption-I).
The exact frozen command therefore cannot load its dependency closure. If the
worker reaches that import, it becomes `runtime_unobserved`, the child wrapper
exits 2 without candidate bytes, and the controller fails with M still empty.
That chain is strongly consistent with the observed timing, memory and exit.

The exact historical worker PID, child exit and exception were not retained.
We cannot prove that the import error was the first exception, exclude an
earlier post-reservation failure, recover a valid M, or claim an observed frozen
source-set digest from these facts. The dependency defect is proven; its status
as the precise historical cause is a strong inference.

Independent source-only review by `i7_current_storage_count` agreed with that
distinction and found the test gap: the command test only asserted `-I` syntax;
the isolated public-worker test replaced `_production_dependencies` with a mock;
other isolated checks imported only modules without numerical dependencies.
Earlier passing test counts remain true but do not establish production viability.

An earlier observer-only preparation (session 66451) failed before the launcher
was called: JSON `busctl monitor` did not print the expected readiness banner.
After rechecking absent UNIT/M/A, an acknowledged D-Bus subscription captured
the actual single invocation. No measurement was restarted to fix observation.

## Disposition and next work

Preserve the frozen checkout, consumed M, failed service and original inspection.
Do not install dependencies into the frozen execution environment, change `-I`,
rewrite source bindings, reset the unit, substitute a new slot, issue A, or
claim permission to launch native work. Any replacement attempt requires an
explicitly reviewed new scope; this report supplies no such authority.