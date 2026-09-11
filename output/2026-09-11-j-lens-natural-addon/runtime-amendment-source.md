# Explicit runtime recovery: source and failed-admission record

11 September 2026. Source/fabricated fixtures only; no recovery stage, model,
tokenizer, scientific archive, reader, checker or grader was run by this task.
The original forward.py and token-preflight files remain unchanged.

## Preserved original failure

Service `j-lens-natural-addon-forward-20260911-MMxZIs.service`, invocation
`fea7e5a7bf0b48ea884853c252572f47`, PID4178933, ran05:38:04–05:38:07UTC
and exited1; MainPID0, failed/failed. Its journal traceback stops at the
runtime-equality assertion in original forward.py:325, before np.load,
load_model or capture. Numerical libraries were imported but no model call or
numerical direction-array load occurred. The directory contains only the two
original records, frozen here without editing:

| File | SHA256 |
|---|---|
| forwards/attempt.json (artifact not distributed in this public snapshot) | `c313921dcdb033cd973dbbd8884d80feef1ca80e79362d02b453c8e9df2bf6ba` |
| forwards/failure.json (artifact not distributed in this public snapshot) | `7f81cc5fad6f71b7d8ee901a642100cce1040dae9ad8fc1c2d440db2616fe1c6` |

Independent read-only diagnosis checked /var/log/dpkg.log, installed package
versions/changelogs, interpreter build text, service/journal metadata and
file hashes. At05:33UTC, after the successful05:13:50UTC preflight, distro
Python changed3.12.3-1ubuntu0.15→0.17 and glibc2.39-0ubuntu8.8→8.9.
These packages contain real patches, not merely changed timestamp text.
The four declared numerical package versions did not change. This failure
provides no scientific evidence for or against J-Lens.

## Thin wrapper and exact provenance

The complete frozen main amendment (artifact not distributed in this public snapshot)
was read; SHA `5bcc78c413c319e1ccf329066d01699a9c56bfdbb7a8bb0aa4ecc5abb9a8a72c`.
The original producer was reread completely; SHA
`23c9c7cf77f2239bd16b219c440fbf056628730cd2bdf51f66b812ab0e8f580c`.

| New source | SHA256 |
|---|---|
| [forward_runtime_amendment.py](forward_runtime_amendment.py) | `679834b6ff4b2e6b37ef4d60c774c994dcf7bffdf4252d388945034d7e70f409` |
| [test_forward_runtime_amendment.py](test_forward_runtime_amendment.py) | `af7b0b9f8d493fac10dd768cca93f458059616ec9f77c2668faf14650d7d8834` |

The wrapper executes only hash-verified original helper source bytes under
their original file identity. It never calls old preflight/forwards entrypoints.
Original frozen_tokens receives the original OUT, preflight receipt SHA and
original expected input pins. No copied receipt, retokenization, source edit,
version spoof or downgraded interpreter is involved.

The new exclusive directory is forwards-runtime-amendment. The CLI takes only
--dataset-sha, --pairs-sha, --preflight-sha; no positional stage argument.
JLENS_NATURAL_ADDON_RUNTIME_AMENDMENT_RELEASE=1 is separately required. The
existing/failed directory guard precedes metadata or scientific reads.

The wrapper requires the exact old/new Python build strings, unchanged
numerical versions, seven pinned amd64 Python/libc package versions and the
two independently checked interpreter/libc binary hashes. It checks these
before the archive and model, and after capture before declaring success.
Original input pins and helper/amendment/failure hashes are rechecked;
runtime changes or errors preserve a failed new stage without retry.

The receipt retains schema jlens_natural_addon_forwards_receipt_v1 and original
input_pins, outputs, model/scope fields. source_sha256 is the actual wrapper;
base_producer_sha256 identifies the original helper. Additional fields bind
runtime_amendment_sha256, failed_admission_sha256s and the unchanged
preflight_runtime. runtime records the actual new process. runtime_transition
contains preflight_python, new_python, old_preflight_source_sha256,
package_versions and binary_sha256s. Main owns downstream contract adaptation.

Scientific kernels and shapes remain unchanged: prefix_end,32×1×1024 h32,
32×1×4 score64,16×1×4 left-minus-right gaps. Model/adapter/mean/U32, BF16 eager
eval/no_grad, no refit/decoding and original resource bounds remain fixed.
This does not assert bitwise equivalence to the previous OS runtime.

## Fabricated verification

Full self-read of both new files;11 tests PASS twice, final0.026seconds
(0.108seconds command wall), using /usr/bin/python3,60second timeout,
CUDA hidden, offline flags and numerical threads1. Tests use only temporary
fabricated JSON, arrays and helper modules; all model/host queries and archive
loads are mocked. They cover exact transition failures, changed source/failure
pins, package queries, stage/admission guards, original token location,
receipt/source/runtime identities, output shapes and post-capture drift.

The best-practices skill prompted current official checks of
[module construction](https://docs.python.org/3.12/library/importlib.html#importlib.util.module_from_spec)
and [binary file hashing](https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest).
Compiling the already verified helper bytes deliberately avoids stale cached
bytecode; this is a provenance choice, not a claim of scientific equivalence.
Main full review and a separate fresh resource admission remain required
before any first actual model pass under this amendment.
