# Runtime guard and bounded archive writer — implementation receipt

Codex — Spectral Optimizer Investigation · 10 September 2026

**Author checks PASS: 27 fabricated CPU/mock tests. This is implementation
preparation, not acquisition admission, a scientific result, or independent
acceptance.** No real checkpoint, source experiment array, image data, GPU
configuration, service launch, or new acquisition was accessed or performed.
Only the new guard, its tests and this receipt were authored by this leaf.
No old source was changed and no commit was made by this leaf.

## API and caller boundary

`experiments.spectral_component_utility_guard` is import-inert. It exposes:

- `Run(path, device='cuda:0')`: requires a caller-created, absolute, empty,
  non-symlink real directory. `device='cpu'` is exclusively a fabricated-test
  mode. The object owns an open directory descriptor; `close()` or a context
  manager releases it.
- `check()`: checks the 1200-second cooperative deadline on every invocation;
  host high-water RSS, free disk and PyTorch allocator high-water usage at
  most once per second. Any failure terminalizes the writer.
- `save(name, value, kind='json'|'npz')`: exclusive direct-child creation;
  returns exactly `{path,size_bytes,sha256}`. JSON accepts only finite plain
  JSON values; NPZ accepts a bounded dictionary of finite, plain numeric
  NumPy arrays, never object/structured/string/complex/subclass/metadata dtypes.
  Per-artifact limits are 16 MiB JSON and 256 MiB NPZ, including headers and
  trailers. NPZ is an ordinary uncompressed ZIP of NPY members, written with
  explicit contexts and `allow_pickle=False` at the NPY writer.
- `used`, `receipts`, `started`, `serialization_seconds`: actual archive
  byte accounting, copy-isolated receipts, monotonic origin and cumulative
  serialization/validation/hash time including failed save attempts.
- `write_failure(exc)`: one exclusive `failure.json` attempt, retaining all
  partial artifacts and reporting prior successful receipts. The reserved
  footer bypasses expired cooperative/resource checks, not byte limits or
  directory identity. No scientific save may follow a failure.
- `configure(device='cuda:0')`: explicit current-runtime validation and
  deterministic configuration; this does not launch a process or service.

The **main runner owns** the `/dev/RECONFIGURE_FOR_LOCAL_STORAGE` mount check, exact big/tmp acquisition
directory, at-least-16-GiB launch disk check, exclusive attempt identity,
committed source/input binding and prospective byte inventory. The guard does
not create `attempt.json`, discover inputs, load scientific state or authorize
execution. It retains a 1-GiB runtime free-disk margin after admission.

To keep final results from listing their own receipt, construct the result
dictionary with `receipts: run.receipts` before calling
`run.save('results.json', result)`. `receipts` returns a deep copy; only after
successful writing is the final file's receipt appended internally. This exact
convention has a fabricated round-trip test.

## Enforced resource and output contracts

Whole output cap is exactly 8,589,934,592 bytes. Ordinary artifacts stop before
the reserved 1,048,576 bytes; the failure footer can use at most that reserve.
Every stream write is checked before exceeding its remaining file/archive
allowance. Successful receipts hash the same exclusive open file, after fsync.
Short writes, file identity changes, external inventory changes, aliased or
nonregular files, reuse and replacement of the archive directory fail closed.
Failed partial files are preserved rather than removed, overwritten or retried.

`configure` requires the exact cgroup-v2 leaf and current systemd unit
`spectral-component-utility-001.service`, with 8-GiB memory, zero swap,
`cpu.max='100000 100000'`, Type=exec, RuntimeMaxUSec=30min, Restart=no and
KillMode=control-group. MainPID must be the current PID; the current 32-hex
`INVOCATION_ID`, service InvocationID and proc/service cgroup must agree.
The service must be active/running. Subprocess observation calls have a
10-second timeout and cannot launch or alter services.

Device is explicitly cuda:0 and exactly NVIDIA GeForce RTX 3090, with at
least 8 GiB free. Visible compute-client rows may contain PID2101 at <=512 MiB,
PID8861 at <=128 MiB, and self at <=4096 MiB, with no duplicate or unknown
rows. No other client's work is displaced. The initial client observation is
not a reservation or ongoing claim that unrelated usage cannot change.

PyTorch must be `2.11.0+cu128`, NumPy `1.26.4`. All four OMP/OpenBLAS/MKL/
NumExpr thread environment values must already equal `1`, and
CUBLAS_WORKSPACE_CONFIG must already equal `:4096:8`. Both torch thread
counts become1; deterministic algorithms are enabled; cuDNN benchmarking and
both cuDNN/matmul TF32 flags are disabled and read back.

The 4-GiB per-process fraction is a **PyTorch caching-allocator limit**, not a
hard whole-process/device GPU-memory limit. Runtime checking also reads
PyTorch's maximum allocated bytes. Non-allocator CUDA allocations and kernel
work cannot be fully capped by this API. Likewise cooperative checks cannot
interrupt a single blocked native call; the independent 30-minute systemd
bound and no-restart/control-group kill policy provide the outer time limit.
A hard kill, host OOM or disk failure can prevent the best-effort footer.

## Exact configure receipt schema

Top-level keys are exactly:

```text
unit, pid, invocation_id, cgroup, effective, service, gpu_clients,
preexisting_gpu_clients, device, device_name, gpu_free_bytes_at_configure,
gpu_total_bytes, gpu_allocator_limit_bytes, gpu_allocator_fraction,
torch_version, numpy_version, deterministic
```

`effective` is exactly `{'memory.max':'8589934592','memory.swap.max':'0',
'cpu.max':'100000 100000'}`. These are the observed leaf controls, not a
measurement of available RAM or an inventory of ancestor cgroups.
`service` has exactly Type, RuntimeMaxUSec, Restart, KillMode, MainPID,
InvocationID, ActiveState, SubState and ControlGroup, with the values above.
Both client lists contain sorted `{pid:int,memory_mib:int}` rows;
`preexisting_gpu_clients` excludes self. The allocator limit is4294967296;
fraction is that limit divided by the observed device total bytes.

`deterministic` has exactly thread_env (the four named string values),
cublas_workspace (`:4096:8`), intraop_threads=1, interop_threads=1,
deterministic_algorithms=true, cudnn_benchmark=false,
matmul_allow_tf32=false and cudnn_allow_tf32=false.
Both version fields are plain Python strings, not TorchVersion subclasses.

## Fabricated checks and provenance

Command, CUDA hidden and CPU thread environment bounded throughout:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 90s python3 -m unittest discover -s tests -p 'test_spectral_component_utility_guard.py' -v
```

Final run: **27 tests, 0.504 seconds, OK** (1.808-second enclosing command).
`git diff --check` also passed. Tests cover import inertness; exact successful
runtime receipt; cgroup, service, invocation, PID, versions, environment,
GPU-client and free-memory rejection; strict JSON/NPZ schemas; noncontiguous
and empty numeric round-trips; receipt hash/copy identity; final-receipt
exclusion; exclusive names; symlink/external/replaced-directory rejection;
per-file and cumulative ZIP-inclusive limits; short writes; retained partials;
deadline/host/disk/mock-allocator checks; terminal failures; reserved exclusive
footer; serialization accounting. All process, cgroup, service and GPU
observations/configuration in `configure` tests are fabricated mocks.

Initial 25-test pass exposed a cleanup warning on intentionally failed
NumPy1.26 `savez` writes. The context-managed ZIP/NPY path removes that warning
while preserving partial files and exact byte caps. Later 25-test and final
27-test runs passed without that warning. No tolerance or scientific rule was
changed to obtain these results.

Source SHA256:
`77e2f00fa113a92592cfd9f543ce4a9ab730cc097b44c60c69fd2b24f87e5bce`

Test SHA256:
`c51613b5a2887b9767cad178e4a3e6dca3013d5682a1636884e44285c87891d4`

## Documentation influence and remaining limits

Implementation-validation guidance informed the explicit allocator distinction
and the context-managed, pickle-disabled standard NPZ writer. Sources checked:
[NumPy1.26.4 NPY/NPZ implementation](https://github.com/numpy/numpy/blob/v1.26.4/numpy/lib/format.py),
[NumPy write_array API](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.write_array.html),
[Python ZIP context managers and streaming writes](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile),
[PyTorch2.11 allocator fraction](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.set_per_process_memory_fraction.html),
and [systemd resource controls](https://github.com/systemd/systemd/blob/main/man/systemd.resource-control.xml).

This is a strict cooperative single-writer guard, not a filesystem quota or a
defence against a malicious concurrent process editing the archive. Numeric
semantic shapes, lineage, mathematical identities, prospective inventory,
actual runtime feasibility and all acquisition/admission decisions remain with
the main runner and independent reviewer/auditor. No real configure or parent
restoration has been admitted or exercised by this implementation work.
