# Exact source and runtime-environment binding

Status: adopted with amendments by [source-environment-decision.md](source-environment-decision.md).
This specifies a read-only binding component;
it supplies no scientific values, run authority, or evidence that an optimizer
step occurred. All mappings have exactly the displayed insertion order. Every
leaf is `None` or an exact built-in `bool`, `int`, `float`, or Unicode-scalar
`str`; containers are exact `dict`/`list`. There are no tensors or path objects.

## Public API

`source_environment.py` should expose only:

```text
SourceEnvironmentError
validate_sources(value, *, profile) -> value
validate_environment(value, *, profile) -> value
collect_verified_sources(repo_root, *, profile) -> owned dict
collect_runtime_environment(repo_root, *, profile, runtime_role) -> owned dict
validate_state_environment(core, environment, *, profile) -> None
```

The two validators are pure schema/policy checks. They cannot establish that a
Git checkout, file, process, or device matches the supplied claims. The two
collectors observe those things read-only and then call the validators. The
anchor validator must compare persisted records directly to independently
supplied expected snapshots, then call `validate_state_environment`; a digest
match alone is not structural validation.

`profile` is one of the three existing exact profile strings. `runtime_role` is
`native_source`, `cpu_audit`, or `fixture_cpu`. Scientific permits the first two;
fixtures permit only `fixture_cpu`. An anchor requires `native_source`, never an
audit or fixture snapshot.

## `sources` exact layout

```text
sources = {
  "schema": "i7_source_binding_v1",
  "profile": str,
  "repository_revision": lowercase Git object ID,
  "git_object_format": "sha1" or "sha256",
  "repository_root_realpath": canonical absolute ASCII str,
  "worktree_status": "clean_including_untracked_v1",
  "files": [source_file, ...],
}
source_file = {
  "path": canonical repository-relative POSIX ASCII str,
  "role": str enum,
  "git_mode": "100644" or "100755",
  "size_bytes": nonnegative exact int,
  "sha256": 64 lowercase hex str,
  "git_blob_oid": lowercase Git object ID for the declared object format,
}
```

`files` order and membership equal the profile's compiled constant; callers
cannot add paths. Duplicate, absolute, empty, dot, dot-dot, backslash, symlink,
submodule and nonregular entries fail. A Git revision commits the entire tree;
the explicit list identifies the load-bearing subset whose working bytes are
also independently SHA-256 checked. Tests and reviews remain commit-bound but
are not misrepresented as executing producers.

Reads are capped at 2 MiB per required file and 16 MiB for the complete listed
source set. Git metadata commands have a 1-MiB output cap; a `cat-file blob`
command has the corresponding 2-MiB file cap. Every command has a five-second
timeout. Exceeding a cap is failure, never truncation followed by hashing.

For the scientific profile the literal prefix `I7/` below expands to
`output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-007/`.
The required ordered `(role,path)` membership is:

```text
canonical_filter        spectral_filter.py
binding_codec           I7/artifact_store.py
binding_codec           I7/identity_codec.py
binding_codec           I7/plan_bindings.py
binding_codec           I7/verified_plan_load.py
binding_codec           I7/data_probe_bindings.py
state_codec             I7/state_core.py
binding_codec           I7/cuda_identity.py
source_binding          I7/source_environment_schema.py
producer                I7/response_math.py
producer                I7/loss_measurements.py
producer                I7/measurement_assembly.py
independent_auditor     I7/independent_numerics.py
independent_auditor     I7/measurement_audit.py
runtime_guard           I7/runtime_guard.py
source_binding          I7/source_environment.py
producer                I7/source_capture.py
envelope                I7/anchor_envelope.py
envelope                I7/artifact_envelopes.py
phase_controller        I7/phase_controller.py
scientific_entrypoint   I7/scientific_runner.py
audit_entrypoint        I7/audit_runner.py
contract                I7/common-state-design.md
contract                I7/analysis-spec.md
contract                I7/anchor-schema.md
contract                I7/measurement-schema.md
contract                I7/numerical-contract.md
contract                I7/assembly-roundoff.md
contract                I7/resource-contract.md
contract                I7/identity-binding-contract.md
contract                I7/data-probe-contract.md
contract                I7/verified-load-decision.md
contract                I7/source-environment-proposal.md
contract                I7/source-environment-decision.md
contract                I7/anchor-envelope-contract.md
contract                I7/envelope-contract.md
contract                I7/phase-controller-contract.md
contract                I7/audit-envelope-contract.md
```

The not-yet-written envelope, controller, scientific-entrypoint, audit-entrypoint
and three closure contracts are mandatory missing members, not optional `None`
entries. Scientific collection therefore must fail until they exist, are
reviewed, committed, and the worktree is clean. If parent chooses different
filenames, change this list before implementation and version the schema after
freeze; do not discover future files by glob. Historical I3/I6 harnesses,
reviews, proposals not adopted as contracts, result JSON, caches and test data
are deliberately excluded from explicit execution membership.

The fixture profile uses a separate two-file constant
`fixture_source.py`, `fixture_contract.md` for temporary-Git tests. Fixture
membership can test the collector but cannot satisfy scientific validation.

## `environment` exact layout

```text
environment = {
  "schema": "i7_environment_binding_v1",
  "profile": str,
  "runtime_role": str,
  "repository_root_realpath": canonical absolute ASCII str,
  "python": {
    "implementation": str, "version": str, "version_info": [int,int,int],
    "cache_tag": str, "executable_realpath": canonical absolute str,
  },
  "libraries": {
    "numpy_version": str, "torch_version": str, "torch_git_revision": str,
    "torch_cuda_build": str or None, "cudnn_version": int or None,
  },
  "operating_system": {
    "sys_platform": str, "system": str, "release": str, "machine": str,
    "libc_name": str, "libc_version": str,
  },
  "safe_environment": {
    "CUBLAS_WORKSPACE_CONFIG": str or None,
    "CUDA_VISIBLE_DEVICES": str or None,
    "MKL_NUM_THREADS": str or None,
    "OMP_NUM_THREADS": str or None,
    "OPENBLAS_NUM_THREADS": str or None,
    "PYTHONHASHSEED": str or None,
  },
  "torch_settings": {
    "num_threads": int, "num_interop_threads": int,
    "default_dtype": str, "default_device": str,
    "deterministic_algorithms": bool, "deterministic_warn_only": bool,
    "grad_enabled": bool, "inference_mode_enabled": bool,
    "autocast_cpu_enabled": bool, "autocast_cuda_enabled": bool,
    "autocast_cpu_dtype": str, "autocast_cuda_dtype": str,
    "float32_matmul_precision": str,
    "cudnn_enabled": bool, "cudnn_benchmark": bool,
    "cudnn_deterministic": bool, "cuda_matmul_allow_tf32": bool,
    "cudnn_allow_tf32": bool,
    "allow_fp16_reduced_precision_reduction": bool,
    "allow_bf16_reduced_precision_reduction": bool,
  },
  "cuda": {
    "initialized": bool, "visible_device_count": int or None,
    "current_device": int or None, "driver_version": str or None,
    "devices": [cuda_device, ...],
  },
  "rng_layout": {
    "python": {"state_version": int, "internal_length": int},
    "numpy": {"algorithm": str, "keys_dtype": str, "keys_length": int},
    "torch_cpu": {"dtype": str, "state_length": int},
    "torch_cuda": [cuda_rng_layout, ...],
  },
}
cuda_device = {
  "index": int, "name": str, "uuid": str,
  "total_memory_bytes": int, "capability_major": int,
  "capability_minor": int,
}
cuda_rng_layout = {
  "index": int, "name": str, "uuid": str,
  "dtype": "torch.uint8", "state_length": positive int,
}
```

No PID, timestamp, resource counter, arbitrary environment mapping, command
output, hostname, username or secret-capable value is collected. The six-name
environment whitelist is exact. Values are read with `os.environ.get`; no
environment or Torch setting is modified.

Scientific policy requires CPython 3.12.3, NumPy 1.26.4, Torch
2.11.0+cu128 with its exact nonempty `torch_git_revision`, CUDA build 12.8 and
cuDNN 91900. Both scientific roles require Linux x86_64, CPU default device,
float32 default dtype, one intra-op thread, deterministic algorithms enabled
without warn-only mode, grad enabled, inference mode off, both autocast modes
off, float32 matmul precision `highest`, cuDNN benchmark off, both TF32 flags
off, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; OMP/OpenBLAS/MKL thread variables
are `1`. These reproduce the prior explicit setup. Inter-op thread count, cuDNN
enabled/deterministic flags, autocast dtypes and both reduced-precision-reduction
flags are recorded but must be populated from a reviewed preflight and pinned in
the expected snapshot; this proposal does not invent different settings. The
launch command must likewise freeze the remaining whitelisted values rather
than accept whatever happens to be present.

`native_source` additionally requires CUDA already initialized before the
collector, exactly one visible device, current logical index 0, exact name
`NVIDIA GeForce RTX 3090`, a nonempty stable UUID, positive memory, capability
and RNG-state length, and matching device/RNG identities. The driver, UUID,
memory, capability, OS release/libc and executable path are observed and then
pinned by the independently approved expected snapshot; they are not invented
here. `cpu_audit` requires CUDA uninitialized both before and after collection,
`visible_device_count/current_device/driver_version=None`, and empty device and
CUDA-RNG lists. It must not query a CUDA device. `fixture_cpu` has the same CUDA
non-initialization shape but cannot validate as scientific.

CPU-only remediation after the failed engineering inspection adopts one shared
UUID representation converter in cuda_identity.py. A native Torch
_CUuuid object, bare UUID text and NVIDIA GPU-prefixed text normalize to exact
lowercase GPU-prefixed8-4-4-4-12 hex. The RNG-core, environment collector and
runtime guard use that converter; the driver query and wrapper's expected GPU UUID then
match the same representation. Unknown object types and malformed/ambiguous
text fail. This changes representation handling, not RNG values or device
identity. It does not authorize native collection or another inspection.

The pure membership constants and exact source/environment validators live in
source_environment_schema.py and are re-exported by source_environment.py.
The engineering controller uses the same validators without importing Torch or
NumPy. There are 45 mandatory scientific source files after this extraction.

## Collection and verification semantics

`collect_verified_sources` canonicalizes the repository root without symlinks;
uses argument-vector Git subprocesses with fixed timeouts; requires `HEAD` to be
a commit, `git status --porcelain=v1 -z --untracked-files=all` to be empty and
no submodules; and obtains object format, modes and blob IDs from Git. Each
required worktree file is opened below a held root descriptor with component and
leaf no-follow checks, bounded read, `fstat` before/after, regular-file and link
count one checks. Its bytes must equal the committed blob bytes, then supply the
independent SHA-256. Repeat HEAD, status and file signatures after collection;
recheck ancestor/root directory device, inode and mode without treating ordinary
sibling creation as identity change. Leaf source opens include `O_NONBLOCK` so a
substituted FIFO is rejected after `fstat` without blocking. Any identity or
source-file race fails. This is cooperative local verification, not protection from a
hostile same-user process or modified Git binary.

`collect_runtime_environment` first requires the role's CUDA initialization
state, snapshots Python/NumPy/Torch CPU and (native role only) all CUDA RNG
states, reads only the fields above, then requires every RNG byte and CUDA
initialization state unchanged. Native driver text is a bounded exact
`nvidia-smi` query after the context already exists. CPU roles invoke no CUDA
query other than `torch.cuda.is_initialized()`. The collector does not seed,
draw, synchronize, allocate a model, change flags, load data, or write files.

`validate_state_environment` performs only saved-value cross-checks. Parameter,
moment, observer and model native devices must be the single environment device;
the core's Python/NumPy/Torch CPU RNG layout and every CUDA index/name/UUID/state
length must equal `rng_layout`; the core CUDA list and environment device list
must be complete and ordered. Scientific CUDA cores require `native_source`.
This comparison does not inspect current hardware and must not be called a
foreign-platform replay certificate; actual runtime collection and the existing
state-core continuation check remain separate.

## Anchor integration and smallest next implementation

At anchor capture, `bindings.sources` and `bindings.environment` are the exact
records above. `anchor_envelope` receives independently retained expected
snapshots, validates both, compares them by exact typed structure, checks
`sources.repository_root_realpath == environment.repository_root_realpath`, and
calls `validate_state_environment` after `state_core.validate_core`. A caller-
supplied self-consistent record without the expected snapshot is insufficient.

The smallest next unit is only `source_environment.py` plus
`test_source_environment.py`: strict primitive schemas; a temporary clean Git
fixture covering dirty/untracked/symlink/hardlink/blob/race failures; CPU-only
runtime collection with CUDA hidden; malformed role/version/flag/RNG tests; and
saved core/environment cross-binding. Do not yet implement the future runner,
phase controller, complete artifact envelopes, scientific manifest, data access,
GPU test, or source execution. Scientific collection is expected to fail while
the mandatory future paths are absent.
