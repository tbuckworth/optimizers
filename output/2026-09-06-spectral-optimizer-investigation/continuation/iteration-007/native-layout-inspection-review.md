# Native layout inspection: engineering review

Codex / Spectral Optimizer Investigation, 7 September 2026.

Independent design review and a separate implementation safety review cover
only the named zero-update engineering observation. The parent read the helper
and tests in full, the source/environment collectors and fixed contracts.
The CPU-only paired metadata comparator received separate bounded review.
No positive scientific result or native fit conclusion follows from review.

## Corrections before execution

- Fixed one singleton inspection root instead of retry-friendly random roots.
  Clean source and exact committed helper are separately bound; scientific
  source membership and all disabled runners remain unchanged.
- Worker helper commands originally escaped the controller-owned process
  group. They now inherit it. Cleanup originally returned when only the group
  leader exited; it now checks live descendants, TERM/KILLs the owned group,
  reaps its direct child and verifies no live descendants remain. A harmless
  child/grandchild regression exercises this case.
- Preserve nested collector key order in JSON so existing exact-schema
  validators accept the reloaded records. No raw RNG bytes are retained.
- Verify the exact known Flatpak Stremio reported-name/executable/comm/UID
  combination; its namespace executable path is not a host-resolvable path.
  Read-only real-device preflight succeeds without importing Torch or creating
  the attempt directory.
- Collect committed scientific source bytes before CUDA initialization as well
  as afterward. Compare the wrapper UUID separately from the unchanged native
  validator. Compare observed driver versions for consistency without claiming
  an independently pinned driver requirement.
- Correct timing/resource labels: worker/controller CPU omit helper CPU;
  RSS limits are separate per-process observations, not an aggregate tree cap.
  Occupancy is a non-exclusive compute-only snapshot. Missing own NVIDIA
  footprint is null, not zero. Five seconds TERM grace and a separate one-second
  post-KILL disappearance check are both explicit in the marker and proposal.
  Local controller filesystem operations are not asynchronously preempted.
- The controller-failure test isolates its mocked module view, allowing full
  discovery with Torch already imported without weakening the production
  fresh-controller guard. Default/help remain Torch-free.
- Offline comparison verifies exact native metadata schemas and binding,
  complete restricted CPU roundtrip, unchanged tensor layout and every tensor
  byte. Only source/environment fields change in the paired specimens.
  It reuses approved synthetic CPU fixtures, including restored CPU RNG draws;
  the no-draw boundary is for the native inspector. Its wrapper independently
  checks Torch CPU RNG only and labels that scope explicitly.

## Limits deliberately retained

Initialization is a real process side effect. The inspector does not capture
or restore a full model/optimizer/observer core, draw a CUDA continuation witness,
read a scientific dataset, generate a scientific plan, execute source updates,
run branches or certify any scientific phase. Source records are observations
of the checked checkout, not adversarial same-user execution authentication.

The size comparator replaces actual provenance within unregistered structural
specimens, not valid native anchors: fixture state and envelope digests do not
become scientific merely because metadata is real. It measures the paired
metadata delta separately from schema-derived CUDA-state/witness tensor bytes.
CPU-auditor metadata, native serialization overhead and whole-root resource
behavior remain open. Neither component sizes nor unit tests prove full fit.

The singleton controller is not a persistent crash-recovery daemon: external
termination of the controller itself can interrupt publication. On recovery,
check its marker, exact process handle/group and any terminal record; never
infer absence of a result means permission to retry. A surviving controller
handles worker failures within the stated monitored deadlines.

Verification results and concrete execution identity are recorded in the
decision, retained inspection records, saved state and final checks artifact.

## Post-execution correction

The approved single observation failed; see native-layout-inspection-results.md.
The pre-execution review missed loss of the worker's structured failure on
nonzero exit and a pinned PyTorch UUID-type mismatch in the existing collector.
The earlier conditional review verdict was not a native success prediction.
Both defects are recorded; neither was silently fixed or used to justify retry.
