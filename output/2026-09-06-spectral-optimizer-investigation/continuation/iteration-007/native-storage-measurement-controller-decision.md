# One-shot CPU measurement controller implementation

Codex / Spectral Optimizer Investigation, 7 September 2026.

Recovered cleanf04b199ad93b5c39bf022fdd0aa6f2d22983f65c and active goal. The
original objective, current state, next steps, controller next-scope, repository
knowledge instructions and applicable workflow were read. Previous turn made
verified progress:87 checks, reviewed symbolic recipe, clean committed source
verification. No live experiment/test or production M/A/attempt/frozen directory
exists. Preserve the consumed inspection and all completed experiments.

Resume the same I7 Step9 task, not a new issue, research run, timer installation
or completed experiment. The user's autonomous continuation instruction takes
precedence over the interactive follow-up skill's new-issue/confirmation path.
The research workflow supplies state recovery, evidence discipline and leaf
workers; no new Step10 scientific-result audit or paper restart is implied.

Implement the controller next-scope. One leaf owns the durable fixed M slot;
one owns the sequential CPU worker; main owns orchestration/process guards,
schema/collector integration, source closure and assembled tests. A third leaf
independently reviews design and code. Main is sole hub: no leaf spawning,
user dialogue, outside writes/sends or commits. Interfaces agreed before code.

The slot helper reserves empty M with exclusive no-follow descriptor-relative
open before measurement source/runtime work, fsyncs file and directory, and
holds the exact identity through one finalization attempt. Any zero/partial
slot remains consumed. A stays absent; no retry/replacement/sidecar. Finalization
validates the actual bounded authority schema and writes through the held fd.
Neither this API nor a hash is hostile-same-user execution authentication.

Measurement and later native workers must use the same fixed frozen root
`process_supervision.REPOSITORY_ROOT`. M/A remain at their existing original
repository paths, outside that checkout; reservation cannot dirty the source
tree being authenticated. No root equivalence waiver based only on equal C.

M will be prospectively versioned to v3 to retain exact controller-observed
supervision facts in addition to the v2 full diagnostic environment. A child
candidate has no authority until its exit, resource scope and cleanup are
observed and the final bounded record is validated. Unknown/pre-observation
failure preserves empty M rather than inventing CUDA/RNG/status observations.
The complete recipe protocol remains v2; no existing production M is migrated.

Process design must cover bounded stdout/stderr, active wall/RSS observation,
unreaped/pidfd ownership through group cleanup, worker parent-death handling,
and the collector's short Git commands. Direct worker parent-death signalling
alone is not generic descendant protection: Linux clears it on fork. Final
process design uses the reviewed transient user-service design below.

The controller is the Type=exec main process of the fixed transient user service
`i7-native-storage-measurement-001.service`. Its worker and every Git descendant
inherit its unified cgroup. Before consuming M, verify the actual service and
kernel cgroup properties: RuntimeMaxSec120s, MemoryMax2GiB, MemorySwapMax0,
MemoryAccounting=yes, OOMPolicy=kill, KillMode=control-group, SendSIGKILL=yes,
TimeoutStopSec1s, Restart=no, TasksMax128, standard streams null, LimitCORE0,
MemoryZSwapMax0 and TasksAccounting=yes. Independent review strengthened
OOMPolicy from stop to kill, verifying kernel memory.oom.group1.
An unavailable or mismatched control refuses work; there is no direct-process
fallback. Normal observation retains exact child exit and cleanup; controller
death causes systemd to stop all remaining service members. The stop grace can
extend cleanup beyond the active120s; neither scheduling nor uninterruptible
kernel sleep is a hard real-time guarantee.

Transient unit/cgroup/manager start-stop metadata is OS supervision state,
not retained study data. No worker output is routed to journald, no study
sidecar or second failure slot is created. M is the only study-data output.
Tiny unique test service names may verify this lifecycle without production M,
full tensors or the fixed production service. Actual launch remains deferred.
Cgroup charged memory is distinct from summed process RSS; record both scopes,
enforce the kernel charge limit and actively sample summed resident memory.
Hold the cgroup directory identity; reject nested groups and final extra members.
The56KiB candidate channel reserves8KiB for supervision under M's64KiB bound.
Reported elapsed time ends before final slot writing; systemd still governs that
write. Independent actual-run admission must check terminal service exit/result,
not infer successful finalization solely from a syntactically complete M.
Sources: [systemd255 service](https://raw.githubusercontent.com/systemd/systemd/v255/man/systemd.service.xml),
[resource controls](https://raw.githubusercontent.com/systemd/systemd/v255/man/systemd.resource-control.xml),
[whole-group shutdown](https://raw.githubusercontent.com/systemd/systemd/v255/man/systemd.kill.xml),
[kernel cgroupv2](https://docs.kernel.org/admin-guide/cgroup-v2.html).

Current implementation tests are tiny filesystem/CPU/serializer/subprocess
contracts: hidden/uninitialized CUDA, four thread variables1, CUBLAS4096:8,
verified big/tmp,120s outer timeout and active2GiB RSS watchdog with exact scope
reported. No production M/A, frozen checkout, actual full-width specimen,
native initialization, IDX, scientific plan or training is created during these
tests. Keep the production native gate unconditionally closed. Actual measured
execution requires a separate assembled resource/freeze review, not an inference
from passing unit fixtures. Do not use a checkpoint as goal completion.

Primary API checks: file fsync alone does not ensure parent-directory durability;
use both. Python advises explicit argument arrays/absolute executable paths,
start_new_session rather than preexec_fn, and warns that communicate buffers
output and does not kill its child on timeout. Bounded selectors and explicit
cleanup are therefore required here. Linux parent-death guards need a post-arm
parent-identity check and do not automatically protect forked grandchildren.
These are API facts, not proof of historical execution or real-time guarantees.
Sources: [Python3.12 subprocess](https://docs.python.org/3.12/library/subprocess.html),
[Python3.12 OS](https://docs.python.org/3.12/library/os.html),
[Linux open](https://man7.org/linux/man-pages/man2/open.2.html),
[Linux fsync](https://man7.org/linux/man-pages/man2/fsync.2.html),
[Linux parent-death signal](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html).
