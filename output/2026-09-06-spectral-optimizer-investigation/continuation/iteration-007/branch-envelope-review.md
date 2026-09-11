# Complete branch producer checkpoint

Codex — Spectral Optimizer Investigation — 6 September 2026.

The synthetic pipeline now saves complete six-branch results linked to the actual
source step. It checks native gradients, parameter/moment endpoints, shared
observer state, independent clone storage, execution-order equality and caller
RNG restoration before declaring a successful artifact. This strengthens the
planned mechanism experiment's evidence trail. It is engineering evidence, not
a new finding about the spectral optimizer or a scientific launch approval.

## Implemented path

[branch_execution.py](branch_execution.py) verifies pinned anchor/witness files
through the existing writer descriptor and restricted CPU loading, then requires
exact equality to supplied decoded values and strict source/context validation.
It applies the established exact built-in safe-global policy before decoding;
no unrestricted fallback or new allowlist entry is introduced.

Fixed profile factories restore one candidate clone. Its actual native mean CE,
raw/current gradients and post-ingest observer match the sealed live witness.
Lagged delivery uses the previous basis, with identity action for a missing basis.
Each defined branch starts from the original complete t-1 anchor, not the
post-ingest candidate state. Every parameter receives a non-null owned gradient,
including the zero branch; AdamW executes once, then gradients are cleared.
The private observer remains at t-1; only the shared candidate observer is t.

Canonical and reverse execution retain live clones concurrently for actual
storage-disjointness checks. Both orders require direct equality of saved native
gradients, masks, parameters, moments, observer and RNG checks. Measurements use
the unchanged assembler, after endpoint validation. Proof hashes are recorded
after actual comparisons; they do not replace those comparisons. The new full
branch envelope is sealed in the same store as the anchor/witness. Failure
terminals that store and retains completed artifacts; it does not restart a step.

[artifact_envelopes.py](artifact_envelopes.py) validates the complete ordered
producer representation against explicit candidate/measurement expectations.
It cross-binds parameter entries to the redundant flat endpoint, moments and
counters, raw endpoints to float64 displacements, before-loss/gradient hashes,
all fifteen pair/contrast memberships and domain masks, complete references and
all proof metadata. Unsupported types, aliases and tensor representations are
rejected before cloning. The source manifest now requires the new producer and
contract; absent future phase/audit files still prevent scientific collection.

## What the independent checks establish

[test_branch_pipeline.py](test_branch_pipeline.py) reloads the complete branch
artifact, extracts raw saved values and independently recomputes numerical
measurements and AdamW updates using the existing NumPy auditor. It also directly
checks that the ORIGINAL live source objects remain unchanged by all branches.

- Ordinary synthetic path: six defined branches, 24 parameter-level AdamW
  checks, all 390 specified independent contrast checks and exact audit counts.
- Engineered synthetic edge case: an orthonormal prior basis on inactive model
  coordinates gives an actual zero lagged direction and positive current norm.
  The restored branch remains undefined; five branches, 20 AdamW checks and all
  260 unaffected independent contrast checks remain. This explicitly constructed
  starting state is not presented as a continuation of the warm-up trajectory.
- Separate caller-state test: an earlier anchor is replayed while the caller has
  a different RNG state. Branches use the anchor RNG; completion restores the
  caller RNG, without mutating the live source model, optimizer or observer.

The independent adversarial suite also exercises actual missing-basis identity
action, malformed references, unknown/reordered keys, incorrect scalar types,
aliases, grad-bearing/view tensors, false proof flags, and jointly malformed
expected/payload comparison masks. A noncurrent endpoint with consistently
rehash-adjusted proof but stale measurements is rejected by raw displacement
cross-binding. These are deterministic CPU fixtures, not a multi-seed study.

## Findings corrected during implementation

The initial live-storage gate incorrectly treated model Parameters as ordinary
saved Tensors. It also required the canonical filter's candidate gradients to
own their storage. The filter actually assigns disjoint contiguous views from a
flat buffer. The live check now measures those occupied spans while keeping
strict ownership for saved tensors and newly assigned branch gradients. It
does not change the filter or clone away an unexpected overlap.

Parent review found that native execution initially selected the CPU saved
candidate copy, and that requiring caller RNG to equal every anchor would fail
when branch analysis follows several completed source trajectories. Native
assignment now uses the native candidate; explicit caller-RNG restoration is
separate from the invariant that every branch preserves the anchor RNG. The
parent initially misread clone_tree as retaining device placement; inspection
confirmed it already makes CPU copies. The execution-input issue was separate.

Additional review added actual assigned-value capture, model-mode and post-step
null-mask checks, safe-global preflight, input/expectation ownership separation,
complete comparison memberships and direct endpoint/displacement bindings.

One adversarial run had 16 passing tests and a zero-lagged-case failure. The
expected-candidate validator incorrectly required direction_error=0 for an
unscaled zero-target vector, while the unchanged response arithmetic specifies
None. The corrected validator accepts None only at zero target and zero error
for positive unscaled targets. The unchanged case and final expanded adversarial
suite pass; no null rule, numerical threshold or fixture was weakened.

Final review additionally required a post-seal anchor-RNG comparison before
caller restoration, so serialization cannot consume RNG unnoticed. A failure
after sealing must retain the branch file and terminal record, not erase it.

## Verification

The final parent CPU suite passes 236 I7 tests (91.813 seconds), plus 17 repository
tests (1.279 seconds) and six reminder tests (2.786 seconds): 259 total. This adds
25 I7 tests: 18 independent adversarial envelope tests, four execution/failure
tests and three parent raw-file-to-independent-audit integrations. Knowledge lint
passes for 14 pages and 11 indexed content pages; whitespace checks pass.
The post-seal RNG-injection regression preserves one indexed branch artifact,
restores caller state, retains the terminal record and rejects retry before forward.
Commands, source hashes and scope are in branch-envelope-checks.json (artifact not distributed in this public snapshot).
Prior check records remain historical and unchanged.

## Limits and next work

The pure validator establishes consistency with separately acquired expected
candidate/measurement values; it does not prove control-flow history or independently
reproduce native CUDA arithmetic. Producer proof flags are attestations. Numerical
checks from the saved file are independent of the producer's flags, but the
complete independent-audit envelope remains unfinished. Inputs are locally
trusted; restricted loading is not a hostile-memory-metadata sandbox or protection
against malicious same-user rewriting/import substitution.

Plans still use a separate tiny fixture store. No single-root scientific phase
controller, immutable all-phase manifest or runner entrypoint is exercised. The
earlier 954.324516-MiB projection is unchanged and incomplete; no complete storage
fit or runtime certificate follows. Full-width artifacts, adverse audit records,
all-phase resource accounting and separate native-pilot review remain required.