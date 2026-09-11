# Independent source review

Date: 2026-09-09
Scope: prospective selectivity-boundary acquisition; no scientific data read or
execution performed.
Status: **CONDITIONAL / NOT YET LAUNCH-CLEARED.** The reviewed implementation is
scientifically coherent, but it was still changing and had no handed-off fixture
or implementation file at the requested close of this review. One registered
summary omission remains in the reviewed bytes.

Reviewed receipts at close:

- `experiments/spectral_selectivity_boundary.py` (860 lines), SHA-256
  `40d6c7a5eb8936274942a293ef45fe95b0bfcc03655c8f79861cd11c81765a7f`
- `output/2026-09-09-spectral-selectivity-boundary/protocol.md`, SHA-256
  `2666a79a9afd6baca82a463ae793c62eeb76e2910c57d8b9a89f01d00ddb4d0a`
- pinned canonical filter `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`
- pinned I9 core `706851828b7120bea0d7a41bd0fa75474ac800b2df71087941d0140c8d1970ec`

The proposed fixture and implementation files did not yet exist. I therefore
did not run a test suite. The only executable check I made was importing the
inert module and calling its analytic `byte_inventory()`; it reported an upper
estimate of 2,416,202,728 bytes against the 3,221,225,472-byte cap, before the
separate 1 MiB failure reserve. This did not read MNIST or run a model.

## Findings that pass source inspection

The plan generator uses disjoint per-class train/held-out IDs and independent
named PCG64 streams. Warmup batches contain only the 4,950 majority examples.
The code performs one clean, raw-delivery, observer-recorded 100-step warmup per
seed and deep-snapshots model, gradients, AdamW, observer and RNG. Every cell and
policy restores that same snapshot and checks its digest. The raw policy then
drops the observer; native and norm-control policies each observe their own raw
gradient exactly once per update. At update 101, raw-gradient hashes are checked
across all three policy forks within a cell and post-observation tracker hashes
across the two filtered forks.

The norm control implements the registered law using float64 norms and a
float32 delivery cast. A zero raw norm requires a zero native target; a zero
target produces an explicit zero gradient followed by an ordinary AdamW step.
The post-cast relative tolerance is `10 * eps(float32)` and failure is not
clipped or hidden. This matches the **input gradient norm**, not AdamW's eventual
parameter displacement.

The Sham construction is correct. Within each eligible true digit, it preserves
the Shared patch count exactly and applies largest-remainder allocation over
poison status, with ascending status as the exact tie-break. It samples without
replacement inside strata, preserves the identical Shared/Sham target vector,
matches total patch count, and excludes true digits 0 and 8. This produces
conditional near-independence rather than exact iid independence, as stated.
The later-added bindings hash the actual normalized train inputs, clean train
inputs, and both held-out input tensors, not only their masks.

Native diagnostics occur only at updates 101, 500 and 2000, after the current
training gradient has updated the observer and fixed the action, but before
AdamW changes parameters. `core.snapshot` includes `.grad`, model modes,
optimizer state, complete observer state and Python/NumPy/CPU/CUDA RNG states.
The before/after probe checks therefore cover the relevant live state;
`autograd.grad` does not accumulate into `.grad`. The post-update snapshot is
taken before gradients are cleared, preserving the applied action.

The saved pre/post states define the actual displacement. The reported split
uses the exact AdamW decay-first operation `theta_decay=(1-lr*wd)theta_before`,
then defines adaptive movement as `theta_after-theta_decay`; the two components
telescope to the observed movement. Numerical-span off-components are computed
from an orthonormalized copy of the stored `V`, while probe retention continues
to use the native `V(V^T g)` action. Signed local utility
`-mean(g_probe)^T delta_theta` and finite probe CE change are both retained and
are not forced to agree.

The cue population is implemented correctly: the primary mask excludes true 0
and rare 8, leaving exactly eight majority nonzero digits and 4,000 held-out
images. Patched and unpatched rates use the same examples and prediction-
independent masks. The declared Shared-minus-Sham, policy-minus-raw interaction
is formed with the right signs.

Static accepted IDX hashes, canonical/core hashes, current scientific source,
fixtures and protocol are intended to be source-pinned; on-disk hashes are
compared with the current commit before output creation and rechecked before
completion. The output directory is exclusive, files use `xb`, the roster must
contain exactly 36 trajectories and 36 native diagnostics, and no resume,
fallback, selector or retry path exists.

## Required before launch

1. **Registered patched classification is missing from the automatic summary.**
   `endpoint_metrics()` includes unpatched rare/majority/balanced CE and
   accuracy, cue excess, ASR and wrong-target fit, but omits the protocol's
   patched true-label CE/accuracy with the same group breakdown. Raw curves do
   contain these values, so no evidence would be lost, but `results.json.summary`
   and its automatic policy contrasts silently omit a registered endpoint.
   Add distinct patched rare, majority-macro and balanced-total CE/accuracy keys
   and fixture their aggregation before freezing.

2. Freeze the final runner, protocol, fixture and implementation receipts, then
   run only the producer's fabricated fixtures. At minimum they should cover
   exact Sham quotas/ties and exclusions; diffuse expected/realized changes;
   patch nonmutation; warmup snapshot restoration and first-action equality;
   native/norm/zero-gradient semantics and cast tolerance; probe `.grad`/Adam/
   observer/RNG neutrality; diagnostic timing and movement telescoping; primary
   cue masks/interactions; null summaries; capped JSON/NPZ/tensor writes; and
   exact roster rejection. The reviewed source hash is not a final receipt.

3. Before the one launch, independently confirm that the prior live acquisition
   is terminal and the RTX3090 has no unapproved compute occupant. The runner
   verifies its own systemd CPU/RAM/swap/runtime/no-restart limits and requires
   8 GiB free, but the reviewed bytes do not themselves establish prior-stage
   terminality or exclusive GPU use. A main-owned launch guard can satisfy this
   operational condition without changing scientific code.

## Interpretation limits to preserve

The wrong-label probes are the first 32 changed positions in a class-blocked
training plan. They are a fixed convenience subset—often true digit 0 in
Diffuse and true digit 1 in Shared/Sham—not representative of all wrong
examples. The added true-label counts make this visible. Their coherence,
retention and local utility must not be extrapolated to the wrong-example
population or compared across cells as if class composition were matched.

Rare digit 8 changes both class identity and support and is absent from warmup,
so this is rare-class acquisition under a stated shift, not a frequency-only or
counterfactual-memorization result. Shared and Sham hold targets and patch
prevalence/class margins fixed but patch different concrete images. Patch excess
measures behavioral cue sensitivity; its interaction does not identify a
covariance mode or prove a defense. Diffuse is not severity-matched to Shared.

Finally, input retention is not update retention. Carried Adam moments,
coordinatewise preconditioning and weight decay can move the actual parameter
step outside the current native span. The diagnostic's actual movement and
finite losses address that locally, but only on training probes at three native
states. They are not held-out directional utility, a long-horizon mediator, or
evidence that a synthetic cue represents human-values misalignment. Low cue
sensitivity with stalled common or rare learning remains a failure, not
selective protection.
