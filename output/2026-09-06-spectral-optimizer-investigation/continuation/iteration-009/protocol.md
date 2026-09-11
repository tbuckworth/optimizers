# I9: neural covariance sources and same-state update utility

## Scientific purpose

I8 showed that the actual filter can preserve useful learning when useful batch
variation identifies a subspace. It also showed that Adam-induced temporal
geometry can succeed without batch variation. I9 measures those candidate
sources in a neural setting, and tests the utility of the *delivered* AdamW
update from identical states. Conditional one-step effects are not long-run
mediation, and a null immediate-loss result does not disprove selective learning.

## Fixed acquisition

- Fresh bundles: seeds 100, 101, 102 under RNG namespace
  `SeedSequence([20260907,9,seed,stream])`. All random plans saved before training.
- Cached MNIST training IDX files only. Per seed: 5,000 training, 5,000 disjoint
  clean validation, 5,000 disjoint auxiliary examples. No official-test file.
- Fixed .9 uniform replacement of training labels, allowing a replacement to
  equal the clean label. Report actual replacement and incorrect-label fractions.
  Soft comparator is `q=.1*one_hot(clean)+.9/10`, **not** .1 true-label mass total:
  the true class receives .19. Do not redraw labels during learning or probes.
- Model: 784–64–ReLU–10, 50,890 parameters, no dropout/batch norm/augmentation.
  Float32 model and native basis, float64 scalar reductions and native small solve.
- Two source policies: raw AdamW with a passive canonical observer, and actual
  current hard rank32 filtering. Same initialization, corruption and training
  batches within each seed. Raw observer sees raw gradients but does not alter
  the update delivered to AdamW.
- 2,000 steps per source, batches of64 with replacement. AdamW lr .001,
  weight decay .01, betas (.9,.999), epsilon1e-8, foreach/fused false.
- Native stable filter: rank32, decay .99, warmup100, hard weighting, no
  normalization/adaptive rank, relative eigenvalue tolerance1e-8, absolute floor0,
  scheduled stabilization every100 observations. Canonical source unchanged,
  SHA256 `9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943`.
- Source anchors are after completed steps100,500,1500,2000: 24 in total.
  All future training batches are predetermined; probes use distinct RNG streams.
  Raw/current warmup state must agree through step100.

## Complete anchors and measurement neutrality

Save independently CPU-cloned model parameters/buffers, parameter gradients,
module modes, AdamW state and hyperparameters/counters, every native observer
attribute except live model/optimizer/parameter aliases, Python/NumPy/Torch CPU
and CUDA RNG states, training position, initialization/split/corruption and probe
plans. Restore exact device placement (native singular values are CPU-resident).
Snapshots must load with `torch.load(weights_only=True)` and reproduce the same
next update. No model-only snapshot may be described as complete state.

Probes use `autograd.grad` and cloned fork state; they must leave the source
state and RNG unchanged. Independently compare complete digests before/after
every anchor probe. Synthetic tests establish round-trip, next-step equivalence,
probe neutrality and passive-observer raw-update identity before neural execution.

## Frozen-state covariance and gradient components

At each anchor, before any probe observation, freeze the previous mean a and
native linear action P=`V(V.T.)`. Draw32 independent training-batch pairs g,g'
from the fixed noisy dataset, 64 examples per batch, with replacement. The
same pair plans are used at corresponding raw/current-source anchors.

Retain every pair's full and P-applied quantities:

```
fresh    = .5 ||g-g'||²
surprise = (g-a) dot (g'-a)
innovation = .5 (||g-a||² + ||g'-a||²) = fresh + surprise
```

Their expectations separate conditional batch covariance from squared mean
surprise. Individual/average surprise estimates may be negative. Preserve signs
and raw pair values; do not clamp or turn unstable surprise ratios into evidence.
Report seed-first means by source and anchor, fresh/innovation and
surprise/innovation, retained/full fresh energy, and retained/full innovation.
Conditional pair uncertainty is not uncertainty across independent training runs.
Pre-probe P is fixed for these estimators; no self-inclusive covariance claim.

On the same paired inputs, accumulate mean clean, fixed-noisy and soft-q
gradients, and their signed Gram matrix, projection retention and components
`soft-clean` and `fixed-soft`. Separately measure auxiliary clean and soft-q
gradients on1,024 auxiliary draws. The soft-q gradient is a hypothetical
fresh-label comparator at fixed parameters, not the conditional gradient of the
realized noisy dataset or proof that fresh label covariance is isotropic.

## Same-state counterfactual updates

A distinct, predeclared batch of64 supplies the update input g0. Clone the
complete anchor; let the canonical observer see g0 once before projecting,
producing the native current action and p0. Save g0, p0, the current basis and
counterfactual displacement vectors. This *current* action is different from
the *previous* action used by frozen-state covariance probes.

From identical parameters, moments and step counter, apply four real AdamW
forks: raw g0; native p0; raw rescaled to norm(p0); native rescaled to norm(g0).
These input-norm controls do not claim to match actual AdamW movement.

For raw/native total displacements Delta, define data displacement
`delta = Delta + lr*wd*theta_before`, separating the identical decoupled decay.
Add two **artificial direct-displacement controls**: rescale raw data delta to
the native data norm, and native data delta to the raw data norm, then add back
the same decay. This is not an optimizer input-scale search and does not produce
a new optimizer state or train a trajectory. It isolates a local norm/direction
comparison of the already-delivered data steps at the fixed anchor.

Zero/undefined norm cases get explicit null reasons, never silent substitution
or an enormous unbounded multiplier. All vectors and losses must be finite;
nonzero norm matches are checked after float32 application, with float64 norms.

Evaluate before/after finite losses on the same separately drawn1,024 training
examples (clean, fixed noisy, soft-q labels) and1,024 auxiliary examples (clean,
soft-q). Evaluate model logits in float32, reduce cross-entropy in float64.
Also retain signed first-order utilities `-g_component dot delta_data`, norms
and step energy outside previous/current basis. Actual stored displacement
vectors permit independent audit beyond I8's sparse energy snapshots.

## Primary interpretation and secondary learning curves

Primary local-utility contrasts are auxiliary clean finite-loss change:

1. native data direction at raw data-step norm minus actual raw update;
2. actual native update minus raw data direction at native data-step norm.

Negative favors native direction. Primary source is current32; average anchors
1500/2000 within each seed, then report all three seed values and their mean.
Retain the two contrasts separately. Other anchors, raw-source states, soft-q,
fixed-label loss and input-norm controls are mandatory secondary measurements.
No choice of favorable source/window/loss after outcomes. Equal data-step norm
with common decay does not imply equal total-step norm; document that distinction.

Covariance-source estimates are companion mechanistic measurements, not a
separate search for a favorable primary p-value. High retained covariance energy
alone does not mean utility. Useful independent-gradient enrichment plus a
seed-consistent matched-data-step directional advantage supports a useful-spike
account locally; surprise-dominated capture with utility instead motivates
optimizer/observer-memory ablations. Mixed signs or weak estimates remain
inconclusive. These local outcomes do not uniquely identify long-run causes.

Save clean validation CE/accuracy and clean/fixed training CE/accuracy at
0,100,...,2000. These descriptive curves show whether useful learning continues
while corrupted-label fitting slows. Report full curves and both best-validation
and endpoint quantities without an official-test or generalization claim. Do
not tune ranks/seeds/stopping metrics to make the mechanism look favorable.

## Resources and provenance

One desktop RTX3090; no foreign training process at launch. Existing desktop
display/Stremio use is left untouched. Require >=8GiB free GPU memory and
>=16GiB available host RAM. One CPU thread; deterministic PyTorch algorithms,
TF32 and cuDNN benchmarking disabled; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

New exclusive output directory on verified `/tmp/spectral-experiment-artifacts` (`/dev/RECONFIGURE_FOR_LOCAL_STORAGE`).
No bulky artifact goes on the nearly full workspace. Budget <=512MiB artifacts,
<=6GiB cgroup host memory and <=4GiB PyTorch GPU allocation; 1,500s cooperative
wall deadline and1,800s systemd whole-process cap, including probes, snapshots,
validation and serialization. Check bounds during training and around writes.
Persist complete anchors and compact progress as they finish. Partial failures
remain partial evidence; no automatic restart or overwrite after a cutoff.

Prospective source/configuration and dataset hashes are recorded before launch;
the source/protocol must be committed. A distinct tiny synthetic GPU smoke may
verify the target device's snapshot neutrality without MNIST or outcome tuning;
it is an implementation check, not an extra training seed. No legacy I7 module,
consumed measurement filename, source freeze or service is modified.

## Documentation check

The best-practices skill was used to verify
[AdamW decoupled decay](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html),
[probability-target cross entropy](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cross_entropy.html)
and [tensor/basic-type state serialization](https://docs.pytorch.org/docs/2.11/notes/serialization.html)
against installed PyTorch2.11 semantics. Those are API facts, not evidence for
the scientific hypotheses. Modal identity was read-only verified with no own
project apps or spend; its platform retry behavior is a future cloud concern,
not a reason to delay this locally available scientific acquisition.
