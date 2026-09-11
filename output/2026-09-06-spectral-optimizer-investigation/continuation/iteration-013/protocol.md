# I13 — preserve the EMA mean, project innovations

Prospective protocol,7 September2026. Follows completed I12 and the
[candidate design](../iteration-012/next-mean-design.md). This is an autonomous,
in-scope mechanism experiment, not a production optimizer change or a request
for another user approval. No completed, live or consumed failed experiment
may be restarted. No source training or cloud compute is required.

## Scientific question

The native filter learns directions of centered gradient variation but projects
the entire raw gradient. Does retaining its EMA mean outside that space restore
useful learning? Does it also restore persistent-label memorization? Compare
against a control with the same direct current-gradient leakage but without the
historical outside-mean term. A possible constructive improvement deserves a
direct test; I12's failed v-reset remedy does not refute all filtering variants.

## Exact policies

Keep the original observation order and frozen native code. Compute raw batch
gradient g; ingest it once via the canonical observer; then deliver to inherited
AdamW. With beta=.99, post-ingest mean `mu=beta*mu_old+(1-beta)*g`, and native
post-ingest linear projection action P (rank at most32):

- `mean32`: `h=Pg+(mu-Pmu)`.
- `leak01_32`: `h=Pg+.01*(g-Pg)`.

At the same state the exact-real-arithmetic difference is
`h_mean-h_leak=.99*(mu_old-Pmu_old)`. Equivalently mean32 preserves the current
EMA mean and projects the innovation. The `.01` term is fixed by the existing
mean decay, not tuned. Numerical residuals must be retained and checked under
a prospectively tested float32 tolerance. The same native P is used for every
term within a step; no extra observer ingestion or recomputation from another
arm's trajectory. Later policy paths have different means and bases.

P is implemented through the actual retained basis/effective rank, not an
invented full covariance matrix. Approximate basis orthogonality does not break
the algebraic identities for a common linear action P. Do not call the EMA
clean or population-unbiased; self-inclusive P depends on g, and the inherited
mean contains fixed-label history. The `.99^H` direct old-mean coefficient is
not a causal fraction of later behavior.

## Parents, references and plans

Six original current-trained I9 complete states: seeds100/101/102 × parent
steps1500/2000. All weights, modes, buffers, gradients, Adam moments/counters/
group options, observer state and RNG are inherited exactly. No I12 reset
endpoint is a parent. Use the exact six saved I10 plans, including all500×64
batch indices and redraw masks/digits. Fixed corruption and data partitions
remain those bound by the original I9 plans; no regenerated plan replaces them.

Cross fixed/soft/redraw targets with mean32/leak01_32:
**36 new branches ×500 updates =18,000 new training updates**.
The **36 I10 raw/current32 inherited-state comparators** already exist for these
same parents/targets/plans. Hash-bind their full curves/finals and never rerun
their learning trajectories. No raw-trained source, frozen policy, width/LR
sweep, new seed, new split, target selection or official MNIST test set.

Each h0 full evaluation must exactly equal the relevant I10 h0 before any
training step. All new policies advance Adam and observer counters by500 from
their actual saved counters. The original I9 basis is a frozen leakage reference.
All source/input/plan/comparator bytes must be rechecked after acquisition.

## Four separate primary effects

Utility U is negative auxiliary clean CE or auxiliary accuracy (higher better).
At h500, use `U(mean32)-U(current32)` separately for soft and redraw targets,
giving four primaries. Compute per parent, average the two parents within seed,
retain all three seed values, then their mean. No composite score, p-value,
outcome-selected horizon or pooling across target regimes.

Mandatory supporting contrasts at h0,1,10,50,100,250,500:

- mean32 minus leak01_32, testing the historical-mean addition beyond direct leak;
- both new policies versus raw and native current, separately by metric/target;
- each policy's absolute clean progress from h0, on auxiliary and validation;
- all training fixed/soft/clean losses and accuracies, confidence summaries,
  and realization residual `R_zeta=L_fixed-L_soft`;
- fixed-target protection cost: additional realization fitting and clean utility
  versus both raw and native current, not hidden behind soft/redraw gains.

A favorable relative contrast alone is not restored learning. Useful recovery
requires absolute progress on the claimed metric; classification and CE may
disagree. A mean32 gain shared by the leak control is not historical-mean
specificity. A gain accompanied by raw-like memorization is a tradeoff, not
selective denoising. Three reused seed bundles provide conditional evidence,
not independent benchmark validation or a tuned performance recommendation.

## Required mechanism measurements

Each successful training step saves raw/applied norms, total and nominal decay-
adjusted data displacement norms/leakage, observer counts, and norm/energy
diagnostics for g, Pg, g-Pg, mu-Pmu, mu_old-Pmu_old, h_mean and h_leak. Retain
algebra residuals and verify the observer advanced exactly once. First raw-
gradient and post-observation fingerprints must agree between the two new
policies within each seed/parent/objective; their delivered gradients differ.

### Frozen alignment probes

At each scheduled evaluation, on an isolated complete-state clone without
ingesting a new observer gradient, measure the incumbent `mu-Pmu` against:
fixed, soft and clean training gradients and their fixed-minus-soft difference,
plus an auxiliary clean gradient. Use the **existing I9 anchor's1024 loss_train
and utility_aux indices**, unchanged across policies and horizons. Save signed
dots, cosines with explicit zero-vector cases, norms and the underlying CPU
vectors/digests. The incumbent complement is not claimed to equal the next
step's post-ingest historical-mean term. Positive gradient alignment is not a
guarantee of a useful Adam step.

Storage/audit limit: retain mean, projected mean, complement and gradient vectors,
but bind the incumbent basis by digest rather than duplicating it in every
alignment record. H0 and final bases can be checked against retained full states.
Intermediate new-branch states at1/10/50/100/250 are not retained: their signed
alignment arithmetic is auditable, but the projection cannot be independently
recomputed from V. Do not claim full basis or forward replay at those horizons.

The common h0 state is identical across targets/policies: acquire six physical
h0 alignment records and reference them in all36 branches. The remaining
36×6=216 nonzero-horizon records are branch-specific. Additionally probe the
36 saved I10 comparator final states once, for matched h500 interpretation;
these are new state diagnostics, not new comparator training. Total planned
physical alignment records:258, if all branches remain finite. Every probe
restores caller RNG and proves caller-state neutrality.

### Common-state actual-displacement probes

For each of18 parent×target groups, use the first saved I10 branch batch and
its exact targets. On separate scratch copies, ingest the same raw gradient
once and obtain current32, mean32 and leak01_32 deliveries. Compute their actual
inherited-Adam displacements (54 isolated diagnostic Adam calls total), never
advance a persistent training parent. Then add four artificial reciprocal
data-step-norm probes: mean at current norm, current at mean norm, leak at
current norm, current at leak norm. Re-add identical nominal decoupled decay,
apply parameter quantization and verify achieved data norms as in I9.

Use existing1024 loss_train/loss_aux indices for fixed/soft/clean training and
clean/soft auxiliary losses. Retain all three actual and four artificial
displacements, their delivered gradients, common losses, post-ingest basis,
norm controls and CPU tensor evidence. Zero source norms or the unchanged I9
rescale limit yield explicit undefined diagnostics, not substituted steps.
These are diagnostic-only local directional comparisons: they do not control
step magnitude throughout a500-update learning trajectory.

## Failure policy and acquisition integrity

Preflight h0 mismatch/nonfinite, topology, algebra-invariant violations,
serialization/storage/provenance errors, eigensolver/CUDA errors, resource limits
and generic exceptions abort the whole acquisition. Resource callbacks remain
outside branch numerical catches. Pre-training common-state probes and reused
baseline-state probes are prerequisites; a failure there aborts the unit rather
than pretending an intervention branch outcome was measured. Per-horizon probes
on a new learning branch participate in its explicit numerical-failure policy;
structural/provenance failures still abort globally. No automatic restart.

## Engineering checks and resources

The best-practices validator checked the matching PyTorch2.11 primary docs:
[AdamW state loading](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)
does not independently verify parameter-ID order; explicit topology and full-
state digest checks are required. [autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html)
returns gradients without accumulating into `.grad`; the existing clone-based
probe helper additionally restores RNG. Neither fact alone proves whole-state
neutrality, so test it directly.

CPU synthetic tests must cover both identities and native update order,
inherited-current/raw agreement, old-state/RNG neutrality, probe tensor/scalar
consistency, reciprocal quantized norms, exact h0 rejection, numerical sealing
and propagation of resource/structural failures. One source-identical synthetic
GPU smoke checks all six new policy×target cells, probes and serialization,
without using real outcomes to modify the recipe.

One exclusive root under verified `/tmp/spectral-experiment-artifacts` on `/dev/RECONFIGURE_FOR_LOCAL_STORAGE`.
Smoke/full share a2GiB new-artifact cap; no bulky data on the smaller repo mount.
Full1500s cooperative/1800s whole-unit wall limit; smoke100s/120s. Both6GiB
cgroup RAM, zero swap,4GiB TorchGPU allocation, CPUQuota100%, one numerical
thread, Restart=no. Require16GiB available host RAM and8GiB free GPU at launch.
Leave GUI/Stremio and unrelated processes untouched. Preserve partial artifacts
on failure. No cloud charge/reservation or automatic fallback; the user's
cumulative $100 experiment budget remains entirely available.
