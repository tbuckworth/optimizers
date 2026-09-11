# I12 — Adam moment content and counter interventions

Prospective protocol, 7 September 2026, following completed I10/I11. This is
autonomous in-scope research, not a researcher-plugin approval workflow. No
completed, live or failed experiment is restarted. Desktop RTX3090 only;
cloud spend/reservations remain $0 of the user's cumulative $100 budget.

## Question and scope

Does inherited second-moment state disproportionately restrict filtered
adaptation after persistent corrupted labels are replaced by soft expected
targets or fresh redraws? How does that differ from first-moment history and
the bias-correction counter? These are dynamic intervention effects, not a
causal mediation decomposition or a fraction of the filter's benefit explained.

Use **all six late current-trained I9 complete states**: seeds100/101/102,
parent steps1500/2000. Cross soft/redraw targets with raw/current32 policies.
Use the **exact six saved I10 batch/redraw plans**. The24 inherited-state
comparators already exist in I10 and are hash-bound inputs, never rerun.
For each seed/parent/objective, use identical saved batch indices and, for
redraw, identical replacement masks/digits across all four interventions,
both policies and the inherited reference. No per-arm regeneration or RNG
advancement replaces the recorded plans. Soft-q is deterministically rebuilt
from the same clean labels and same indexed examples in every arm.
No raw-trained source, fixed-target branch, frozen-policy branch, changed
learning rate, new source training, new data split or outcome-selected subset.

Add four interventions to separate copies of each full I9 state:

| Arm | First moment m | Second moment v | Adam counter |
|---|---|---|---|
| zero_m | zero | inherited | inherited |
| zero_v | inherited | zero | inherited |
| zero_mv | zero | zero | inherited |
| fresh_adam | zero | zero | zero |

This gives **96 new branches ×500 updates =48,000 planned new updates**.
The first three plus the existing inherited arm form the fixed-counter two-by-
two moment-content factorial. Fresh-versus-zero_mv isolates the counter change
conditional on empty moments; fresh_adam itself bundles three changes.
Never reset a counter while retaining nonzero moments.

Change only exp_avg, exp_avg_sq and step as specified, preserving tensor
dtype/shape, parameter IDs/order and group options. All other state remains
exact: weights, buffers, gradients, modes, observer and all RNG streams.
Validate the actual non-AMSGrad AdamW topology and recompute complete-state
digests before/after copying. Save each of24 edited parent states once, plus
its mutation proof. Original parents and I10 baseline artifacts stay immutable.

## Execution and measurements

Restore each edited full state exactly. Its neutral h0 evaluation must equal
the corresponding recorded I10 h0 evaluation **before any new update**; edits
to optimizer history cannot change predictions. Original I9-parent V remains
the frozen diagnostic reference. Raw/current observer behavior and numerical
steps use unchanged I9/I10 helpers, including self-inclusive current delivery.
Adam counters diverge deliberately, while observers continue from their original
counter. No counter is inferred from the local branch horizon.
All pre-update state/evaluation checks are engineering prerequisites: an h0
nonfinite or mismatch cannot be attributed to a moment-induced learning step.

Horizons **0,1,10,50,100,250,500**. Retain complete train/auxiliary/validation
evaluations, clean CE/accuracy, soft and fixed-label losses, realization residual,
confidence summaries, every successful step diagnostic, and one final complete
state per successful branch. Initial mutation records contain moment/counter
checks. All I10 baseline trajectories and corresponding source/plan/input hashes
are bound and retained as references, not newly acquired data.
Retain the first successfully delivered step1 gradient's digest, without an
extra forward/update. Within seed/parent/objective/policy it must match across
all four moment interventions, since weights, observer and targets initially
match. Subsequent gradients may differ as learning paths diverge. A numerical
failure before that measurement leaves the digest unavailable, not fabricated.

## Four separate primary contrasts

Define utility U as −auxiliary clean CE or auxiliary accuracy; higher is better.
For an intervention a, define `B_a=U(current32,a)−U(raw,a)`.
For each of soft/redraw and CE/accuracy, the primary at h500 is

`Delta_v_minus_m = B_zero_v − B_zero_m`.

Compute per parent, average parent1500/2000 within each seed, retain all three
seed values, then their mean. Positive means v deletion improves the relative
current-versus-raw gap more than m deletion. In the balanced fixed-counter
factorial this equals the marginal v-removal effect minus the marginal
m-removal effect: the inherited and zero_mv terms cancel. It does not isolate
v from the known transient caused by keeping old m and the old counter.

There are four primaries, no p-value, composite score, selected favorable metric
or pooled objective. This is adaptive research on reused MNIST splits and
states, not an independent replication or a tuned performance benchmark.

Mandatory supporting contrasts, all horizons and all individual parents/seeds:

No retrospective best-horizon selector is primary. Earlier horizons reveal
transients and later path dependence; do not quietly substitute one for500.

## Numerical failures are outcomes, not discarded cells

Do not clip steps, tune the learning rate, stop at a finite loss/amplitude
threshold or retry an adverse branch. A v-only reset can legitimately create a
large finite transient. Only an **explicit nonfinite numerical detection** in
objective, gradient, parameters, optimizer/tracker state, displacement, scalar
diagnostic or evaluation may stop that individual branch.
Branch-local outcome handling begins only with update1, after a valid h0 seam.

The core catches only its own nonfinite checks and tightly specified existing
I9/I10 finite-check failures. Topology, target validity and seam errors are not
numerical outcomes. Resource callbacks run outside the numerical catch. Generic
exceptions, CUDA faults, eigensolver exceptions, OOM, timeout, storage, serialization,
hash/parent/plan mismatch or inability to seal evidence terminate the entire run.
The exact JSON nonfinite-number rejection from the existing step diagnostic is
also an explicit numerical detection; other JSON/serialization errors are not.

On a branch-local numerical failure, record exact attempted step and stage,
number of successful step diagnostics, partial evaluations and the failed
terminal state. Also retain the last successfully evaluated complete state and
its horizon; this is **not** claimed to be the immediately pre-failure state
when intervening steps occurred. Bind terminal bytes by file hash even if a
nonfinite scalar prevents a normal complete-state digest. Record that limitation,
never replace NaN with a finite outcome or label the failed state successful.
Only after sealing these artifacts may the next branch restore an independent
parent copy. Missing scheduled endpoints remain missing.

Any primary requiring a missing paired parent contribution is unavailable:
no survivor-only three-seed mean, no imputation and no selected replacement
seed. Fully observed contrasts retain their declared status, but the overall
mandatory study is scientifically incomplete if any branch fails, even when all
96 attempts are terminal and the acquisition has finished. Report numerical
failure counts and identities before successful-case outcomes. Keep all evidence.

## Predictions and interpretation boundaries

The inherited-v hypothesis predicts positive Delta_v_minus_m, preferably across
seeds/regimes with absolute improvement in current adaptation. Similar changes
in raw and current instead imply generic adaptation effects, not filter-specific
restriction. Persistent current deficits after edits favor ongoing projection
or weight/feature geometry as competing limits, without identifying either alone.

The reviewed reset algebra (artifact not distributed in this public snapshot) predicts startup
amplitude differences: zero_mv with an old counter is about2.79–2.94 times a
fresh Adam step here when epsilon is negligible; zero_v with inherited m may
be more extreme. A negative zero_v result is therefore not proof that inherited
v was beneficial. Direct EMA coefficients are not causal effect fractions;
early differences can persist through changed weights and future gradients.
Redraw may rebuild v differently from soft targets despite equal conditional
mean gradients. This proposed pattern is not yet a measured mechanism.

Preserve I8's constructive selective-learning result, I9's adverse local probes,
I10's target/source boundaries, I11's protection without a mean best-stop gain,
and earlier I4/I6 contradictions. No optimizer default or literature novelty claim.

## Engineering checks and resource envelope

The best-practices validator checked current primary
[PyTorch AdamW state documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).
Loading associates parameter IDs by order without additional verification;
explicit topology/mutation checks and full-state restoration are therefore
required. Deleting selected moment values, not arbitrary optimizer metadata,
is a deliberate experiment rather than a recommended training practice.

CPU synthetic tests cover the exact mutation mask, unchanged caller state/RNG,
first-step recurrence/amplitude, baseline finite-path equivalence, neutral
evaluation, early seam rejection, branch-local nonfinite handling and propagation
of resource/structural errors. One source-identical synthetic GPU smoke checks
all16 intervention×objective×policy combinations, serialization, and unchanged
inherited-path agreement with I10. Smoke uses synthetic data only, not real
outcomes to alter configuration. Do not mutate acquisition source after freeze.

One exclusive attempt root under verified `/tmp/spectral-experiment-artifacts` on `/dev/RECONFIGURE_FOR_LOCAL_STORAGE`.
Smoke and full share a **2 GiB new-artifact cap**, accommodating24 edited parent
states,96 final states, partial-failure snapshots and raw diagnostics. Full has
1,500s cooperative /1,800s whole-unit wall limits; smoke100s/120s. Both have
6GiB cgroup RAM, no swap,4GiB TorchGPU allocation, one numerical thread,
CPUQuota100%, Restart=no. Require16GiB available host RAM and8GiB GPU free;
leave desktop apps alone. Preserve partial artifacts on any failure; no automatic
retry, cloud fallback or extra budget reservation. Verify all source/data/parent/
baseline/plan hashes again at acquisition completion.
