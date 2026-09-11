# I17 — directional temporal response at matched stationary gain

Prospective protocol, 8 September 2026. No I17 scientific outcome exists.
This separate continuation implements the delivered [I16 next design](../iteration-016/next-gain-matched-design.md),
under the user's autonomous research authority. No earlier experiment is rerun.

## Question and opposing predictions

Can the learned spectral action usefully route fast and slow responses once
the scalar and spectral policies have the same conditional stationary gain
and the same manual parameter shrinkage? I16 established a strong isotropic
competitor, not that directional routing cannot help. The opposing steelman
is that reducing spectral's high-gain directions restores a better balance.
Neither neural ranking follows from the stationary algebra.

The panel remains adaptively reused MNIST, not fresh-task confirmation. Keep
the constructive I8 toy and I15 learning progress alongside I16's mostly
scalar-favorable results. No production default change is in scope.

## Fixed acquisition and reference membership

Use the same six complete I14 SGDm h100 parents: seeds 200/201/202, clean and
fixed-corrupted training targets, LR .03, rank32 stable observer, and the same
data splits, fixed labels and saved batch plans. New policies are exactly
`scalar_k0p5`, `scalar_k0p9`, `scalar_k1`, `spectral_mean_projected`.
Each continues global updates101–2000: 24 branches, 45,600 new updates.
Reuse I16's six `k0` trajectories without training or checkpoint-forward replay.
All five logical policies retain horizons100/250/500/1000/1500/2000.

No additional learning rate, target, seed, rank, horizon, schedule or dataset
may be added after an outcome is seen. A failed or consumed attempt is not
restarted. This protocol does not authorize a fresh-panel follow-up yet.

## Exact data update and retained state

Ingest raw gradient g exactly once with the unchanged observer, obtaining
post-ingest mean mu=.99*mu_old+.01*g and native action A_t. Use the native
basis action, including its finite-precision defects; do not repair its basis.

Scalar k recurrence:

```
h = k*g + (1-k)*mu
b = .9*(k*b_old) + h
d = (1-.9*k)*b
```

Spectral mean/projected-history recurrence:

```
h = native_observer_delivery + (mu - A_t(mu))
b = .9*A_t(b_old) + h
d = b - .9*A_t(b)
```

Retain b, not d, in the initialized optimizer momentum state. Apply common
manual parameter multiplication .9997, then one data addition `theta += -.03*d`.
The native delivery is mathematically A_t(g); preserve I15's exact operation
ordering and separately log native-minus-A_t(g) rather than refactoring it.
No transient learning-rate changes, post-step corrections, clipping, resets,
observer replacement or added training forwards. Explicit no-grad tensor
operations reproduce initialized PyTorch SGD's multiply-then-add buffer
recurrence; synthetic tests compare against a native SGD reference. The
optimizer group remains .03/.9/no dampening/no Nesterov/no internal decay.
Expose scalar k0 solely for synthetic bitwise I16 parity; reject it in real
acquisition. Reuse, rather than rerun, its six existing scientific trajectories.

With fixed exogenous gradients, scalar DC gain is1. For a fixed invertible
I-.9A, spectral normalization also cancels stationary gain. Only for an ideal
fixed orthogonal action does this become independent lag9 momentum inside
the span and lag99 EMA outside. Moving operators, inherited h100 buffers,
finite precision and endogenous gradients prevent an exact fixed-kernel
description of the neural run. In particular, matching stationary gain does
not equalize immediate movement, total update energy or finite-time decay
effects. The intermediate scalar kernels' longer mean lags do not imply
lower white-noise variance (see the delivered algebra check).
The pre-outcome moving-action counterexample (artifact not distributed in this public snapshot) makes
the stationary-versus-realized-gain distinction exact. Its alternate
constant-preserving recurrence is a future control, not an extra I17 arm.

## Selection and all eight primary comparisons

For each seed and training target, jointly select scalar (k,h) from four k
values and six horizons using (separately) minimum validation CE and maximum
validation accuracy. Break ties by earliest horizon, then lower k. Select the
spectral horizon using the same metric, with earliest-horizon ties. Auxiliary
outcomes are never used for selection. The common h100 is one inherited state,
not four independent observations.

Report spectral-minus-selected-scalar utility for two targets × two selectors
× two auxiliary metrics: eight primaries, all three seed effects and arithmetic
mean. Utility is minus clean auxiliary CE or clean auxiliary accuracy. Record
selected k/h, validation value and both auxiliary metrics for every selection.
Do not construct a favorable composite or pool metric-disagreeing findings.
Twenty-four scalar choices versus six spectral horizons are unequal search
opportunities, not an oracle guarantee or equally tuned whole-program test.

If a required scalar branch fails, the affected joint-selection contrast is
unavailable; do not select over survivors. Missing required seeds make the
aggregate unavailable. Retain all failures and existing references separately.

## Required secondary evidence

Retain all fixed-policy curves, per-k selected comparisons, endpoint contrasts
and change from h100. Save the existing train clean/fixed/soft CE, clean/fixed
accuracy, R_zeta=fixedCE-softCE, confidence and true-label probability, plus
validation/auxiliary clean CE, accuracy, confidence and true-label probability.
Distinguish useful progress from preservation against another arm's decline.

Log per-update raw/postmean/applied gradient, old/transformed/new recurrent
buffer and normalized delivery geometry; actual total/data/nominal-decay
displacements; data path length and squared energy; signed raw-gradient dot
data-step; observer count; mean, buffer and normalization residuals; native
action defects. Give energy/path summaries for101–2000 and1001–2000 within
each branch, then equal-seed means. These are descriptions, not semantic
denoising scores or mediation fractions. Small per-step algebra defects do
not bound accumulated nonlinear learning effects.

## Provenance and neutral parent checks

Pin I16 audit, summary, report corroboration, collection and completion;
bind all six reused k0 JSON curves and declared checkpoint hashes. Retain
the transitive accepted I14 parent/plan/data and I15 provenance. Only load the
six used old full parents, never unused old terminal tensors.

Before any scientific update, restore every parent model/optimizer/observer/
RNG state, verify its full digest and evaluate h100 neutrally against the
existing I14 and reused I16 references. Repeat on each branch restore. This
neutral admission check is not a repeated training step. Across the four
new branches from a parent, the first raw gradient, post-observer and old
buffer digests must agree; later trajectories need not share gradients.

Save model-only intermediate states, full terminal states, and full typed
nonfinite failure states with explicit missing endpoints. Only typed numerical
failures may seal one branch and allow others to continue. Structural, source,
seam, resource, CUDA, eigensolver and serialization errors abort acquisition.
No retry or outcome-dependent repair. Independent CPU analysis will hash and
derive scalar results/state digests without forwards or optimizer replay;
losslessly archive the original JSON. Analysis source and synthetic tests must
be reviewed before scientific acquisition.

## Resource and admission gates

One exclusive `spectral-i17-001.` root under verified `/tmp/spectral-experiment-artifacts`
on /dev/RECONFIGURE_FOR_LOCAL_STORAGE, with a declared runtime/ directory counted in the shared3GiB
logical artifact cap. Local RTX3090 only, one sequential worker,6GiB host,
zero swap, one numerical thread/CPUQuota100%,4GiB Torch GPU cap, Restart=no.
Leave unrelated users untouched. Expected cloud spend0; cumulative cap$100.

One synthetic smoke: two100-step synthetic warmups and eight10-step branches,
280 updates total, synthetic seed217 (generator2026090817). Time each policy
separately. Prospective forecast:

`1.5 * max_policy(total_policy_update_seconds / 20) * 45600 + 60 <= 1800s`.

The conservative maximum uses two10-step branches per policy. Smoke must
complete with zero numerical failures and all first-step pairs, source hashes
and artifacts valid. Smoke100s cooperative/120s unit; confirmation1800s
cooperative/2100s unit. Freeze protocol, predictions, core, runner, tests and
imported source closure before smoke. Main records exact roots, commit,
service names, PIDs, invocation IDs, attempts and completion receipts. No
GPU work is admitted merely by writing this protocol.
