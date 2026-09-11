# I16 — scalar temporal-response controls

Prospective protocol,7September2026. The user authorized ongoing autonomous
hypothesis testing; this bounded continuation follows the delivered and reviewed
[I15 findings](../iteration-015/results.md). Implement and validate this design,
freeze all acquisition sources, then admit one synthetic smoke and the specified
confirmation only if its prospective checks pass. No old trajectory is restarted.

## Question and exact scope

Does I15 mean/projected-history offer useful spatial selectivity beyond a strong
family of isotropic temporal-response controls? Its favorable fixed-label
accuracy/progress, worse selected CE than raw, and clean underfitting all remain
part of the question. A scalar success is informative evidence, not a reason to
discard the constructive I15 result.

Use exactly the six I14 SGDm h100 parents: seeds200/201/202 × clean/fixed target,
same MNIST MLP, data splits, labels, batch plans and global horizons. Continue
each under k=0,.5,.9 for global updates101–2000, giving18newbranches and34,200
newupdates. Reuse six I14 raw trajectories as k=1 and six I15 mean/projected
trajectories as the spectral comparator. No other rate, rank, target, seed,
model, dataset, schedule or cloud allocation is added.

## Exact scalar recurrence

Compute raw minibatch gradient g once. Ingest it once through the unchanged
canonical stable rank32 observer, with decay beta=.99 and inherited warmup100.
The observer's mean is post-ingest mu=.99*mu_old+.01*g. The native projected
gradient is not the scalar delivery; restore the specified scalar mixture:

\[
h_t=k g_t+(1-k)\mu_t,\qquad
\widetilde b_{t-1}=k b_{t-1},\qquad
b_t=.9\widetilde b_{t-1}+h_t.
\]

Operationally copy k times the old, initialized SGDm buffer into the live
buffer, deliver h, apply the unchanged manual parameter multiplier .9997,
then let torch SGD with lr=.03,momentum=.9,dampening0,noNesterov,nointernaldecay
take its ordinary step. Do not modify optimizer-group momentum, reset buffers,
rescale the completed step, clip, reinitialize the observer or add callbacks.
Use literal raw at k=1 and literal postmean at k=0 to avoid gratuitous endpoint
roundoff. k=1 is exposed only to synthetic parity tests; reject real k=1 runs.
No actual I14 update or endpoint forward is replayed for parity.

This is not purely a step-size intervention. On a common fixed stream, its
transfer is

\[
G_k(L)=\frac{k+(1-k)(1-.99)/(1-.99L)}{1-.9kL},\qquad
G_k(1)=\frac1{1-.9k}.
\]

The k grid gives DC gains1,1.818...,5.263...,10. Ideal fixed-P spectral
mean/projected has gain10 inside and1 outside. Actual moving actions and
different endogenous streams do not obey a fixed oracle transfer, and scalar
k changes bandwidth, mixing, amplitude and duration response simultaneously.

## Eight primary estimands and selection rules

For each seed and target, jointly select (k,h) from all four scalar k values
and h100/250/500/1000/1500/2000 using two separate validation selectors:
minimum clean CE and maximum clean accuracy. Ties resolve by earliest horizon,
then smaller k. The shared h100 appears in each k cell but is one inherited
state, not four independent observations. Select the spectral arm separately
over its six horizons using the corresponding validation metric and earliest
horizon ties. Never select using auxiliary outcomes.

For each target × selector × auxiliary metric, report
U(spectral selected)-U(scalar selected), where U=-auxiliary cleanCE or auxiliary
cleanaccuracy. These are eight primaries; give all three paired seed values
and arithmetic mean, with no composite or result-selected horizon. Retain the
chosen k,h,validationvalue and BOTH auxiliary metrics for every choice.

“Best scalar” means the validation-selected procedure, not an auxiliary oracle
or a guaranteed upper bound. More scalar choices may help validation fit but
can also increase selection noise. Both the stronger scalar search and its
finite-sample limitation must be stated. If any required scalar member fails,
the corresponding joint envelope contribution is unavailable, not reselected
over survivors; a missing required seed makes its aggregate unavailable.

This study follows observation of I14/I15 outcomes on the same data/seed panel.
Its within-study validation-to-auxiliary procedure is a valid conditional
comparison, but the wider research sequence is adaptive, not a prospectively
fixed program or fresh-dataset confirmation. Do not overstate generalization
or erase the separate scope of the prior results.

## Mandatory secondary evidence

Retain all five logical policies: four scalar k values plus spectral. Report
every scheduled cell and fixed-k endpoint contrast, paired seed values and
equal-seed means, absolute progress from common h100, and separately selected
per-k comparisons. No weak k or adverse target is dropped.

At each horizon retain train clean/fixed/soft CE, clean/fixed accuracy,
R_zeta=fixedCE-softCE, confidence and true-label probability; validation and
auxiliary clean CE/accuracy/confidence and true-label probability. Explain
whether an advantage is actual branch progress, preservation against another
arm's deterioration, or both. All sampled horizons remain available; no later
duration extension is hidden in this study.

Every new update records raw gradient/postmean/applied delivery/old/scaled-old/
new-buffer norms and squared energies; actual total/data/nominal-decay
displacements, signed raw-gradient dot data-step, loss and observer count.
Check postmean, mixture and buffer recurrence with homogeneous float32 bounds;
retain actual-versus-ideal data-step and manual-decay defects. Record current
action-complement energies if available as descriptive geometry, never semantic
or mediation fractions. Aggregate data-step squared energy, path length (sum
of norms), raw-gradient dots, and any complement-energy ratios separately over
101–2000 and1001–2000, within branch then equally over seeds. Small local
residuals do not bound accumulated nonlinear utility sensitivity.

## Provenance and failure discipline

Before ANY new scientific update, restore all six parent model/optimizer/
observer/RNG states, verify full state digest and neutral h100 evaluation
against bound raw/current/spectral references. Repeat the exact check on each
branch restore. A neutral admission forward is not a repeated training step.
At the common first new update, raw gradient/postobserver/oldbuffer digests must
agree across k values; applied and newbuffer digests need not. Later streams
are endogenous and must not be forced equal.

Save complete new terminal state, model-only intermediate states and complete
state on typed nonfinite failures, with explicit missing endpoints. Only typed
nonfinite arithmetic may seal one branch and permit independent branches to
continue. Structural/source/seam/resource/CUDA/eigensolver/serialization errors
abort acquisition. Preserve partial outputs. No retry or result-driven change.
An independent CPU audit will hash artifacts, derive state digests and scalar
aggregates without forward/optimizer replay; original JSON is losslessly archived.

## Resource admission

Use one exclusive root with prefix spectral-i16-001. under verified
/tmp/spectral-experiment-artifacts on /dev/RECONFIGURE_FOR_LOCAL_STORAGE. Explicit runtime/ child is included in the
3GiB shared logical file cap and separately inventoried; scientific phase
membership must match exactly. Unit caps:6GiBhost,zeroSwap,CPUQuota100%,one
numerical thread,4GiBTorchGPU,Restart=no. Leave unrelated GPU users untouched.
No cloud money is spent; cumulative authorized cloud cap remains$100.

The one synthetic smoke uses two100-update warmups and six10-update scalar
branches (260updates total), seed216. Time new scalar updates separately.
Admission formula1.5*(scalar_update_seconds/60)*34200+60 must be<=1800s.
Smoke100scooperative/120sunit; confirmation1800scooperative/2100sunit.
Freeze protocol/predictions/core/runner/tests and their imported source closure
before smoke. Record exact root,PIDs,invocations,commit,attempts and completions.
