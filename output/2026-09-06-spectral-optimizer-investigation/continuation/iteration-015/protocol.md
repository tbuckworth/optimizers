# I15 — SGDm history transport and mean restoration

Prospective protocol, 7 September 2026. This is a design for implementation
validation, not a launched study or a production change. It follows the
three-seed [I14 result](../iteration-014/mathematical-interpretation.md) and the
related-work review (artifact not distributed in this public snapshot). Completed I14 learning trajectories are
immutable and must not be rerun.

## Question and fixed scope

Did SGDm help the native filter because its momentum buffer carries previously
admitted directions outside the current rank-32 action? Can the explicit EMA
mean complement replace that carried history?

Use the six I14 SGDm h100 states: seeds 200/201/202 × clean/fixed target. Each
state already contains the exact model, `.03`-SGDm buffer, canonical observer,
RNG and modes after the common unfiltered warmup. Continue the exact saved I14
batch rows 101–2000 with the same clean or fixed labels, coefficient `.01`,
manual decoupled decay, architecture and data splits.

The four logical factorial cells are delivery `{current, mean}` × history
`{native, projected}`:

- `current_native`: reuse the six completed I14 current32 trajectories.
- `current_projected_history`: six new branches.
- `mean_native`: six new branches.
- `mean_projected_history`: six new branches.

Thus acquisition is exactly **18 new branches × 1,900 updates = 34,200 new
updates**. The six completed I14 raw SGDm trajectories are separate references.
No native-current or raw training is repeated; no seed, rank, rate, decay,
target, horizon or split is added after outcomes.

## Exact step semantics

At global step $t>100$, compute the raw minibatch gradient $g_t$. Ingest it
once through the unchanged canonical observer (rank 32, decay .99, stable
update), obtaining its actual post-ingest native delivery and mean. Define the
implemented linear action

\[
 A_t x = V_t(V_t^\top x),\qquad
 \mu_t=.99\mu_{t-1}+.01g_t.
\]

`current` delivers the literal native filtered gradient $h_t=A_tg_t$, checked
against the explicit action. `mean` exactly matches I13's post-ingest policy,

\[
 h_t=A_tg_t+(\mu_t-A_t\mu_t).
\]

Let $b_{t-1}$ be the inherited live SGDm buffer before this step. `native`
uses the PyTorch recurrence

\[
 b_t=.9b_{t-1}+h_t.
\]

`projected_history` changes only the carried term:

\[
 b_t=.9A_tb_{t-1}+h_t.
\]

Operationally, replace each pre-existing buffer by the unflattened
$A_tb_{t-1}$, then let the unchanged torch-SGD step add $h_t$. Never project
the whole updated buffer: doing so would erase mean's newly delivered outside
complement. Apply the same manual parameter multiplier
$1-.03\times.01$ after observation/delivery and before `optimizer.step`, as in
I14. Persist the resulting buffer. There is no reset, rescaling, clipping,
second moment, Nesterov term or extra observer call.

$A_t$ is the actual native action, not an assumed exact orthoprojector.
Eigenvalue pruning, finite precision and imperfect orthogonality can make
$A_t^2\ne A_t$. Save $\lVert V_t^\top V_t-I\rVert$, action-idempotence residuals
and homogeneous backward-error bounds. Terms such as $x-A_tx$ are complements
under this action; call them orthogonal only within measured numerical error.

## Frozen estimands

Utility $U$ is negative auxiliary clean CE or auxiliary clean accuracy.
For each target and metric at h2000, report three separate primary families:

\[
\begin{aligned}
H &= U(\text{current,native})-U(\text{current,projected}),\\
M &= U(\text{mean,projected})-U(\text{current,projected}),\\
S &= [U(\text{mean,projected})-U(\text{current,projected})]
   -[U(\text{mean,native})-U(\text{current,native})].
\end{aligned}
\]

Positive $H$ says native carried history helped current delivery; positive
$M$ says explicit mean improved the history-projected cell; positive $S$
says mean was more useful after carried history was removed. These are 12
registered estimands (three families × two targets × two metrics), not a
composite. Retain all three paired seed values and their arithmetic mean. A
missing contribution makes its estimand unavailable; never survivor-average.

Mandatory secondary results include all four factorial cells plus raw at
global h100/250/500/1000/1500/2000; the complementary mean effect under native
history and history effect under mean delivery; auxiliary and validation clean
CE/accuracy/confidence; absolute progress from h100; and both auxiliary metrics
under minimum-validation-CE and maximum-validation-accuracy selection over all
six scheduled horizons, with earliest-horizon ties. No selector replaces an
adverse endpoint.

For fixed labels, retain training clean/fixed/soft CE and accuracy,
$R_\zeta=L_{\rm fixed}-L_{\rm soft}$, confidence and true-label probability.
Report whether any clean benefit is preservation against raw/native decline,
positive branch progress, or both. Mean-restored realization fitting is a
tradeoff, not semantic denoising.

## Mechanism records and limits

Every new step retains norms and squared energies for $g_t,A_tg_t,\mu_t$,
$A_t\mu_t,\mu_t-A_t\mu_t,b_{t-1},A_tb_{t-1},b_t,h_t$, total/data/decay
displacements, and current-action complement energy. Check the native-delivery,
mean-definition and selected buffer-recurrence identities with homogeneous
float32 bounds. Record signed $g_t\cdot d_t$,
$(\mu_t-A_t\mu_t)\cdot d_t$, and
$(b_{t-1}-A_tb_{t-1})\cdot d_t$, alongside all component norms and explicit
zero-vector cases. Here $d_t$ is the decay-subtracted data displacement.

Aggregate energy ratios within branch, then equal seeds. Report cumulative
steps 101–2000 and the exact late window 1001–2000 separately. Dots/cosines describe orientation only jointly
with amplitude; neither signed alignment nor outside-energy fraction is a
clean-utility or causal-mediation fraction. First-step raw-gradient,
post-observer, old-buffer and action digests must agree across the three new
arms at a common parent where the corresponding quantity is policy-independent.
Later paths are endogenous and need not agree.

At each scheduled horizon save the scalar evaluation and model state; retain
complete terminal model/optimizer/observer/RNG state. The h100 evaluation and
complete parent digest must exactly match both bound I14 raw/current records
before any new update. Complete these seam checks for all six parents before
the first scientific update. Hash-bind the accepted I14 runtime supplement, all six
parents, all 12 reused raw/current curve JSONs and their checkpoint records,
three confirmation plans, source
dataset hashes and fixed-corruption counts. Independently recompute the I14
tree digests. Load only the six used canonical h100 tensors, not unused I14
terminal tensors. The neutral h100 forward is new-branch admission; it does not
rerun an I14 trajectory, endpoint forward or optimizer update.

## Failure and resource discipline

An h100 seam, reference, plan, source, topology or artifact mismatch aborts the
study before training. After a valid seam, only explicit nonfinite arithmetic
may seal one branch and allow independent branches to continue. Preserve its
partial curve/diagnostics and failed state; all affected primaries become
unavailable. Resource, serialization, eigensolver and generic callback errors
abort acquisition. No retry, finite-loss cutoff or result-dependent amendment.

Use one new exclusive root under verified `/tmp/spectral-experiment-artifacts` on `/dev/RECONFIGURE_FOR_LOCAL_STORAGE`.
Declare an explicit `runtime/` child prospectively, count all of its files in
the shared **3 GiB** cap, and audit its entries separately; scientific phase
directories must equal their exact artifact manifests. Limits may be no wider
than I14: 6 GiB host, zero swap, one CPU thread/CPUQuota100%, 4 GiB Torch GPU,
1,800 s cooperative and 2,100 s unit ceilings. The source-identical smoke uses
two synthetic clean/fixed parents, each with a 100-step warmup, then all six
target×new-arm cells for 10 updates: exactly 260 updates. It has a 100 s
cooperative and 120 s unit ceiling. Time the 60 new history updates separately
from their cheaper raw warmup. Before real acquisition require
`1.5 × (history_update_seconds / 60) × 34200 + 60` below the 1,800 s ceiling;
the extra 60 s is reserved for admission/evaluation/serialization. Runtime
invariants read device scalars and synchronize the measured updates. This is a
prospective admission estimate, not a guarantee; the hard limits still apply.
No
cloud instance or paid reservation; cumulative authorized budget remains
$0/$100. Freeze implementation, tests, this protocol and predictions before
the first attempt. This is autonomous validation, not an approval ceremony.
