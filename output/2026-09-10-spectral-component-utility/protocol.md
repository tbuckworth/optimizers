# Signed objective changes under a common-state spectral action

Codex — Spectral Optimizer Investigation · 10 September 2026

**Design fixed for implementation; helpers reviewed, not frozen for
acquisition. No real checkpoint/model replay has run for this diagnostic.**
Fabricated CPU helper checks are recorded in [main acceptance](helper-acceptance.md).
The [source inventory](source-inventory.md) and [mathematical assessment](design-math.md)
motivate a new local objective-accounting question, not another efficacy sweep.
Independent design review and exact source/resource checks precede execution.
These are internal scientific checks under existing autonomous authority, not
a request for new user approval.

This is outcome-informed analysis of reused trained states and task data, not
a fresh confirmation or blind model-selection study. Fixing this diagnostic's
choices before its computation does not remove that adaptive provenance.

## 1. Question and estimands

At the same saved native model, Adam state and observer, does retaining the
filter rather than bypassing it restrict consistency learning, softened-target
learning or realized-label fitting? Does that accompany useful or harmful
clean prediction changes? This contrasts one current action, not full policies
trained from different initial conditions or a mechanism of the endpoint gap.

For a fixed training evaluation panel I and the exact uniform support of all
25 shifts (dy,dx) in {-2,-1,0,1,2}², compute mean logits separately per image:

```text
zbar_i = mean_T f_theta(T x_i)
A(z) = logsumexp(z)
q_i = 0.1 one_hot(true_i) + 0.9 uniform_10
epsilon_i = one_hot(assigned_i) - q_i
S = mean_i [A(zbar_i) - q_i dot zbar_i]
F = -mean_i [epsilon_i dot zbar_i]
C = mean_i [mean_T A(z_iT) - A(zbar_i)]
L = mean_i,T CE(z_iT, assigned_i) = S + F + C
```

Differentiate all means, including zbar; no detached teacher. Also retain the
finite scalar split S = 0.1 S_true + 0.9 S_uniform, with those two mean-logit CE
terms separately recorded. S is not clean CE; F includes correctly assigned
examples and can have either sign; C can vanish for wrong or constant predictions.
On a separate nontraining panel R, define H_O as original-image true-label CE
and H_T as mean true-label CE over all 25 views, not CE of mean logits. H_O
accuracy is original-image argmax accuracy. H_T accuracy is the equally weighted
mean of per-view argmax accuracies, not accuracy from mean logits or a majority
vote. H_O is primary competence, H_T secondary.

For each actual raw/native endpoint theta_a and each objective J, record
E_J(a) = J(theta) - J(theta_a) and U_J(a) = -grad J(theta) dot (theta_a-theta),
plus E-U. Positive means decrease of the named objective, not automatically
useful learning. D_J = E_J(native)-E_J(raw), with its derivative counterpart.
The identity E_L=E_S+E_F+E_C must hold; this does not decompose D_H into causes.
The action gradient below is not required to equal the evaluation-panel gradient.

## 2. Fixed states and panel selection

Use all twelve native states in parent-inventory.json (artifact not distributed in this public snapshot):
seeds 202609171, 202609172, 202609173 × prior training mode none/translate ×
warmup100/final56304. Fixed order is seed, none then translate, warmup then final.
Do not select by performance, change a seed, or reconstruct missing intermediate
states. Final parents differ from warmups in weights/moments/history; those
between-parent contrasts are descriptive, not observer-only interventions.

Use original strong-study plan IDs, fixed assigned/true labels, and original
training IDX inputs. No official-test file, download or new split. With NumPy
1.26.4, Generator(PCG64(SeedSequence([40,seed]))) permutes local positions 0…49999.
The first 128 positions form two disjoint ordered action batches of 64; the next
256 positions form I. R is the first 128 IDs in the original plan's reporting
order. No class-balance correction or rejection after reading outcomes.
Stream [41,seed] draws integers(-2,3,size=(2,64,2),dtype=int8) once, last axis
(dy,dx), for the action images. Save all selected positions/IDs/labels/shifts.

The identical two action batches, assignments and shifts are reused across all
four parents within a seed. **Every action input is translated**, even when its
parent was trained without augmentation. In those parents this is a hypothetical
onset-of-augmentation action, not the original unaugmented trajectory's next step.
Evaluation enumerates each of the 25 shifts exactly with equal weight; shift
order is lexicographic (dy outer, dx inner). Translate raw pixels before the
canonical standardization. Keep all image/view weights fixed before and after.

## 3. Faithful actions, not separate component optimizers

Load the exact strong snapshot through a new strict adapter, not I9 restore.
Freshly verify parent byte hash/size, schema and all tensor shapes, dtypes,
parameter order, settings and counters. Canonical ownership aliases must refer
to the reconstructed model/optimizer. Restore and snapshot round-trip tree
identity must pass; isolate/restore RNG state around restoration. Verify baseline
reporting logits using the first original 500-image chunk and the producer's
FP32 batching semantics, against the pinned saved original readout. Exact
agreement is expected under the pinned environment; preserve any discrepancy
as failure, not a reason to relax tolerances after seeing scientific results.

At each parent/action batch compute one ordinary FP32 mean assigned-label CE
gradient. Raw and native private joint copies use that identical gradient and
the inherited AdamW configuration/counters. Raw bypasses the observer. Native
calls the unchanged filter once (current-input observation then projection),
then AdamW once. Do not change rank, decay, warmup, learning rate, normalization,
old moments or current-input self-inclusion. Discard private updates afterward.
No candidate becomes the next candidate's parent; no training continuation.

Save exact FP32 before and after parameter vectors. A third **decay-only**
endpoint is the actual rounded theta*(1-lr*weight_decay), not Adam on zero input.
For each endpoint also evaluate the materialized FP32 path point
theta + 0.1*(endpoint-theta). Every derivative dot uses that path point's actual
materialized FP32 vector minus the parent, subtracted in FP64; do not substitute
0.1 times the full displacement. Full raw/native endpoints are the actual optimizer
outputs; the 0.1 paths are geometric checks, not additional optimizer steps.
There are no component-Adam, component-removal or norm-rescaling arms.

Save actual total/data displacements, norms, inherited first/second moments,
per-parameter counters, raw/delivered gradients and post-native basis. Use these
for checkable dot products, projection and Adam arithmetic. Describe smaller
motion if observed; this design does not identify a direction-only effect.
Common decay cancels in the within-parent linear contrast, but retain absolute
decay/data accounting. Finite data-only effects compare each endpoint to its
corresponding decay endpoint at the same path fraction, not to a fictitious
zero-momentum parent.

## 4. Readouts and interpretation before looking

For each of 12 parents, compute baseline gradients for S,F,C,L,H_O,H_T on the
fixed panels. S/F/C/L use the same model and I/view grid. Objective arithmetic
is FP64 from FP32 logits. Before component arithmetic, differentiably subtract
each image/view's class-0 logit from every class, then average views. This fixed
logit-offset convention leaves the mathematical S/F/C/L objectives unchanged
and limits cancellation from large common offsets; no teacher/mean is detached.
Store original FP32 logits, not only the shifted values. Parameter derivatives
remain FP32. After derivative calculation, save detached scalar/logit values
and drop all original objective/logit graph references before private actions;
the last autograd call need not free separate earlier evaluation graphs.
At baseline and
each of the 12 materialized endpoints (2 batches × 3 actions × 2 path fractions),
save I's 25-view logits, R's original/25-view logits and scalar objectives.
On I, report true/assigned-label CE and accuracy both on original images
(the (0,0) view, index 12) and as means over all per-view CE/argmax outcomes.
Do the same for the actually-wrong subset, using its fixed assigned!=true mask.
Original-image subset fit is the primary wrong-fit proxy matching the earlier
study; mean-per-view fit is secondary. Neither is a mean-logit accuracy metric.
If an evaluated subset is empty, record unavailable/count zero, not an invented
score. Mean-logit true/uniform CE components accompany every S readout.

Primary local explanatory readouts are D_HO and D_C at full step for the three
**translated final parents**, averaged over the two action batches within each
parent. List all three seed results and absolute raw/native effects; do not
replace them with warmup, another component or a favorable path fraction.
S/F/L, H_T, accuracy, derivative and 0.1-path results are predeclared secondary
views, as are every other training-mode/stage cell. Show all cells, no pooled
12-parent or 24-action independent-seed claim, no significance/equivalence claim.

The local lost-consistency hypothesis is supported descriptively when raw
decreases C and improves H_O, while native gives less improvement in both;
consistent seed, derivative and smaller-path signs strengthen that account.
If C improvement is not reduced despite an adverse H_O contrast, this local
consistency explanation is contradicted at these states. If raw itself does
not improve C, calling the result lost useful consistency learning is unsupported.
Lost S progress, extra F fitting, uniform-confidence effects, mixed seeds or
state reversals remain distinct outcomes, not failed attempts to obtain a win.

The constructive alternative is less realization fitting while useful clean
progress survives or improves. Pair F with actual wrong-assignment fit and clean
competence; neither its sign nor reduced noisy-label accuracy proves protection.
Any favorable local outcome leaves the adverse full-trajectory bridge intact.
No component is established as a causal mediator by these associations.
Report raw signed values even when tiny. Mark absolute finite CE/objective
changes or paired contrasts <=1e-8 nats as numerically small, rather than using
their sign as positive/negative evidence; use <=1e-10 for linear dot-product
utility. These are interpretation guards, not equivalence margins. Do not
rescale a step or choose an earlier state merely to obtain larger effects.

## 5. Integrity, computation and failure scope

Implement inert new files only. Preserve every frozen old source. Before a real
load, fabricated CPU fixtures must cover: strong-schema round trip/ownership,
RNG isolation, correct and rejected model/optimizer/tracker structures, actual
canonical raw/native step fidelity including repair, exact 25-view objective
identity, its differentiated counterpart, S_true/S_uniform split, label-independent
C at a fixed model, the m=1 degeneracy, private-state immutability and output caps.

Predeclared arithmetic tolerances: scalar S/F/C identities absolute/relative
1e-10; FP32 gradient sum residual norm <=1e-6 + 5e-5*norm(direct L gradient).
Independent saved-array checks of projection, materialized paths and Adam math
use absolute1e-7/relative5e-5; dot products and metric recalculation use FP64
absolute/relative1e-10. Numerical disagreements are reported and preserved;
no after-outcome tolerance relaxation. These tolerances must pass adversarial
fabricated fixtures before source freeze, not be calibrated on real parents.

Pin this protocol, all new code/tests, exact old dependencies, parent receipts,
original data and input plan/readout receipts to a committed launch manifest.
Independent audit verifies saved objectives/identities/dots, materialized endpoint
and Adam algebra, projection from recorded post-basis, panel membership and
hashes. It does not replay the neural model or independently reconstruct the
entire canonical streaming-observer history; label that boundary plainly.

Proposed local acquisition: 1 CPU, 8 GiB host, no swap, <=4 GiB GPU allocated,
20-minute cooperative/30-minute hard deadline, Restart=no, KillMode=control-group,
8 GiB archive cap including failure reserve. Require verified large-volume
mount, >=16 GiB free disk, >=8 GiB free GPU and no unknown compute clients.
No paid resources or displacement of existing desktop work. Budget must include
24 post-native rank200 bases (upper4,514,803,200 bytes), all vectors/logits,
receipts and failure reserve; exact implementation inventory precedes admission.
One separately bounded audit: 1 CPU, 2 GiB, no swap, 15-minute cooperative/
20-minute hard deadline, <=64 MiB output outside the immutable acquisition.