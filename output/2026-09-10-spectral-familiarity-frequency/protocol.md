# Proposed fixed protocol: familiarity under a paired frequency swap

Codex — Spectral Optimizer Investigation · 10 September 2026

**Not executable admission.** No scientific run, real-data smoke
test, model load or source mutation has occurred for this study. Implementation
and independent review must precede a separate source/inventory freeze and
once-only local acquisition decision. Any substantive change below requires
a dated amendment before scientific data are inspected. No parameter sweep
or outcome-dependent roster change is authorized by this protocol.

## 1. Question and estimand boundary

Does the native filter's effect on correct learning differ between previously
exposed and omitted classes, when each is subsequently low- or high-frequency?
Can it preserve acquired rare competence without blocking useful unfamiliar
learning, particularly while resisting fixed wrong labels elsewhere?

This is a learning-level full-history intervention. Familiarity changes model,
Adam and observer states together. Frequency swaps exposure with a second
focal class. Neither pure observer mediation nor isolated-frequency causality
is identified. The two classes are repeated conditions within three seed
bundles, not six independent seeds. No semantic generalization, alignment,
privacy, safety efficacy or language-model performance claim follows directly.

## 2. Fixed roster and data

Seed bundles: **202609181, 202609182, 202609183**. Every seed crosses both focal
digits, **A=3 and B=8**. Background digits are 0,1,2,4,5,6,7,9. These are fixed
examples of class identity, not a random sample of tasks or a ten-class survey.

Use only the existing pinned official MNIST training IDX, never the official
test set. The unchanged input pins are images
`ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db`
and labels
`65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5`.
Before any run, the new inventory must resolve exact paths and verify hashes.

For each seed and true digit, use one predeclared seeded permutation of IDs:
first 550 warmup-only, next 550 continuation-only, next 500 reporting-only.
All three pools are disjoint within a seed; use the same pools in every cell
of that seed. Different seeds may overlap, as in earlier studies. This makes
available support equal across classes/frequencies. Realized unique exposure
will still differ with sampling frequency: record it, do not call it matched.
Warmup-only images never appear in continuation or reported held-out metrics.
Reporting images do not guide stopping, selection or hyperparameters.

Draw occurrences with replacement using independently named, deterministic
PCG64 streams for splits, warmup indices/slots, continuation indices/slots,
and corruption. Freeze the numerical stream-tag registry in inert source
before acquisition. Build all numeric plans before training. Keep true labels
unchanged; explicit class-role metadata replaces hard-coded digit-8 helpers.

## 3. Model, policy and two new parents per seed

Reuse the immutable I9 784–64–10 ReLU model (50,890 parameters), CPU-seeded
initialization, NumPy FP32 division by FP32 255, and AdamW: learning rate .001,
weight decay .01, betas (.9,.999), epsilon 1e-8, foreach/fused false. No scheduler,
augmentation, clipping, normalization, optimizer reset or rank adaptation.

Canonical native observer: stable global centered hard rank32, decay .99,
raw-gradient covariance, repair interval100, relative eigenvalue threshold
1e-8 and absolute floor0. It observes each current raw gradient before delivery
of the projection; Adam then steps once. Preserve the exact filter/core
source pins identified in [source feasibility](source-feasibility.md).

Start both warmups from the exact same initialized model for that seed. Use
**100 clean, raw-delivery Adam updates**, batch64, with the observer recording
one raw gradient per update. Construct 6,400 warmup slots: 712 focal slots and
711 for each of the eight background classes. One parent fills the focal
slots with class3; the other fills them with class8. Identical seeded slot
permutation and background occurrence IDs in both; focal IDs come from the
respective warmup-only class pool. This is a class replacement, not extra
warmup compute. The omitted focal class supplies no positive example but
still receives negative-class gradients through multiclass CE.

Save both complete parents: model specification/parameters/modes, gradients,
Adam state and counters, observer mean/basis/eigenvalues/count/repair state,
and Python/NumPy/Torch/CUDA RNG. At the fork both Adam and observer counts
are100; their numerical values differ between the two warmup histories.

Fork raw and native from the same complete parent for every continuation cell,
without aliases or moment/bias-clock resets. Raw may discard its observer only
after exact fork verification; it bypasses filtering throughout continuation.
Native records and delivers steps101 through2000. Initial raw batch gradient
must match between the paired policies before their first differing delivery.
State forks, not merely matching model weights, are the pairing unit.

## 4. Two exact continuation schedules

Each continuation has **1,900 updates of64**: nineteen100-update blocks.
Each6,400-slot block has the following fixed quotas:

| Schedule | Class3 | Class8 | Each of eight background classes |
| --- | ---: | ---: | ---: |
| A-low / B-high | 64 (1%) | 640 (10%) | 712 (11.125%) |
| A-high / B-low | 640 (10%) | 64 (1%) | 712 (11.125%) |

Build64 permanent A slots,64 permanent B slots,576 exchangeable focal slots
and712 slots per background class, then apply one common seeded permutation
of all6,400 slots per block. Fill exchangeable slots with the high-frequency
focal class. For each focal class draw640 indexed occurrences per block: the
first64 occupy its permanent slots in both schedules; the remaining576 occur
only in its high-frequency schedule. Background slots, indices and ordering
are identical across the swap. No deliberate class grouping or burst arm.

Each focal class receives1,216 or12,160 continuation occurrences. Its low
occurrences are a position-matched subset of its high occurrences; each
background class receives13,528 in either schedule. The schedules do **not**
have the same complete multiset. Changing focal frequency necessarily changes
competitor-focal exposure here; neither extra exposure nor this coupling is
removed by sharing RNG. Both familiarity parents, label settings and policies
reuse the same schedule-specific occurrence arrays for each seed.

## 5. Two label settings and physical count

Both focal classes are always correctly labelled. Warmup is always clean.
For continuation, compare:

- **Clean:** all original labels.
- **Diffuse background:** independently select each background training ID
  with probability .9 and assign it a uniform label among the eight background
  digits, including its true label. Unselected IDs retain truth. The expected
  actually-wrong fraction is .9×7/8=.7875 *within background*, not across the
  whole sampled stream. Wrong assignments are fixed per original ID, shared
  across histories/schedules/policies, and never target either focal class.

Record selected and actually changed counts/rates per true class, pool and
sampled stream. Repeated occurrences retain the same assigned label. This is
not exactly the older nine-background .8-actually-wrong law, nor the larger
strong-regime study. There are no shared-cue, sham, patch, norm-matched or
augmentation arms in this study.

Physical roster:3 seeds ×2 warmup histories ×2 frequency assignments ×2 label
settings ×2 policies = **48 continuations**, plus6 shared new warmups.
Each trajectory reports both focal classes, giving96 logical factorial rows.
No duplicate run is needed for the second class. Total physical training
updates:6×100 +48×1900 = **91,800**. Execute every registered branch regardless
of warmup competence or intermediate results.

## 6. Readouts, contrasts and competence annotation

Save FP32 logits for all5,500 continuation-pool images and5,000 reporting
images at initialization (once per seed), warmup boundary (once per parent),
and absolute steps200,300,…,2000 (19 per continuation). Use deterministic
evaluation, no input transforms. Recompute scalar CE using FP64 log-sum-exp
from saved logits; retain per-class count/correct-count/CE-sum denominators.
Model-evaluation batching must be frozen in source and never altered by policy.

**Primary behavioral readouts:** focal reporting true-label CE and accuracy
at step2000, shown jointly, per seed/class/cell. Warmup values and changes
since warmup are mandatory. Also show the full fixed curves, eight-background
macro reporting CE/accuracy, and background actually-wrong training-subset
assigned-label AND true-label CE/accuracy. Wrong-target fitting is a measured
behavior, not a pure memorization score; report undefined subsets as undefined.
No selected checkpoint, time-to-threshold winner, seed exclusion, combined
utility score or post-hoc subgroup is a primary outcome.

For focal digit d, seed s, familiarity f, frequency r and label setting c,
define positive-as-beneficial policy contrasts:

    Δ_CE(s,d,f,r,c)  = CE_raw(T) − CE_native(T)
    Δ_acc(s,d,f,r,c) = accuracy_native(T) − accuracy_raw(T)

Also report each policy's own change from its parent:

    G_CE(policy)  = CE_parent(H) − CE_policy(T)
    G_acc(policy) = accuracy_policy(T) − accuracy_parent(H)

Because the parent is identical within a policy pair, Δ = G_native − G_raw.
This cancellation **does not** adjust away differences in starting competence,
representation or ceiling across familiarity parents. No division by headroom
or baseline loss. For each metric and seed/digit/label setting report:

    J_low  = Δ(familiar,low)  − Δ(omitted,low)
    J_high = Δ(familiar,high) − Δ(omitted,high)
    K      = J_low − J_high

These are descriptive paired interactions under the specified exposure swap.
Show the component cells, not just J/K; average the two digit readouts within
each seed before any three-seed summary. Retain both digit-specific results
and all three seed-level values. No uncorrected significance claims from96
rows or treatment of checkpoints/classes as independent replications.

Predeclare a descriptive **learned-competence annotation** for each exposed
focal class at H: held-out accuracy≥50% AND true-label CE below that class's
value at the same initialized model. This is a modest operational criterion,
not a statistical threshold or scientific fact about mastery. Report the
underlying values and annotation for every seed/class. Failure means that
cell cannot support a claim about preserving demonstrably acquired competence;
it does not drop the cell, stop the study, extend warmup, change seeds or
license another run. No competence matching by checkpoint selection.

Better Δ alone can mean less deterioration. Distinguish reduced forgetting,
continued useful gains, and damage to the raw comparator. A favorable J/K
without useful absolute performance is not evidence of a useful policy. Even
perfect agreement cannot attribute a mediator to covariance memory alone.

## 7. Evidence, audit and prospective resource limits

Save3 initial states,6 complete warmup parents and48 final states; numeric
plans with all split IDs, slot/occurrence arrays and label assignments; source/
environment/argument hashes; exact first-action fork/gradient/clock checks;
all scheduled logits and scalar records; terminal wall/resource/byte receipts.
No dense P×P matrix, per-example gradient bank, intermediate full-state series,
extra local utility probe or replay of an older experiment is needed.

Prospective independent audit must rederive roster/splits/quotas, disjointness,
couplings, fixed labels, competence annotations, all scalar metrics and paired
contrasts from persisted evidence, independently of producer summary helpers.
Decode saved snapshots sequentially to check actual tensor/state schema,
parameter ordering, Adam counters and native observer clocks; opaque byte
hashes alone do not establish these properties. Check producer non-alias and
first-action receipts separately: a persisted snapshot cannot retrospectively
prove that live objects never aliased. It is a saved-evidence audit, not a
replay of neural forward/backward or the complete streaming-observer trajectory.
Validate both producer and audit with fabricated
CPU fixtures and mutation tests before real data. No automatic retry on fail.

Conservative inventory: (3 +6 +48×19) ×10,500×10×4 =386,820,000 logit bytes;
57 states ×7,597,528 bytes =433,059,096;256MiB reserve for numeric plans,
metrics, headers and receipts. Total **1,088,314,552 bytes**, about1.014GiB,
under a **2GiB acquisition-artifact cap**. This intentionally budgets every
state as a full rank32 state, including much smaller raw/initial states.
The implementation inventory must enumerate all actual artifacts and confirm
the bound; compression is not required for admission.

Proposed acquisition envelope: one local RTX3090, one CPU math thread,
16GiB host/no swap,8GiB allocated-GPU ceiling,25-minute cooperative and
30-minute hard non-restarting service limit. Planning allowance10–20minutes,
based on source/size and analogous completed studies, not a benchmark or
promise. A separate once-only saved-evidence audit is provisionally one CPU,
4GiB/no swap, five-minute hard limit; finalize its inventory before admission.
Check live owners/free resources before any future launch; do not interfere
with unrelated work. No cluster/Modal run or paid reservation is selected.

## 8. Preparation if this deferred study is explicitly resumed

This prospective protocol adds no empirical finding and is not a gate before
reporting the already useful protection and learning results.
