# Clean multiview observer comparison — fixed prospective protocol

Codex — Spectral Optimizer Investigation · 10 September 2026

## Question and scope

Does a four-view observation stream improve actual clean learning under the
stable rank-32 filter while retaining single-view gradient delivery, and is
it competitive with simply delivering the four-view mean to raw AdamW?
This implements the selected[design](../2026-09-10-spectral-augmentation-state/design-decision.md).
Twelve fresh trajectories only. No noisy-label panel, saved-state replay,
mean-complement restoration, rank/LR tuning, cloud run or capability speedrun.
Prior diagnostics are complete and consumed; a clean gain would not certify
retained anti-memorization protection or a safety benefit.

## Exact data plan

Fresh paired seeds **202609161,202609162,202609163**. Existing pinned MNIST
training IDX only; never official test data. Reuse the accepted balanced split:
PCG64 SeedSequence([0,seed]) shuffles source IDs within each true digit0..9;
first500/class training, next500/class heldout. Class-major arrays of5000each.
All labels are true. Train/eval source IDs are disjoint, all classes eligible
from the first update. Batch64 source occurrences sampled with replacement,
4000updates, stream1. Stream2 independently samples first-view (dx,dy) shifts
uniformly from −2..2, int8[4000,64,2]. Stream3 samples three additional views
int8[4000,3,64,2], with independent draws for each occurrence, not one shared
shift per batch/image identity. Stream4 independently fixes secondary heldout
shifts int8[5000,2]. Each stream uses PCG64(SeedSequence([stream,seed])).

Save the actual eight plan arrays: train_ids,eval_ids,train_labels,eval_labels,
occurrences,shifts,extra_shifts,eval_shifts. Reuse unchanged zero-fill integer
translation on float32 pixels in[0,1]; no interpolation/wrap or new transform.
The first view/initialization/occurrences and heldout readouts are shared by
all four arms of a seed. Extra views never influence single-view arms.

## Model, policy and numerical conventions

Accepted784→64ReLU→10 MLP,50,890parameters. Same seeded initialization and
unchanged AdamW: lr0.001,betas(0.9,0.999),eps1e−8,weight_decay0.01,
foreach=False,fused=False. Same stable centered temporal observer: rank32,
decay0.99,warmup100,hard filtering, no normalization, adaptive=none,
stabilize_every100,relative_eig_tol1e−8,absolute_eig_floor0. No default changes.

For a batch at current fixed parameters, calculate separate float32 mean-CE
gradients g1,...,g4 using autograd.grad. No model/Adam update between views;
release each graph. Four-view average is sequential FP32 sum g1+g2+g3+g4,
then division by4. Do not average logits or introduce an EMA outside the span.

| Policy | Number of views | Observed once/update | Delivery through100 | Delivery from101 |
|---|---:|---|---|---|
| raw1 |1|none|g1|g1|
| native1 |1|g1|g1|P1 g1|
| observer4 |4|mean(g1..g4)|g1|P4 g1|
| raw4 |4|none|mean(g1..g4)|mean(g1..g4)|

P is the **post-observation** canonical hard action; rank-zero V=None uses
the canonical identity fallback. Advance observer counter once, update its
SVD once on the specified observation, then project the specified delivery
gradient only after warmup. Do not ingest g1 again after observing its mean.
Advance Adam exactly once, preserving ordinary carried moments and decoupled
decay. The native1 dispatcher must match the unchanged filter_grad path in
synthetic fixtures. Track all parameter Adam counters; they must agree with
the optimizer update number. No clipping, normalization, gain compensation,
moment reset/projection or delivery change beyond the table.

Initial model hashes must match across all four arms. At update100, raw1,
native1,observer4 must have identical model/Adam learning-state hashes and
predictions, even though their observers differ. raw4 has a distinct warmup
trajectory; do not claim matched-state isolation against it. After101 all
comparisons are whole-policy trajectory effects. Rotate sequential arm order
by seed index through(raw1,native1,observer4,raw4); wall times are descriptive,
not a randomized hardware benchmark.

## Readouts and fixed comparisons

Evaluate at0,100,200,400,...,4000 (22 states). At each readout retain complete
FP32 logits on original training5000, original heldout5000, and the fixed
translated heldout5000. Calculate mean CE in float64 and top1accuracy; labels
remain true. Primary: **original heldout CE and accuracy at4000**, with all
three paired values and means. Also report absolute improvement from100 for
each policy and complete curves. Secondary translated readout is not selected
in place of an adverse primary. Training fit and original/translated agreement
are descriptive; no new noisy-label claim is available.

Predeclared comparisons, using positive signs for greater useful improvement:
observer4−native1 (observation-policy change), observer4−raw4 (extra-view control),
observer4−raw1 (ordinary baseline), raw4−raw1 (delivery-averaging benefit),
native1−raw1 (incumbent cost). For CE, reverse loss differences; for accuracy,
ordinary differences. Report both metrics even if signs differ, every seed,
endpoint and within-policy warmup change. No best-checkpoint/LR/rank/viewcount
selection or pseudo-replication across views/epochs. Three seeds give limited
descriptive evidence, not a significance/optimality claim. A smaller relative
deficit without actual candidate progress is not recovered learning.

The candidate can improve, tie, worsen, or exchange original/translated
performance. An observer4 gain over native1 but loss to raw4 is recipe progress,
not an advantage over using the extra views directly. Lack of clean efficacy
does not erase earlier conditional filtering/protection results. No pass/fail
outcome automatically terminates the overarching investigation.

## Artifacts and independent verification

Save one initial snapshot per seed; full warmup/final state for each arm;
one logits NPZ and one stream NPZ per arm; each seed plan NPZ plus hash metadata;
provenance and final result JSON. Exact stream keys:
losses float64[4000,4], unused views zero; gradient_evaluations,observer_steps,
adam_steps,basis_rank int64[4000]; observation_norm,delivery_norm,applied_norm,
data_step_norm float64[4000]. Raw observer fields are zero, not inferred
observations. data_step_norm is the float64 norm of the rounded FP32 actual
after−decay_base displacement, where decay_base is actualFP32 theta(1−lr·wd).
It is not Adam with zero gradient. Norm fields do not certify useful delivery.

Independent NumPy-only auditor hashes every artifact/source and independently
reconstructs all plan streams/labels, logit metrics, endpoint/warmup summaries,
counter/count/availability invariants and matching warmup predictions. It hashes
full states but does not unpickle or replay training, autograd or SVD history;
state-digest assertions remain producer assertions supported by reviewed code.
Metric tolerance abs/rel1e−10. Exact array dtype/shape/finite/header inventory,
no-pickle NPZ access and direct-file/size caps. No tolerance relaxation after
acquisition. Missing/failed outputs remain explicit; no automatic retry.

## Work and resource envelope

4000updates ×64underlying occurrences =256,000 per arm. Single-view arms use
4000batch-gradient evaluations, four-view arms16000; **120,000 total** across
the12trajectories (72,000 extra versus all-single-view),7,680,000transformed
training-example evaluations. Evaluations add forward-only work, not counted
as backward passes. Record actual backward counts, synchronized training,
augmentation, evaluation and total times; extra views are not free.

One new local unit `spectral-multiview-clean-001.service`:1CPU,16GiB hostRAM,
noSwap,8GiB GPU,1GiB archive,900s cooperative/1200s hard,Restart=no. Preserve
accepted desktop GPU clients. Fresh validated mktemp parent on /tmp/spectral-experiment-artifacts,
exclusive child/attempt, committed source/protocol/fixtures before acquisition.
Exact byte inventory required in preflight before launch. One bounded independent
audit thereafter:1CPU,2GiB/noSwap,300s cooperative/360s hard,32MiB output outside
archive under new `spectral-multiview-clean-audit-001.service`. Cloud$0/$100.
