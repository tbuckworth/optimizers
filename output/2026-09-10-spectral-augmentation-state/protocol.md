# Saved-state augmentation diagnostic: useful direction versus step size

Codex — Spectral Optimizer Investigation · 10 September 2026

## Scope and hypotheses

This is the next task selected after the user's clean and wrong-label
augmentation questions. Those acquisitions/audits are complete and consumed.
Use only the **six clean native update100 snapshots**: seeds202609141/142/143
× original/translated warmup. Exact parents, source results/audit and plans
are registered in parent-pins.json. No old experiment is resumed or repeated;
no noisy parent, new training trajectory or optimizer-default change is added.

Question: at these fixed parents, does native's local useful-learning cost
persist when **actual Adam data-step norms** are equal, or is it recoverable
by a parameter-space scale control? Geometry separately asks how much the
recorded span admits the gradient mean and variation across transformed views.
Neither local response nor gradient retention explains the full4,000-step
trajectory by itself. Keep useful native learning/protection from prior runs.

Live hypotheses: scale-limited local learning; adverse direction at equal
scale; beneficial direction with an adverse scale; neither difference; or
mixed behavior across input/readout/parent. No outcome is fixed by construction.
Post-Adam norm matching is a diagnostic counterfactual, **not** a deployable
learning-rate or gradient-normalization variant. It motivates, not certifies,
a subsequent stronger optimizer recipe.

## Fixed data and state

Reuse the original pinned MNIST training IDX and each seed's saved balanced
5,000train/5,000held-out plan. Select training localIDs by
PCG64(SeedSequence([10,seed])).permutation(5000)[:64]. Stream11 draws four sets
of dx,dy integer shifts with shape(4,64,2), integers(−2,3), dtypeint8. Stream12
selects held-out localIDs by permutation(5000)[:256]; stream13 draws one fixed
shift per readout example, shape(256,2), same integer law. Same diagnostic
plan across both warmup modes. Original and translated held-out labels are true;
train and readout source IDs remain disjoint. Store all actual IDs/shifts/labels.

The exact unchanged zero-fill translation function is reused. Five candidate
batch gradients: original64 and four separately translated64-example batches.
No view-mean proposal, extra example panel, alternate transform or tuning.
Two readout objectives: original256 and fixed-translated256 held-out images.
Use fullmeanCE for gradients, retain accuracy and full logits for readouts.

Hash parent bytes before loading; weights_only=True,map_location=cpu, exact
accepted snapshot/model/Adam/tracker schema, finite values and update100
required. Use the unchanged accepted MLP/Adam/filter code. Saved parents and
all original files remain immutable. All prospective responses start from a
fresh private copy of the same parent. No response is chained into another.
Check the in-memory original parent and disk hashes after measurement.

## Geometry — empirical finite-grid identity

At each fixed model, collect each example's gradient for its original image
and its four views using autograd.grad, without accumulating .grad. Collect
the exact five batchmean gradients separately and verify agreement with the
per-example mean to FP32 tolerance. No covarianceP×P array is constructed.

For translated gradients gᵢᵣ (B=64,R=4), define μᵢ=meanᵣgᵢᵣ, μ=meanᵢμᵢ,
T=meanᵢ,ᵣ‖gᵢᵣ−μ‖², B_between=meanᵢ‖μᵢ−μ‖²,
W=meanᵢ,ᵣ‖gᵢᵣ−μᵢ‖². **T=B_between+W** exactly on this empirical grid.
Apply the same decomposition to Q_old Q_oldᵀg to measure actual operator-
retained components; use its small Gram matrix to compute norms without
formingP×P. Also report ‖Q_old Q_oldᵀμ‖²/‖μ‖² and original-image mean/variation
retention. This honors small recorded non-orthogonality instead of silently
reorthogonalizing the archived basis; for an orthonormal Q it equals coordinate
energy ‖Q_oldᵀμ‖². Record basis Gram error separately.
Zero denominators yield null/explicit unavailable, not perfect retention.
Four-view μᵢ contains residual view randomness: the between term is **not**
an unbiased population signal estimate or semantic-feature label.

This old-Q geometry does not update the observer. Actual native delivery below
uses the candidate-specific updated Q, keeping these two questions separate.

## Actual proposals and matched-size controls

For each of five batch gradients, private native tracker.filter_grad observes
that candidate **once**, update100→101, then projects before actual AdamW.step.
Raw uses the same original model and carried Adam state without projection.
Record old and each updated basis/rank, applied gradients, parent Adam
moments/counts, theta and actual resulting parameter vectors. Assert each
private Adam count advances exactly once; the auditor reconstructs its reference
response from the archived parent moments/counts. No candidate's observer is reused.

Let θ_decay be the rounded FP32 result of θ·(1−lr·wd), and θ_R,θ_N the actual
raw/native AdamW outputs. Define data displacements d_R=θ_R−θ_decay and
d_N=θ_N−θ_decay, with these subtractions rounded in FP32. Norm matching uses
float64 norms of those rounded FP32 displacements (including the materialized
matched displacement). Derivative utilities instead subtract the archived
parameters after casting to float64. **Decay-only is not Adam with zero gradient**: the latter
would also move carried momentum. Use the same θ_decay for all full responses.

Five full candidate endpoints, in fixed order:
raw θ_R; native θ_N; raw_to_native θ_decay+d_R‖d_N‖/‖d_R‖;
native_to_raw θ_decay+d_N‖d_R‖/‖d_N‖; decay θ_decay.
Compute scales from float64 norms, then materialize parameters in FP32.
Norm-match multiplier>100 or a zero source/target norm is explicitly
unavailable for that matched candidate, without changing raw/native/decay.
No cap clipping, replacement norm or post-outcome threshold adjustment.
Check actual materialized data norms, not just algebraic intended norms.

Read each endpoint at path fractions α=1 and0.1:
θ(α)=θ+α(θ_full−θ), in FP32. Atα=1 retain actual full endpoints exactly.
At0.1 this scales the **whole displacement including decay**: it is a local
nonlinearity check, not another Adam step or full-strength-decay control.
For each objective, retain baseline/after logits, accuracy/CE improvement
L(θ)−L(θ(α)), total signed linear utility−∇L(θ)·[θ(α)−θ], and data-only
utility relative to the identically scaled decay candidate. Positive CE
improvement/utility is favorable. No-step baseline and decay comparator are
both explicit. The two finite path fractions are not tuning candidates.

## Primary contrasts and interpretation

Report all candidates/absolute utilities first. Atα=1, the two directional
contrasts are native_to_raw−raw (equal raw data norm), and
native−raw_to_native (equal native data norm), for both clean objectives.
Also report native_to_raw−native as scale recovery and native−raw as the
unmatched policy contrast. Use CE improvement and signed linear utility;
accuracy is retained as a less sensitive local readout. α=0.1 checks whether
finite/linear signs change, without selecting whichever looks favorable.

Average the four translated draws **within parent/seed first**, leaving the
original draw separate. A four-view mean/contrast is unavailable unless all
four required draws are available; retain each draw and availability count,
never silently average an available subset. Report all three seed values and means separately by
warmup mode, candidate input mode and readout objective. Do not treat30probe
batches, six parents or600readouts as independent training replicates. A
comparison between warmup modes changes model, Adam and observer together;
it cannot isolate observer history. Mixed signs and unavailable matches stay
visible. This measurement does not establish long-run mediation, semantic
selection, a tuned optimizer advantage or an alignment benefit.

## Verification, artifacts and resource envelope

Freeze source/protocol/fixtures and exact parent pins before acquisition. New
NumPy-only auditor independently reconstructs diagnostic IDs/shifts, finite-grid
geometry, projection/delta algebra, carried-Adam reference responses, matched
norms, logit metrics and contrasts from archived arrays. It does not regenerate
autograd tensors, SVD history or model forward passes; independent algebra is
not an independent training replication. Synthetic exact/analytic fixtures
cover those boundaries before admission.

Tolerances: metric float64 comparisons1e−10absolute/relative; geometric identity
1e−9relative with1e−10absolute; basis orthogonality maxerror1e−3; FP32 gradient,
projection/Adam reference comparisons use explicit measured L2 error bounds
at most5e−5relative plus1e−6absolute (Adam reference tolerance scales with its
data-displacement norm, never the much larger parameter-vector norm); matched materialized data norm5e−5relative
plus1e−7absolute. Record errors and do not silently relax after a scientific
attempt. A failure preserves partial evidence, not an automatic retry.

Store exactly six bounded parent NPZs (under128MiB each), three diagnostic plans,
source/parent receipts, schema/validity metadata and final results. Full per-
example gradients, basis/moments, proposals and logits fit below1GiB in total;
noP×P matrix, checkpoint duplication or new trajectory history is needed.
New unit `spectral-augmentation-state-001.service`: localRTX3090,1CPU,
16GiB host/no swap,8GiB GPU,1GiB archive,900s cooperative/1200s hard,
Restart=no. Fresh validated big-volume parent and exclusive attempt guard.
One independent audit thereafter: CPU-only1CPU/2GiB/no swap,300s cooperative
/360s hard,32MiB output cap, new exclusive JSON outside the source archive.
No paid
compute, cloud call, existing GPU-user displacement or experiment restart.