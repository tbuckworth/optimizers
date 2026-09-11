# I10: longer-horizon branches and fixed-label realization fitting

## Question and favorable hypothesis

I9 preserved endpoint accuracy but its late matched-displacement clean-CE
direction contrasts were adverse. A useful restriction could still change
subsequent learning: preserve shared softened-label structure while slowing
fitting of the particular fixed corruption realization. Test explicit horizons,
both classification and probability loss; a favorable outcome is not assumed.

The finite-input decomposition is L_fixed=L_q+R_zeta, with
q_i=.1 one_hot(y_clean,i)+.09 and R_zeta=-mean(zeta_i dot log p_i).
Soft-q removes realized corruption and fresh label-draw gradient variance.
Fresh redraw retains categorical label noise but removes persistent corrupted
targets, also changing their temporal correlation. Fixed-versus-redraw and
redraw-versus-soft therefore answer different questions. All use known clean
labels for mechanistic controls, not deployable baselines.

## Complete fixed design

All24 I9 anchors: seeds100,101,102; source raw/current32; completed steps
100,500,1500,2000. Verify parent file hashes, binding metadata and complete-state
digests against committed I9 records. Preserve parameters, AdamW moments/counters,
observer and all RNG exactly at each start. No moment resets or source retraining.
Same5000-image train/validation/auxiliary split and architecture; training IDX
only, no official-test data.

Every anchor has the full3×3 factorial:

| Training target | Raw AdamW | Frozen old-P32 | Actual current32 |
|---|---|---|---|
|Fixed I9 corrupted labels|yes|yes|yes|
|Soft-q targets|yes|yes|yes|
|Fresh categorical redraw each occurrence|yes|yes|yes|

500 new updates per branch. There are216 logical cells. Step100 raw/current
parents are digest-identical: execute those nine branches once per seed and
reference the same artifacts for both source labels. Thus189 unique branches,
94,500 updates; do not present aliases as independent evidence.

Frozen policy applies parent V(V.T g) without observations, normalization or
repair; observer explicitly inactive. Raw uses the I9 passive native observer,
discarding its projection. Current32 observes each branch gradient once before
projecting. Original AdamW lr.001, weight decay.01, betas(.9,.999), eps1e-8.
Frozen gradient projection does not confine Adam's delivered displacement.

For each(seed,parent step), use SeedSequence([20260907,10,seed,parent_step,stream]).
Stream0 chooses500 batches of64 training indices with replacement. Streams1/2
choose independent .9 replacement masks/uniform digits for each occurrence,
including repeated examples within a batch. All nine arms and both source
states share plans. Save all plans before execution. Do not redraw labels
inside optimizer/evaluation code. Redraw's expected objective is L_q, not one
fixed realization objective. These are new batches, not I9's completed future.

## Measurements and state

Full train/auxiliary/validation evaluations at horizons0,1,10,50,100,250,500.
Float32 chunked forward passes, float64 loss reductions. Training: clean/fixed/
soft-q CE, clean/fixed accuracy, signed R_zeta=L_fixed-L_q. Auxiliary: clean/soft
CE and clean accuracy. Validation: clean/soft CE and clean accuracy. All clean
splits: mean max probability and mean true-class probability, descriptive
confidence rather than claimed calibration. No auxiliary training gradients.

Every update records objective loss, raw/applied gradient norms, actual total
and decay-subtracted data-step norms, leakage against frozen/current relevant
bases, observer step. These are not matched movement controls; branches evolve
different states. Data displacement follows I9's
nominal convention: total_delta + lr*weight_decay*theta_before; it is not a
separately rounded isolated-decay simulation. Preserve all step diagnostics,
fixed-horizon curves and one
complete final snapshot per unique branch, bound to parent/plan/objective/policy.
Full final state keeps later investigations possible without retraining.
Frozen-basis convention belongs to branch metadata; parent observer is unchanged.

Evaluation must preserve parameters, buffers, modes, gradients and RNG.
Every restored starting digest must equal its parent, and all nine horizon0
measurements must match. Stream files; avoid duplicating initial/frozen bases.
At completion verify I9 inputs and all scientific source hashes unchanged.
Complete snapshots are not generic checkpoints from arbitrary untrusted sources.

## Primary estimates and interpretation

Primary source current32; parent steps1500/2000 averaged within each seed first,
then all three seed values and their mean. Primary horizon500. Define:

B_CE(o) = auxiliary_clean_CE(raw,o) - auxiliary_clean_CE(current32,o).
B_acc(o) = auxiliary_clean_accuracy(current32,o) - auxiliary_clean_accuracy(raw,o).

Positive favors native. Report six estimates separately: B_CE(fixed),
B_acc(fixed), and fixed-minus-soft and fixed-minus-redraw interactions for
each metric. No post-outcome metric/comparator choice, composite success score,
or multiplicity-free p-value. Baseline cancels because each parent is shared.
Here raw means removal of filtering from an inherited current-trained Adam
state, not an end-to-end unfiltered baseline.

Predict a growing fixed-label benefit and less realization-specific fitting,
with smaller benefits when persistent label realization is removed. A redraw
benefit exceeding soft-q suggests fresh-variance effects. A benefit under
soft-q also points to broader optimization/subspace effects. Frozen success
implicates retained geometry; live-only success motivates adaptation mechanisms,
not proof of one isolated cause. All states inherit corruption-trained features
and moments. Opposite interactions or persistent harm weaken the account.
Accuracy/CE disagreement remains disagreement; I9 adverse evidence is retained.

All other horizons/anchors/source states/frozen benefits and full factorial
results are mandatory secondary measurements. No state/seed/rank/horizon
selection to manufacture a favorable result. Three seed bundles reuse one
dataset;189 branches are not189 independent replications or a new benchmark.

## Implementation, resources and safeguards

Reuse unchanged I9 state/optimizer helpers and canonical spectral_filter.py.
New branch/evaluation code gets focused synthetic CPU tests and independent
review. Separate bounded synthetic GPU smoke exercises all nine combinations
and complete-state roundtrip before full execution; identical committed source
hashes in smoke/full. No real branch outcomes used to adjust configuration.

One available desktop RTX3090; leave GUI/Stremio alone. Require8GiB free GPU and
16GiB available host memory. One numerical CPU thread, deterministic PyTorch,
TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8. Cooperative1500s/whole-unit1800s cap,
6GiB cgroup RAM,4GiB PyTorch GPU allocation,2GiB new artifacts on verified
/tmp/spectral-experiment-artifacts (/dev/RECONFIGURE_FOR_LOCAL_STORAGE). This explicit I10 allowance accommodates189
complete final snapshots; I9's512MiB budget is unchanged. Exclusive outputs,
Restart=no, preserve partial evidence without automatic retry.

The best-practices skill checked current PyTorch2.11 primary documentation for
[AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html),
[probability-target CE](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cross_entropy.html)
and [weights-only loading](https://docs.pytorch.org/docs/2.11/notes/serialization.html).
These support API correctness, not the scientific mechanism. Restricted loading
does not replace exact locally produced parent/source provenance checks.
