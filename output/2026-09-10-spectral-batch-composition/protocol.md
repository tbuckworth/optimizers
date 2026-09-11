# Same-exposure batch composition: can useful rare learning recover?

Prospective protocol, 10 September 2026. **Not acquired.** This is the selected
bounded continuation of the safety-first investigation, not a rerun of the
completed selectivity or J-Lens experiments. Main owns admission and launch.

## Construct and fixed roster

The strongest hypothesis is that concentrating rare correct examples within
batches can make their shared learning signal more usable, without increasing
their total loss weight. First test useful recognition and preservation, not
just movement of a covariance statistic. Reordering also changes ordinary
optimization and Adam history; this is a schedule-policy interaction, not a
covariance-only intervention. The prior design (artifact not distributed in this public snapshot)
and [checked prior result](../2026-09-09-spectral-selectivity-boundary/results.md)
motivate this new test. The prior interpretation suggested reused parents;
this protocol follows the selected design's **fresh three seeds** instead.

Exactly seeds **202609121, 202609122, 202609123** × cells **clean, diffuse** ×
schedules **interleaved, grouped** × policies **raw, native32, norm_raw**:
36 trajectories, including 12 native trajectories. No tuning, replacement
seed, success-dependent extension, selected stopping or old reference replay.

Unchanged accepted setup: 784–64–10 ReLU MLP, all 50,890 parameters; float32
pixels/255 using CPU NumPy normalization; AdamW lr .001, wd .01, betas(.9,.999),
eps1e-8, foreach=False, fused=False. Current stable centered global hard rank32,
beta .99, raw covariance, repair100, relative_eig_tol1e-8, absolute_eig_floor0.
Norm_raw uses the same own-state native-action norm function along its raw
gradient, with float64 norm/scaling and float32 delivery. Explicit zero remains
a real gradient and Adam step. Preserve the accepted 10×dtype-epsilon
post-cast relative norm guard. No schedule, clipping, augmentation or changes
to the production filter/core or completed study source.

## Data, warmup and exact schedules

Use the same pinned official **training** IDX files and accepted make_plan
recipe on the new seeds: 550 training IDs per non-8 digit, 50 digit-8 IDs,
500 disjoint held-out IDs per digit. PCG64 with SeedSequence([seed, stream_id])
streams0–5 retain split/warmup/continuation/diffuse/unused-poison/unused-sham
semantics. Unused patch allocations do not enter inputs, training or outcomes.
Save the underlying complete plan for direct reconstruction. No official test
data, patches or true-label changes for rare digit8.

One new 100-step majority-only clean raw warmup per fresh seed, with the
unchanged observer recording its raw gradients. All 12 branches within a seed
restore that exact full model/Adam/filter/RNG state. Absolute horizon2000,
batch64; 1900 continuation updates per branch. This is 300 physical warmup
plus68,400 continuation updates. Evaluation/probe randomness never consumes
the training streams.

Take each original 50×64 block of continuation indices: **38 blocks, no short
final block**. Treat draws as occurrences, retaining repeated example IDs.
For zero-based block b:

1. Permute its flat3,200 occurrences using PCG64(SeedSequence([seed,6,b])).
   Split this order into rare and non-rare lists using the saved true labels.
   Their within-list orders are shared by both schedules.
2. Draw the same permutation π of the50 batch positions for both schedules
   using PCG64(SeedSequence([seed,7,b])). Let R be this block's rare count.
3. Interleaved rare counts start at floor(R/50) each, with one extra in the
   first R mod50 positions of π. Grouped counts start at zero; fill up to64
   rare slots in each successive position of π until R is exhausted.
4. In chronological batch order0…49, consume the corresponding counts from
   the common rare list and fill the remaining slots from the common non-rare
   list. This preserves within-group order across the block before within-batch
   shuffling; neither schedule gets an independently drawn example stream.
5. Within each batch t, apply the same64-slot permutation to both recipes,
   using PCG64(SeedSequence([seed,8,b,t])). No batch depends on model outputs.

Persist both full schedules, per-batch rare counts, all block permutations and
count vectors (or their reproducible named-stream definitions plus exact array
hashes), and each block's multiset/count checks. Require both schedules to have
the exact same occurrence multiset as the original block. Require interleaved
counts differ by at most1, grouped counts fill64 except at most one partial
batch, and correct total rare exposure. Handle R=0, R=3200 and intermediate
extremes in fabricated tests; never reject a real seed because the contrast is
small. Both are interventions: interleaved is not the previous iid baseline.

At fixed θ the average of these50 batch gradients is identical in exact
arithmetic, since the same3,200 equal-weight losses appear. It is not identical
along divergent training trajectories. In the deterministic two-group model,
g_t=μ_c+q_t(μ_r−μ_c), and centered block covariance is

    C_block = Var_block(q_t) (μ_r−μ_c)(μ_r−μ_c)ᵀ.

This gives a constructive selection signal, not a prediction of neural rescue.
Finite within-group variability, moving means and Adam feedback remain.

## Behavioral outcomes and prospective interpretation

Save train and held-out unpatched float32 logits at0,100,200,…,2000; share exact
initial/warmup predictions across all branches. Score true-label CE/accuracy
for every digit, rare8, majority macro and balanced total. Separately score
assigned-target fit and actually changed-target fit on training examples.
Report all seed curves and changes from warmup, with fixed2000 primaries.

For every policy/cell, report grouped-minus-interleaved rare/common accuracy
and CE. For both schedules/cells, report native-minus-raw and
native-minus-norm_raw. Primary schedule-policy interactions are

    I(cell,metric) = [M(grouped,native32) − M(interleaved,native32)]
                    − [M(grouped,raw) − M(interleaved,raw)].

Report analogous norm_raw-minus-raw interactions as secondary, all endpoint
class metrics and wrong-target fitting, every seed, mean/sampleSD/sampleSE and
sign counts. Positive is favorable for accuracy, negative for CE. No composite
victory score, equivalence/noninferiority declaration or selected checkpoint.

A constructive result means actual improvement in rare learning from warmup
and versus interleaved native, while common competence and diffuse protection
remain useful. Quantify any common-accuracy/CE or wrong-label-fitting costs;
no unregistered tolerance may turn a cost into “no sacrifice.” Similar schedule
benefits in raw weaken a filter-specific account. Changed retention without
recognition weakens geometry-alone explanations. Finish the fixed roster
regardless of these outcomes; do not tune away nulls or adverse cases.

## Six count-selected native diagnostic events per trajectory

In each of zero-based blocks **0,7,37**, select two distinct chronological
batch positions: high = first argmax of rare count; low = first argmin among
positions excluding high. These are chosen from schedules only, before any
training. If counts are equal, record that explicitly rather than implying a
high/low contrast. Absolute update =101+50b+t. Execute diagnostics in increasing
update order and retain their high/low labels. 12 native trajectories ×6 =
**72 events**. No raw/control diagnostic branch or counterfactual update replay.

At the pre-update model after its single actual training-gradient observer
update, preserve model/Adam/filter/RNG and .grad exactly around probes. Use
autograd.grad on the mean CE, not backward into training gradients. Save the
mean gradient (float32 P-vector), inputs, IDs, true/assigned targets, pre/post
logits for each group. **No per-example-gradient/coherence estimate is made.**

Probe membership is fixed across schedules/policies and independent of model
outputs. Use PCG64(SeedSequence([seed,9,group_id])):

- Group0 majority: in ascending true-digit order excluding8, randomly permute
  that digit's550 training positions and take6; concatenate all9 strata (54).
- Group1 rare8: randomly permute its50 training positions and take32.
- Group2 wrong: randomly permute all actually changed Diffuse positions and
  take min(32,count); reuse identical inputs with assigned versus true targets.
  Clean wrong groups are absent/null, not zero. Save true-label counts.

Majority and rare8 probes always use **true labels**, including under Diffuse.
Only wrong_assigned uses changed labels; wrong_corrected uses true labels.

These are small randomized **training** subsets, not held-out causal estimates.
The majority probe is stratified, the wrong probe is not the earlier first32
class-blocked subset. Report each mean's norm/native-action retention,
rare-minus-majority mean retention, wrong-assigned-minus-corrected retention,
actual Adam movement with decay/adaptive decomposition and numerical-span
off-component energy, signed −mean_gradientᵀΔθ and finite before-minus-after CE.
Mean differences use explicit float32 subtraction before float64 summaries.
Persist complete before/after state plus raw/applied training gradients and
parameter order for independent arithmetic checking without rerunning a model.
Retained gradient energy is not signed utility, safety or trajectory mediation.

## Execution and evidence bounds

Immutable helper imports from accepted selectivity code and I9 core are allowed;
no mutation/monkeypatch of their globals and no calls to their acquire/main.
New schema must distinguish schedule, seed, cell, policy and absolute update.
Pin every executed source/import, protocol, fabricated tests, environment and
input file. Check committed bytes at launch and unchanged bytes at completion.

Local RTX3090 only,1CPU math thread,16GiB host/no swap,≤8GiB GPU allocation,
30min hard/25min cooperative deadline,≤3GiB total acquisition output,≥1GiB free
disk reserve on mounted /private-artifacts/storage. Analytic inventory must cover3warmup,
36final and144diagnostic full states plus72events' mean/action gradients,
logits and metadata before launch. No paid compute or credentials required.
Use one exclusive parent/acquisition-001 and one non-restarting service,
`spectral-batch-composition-001.service`; no retries or source changes mid-run.

Freeze the independent checker/tolerances before outcome access. Once complete,
one CPU-only saved-data audit checks sampler reconstruction, all source and
artifact hashes, roster/pairings, logits/CE/counts/contrasts, actual actions,
Adam arithmetic, mean-gradient geometry and finite probe losses. It does not
repeat training/inference/gradient estimation or replay an earlier audit.
If acquisition/audit fails, preserve evidence and report the precise defect;
do not silently rerun, relax tolerances or certify a partial result.

## Implementation documentation check

Explicit named PCG64/SeedSequence streams and persisted batch arrays follow
[NumPy's stream guidance](https://numpy.org/doc/stable/reference/random/parallel.html).
We retain the accepted pinned Torch2.11.0+cu128 environment and deterministic
settings; current [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
warns that seeds do not ensure cross-version/platform identity. No upgrade or
cross-invocation bitwise future-trajectory claim is made. Main used the
best-practices validator before implementation; these checks support data
pairing and provenance, not scientific efficacy.
