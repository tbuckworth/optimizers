# I11: fixed-label duration continuation

Prospective 7 September 2026, after completed and audited I10. This is an
adaptive follow-up informed by I10, not a new independent confirmation of its
original primary. Autonomous in-scope research, desktop RTX3090, zero cloud spend.
No completed/live/failed I7, I9 or I10 experiment is restarted or overwritten.

## Question and fixed design

Does the adverse fixed-label clean-CE contrast from previously filtered states
cross over as raw memorization progresses? Does filtered learning itself improve,
or does a relative gain reflect preservation while raw deteriorates?

Continue **all 63 physical fixed-objective final states** selected from the
hash-bound I10 branches.json, retaining their exact policy. These represent 72
logical cells over all 24 I9 anchors; identical step-100 source aliases remain
one execution, not independent evidence. No soft/redraw extension, new source
training, optimizer reset, new learning rate or outcome-driven subset selection.
Three seeds 100/101/102, original raw/current histories, original parent steps
100/500/1500/2000, and raw/current32/frozen32 continuation policies all remain.

Each state receives 1,500 new updates: 94,500 physical updates total. Combined
with I10 this gives 2,000 updates since its I9 parent. Model, optimizer moments
and counters, observer, modes, gradients and RNG are restored exactly. Frozen32
keeps the original I9-parent operator and unchanged observer, not a newly chosen
I10-boundary subspace. Original I9-parent V is also the frozen reference for all
new step diagnostics; current-basis diagnostics retain their usual meaning.

New plans use local NumPy SeedSequence([20260907,11,seed,i9_parent_step,0]) and
1,500×64 uniform training indices in [0,5000). Save every plan before training.
Plans are shared across all three policies and both source histories for each
seed/parent-step. This is a two-stage shared-stream continuation, not a claim
that the I10/I11 concatenation was one uninterrupted RNG draw. Fixed corrupted
labels, train/auxiliary/validation splits, architecture and all optimizer/filter
settings remain exactly I9/I10. Only training IDX files; no official-test data.

## Seam and measurements

Verify the committed I10 completion hash, manifest/source hashes, branch index,
each selected branch/final artifact, and its I9-parent metadata. Restricted-load
every final state and independently recompute its complete-state digest. At each
start, restored full state must match and a neutral reevaluation must equal
recorded I10 horizon500 exactly. The three policies' I11 starts are **not equal**
to each other; each instead matches its own prior endpoint.

New local evaluation horizons: 0,100,500,1000,1500; cumulative continuation
horizons since I9: 500,600,1000,1500,2000. Keep both fields distinct from
i9_parent_step and Adam/observer counters. Existing I10 evaluations supply
cumulative 0,1,10,50,100,250,500, with the common seam included once.
Use unchanged I10 full-split evaluation and neutral-state checks. Retain every
new step diagnostic and one complete final snapshot per physical lineage.
At completion recheck source, I10/I9-parent and training-file hashes unchanged.

## Primary and mandatory secondary analysis

Two separate primary estimates: cumulative current32-versus-raw benefit at
continuation horizon 2,000 from **current-trained I9 parents 1500/2000**, averaging
those two parents within seed before reporting all three seed values and mean:

`B_CE = auxiliary clean CE(raw) − auxiliary clean CE(current32)`;
`B_acc = auxiliary accuracy(current32) − auxiliary accuracy(raw)`.

Positive favors current32. This is a **cumulative 2,000-update policy-regimen
contrast from shared I9 parents**, not an isolated treatment effect of the extra
1,500 updates: I11's starting lineages have already diverged during I10.
No p-value, composite success score, selected favorable metric or pooled source.
The change from the known I10 horizon500 benefit is descriptive and explicitly
post-I10-conditioned, not a replacement of the adverse I10 primary.

Mandatory: all sources, original anchors, policies, seeds, horizons, clean CE,
accuracy, L_q, L_fixed, R_zeta, confidence descriptors and step diagnostics.
For every policy report absolute changes from both I9 parent/cumulative0 and
its own I10 endpoint/cumulative500. A relative sign crossover is not itself
useful filtered learning. Report crossover only at measured horizons, preserving
all seed signs and later reversals; never infer an exact crossing time.

A descriptive optimistic raw comparator uses the complete sampled grid
[0,1,10,50,100,250,500,600,1000,1500,2000]. Within each seed, average the two
raw-policy curves from the two primary current-trained parents at each horizon
first, then independently select
minimum clean CE and maximum clean accuracy (earliest horizon on exact ties).
Compare current32 at cumulative2000 to these two raw optima, report each selected
horizon and all seed gaps, then mean. This is same-auxiliary-data hindsight,
not an unbiased stopping policy or a continuous-time optimum. Preserve the
I9 parent as the h0 stopping option. Do not select each parent separately.

Favorable steelman: filtering continues useful learning while preventing later
realization fitting, and ultimately improves clean utility beyond raw's
optimistic sampled comparator. Weaker favorable outcome: relative endpoint
protection without continued filtered progress. Adverse outcome: no crossover,
continued harmful restriction, or benefits only from raw collapse. Classification
and probability loss may disagree; preserve both. Source-history reversals,
favorable I8 toy evidence and I9/I10 negative primaries remain in the synthesis.

## Implementation and resources

Reuse unchanged I9/I10 numerical/state helpers; new thin continuation core and
runner only. CPU synthetic split-versus-uninterrupted tests cover all policies,
exact full-state/RNG equality, neutral evaluation and frozen observer preservation.
One bounded synthetic GPU smoke must reproduce split/serialized continuation
and full final states for each policy before the source-identical full run.
No real continuation outcome is inspected to alter the configuration.

New exclusive attempt root on verified /tmp/spectral-experiment-artifacts (/dev/RECONFIGURE_FOR_LOCAL_STORAGE), with
smoke/full children sharing a **1 GiB** artifact cap. Full: cooperative1,500s,
whole-unit1,800s,6GiB cgroup RAM/no swap,4GiB TorchGPU allocation,one numerical
CPU thread/CPUQuota100%,Restart=no. Synthetic smoke:100s/120s with same other
limits. Require16GiB available host RAM and8GiB free GPU; leave GUI/apps alone.
Preserve partial artifacts on failure, no automatic retry. The initial $100
Modal budget remains unspent/unreserved; this local study does not change it.