# Candidate current-stable cross-optimizer discriminator

7 September 2026. This is a design candidate, not an acquired protocol, source
freeze, or launched experiment. It proposes no production-default change.

## Question and smallest factorial

Does the current stable native filter have a neural effect without AdamW's
coordinatewise second-moment transform, and does ordinary momentum materially
change that effect? Run exactly:

```
3 bases × 2 filter states × 2 targets × 3 fresh seeds = 36 trajectories
bases:          SGD, SGDm(.9), AdamW
filter states:  raw/passive observer, native current stable hard32
targets:        clean, fixed 90% label replacement
candidate seeds: 200, 201, 202
```

This is a necessity discriminator, not an optimizer leaderboard and not an
identified mediation fraction. A replicated filtered effect with SGD would
show that AdamW's diagonal second moment is not necessary for an effect in this
setting. Failure under SGD would not prove necessity: finite horizons,
base-specific operating points, and optimization failure remain alternatives.

Use the current 784–64–ReLU–10 model and MNIST split conventions, but initialise
all 12 cells of a seed from the same **fresh** weight state. Do not use I9/I10/
I13 Adam-trained parents. Per seed, freeze one disjoint train/validation/
auxiliary split; one fixed-corruption mask and replacement-digit array; and one
complete 2,000×64 batch-index plan. The clean and corrupted cells use the same
examples and batch indices. Raw and filtered cells consume identical targets in
identical order. Save plans before training. Use no official-test outcome.

## Filter and optimizer semantics

Both filter states instantiate the same current canonical stable observer:
rank 32, EMA decay `.99`, warmup 100, hard weighting, no normalization or
adaptive rank, relative eigenvalue tolerance `1e-8`, absolute floor zero, and
stabilization every 100 observations. It sees each raw gradient exactly once.
The raw arm is passive and delivers the unchanged gradient; the native arm
delivers the canonical post-ingest current projection after warmup. The two
arms must have identical parameters and optimizer state through update 100.

To avoid silently changing regularization along with optimizer geometry, use
the same explicit decoupled decay coefficient `.01` in every base:

- **SGD:** zero momentum, dampening zero, Nesterov false; apply the decoupled
  multiplicative factor `theta <- (1-lr*0.01) theta`, then the data update.
- **SGDm:** momentum `.9`, dampening zero, Nesterov false, buffer initially
  absent/zero with the exact PyTorch first-buffer convention; the same separate
  decoupled decay and then the momentum data update.
- **AdamW:** betas `(.9,.999)`, epsilon `1e-8`, AMSGrad false, maximize false,
  capturable/differentiable false and foreach/fused false; weight decay `.01`.

The implementation must test the SGD/SGDm decay ordering against the stated
formula. Do not pass `weight_decay=.01` to ordinary SGD: PyTorch SGD's coupled
L2 term is not AdamW's decoupled decay. Report total displacement, the nominal
decay displacement, and their data-displacement difference separately.

## Learning-rate operating point

The preferred design uses a **separate raw-baseline-only calibration**. On
separate calibration seeds and disjoint validation data, run a small
predeclared logarithmic grid for each base on clean labels only. Select the rate
with lowest mean clean validation CE at update 2,000; ties within exact stored
precision choose the smaller rate. Reject nonfinite candidates rather than
silently shortening their horizon. Freeze the selected rate before constructing
or inspecting any filtered or 90%-corruption result, and reuse it for all four
scientific cells of that base.

This gives each base a functioning raw operating point and prevents the filter
from selecting its own favorable rate. It does use validation outcomes and adds
calibration work, so the result answers “does filtering help at a competent
raw-baseline rate?” It does not compare separately optimized methods fairly.
The calibration seeds, candidate rates, selection record, and all rejected
candidates must be retained; their runs are preparation, not extra scientific
replicates.

A cheaper alternative is outcome-free initial-step matching: on a frozen
initial model and predeclared clean batches, choose each rate so the median raw
data-step norm matches AdamW at `.001`. That avoids outcome selection and extra
training curves, but it matches only local scale. At the first step SGDm is
effectively SGD; it says nothing about later momentum amplification,
conditioning, or convergence. It is suitable as a sensitivity operating point,
not the sole basis for interpreting an apparent SGD null. Do not run both
calibration schemes unless they are predeclared as separate sensitivity
families; post-outcome choice between them would create another selection axis.

## Outcomes and contrasts

The fixed endpoint at update 2,000 is primary because late corruption fitting
is part of the question. For clean auxiliary CE and accuracy, retain separately
for each base and target

```
B(base,target,metric) = U(native)-U(raw),
Gamma_momentum = B(SGDm)-B(SGD),
Gamma_adaptive = B(AdamW)-B(SGDm).
```

Here CE utility is negative CE; higher is always better. Report all three seed
values and their mean, with CE and accuracy separate and no p-value, composite,
or pooling across clean/noisy targets. The most direct non-necessity evidence is
a seed-consistent favorable `B(SGD, noisy, metric)` accompanied by functioning
raw SGD and no disproportionate clean cost. Optimizer interactions are
descriptive unless their predicted step geometry is also observed.

Separately report a predeclared stopping regime. At updates
`{100,250,500,1000,1500,2000}`, select each trajectory's checkpoint by clean
validation CE for the CE analysis and by clean validation accuracy for the
accuracy analysis, with earliest-checkpoint tie breaking; evaluate the selected
checkpoint on auxiliary data. Never substitute this selected result for the
fixed endpoint. It answers whether an optimizer/filter pair merely has a useful
earlier stopping point, while the endpoint tests preservation under continued
training.

At every checkpoint retain clean validation and auxiliary CE/accuracy, noisy
training CE/accuracy for corrupted cells, raw/applied gradient norms, actual
data-step norm, separately reconstructed decay norm, and step energy outside
the current learned basis. For plain SGD, a projected data step should lie in
the current range up to floating-point error. SGDm may retain motion from past
bases; AdamW generally need not commute with the projector. Similar filter
benefits despite these differences weigh against an Adam-specific account;
different benefits that track leakage or scale support optimizer interaction,
not causal mediation by themselves.

## What I13 changes—and what it does not

I13 found that, on six reused late AdamW states, `mean32` strongly improved
soft/redraw clean utility over native `current32` at horizon 500: mean auxiliary
effects were `+0.1864/+27.28 points` for soft CE/accuracy and
`+0.1276/+23.56 points` for redraw. Absolute soft/redraw progress became
positive in aggregate and approached raw (redraw CE worsened in one seed).
The gain remained large against the one-percent leak control; this is a
whole-policy comparison, not an additive attribution to one component.
These are conditional saved-
state results, not fresh-source optimizer comparisons; see
`analysis-001/summary.json` and its scope block.

That result is a reason **not** to add policies blindly here. The 36-cell study
should retain only raw and canonical native filtering, because its question is
whether the already studied native effect requires AdamW. Adding mean32 would
make a 54-cell joint test of a different algorithm and blur the necessity
contrast. If the native factorial identifies a substantive optimizer
interaction, mean32 transfer can be a separate, smaller follow-up on the most
diagnostic two bases. I13 neither licenses assuming mean32 transfers to SGD nor
requires doubling this grid now.

## Failure and resource discipline

Freeze source, calibration choices, plans, expected membership and byte/time/
memory limits before any scientific outcome. A tiny synthetic CPU suite and one
source-identical synthetic GPU smoke may validate semantics but may not tune
the study. Derive the final time cap from that smoke and the exact 72,000-update
membership; do not copy I13's 18,000-update wall cap unchanged. Use the same
local big-volume/artifact discipline, explicit host/GPU admission checks, zero
swap, bounded GPU allocation, and no automatic retry or cloud fallback.

Nonfinite branch arithmetic retains a typed failed-arm record and makes every
contrast requiring that endpoint unavailable; never replace it with a survivor
mean or an earlier checkpoint. Independent arms may continue after a contained
numerical failure. Source/plan/provenance mismatch, serialization failure,
storage ceiling, CUDA/eigensolver error, timeout, OOM, or other resource failure
terminates the acquisition and preserves the partial record. Retain adverse
finite trajectories without clipping, retuning, or early stopping them.

Before acquisition, a final protocol must replace the candidate seeds,
calibration grids and resource ceilings above with source-bound exact values.
This note itself authorizes no run.
