# I14 independent design review

Codex — Spectral Optimizer Investigation, 7 September 2026.

## Verdict

The [prospective protocol](protocol.md) is a fair, bounded discriminator of
whether the current stable native filter can have a neural effect without
AdamW's coordinatewise second moment. It is not an optimizer leaderboard,
separately tuned method comparison, or mediation design, and the protocol says
so. I found no material blocker after four clarifications were incorporated:
the per-calibration-seed 85% competence floor, the six nonzero selector
checkpoints, the first projected update at 101, and the distinction between 90%
uniform replacement and 81% expected incorrect labels.

This review used the current knowledge schema, the historical cross-optimizer
audit (artifact not distributed in this public snapshot), the
[I13 candidate](../iteration-013/next-cross-optimizer-design.md), and the
[I13 result](../iteration-013/results.md). It involved no model/data run,
experiment, GPU use, outcome inspection, or approval gate.

## Why the comparison is identifiable enough

The primary contrasts are within-base paired effects

\[
B(o,t,m)=U(\mathrm{current32})-U(\mathrm{raw}),
\]

at a fixed 2,000-update endpoint, separately for three optimizers, two target
regimes, and clean CE/accuracy. Raw and filtered arms share initialization,
examples, corruption, batch order, learning rate, decay coefficient, and the
first 100 updates. Therefore a favorable SGD contrast is direct evidence that
AdamW's diagonal second moment is not necessary for that measured effect in
this setting. A null or adverse SGD contrast does not establish necessity:
the tested operating point, horizon, rank, task, and decay remain alternatives.

The clean raw-only calibration is independent of confirmation outcomes and
does not let the filter choose its own rate. Requiring at least 85% final clean
validation accuracy in each of two calibration seeds prevents a finite but
nonfunctional baseline from entering confirmation. Selecting the eligible rate
by mean final validation CE is coherent with the question “does filtering help
at a competent raw operating point?” It does not establish fairness between
separately optimized raw and filtered methods. Boundary-grid selections and
all rejected curves must remain visible because they diagnose operating-point
sensitivity.

The fixed endpoint and the validation-selected results answer different
questions and are correctly kept separate. The endpoint tests continued clean
learning or corruption protection; the six-checkpoint selectors test whether
each trajectory has a useful earlier state. Evaluating both auxiliary metrics
under both validation selectors avoids choosing the reported metric after the
selector is known. Retaining h0 and h100 absolute progress is important because
a favorable filtered-minus-raw endpoint can arise from preservation against raw
deterioration rather than filtered learning.

## Ranked failure scenarios and existing controls

1. **A nominal SGD null is really a broken operating point** — high likelihood,
   high interpretive severity without a guard. The calibration floor, two-seed
   eligibility, frozen grid, and no post-outcome expansion now control the most
   vacuous case. Residual warning signs are a boundary-selected rate, weak clean
   competence near 85%, or negligible fixed-target learning by raw SGD. These
   should qualify the conclusion rather than trigger retuning.

2. **Cross-base interactions are attributed to optimizer geometry although
   regularization changed with the rate** — high likelihood, medium severity.
   A common coefficient does not imply common shrinkage: the per-update factor
   is `1-lr*0.01`. The protocol correctly treats
   `B(SGDm)-B(SGD)` and `B(AdamW)-B(SGDm)` as descriptive. Within-base effects
   remain cleanly paired; cross-base differences cannot isolate momentum or
   adaptive scaling.

3. **Warmup differences contaminate the filter contrast** — low likelihood
   after the explicit check, high severity. Exact model, optimizer, observer,
   RNG, and evaluation equality through update 100, with the first projected
   delivery at update 101, makes the post-warmup policy the intended treatment.
   Any mismatch is a structural abort, not a scientific outcome.

4. **A noisy-label benefit is called denoising when raw never fit the
   realization** — medium likelihood, medium severity. Fixed training CE,
   accuracy, expected-soft CE, confidence, and
   `R_zeta=L_fixed-L_soft` are retained. They distinguish raw deterioration,
   realization fitting, and clean progress, but do not identify a semantic
   clean subspace. The exact realized incorrect-label count must accompany the
   nominal replacement rate.

5. **Selected checkpoints hide an adverse endpoint or failed arm** — low
   likelihood after protocol controls, high severity. The fixed endpoint is
   primary, selector results are separately named, ties choose the earliest
   checkpoint, and a missing arm makes its paired estimand unavailable. No
   survivor mean or earlier endpoint substitutes for failure.

## Mathematical and reporting boundaries

For plain SGD after warmup, the decay-subtracted data update is
`-lr*P_t g_t`, so it lies in the current retained span up to floating-point and
parameter-rounding error. SGDm instead applies its accumulated buffer, which
contains gradients expressed under earlier bases; AdamW additionally applies
coordinatewise moment scaling. Neither update must remain in the current span.
The saved data-step leakage and signed gradient–displacement diagnostics can
show whether optimizer families transform the projected input differently.
Correlation between leakage and outcomes is not a mediation estimate.

The historical evidence already contains real SGD/SGDm/Adam comparisons, but
only under single-seed legacy covariance code and test-selected rates. I14's
fresh plans, stable implementation, independent calibration, and three seed
bundles address those weaknesses without erasing the old mixed results. The
confirmation seeds may reuse examples across seeds, so they are randomized
replications conditional on one reused MNIST source dataset, not independent
dataset replications.

No extra optimizer, rate grid, rank, noise level, or mean-preserving arm is
needed for this question. Adding them would weaken the necessity contrast and
create new selection axes. Subject to implementation tests and frozen source
checks already required by the protocol, the scientific design is ready for a
bounded acquisition.

## Static implementation addendum

I subsequently compared `run_cross_optimizer.py`, `optimizer_core.py`, and
`data_plan.py` against the protocol without executing them. The final reviewed
source implements the registered 18-curve calibration and exact 36-cell
confirmation membership; derives eligibility from both calibration seeds;
re-derives selected rates from completion-bound curves; and copies and
hash-checks the outcome-free confirmation plans. Source, dataset, prior-phase,
attempt, and artifact records are bound across the three exclusive phases.

Raw/current initial states and evaluations are digest-equal. A pair that reaches
h100 must also have identical full-state and evaluation digests there. Because
the policies are untreated duplicates through update100, a pre-warmup numerical
failure is accepted only when both arms have the same nonfinite state,
diagnostics, curve, and failure fingerprint; an asymmetric failure aborts the
phase rather than becoming evidence. The complete smoke must contain all1,320
updates with no numerical failure, and its elapsed time is scaled by each real
phase's update count with the frozen1.5 safety factor before real-data work.

Earlier review findings involving permissive calibration schemas, preflight
writes, incomplete tensor-topology validation, intermediate `OrderedDict`
serialization, version-string serialization, and unmatched warmup failure were
corrected in the reviewed source. The trajectory preflight now checks tensor
dtype, rank, device, feature width, required nonempty splits, label ranges,
finiteness, and batch bounds before model construction or its h0 state write. I
found no remaining material protocol/runner mismatch. This is static
acceptance, not an execution result or an audit of future artifacts.
