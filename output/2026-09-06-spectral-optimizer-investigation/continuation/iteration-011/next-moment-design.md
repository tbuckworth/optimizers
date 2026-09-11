# Candidate after I11: inherited optimizer-state intervention

7 September 2026. Independently proposed mathematical design, **not executed**
and not part of the frozen I11 duration experiment. No optimizer default changes.

I10 shows slow soft/redraw adaptation under projection after fixed-label
training. Existing moments, a restricted incoming mean, changed gradient
variance, and observer history may coexist. A later experiment should distinguish
these instead of treating the `.999^500` coefficient as measured causation.

Minimal component-identifying proposal: use all six late current-trained I9
anchors (seeds100/101/102 × steps1500/2000), soft/redraw × raw/current32, and the
exact recorded I10 plans. Its inherited-moment comparator already exists; do not
rerun it. Add four interventions to each copied starting state:

| Arm | m | v | Per-parameter step counter |
|---|---|---|---|
| m deletion | zero | inherited | inherited |
| v deletion | inherited | zero | inherited |
| both deletion | zero | zero | inherited |
| fresh Adam | zero | zero | zero |

This would require 96 new 500-step continuations. The first three plus the
existing inherited arm form a moment-content factorial at the same counter.
The last two contrast bias correction conditional on empty moments. Keep all
other model/filter/RNG/optimizer state fixed; do not reset counters while
retaining nonzero moments. Fresh Adam is operationally interpretable but bundles
two moment deletions and counter reset; it is not a unique v intervention.

Primary candidates: changes in raw-versus-current benefit under each intervention
relative to the existing inherited benefit, separately for objective and CE/
accuracy, seed-first over both anchors. Preserve early horizons1/10 as well as
the endpoint: mature-counter v deletion can create deliberately artificial
large updates, especially with inherited m. Nonfinite/adverse outcomes must
remain results, not motivate retuning or silent retries. Final resource/stopping
rules and exact estimand multiplicity need a prospective protocol before launch.

Defer calibration and cross-history state exchange. A separate no-update moment
calibration can avoid zero-v shock but needs fresh declared batches and explicit
observer freeze/adaptation semantics. Donor-state exchange is off-trajectory
because raw/current histories have different parameters. Neither should replace
the basic intervention without stating its different estimand.

This proposal motivates the next research decision; I11 currently changes only
training duration and preserves all moments/counters. No paid compute reserved.
