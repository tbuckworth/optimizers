# Fixed paired action-policy analysis

9 September 2026. This makes the numerical aggregation in the frozen
[action protocol](protocol.md) explicit while the batch is still running.
It does not change an arm, horizon, readout or primary metric. The prior
representation study and seed100's first-step diagnostic summary were already
seen; this is not a claim of a wholly outcome-blind research programme. New
branch endpoint outcomes have not been inspected for this specification.

## Admission and estimands

Read the completed 35-state measurement only, with exact seed/policy/step
roster, source hashes and scalar/analyzed/raw receipts. Reuse the accepted
native representation summary with SHA256
`df4f1516f1d1db60c46e2344643f08c4d162892720b8e33e18a1b0069ec09480`.
No model inference, training, new frequency selection or readout fitting.

For each seed100–104 and each fixed endpoint2000/2500, retain individual
values and calculate three contrasts: orthogonal minus archived native;
norm-matched orthogonal minus archived native; and norm-matched orthogonal
minus orthogonal. The last is a within-acquisition policy comparison, whereas
the first two use an archived, noncontemporaneous reference. Do not silently
call all three equally controlled experiments.

Primary metrics are held-out cross-entropy, mean held-out correct-class margin,
and selected-five evaluation R² in final hidden activations. Lower CE and higher
margin/R² are favorable directions; R² is readability, not useful performance.
Report all five differences, their arithmetic mean, sample standard deviation
and standard error SD/√5, with positive/zero/negative counts and metric direction.
Seeds—not checkpoints, classes, frequencies or null draws—are the units.
No p-value, confidence interval, equivalence conclusion or composite success
score is added. A small difference is not proof of preservation/equivalence.

Retain secondary accuracy (fraction; multiply by100 only when labeled percent),
fixed-panel R², pre-attention R², shuffled-row nulls and symmetry/margin-related
diagnostics. Undefined energy-normalized secondary ratios remain explicitly
undefined; they must not become zeros or cause favorable-seed selection.
No AUC is needed for this first endpoint report. All35 new states, including
the common-input step1501 diagnostics, remain in the analysis artifact.

## Interpretation discipline

First-step actions share one raw gradient and one post-update estimator.
Report action and actual Adam displacement separately. For later steps, the
models, raw gradients, legacy estimators and Adam histories diverge. This tests
continuation policies from the common step1500 state, not a fixed projector or
selection from initialization. Norm matching is before Adam and does not match
the actual step norm, optimizer history, cumulative dose or decay balance.

Preserve any rank reduction, norm-match degeneracy, unfavorable behavior and
disagreement between readability and behavior. Read the saved operator geometry
before using the phrase "same span" rather than "numerical span". Even if
orthogonal continuation retains the observed advantage, that supports a
conditional continuation result; it does not establish circuit formation,
semantic clustering, safety or a generally faster optimizer.

The separate [Adam action note](adam-action-interpretation.md) supplies
conditional algebra and hypothesis interpretations, not additional observations.

## Resources and output

Pure JSON aggregation only after measurement completion: one CPU thread,
2GiB RAM/no swap, five-minute service ceiling, new exclusive output directory,
100MiB output ceiling. Hash verification may stream raw/checkpoint files but
must not load a model or reconstruct activations. A successful acquisition is
not a result audit: independently check raw numerical readouts before a report.
