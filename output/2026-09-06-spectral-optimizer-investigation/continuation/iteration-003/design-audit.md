# Iteration 003 prospective statistical and interpretation audit

Prepared on 6 September 2026 before confirmatory results. Reviewed
`best-practices-check.md`, `continuation/README.md` and the complete current
`protocol.md`. This is a design audit, not an implementation validation or
permission to launch. Only this audit file is owned by its author; the parent
and implementation author own any protocol changes and launch decision.

Reviewed protocol SHA-256:
`3aaee82c98bd0ec5cdab20d3a9d3495e749346a0f5ee1fce77183650be0834bc`.
No confirmatory results were available to this reviewer. The design does not
require changing its model, data sizes, paired seeds or optimizer factors for
the stated measurement question. The definitions below should be resolved in
the final prospective text. The protocol author has acknowledged the
aggregation and cross-term recommendations, but this review does not assert
that later protocol amendments or implementation changes are already present.

The proposed paired measurement can establish what happens along the specified
training trajectories: which probe-gradient components their applied
projectors retain, how actual AdamW displacements relate to those projectors,
and how those arms learn at their fixed training budget. It cannot establish
that the retained components are generally useful, that wider estimation alone
causes any observed retention difference at fixed parameters, or that the
three untuned arms characterize the algorithm's optimal performance.

## Definitions to settle before the confirmatory freeze

1. **Freeze the aggregation rule for ratios and signs.** The protocol specifies
   averaging diagnostic steps, which should mean an unweighted arithmetic mean
   of finite step-level ratios within each seed, followed by a paired contrast
   and equal weighting of the three seeds. This is different from a ratio of
   summed energies, which emphasizes large-gradient steps. If both are wanted,
   identify the former as primary and the latter as a secondary energy-weighted
   summary before looking at data. Report finite/null counts and denominators;
   never replace undefined ratios with zero or silently compare unequal sets
   of diagnostic steps. Define the handling of zero displacement when computing
   leakage or cosine, in addition to the existing zero-gradient rule.
2. **Separate preferential attenuation from combined-gradient behavior.** For
   the same probe examples and parameters, `n=c+r_corrupt` exactly, so
   `||n||²=||c||²+||r_corrupt||²+2*cᵀr_corrupt`. The projected version contains
   the analogous cross term `(Pc)ᵀ(Pr_corrupt)`. A positive clean-minus-residual
   retention difference alone does not say whether the combined noisy gradient
   points toward the clean task; cancellation can matter. Either record these
   two cross terms and original/projected n norms with closure checks, or
   explicitly restrict the interpretation to the two marginal retention
   ratios. This is a recommendation about identifiability, not an extra
   counterfactual optimizer intervention.
3. **Name an ascent event precisely.** Preserve current-gradient ascent,
   positive outside-subspace contribution, and a reversal in which the
   within-subspace contribution is descending but the total is ascending as
   different events. Use the frozen numerical sign thresholds for each term.
   A positive outside contribution can weaken a still-descending step; a step
   with positive inside contribution is not evidence that leakage alone
   reversed descent. Summarize both total and nominal-decay-subtracted updates,
   and state which one is primary if a headline frequency is reported.
4. **Respect decomposition uncertainty.** The allowed normalized in/out closure
   residual is 1e-4, while the sign threshold is 1e-6. Thus a valid small
   positive total derivative does not automatically establish a clean
   within-descent/outside-ascent decomposition. A reported leakage-reversal
   event should have sign margins exceeding the measured absolute closure
   residual plus the appropriate numerical threshold, or be marked unresolved.
   No change to the production projection is implied by this reporting rule.

These clarifications prevent later choices of denominators, weighting or event
definitions from changing the headline finding. They do not require changing
the fixed model, sample count, optimizer hyperparameters or paired seeds.

## Checklist for interpreting the retained gradients

- The primary probe uses a new batch draw from the *same fixed training data*
  and noisy-label assignment. It is independent of the current batch draw,
  not independent of the model's previous learning. Chance image overlap with
  the current batch is permitted by the frozen plan. Do not label this an
  unseen-example or fresh-label generalization measurement.
- Applied P is measured after the current raw gradient updates the covariance.
  Current-gradient retention is therefore self-inclusive. The independently
  drawn training probe and disjoint clean probe help distinguish in-sample
  retention from retention on other examples, but are still conditional on the
  learned parameters and basis. The auxiliary pool is disjoint in image IDs,
  not an independent source of optimizer replication.
- `r_corrupt=n-c` is the change caused by the *fixed label assignment* at the
  same parameters and examples. It is neither all minibatch noise nor a pure
  oracle gradient of memorization. In cross-entropy, softmax-probability terms
  cancel in this subtraction, leaving a label difference multiplied through
  the current model Jacobian; that Jacobian remains trajectory-dependent.
- Replacement probability .9 gives expected incorrect-label fraction .81
  because replacement can retain the original digit. Report both realized
  replacement and incorrect fractions, and preserve the zero-corruption
  residual as a null ratio rather than zero retention.
- Squared-norm retention describes how strongly a component is attenuated,
  not whether the retained component leads to a beneficial parameter update.
  The logged clean-probe dot actual displacement connects the measurements to
  local directional change; it remains local, stochastic and distinct from
  realized generalization. A wider estimator can retain more of both clean and
  corruption directions, or neither, without contradicting these definitions.
- As a real-valued projection operator, the approximate float32 basis can
  produce slight numerical violations of an ideal [0,1] retention range.
  Preserve unrounded values and report violations with orthogonality/closure
  diagnostics; do not silently clip ratios. A material violation is a
  measurement-validity concern, not superior retention.

## Checklist for actual-update geometry and causal language

- Momentum alone can make AdamW ascend the current training batch. Iteration
  002's fixed-basis, first-step example establishes possibility; the planned
  neural run measures a different setting with accumulated moments and an
  adaptive basis. Do not attribute all neural ascent events to diagonal
  scaling or projection, even if they resemble that example.
- Keep `gᵀdelta`, finite same-batch loss change, and clean-probe directional
  change separate. They refer respectively to a local linear prediction, an
  actual finite training-batch change, and a different probe objective.
  Curvature can alter a finite-step outcome even with a descending derivative.
- Retain the signed in/out closure residual. Norm leakage alone does not show
  harm, and a positive outside term alone does not prove overall ascent.
  Decay subtraction is nominal arithmetic at the pre-step parameters with
  floating-point error; it is not a replay of an independently run no-decay
  optimizer. The same learned parameters and moments reflect historical decay.
- AdamW uses P=I by definition in this protocol. Its reported subspace leakage
  is therefore exactly zero as a measurement convention, not evidence that
  unfiltered AdamW has better geometric alignment with a spectral basis.
  Baseline ascent frequency and clean-gradient update alignment are meaningful
  comparisons; directly celebrating lower baseline leakage is not.
- Width 32 and 128 receive paired exogenous randomness but develop different
  parameters, gradients, covariances and optimizer states. Their paired
  contrast estimates a difference between the complete specified training
  procedures. It is not a fixed-gradient causal effect of changing estimator
  width while keeping the learned state identical. A separate common-state
  replay would be needed for that claim.
- The protocol separates maximum estimator width from delivered rank 32 and
  records realized ranks. A nominally wider cap need not always be filled;
  actual ranks and warmup/identity steps must remain visible. A pilot failure
  to reach width 128 is a stated timing/representativeness gate failure, not
  automatically a numerical correctness failure or evidence against wider
  estimation.

## Checklist for comparisons, selection and uncertainty

## Pilot and audit boundary

Development seed 9876 may determine runtime feasibility and test invariants,
not accuracy-based design choices. The pilot must keep loss/retention/update
outcomes out of the design-decision output even though it computes some of
them for instrumentation. Timing or implementation corrections require a
dated amendment and retained failure records. Do not retune the 2,000-step
endpoint using the pilot's apparent learning behavior.

Before launch, independently verify instrumentation-on/off trajectory
agreement, probe state invariance, fixed data/RNG provenance, loss-gradient
identities, decay decomposition, rank/projection behavior and validation
selection. This checklist does not assert those gates passed; the current
review is prospective and reads the design only. Commit the final reviewed
protocol/source hashes before the parent's explicit confirmatory GO. No
confirmatory result or implementation source was inspected for this audit.
