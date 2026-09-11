# Mathematical follow-up review — 7 September 2026

## Scope and disposition

**PASS — theoretical synthesis and proposed diagnostics; NO NEW EXPERIMENTS.**

The user requested this strand in addition to the existing investigation. The
research-workflow skill was used to separate primary-source claims, mathematical
identities, existing empirical evidence and provisional hypotheses. Main
integrated the report; separate leaves supplied covariance derivations, a
bounded primary-literature comparison and a hypothesis challenge. A separate
reviewer independently checked the main derivations and final assembled report.

Deliverable: [What the spectral optimizer is selecting](../../../research/spectral_optimizer_mathematical_synthesis_2026-09-07.md).
Supporting work is in the adjacent mathematical-followup notes and bibliography.
The original delivered report/PDF and optimizer implementation are unchanged.

## Independently checked mathematics

- Conditional centering and complete-innovation covariance expectation,
  explicitly excluding native startup, rejected residuals, pruning, numerical
  repairs and noncommutation of expectation with rank selection.
- Independent same-state batch-pair covariance/mean-surprise identities;
  directional scalar estimates avoid any requirement that one cross-product
  matrix realization be symmetric.
- Clean-gradient oracle risk and the differing PCA objective, including the
  independent useful-signal/noise counterexample and favorable isotropic case.
- Frozen-J square-loss kernel normalization, PSD ordering for projected SGD,
  and why the same claim does not hold for post-projection Adam scaling.
- Stationary quadratic gradient variance, gradient and centering transfer
  functions, temporal power, and the orthogonal innovation-admission threshold.
- Exact same-prestate Adam displacement and derivative qualifications, historical
  moment span, separate decay, and why input-norm matching is not displacement
  or direction matching.
- Eigengap-dependent conversion of covariance capture regret to projector
  distance; high capture alone does not establish useful learning.

## Corrections incorporated before final review

The initial supporting drafts were tightened to distinguish ideal covariance
proposals from native truncated updates; the fixed corrupted-training conditional
mean from a population clean gradient; global sign loss from loss of all
coordinate signs/phases; and generic sinusoidal normalization from the Nyquist
exception. Orthogonal admission excludes unresolved ties.

Historical momentum was described using the span of historical gradient spaces,
not their union. Adam derivatives state their positive-denominator domain.
The scalar-control interpretation now preserves the clean larger-step/underfit
counterexample without claiming it causally rules out magnitude effects.
Directly rescaled displacements are explicitly artificial one-step diagnostics,
not implemented I7 policies. Falsifiers require a specified regime, meaningful
effect size and adequate precision; lack of significance is not equivalence.
Final main integration also marks the literature note's estimator-fidelity
comparison as already completed in iteration 005, leaving same-state learning
utility as the unrun question. It does not propose repeating that replay.

## Existing evidence checked, not reexecuted

Final independent review matched the report against iterations 004–006:

- Noisy endpoint versus validation-accuracy-selected contrasts, including the
  reversal between separate three-seed bundles.
- The rank-32 optimum denominator in the wider-storage fidelity comparison,
  and its distinction from total covariance trace and actual learning.
- The clean actual-step norm reversal and heterogeneous noisy lagging results.

These remain existing multi-seed/passive-replay evidence. No previous audit or
training run was repeated and no new performance claim was added.

## Authority and recovery

Repository knowledge lint passed: 14 pages, 11 indexed content pages. Final
local Markdown-link validation passed on 89 links; all nine unique bibliography
keys match the citation registry. The diff check passed. These are document
validation checks, not new scientific experiments.