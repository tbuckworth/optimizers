# Iteration 007 planning decision — no execution approval

**Historical proposal, deferred after independent challenge.** The current
decision is [mechanism first](challenge/decision.md); the contract below is
retained for provenance and must not be executed as the active plan.

Codex parent, Spectral Optimizer Investigation, 6 September 2026.
Workflow checkpoint: decomposition complete; independent design challenges next.
This is an exploratory follow-up proposal informed by completed outcomes, not a
retroactive amendment of iteration 006. No training or synthetic experiment ran
while preparing this decision.

## Decision and scope

The prerequisite [static compatibility check](source-compatibility.md) is now
complete. The common current/AdamW recipe, helper, data hashes, environment and
selectors match nominally. No material configuration difference was found that
requires restarting an earlier study. Cross-harness bitwise trajectory equality
has not been demonstrated; the scalar variants have different arithmetic.

## Changes to the worker recommendation

1. **Do not adopt “directionally stable” versus “mixed/near-zero” labels.** Three
   agreeing signs out of four and a 1-point median do not establish stability;
   small magnitude and inconsistent direction are also different observations.
   Report each signed contrast, mean, median, range and descriptive sample SD
   without a pass/fail effect-size label, equivalence assertion or powered claim.
   The worker's thresholds remain visible as an unaccepted proposal.
2. **Account for the independent rerun before launch.** The researcher workflow
   requires a separate fresh-seed check of the load-bearing comparison, not only
   reaggregation. Budget 16 primary cells plus four separate audit cells, and
   never pool the audit bundle into the primary four-bundle summary.
3. **Do not assume exact Adam state is available in old checkpoints.** Iteration
   006 retained model checkpoints, not the full optimizer/observer training state.
   Iteration 005's saved pre-Adam bundles contain raw/probe gradients and bases but
   only a parameter hash, not the parameters or Adam moments. Its complete raw
   streams could support a separately validated mathematical reconstruction;
   that is not automatically exact runtime state or a finite-loss replay.

These are design corrections, not defects in the completed iteration-006 audit.
The lambda estimates are subjective planning heuristics, not measured information
rates or evidence that a new experiment is certain to be useful.

## Candidate prospective contract for design review

- **Primary bundles:** 60007, 60008, 60009, 60010. **Separate audit bundle:**
  60011. These are new choices; no plan arrays or learning outcomes have been
  generated for this proposal. Specify streams exactly as
  `SeedSequence([20260906, bundle, stream])`, stream IDs 0–6 with the same roles
  as the prior independent rerun. Do not ambiguously inherit the different
  primary-study namespace `[20260906,3,seed,stream]`.
- **Recipe:** same MNIST data preparation, independently sampled 5,000/5,000/5,000
  train/validation/auxiliary splits, nominal replacement .9, 784–64–ReLU–10,
  2,000 AdamW steps, batch 64, LR .001, decay .01, 100-step raw warmup, native
  stable width-32 observer and unchanged four delivery definitions. Within each
  bundle, pair initialization, split, fixed labels and batches across all arms.
- **Primary quantities:** current-minus-AdamW, lagged-minus-current and
  restored-minus-current official-test accuracy at each arm's own earliest
  strict maximum-validation-accuracy checkpoint. All three contrasts are
  mandatory, including each of the four signed primary bundle differences.
- **Mandatory secondary outputs:** both validation selectors, step-100 and
  step-2000 endpoints, selected exposure, test CE, final clean/noisy training
  accuracy, delivered-gradient geometry and actual decay-subtracted displacement.
  Preserve zero-exposure selections and null masks; never drop an adverse cell.
- **Test boundary:** complete all 16 primary training/selection cells before
  opening official test data; evaluate four checkpoints per cell, 64 primary
  test evaluations. The independent four-cell audit has its own train-before-
  test boundary and 16 test evaluations. Freeze both primary and audit plans
  and scientific sources before inspecting primary outcomes.
- **Evidence scope:** adaptive exploratory reuse of MNIST and its official test
  set, conditional on the fixed recipe and bundle generator. A fresh random seed
  does not create an untouched test set. No confidence interval, p-value,
  sign-frequency generalization or new SOTA claim is proposed for four bundles.
- **Stopping:** exactly four primary bundles and the one predeclared audit
  bundle, then stop this MNIST seed-extension line regardless of the observed
  signs. Any gate failure preserves partial evidence and stops for review; no
  automatic retry, replacement seed, pooled old/new estimate or expanded sweep.

## Resource envelope and prerequisites

Observed iteration-006 rates suggest roughly five local RTX-3090 minutes for
16 cells and one minute for the separate four-cell rerun, excluding design and
audit work. These are estimates, not a launch allowance. A proposed cooperative
cap is 600 seconds primary and 180 seconds audit; a reviewed runtime-only pilot
must establish adequate headroom for the final implementation.

Before implementation, three independent assumption/mentor/pre-mortem reviews
must address whether this fixed extension has enough information value, whether
the repeated-test estimand is honest, whether the unpooled audit is correctly
costed, and whether saving a few full common-state anchors in a new study is
worth the added implementation/audit burden. The latter is an open design
choice, not permission to reconstruct or restart old trajectories.

The next checkpoint is a challenged protocol and analysis specification, then
implementation review and a separately recorded pilot/launch decision. No
pilot, dataset access, GPU allocation or new experiment is approved here.
