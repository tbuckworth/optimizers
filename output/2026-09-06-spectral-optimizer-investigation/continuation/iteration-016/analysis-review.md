# I16 scalar-analysis review

Independent prospective review, 7 September 2026. This review inspected only
the frozen analysis source, its synthetic tests, the I16 protocol, and the
acquisition schemas. It did not inspect I16 outcomes, load experiment tensors,
run model forwards, replay optimizer steps, or rerun an acquisition audit.

## Verdict

**PASS for the bounded post-acquisition audit and analysis.** The reviewed
analyzer implements the predeclared estimands and failure discipline without
survivor substitution. This verdict concerns source correctness and synthetic
coverage; it is not an audit result and says nothing about the eventual
scientific outcome.

Reviewed files:

- `analyse_scalar.py` — SHA256
  `9ae6c149c0133b8622b99da82d0768c7c6b4cae1d05b1519aeb2619a79fcd5bc`
- `test_analyse_scalar.py` — SHA256
  `8489f07b18016081a511ed9b643952d509257ffb8edcc2b19368b2542846a68b`

The independent CPU-only invocation of the synthetic analyzer suite passed
10/10 tests in 0.026 seconds with CUDA hidden and one numerical thread.
`git diff --check` was clean for both files.

## Substantive checks

The joint scalar selector requires all four scalar policies for a seed and
target. It searches the fixed 24 `(k,horizon)` candidates and orders ties by
the validation objective, earliest horizon, then smaller `k`. The spectral
comparator is independently selected over its six horizons. The resulting
eight target-by-selector-by-auxiliary-metric primaries retain all three paired
seed values; a missing required scalar member or seed makes the relevant
quantity unavailable rather than inducing survivor reselection. The analyzer
also retains all five logical policies, all scheduled curves, absolute changes
from the shared h100 state, fixed-k endpoints, and validation-selected
spectral-versus-each-k contrasts.

The provenance path now checks the accepted I14 raw references and six parent
state/evaluation seams, then binds each of the six I15 spectral comparators to
its original branch JSON, exact five-checkpoint projection, completion
manifest, six-point summary curve, and I14 h100 state/evaluation. The accepted
I15 step diagnostics used for dose geometry are validated by the frozen I15
validator. New complete terminal states and typed-failure states are CPU
tree-digested; intermediate model-only states remain hash-checked, matching the
protocol's stated boundary.

Numerical failures are represented as unavailable evidence, while structural
errors remain audit failures. Completion branch/update/failure/time totals are
recomputed from the branch index. The analyzer also checks the exact smoke
forecast formula, cooperative wall limits, the 4 GiB Torch GPU ceiling, the
3 GiB artifact cap, exact phase membership, source hashes, attempts, plans,
corruption counts, first-step common digests, and neutral parent seams.

Windowed geometry is computed within each branch and then equally across
seeds. It includes all four scalar policies and the I15 spectral comparator,
with data/total squared-energy sums, path-length sums, integrated and mean
raw-gradient dots, and action-complement fractions. Fields absent from the
legacy I14 raw schema are explicitly null rather than inferred. New scalar
component diagnostics and algebra/decay defects remain separate from these
dose summaries.

## Corrections made during review

The initial draft was not acceptable because it expected the obsolete
summary-row form of `spectral_references`, did not independently bind the
original I15 spectral artifacts, and unconditionally accessed new scalar dot
fields on legacy I14 rows. Those defects were corrected. Follow-up review also
closed exact checkpoint-to-branch projection and h100 seam checks, added the
I15 spectral policy to path geometry, retained sums as well as means for signed
dots, and integrated timing/resource accounting. Synthetic regressions cover
joint tie ordering, no-survivor missingness, direct-reference mutation,
legacy-null fields, typed evaluation failure, state digests, timing totals,
and resource limits.

## Interpretation boundary

The analysis can support a conditional comparison on this fixed panel. It
cannot turn I16 into a fresh-benchmark or equal whole-program tuning-budget
comparison: the I15 mean/projected comparator was chosen after I15 outcomes on
the same panel were known, whereas the I16 scalar grid was fixed prospectively;
within I16 the scalar family receives 24 validation candidates versus six for
the spectral arm. Both asymmetries must accompany any scientific conclusion.
